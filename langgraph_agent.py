# langgraph_agent.py
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from typing import TypedDict, Annotated
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# ESTADO
# ─────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# ─────────────────────────────────────────────────────────────────
# HELPER: convierte tools MCP → LangChain
# ─────────────────────────────────────────────────────────────────
def mcp_tools_to_langchain(mcp_tools_list, session):
    langchain_tools = []
    for mcp_tool in mcp_tools_list:
        tool_name = mcp_tool.name
        tool_description = mcp_tool.description

        def make_tool_func(name, sess):
            async def tool_func(**kwargs) -> str:
                result = await sess.call_tool(name, kwargs)
                return result.content[0].text
            tool_func.__name__ = name
            tool_func.__doc__ = tool_description
            return tool_func

        lc_tool = tool(make_tool_func(tool_name, session))
        langchain_tools.append(lc_tool)
    return langchain_tools

# ─────────────────────────────────────────────────────────────────
# GRAFO
# ─────────────────────────────────────────────────────────────────
def build_graph(llm_with_tools, session, checkpointer):

    async def call_agent(state: AgentState):
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    async def call_tools(state: AgentState):
        last_message = state["messages"][-1]
        tool_results = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f"  🔧 Tool: '{tool_name}' | args: {tool_args}")
            result = await session.call_tool(tool_name, tool_args)
            result_text = result.content[0].text
            print(f"  📦 Resultado: {result_text[:120]}...")
            tool_results.append(
                ToolMessage(content=result_text, tool_call_id=tool_call["id"])
            )
        return {"messages": tool_results}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_agent)
    graph.add_node("tools", call_tools)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    # checkpointer guarda el estado del grafo en SQLite después de cada paso
    # Esto es lo que permite que el agente recuerde conversaciones anteriores
    return graph.compile(checkpointer=checkpointer)

# ─────────────────────────────────────────────────────────────────
# LOOP DE CONVERSACIÓN
# ─────────────────────────────────────────────────────────────────
async def chat_loop(graph, thread_id: str):
    """
    Loop interactivo de conversación.
    thread_id identifica la sesión — el mismo thread_id recupera
    la memoria de esa conversación desde SQLite.
    Si usas un thread_id distinto, el agente empieza desde cero.
    """
    print(f"\n💬 Conversación iniciada (thread: {thread_id})")
    print("   Escribe 'salir' para terminar, 'nuevo' para nueva sesión\n")

    # config es el identificador de sesión para el checkpointer
    # LangGraph usa thread_id para saber qué historial cargar de SQLite
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("👤 Tú: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "salir":
            print("👋 ¡Hasta luego!")
            break
        if user_input.lower() == "nuevo":
            # Generar nuevo thread_id para empezar conversación fresca
            import uuid
            thread_id = str(uuid.uuid4())[:8]
            config = {"configurable": {"thread_id": thread_id}}
            print(f"🆕 Nueva sesión iniciada (thread: {thread_id})\n")
            continue

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        print(f"\n🤖 Agente: {result['messages'][-1].content}\n")

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            langchain_tools = mcp_tools_to_langchain(tools_result.tools, session)
            print(f"✅ Tools cargadas: {[t.name for t in langchain_tools]}")

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            llm_with_tools = llm.bind_tools(langchain_tools)

            # AsyncSqliteSaver guarda el historial en memoria.db
            # Cada vez que ejecutas el script, la memoria persiste en ese archivo
            async with AsyncSqliteSaver.from_conn_string("memoria.db") as checkpointer:
                graph = build_graph(llm_with_tools, session, checkpointer)

                # thread_id fijo = siempre retoma la misma conversación
                await chat_loop(graph, thread_id="sesion_principal")

if __name__ == "__main__":
    asyncio.run(main())