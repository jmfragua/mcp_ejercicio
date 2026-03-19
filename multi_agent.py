# multi_agent.py
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from typing import TypedDict, Annotated, Literal
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# ESTADO COMPARTIDO
# ─────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str

# ─────────────────────────────────────────────────────────────────
# HELPER: crea un sub-agente con tools específicas
# ─────────────────────────────────────────────────────────────────
def make_agent_node(llm, session, tool_names: list[str], system_prompt: str):
    async def agent_node(state: AgentState):
        # Construir tools solo con las permitidas para este agente
        lc_tools = []
        for name in tool_names:
            def make_func(n):
                async def func(**kwargs) -> str:
                    result = await session.call_tool(n, kwargs)
                    return result.content[0].text
                func.__name__ = n
                func.__doc__ = f"Tool MCP: {n}"
                return func
            lc_tools.append(tool(make_func(name)))

        # ✅ agent_llm tiene las tools bindeadas — se usa en TODAS las llamadas
        agent_llm = llm.bind_tools(lc_tools)

        messages_with_system = [SystemMessage(content=system_prompt)] + state["messages"]
        response = await agent_llm.ainvoke(messages_with_system)

        result_messages = [response]
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                print(f"    🔧 [{tc['name']}] args: {tc['args']}")
                res = await session.call_tool(tc["name"], tc["args"])
                result_text = res.content[0].text
                print(f"    📦 Resultado: {result_text[:100]}...")
                result_messages.append(
                    ToolMessage(content=result_text, tool_call_id=tc["id"])
                )

            # ✅ FIX: segunda llamada también usa agent_llm (con tools bindeadas)
            # antes usaba llm sin tools — el modelo no sabía qué argumentos pasar
            final = await agent_llm.ainvoke(
                [SystemMessage(content=system_prompt)]
                + state["messages"]
                + result_messages
            )
            result_messages.append(final)

        return {"messages": result_messages, "next": "orquestador"}

    return agent_node

# ─────────────────────────────────────────────────────────────────
# ORQUESTADOR
# ─────────────────────────────────────────────────────────────────
def make_orchestrator(llm):
    async def orchestrator(state: AgentState):
        system = SystemMessage(content="""
Eres un orquestador de agentes. Tu única función es decidir a qué agente especializado
delegar la tarea del usuario. Responde SOLO con una de estas palabras exactas:

- investigador  → para búsquedas web, noticias, información actual
- calculador    → para matemáticas, fechas, cálculos
- archivero     → para leer o escribir archivos
- responder     → si puedes responder directamente sin tools

No expliques tu decisión. Solo escribe la palabra.
        """)

        messages = [system] + state["messages"]
        response = await llm.ainvoke(messages)

        decision = response.content.strip().lower()
        valid = ["investigador", "calculador", "archivero", "responder"]
        if decision not in valid:
            decision = "responder"

        print(f"\n  🧠 Orquestador delegando a: [{decision}]")
        return {"next": decision}

    return orchestrator

# ─────────────────────────────────────────────────────────────────
# NODO RESPUESTA DIRECTA
# ─────────────────────────────────────────────────────────────────
def make_direct_response(llm):
    async def direct_response(state: AgentState):
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response], "next": END}
    return direct_response

# ─────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────
def router(state: AgentState) -> Literal["investigador", "calculador", "archivero", "responder"]:
    return state["next"]

# ─────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DEL GRAFO
# ─────────────────────────────────────────────────────────────────
def build_multi_agent_graph(session, checkpointer):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    investigador = make_agent_node(
        llm, session,
        tool_names=["web_search"],
        system_prompt="Eres un agente investigador experto en búsquedas web. Busca información actualizada y presenta resultados claros y concisos."
    )

    calculador = make_agent_node(
        llm, session,
        tool_names=["calculate", "get_current_date"],
        system_prompt="Eres un agente calculador. Resuelves operaciones matemáticas y consultas sobre fechas con precisión."
    )

    archivero = make_agent_node(
        llm, session,
        tool_names=["read_file", "write_file"],
        system_prompt="Eres un agente archivero. Gestionas archivos: lees y escribes contenido de forma organizada."
    )

    orquestador = make_orchestrator(llm)
    respuesta_directa = make_direct_response(llm)

    graph = StateGraph(AgentState)
    graph.add_node("orquestador", orquestador)
    graph.add_node("investigador", investigador)
    graph.add_node("calculador",   calculador)
    graph.add_node("archivero",    archivero)
    graph.add_node("responder",    respuesta_directa)

    graph.set_entry_point("orquestador")

    graph.add_conditional_edges(
        "orquestador",
        router,
        {
            "investigador": "investigador",
            "calculador":   "calculador",
            "archivero":    "archivero",
            "responder":    "responder",
        }
    )

    graph.add_edge("investigador", END)
    graph.add_edge("calculador",   END)
    graph.add_edge("archivero",    END)
    graph.add_edge("responder",    END)

    return graph.compile(checkpointer=checkpointer)

# ─────────────────────────────────────────────────────────────────
# LOOP DE CONVERSACIÓN
# ─────────────────────────────────────────────────────────────────
async def chat_loop(graph):
    print("\n🤖 Sistema Multi-Agente iniciado")
    print("   Agentes: investigador | calculador | archivero")
    print("   Escribe 'salir' para terminar\n")

    config = {"configurable": {"thread_id": "multi_agent_sesion"}}

    while True:
        user_input = input("👤 Tú: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "salir":
            print("👋 ¡Hasta luego!")
            break

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        print(f"\n🤖 Respuesta: {result['messages'][-1].content}\n")

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
            print("✅ Servidor MCP conectado\n")

            async with AsyncSqliteSaver.from_conn_string("memoria_multi.db") as checkpointer:
                graph = build_multi_agent_graph(session, checkpointer)
                await chat_loop(graph)

if __name__ == "__main__":
    asyncio.run(main())