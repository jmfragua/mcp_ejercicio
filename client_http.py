# client_http.py
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from typing import TypedDict, Annotated, Literal, Any
from pydantic import BaseModel, Field, create_model
from mcp import ClientSession
from mcp.client.sse import sse_client

load_dotenv()

# ─────────────────────────────────────────────────────────────────
# ESTADO
# ─────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str

# ─────────────────────────────────────────────────────────────────
# HELPER: convierte tools MCP → StructuredTool de LangChain
# StructuredTool permite definir el esquema explícitamente
# así el modelo sabe exactamente qué argumentos pasar
# ─────────────────────────────────────────────────────────────────
def mcp_to_structured_tools(mcp_tools_list, session, allowed_names=None):
    langchain_tools = []

    for mcp_tool in mcp_tools_list:
        if allowed_names and mcp_tool.name not in allowed_names:
            continue

        tool_name = mcp_tool.name
        tool_description = mcp_tool.description or f"Tool: {tool_name}"
        input_schema = mcp_tool.inputSchema or {}
        properties = input_schema.get("properties", {})
        required_fields = input_schema.get("required", [])

        # Construir modelo Pydantic dinámicamente desde el JSON Schema de la tool MCP
        # Pydantic le dice al modelo exactamente qué campos son requeridos y de qué tipo
        field_definitions = {}
        for field_name, field_info in properties.items():
            field_type = str  # default
            if field_info.get("type") == "integer":
                field_type = int
            elif field_info.get("type") == "number":
                field_type = float
            elif field_info.get("type") == "boolean":
                field_type = bool

            description = field_info.get("description", field_name)

            if field_name in required_fields:
                # Campo requerido — sin valor default
                field_definitions[field_name] = (field_type, Field(description=description))
            else:
                # Campo opcional — con valor default
                default_val = field_info.get("default", None)
                field_definitions[field_name] = (field_type, Field(default=default_val, description=description))

        # create_model genera una clase Pydantic en tiempo de ejecución
        ArgsModel = create_model(f"{tool_name}_args", **field_definitions)

        # Crear la función que ejecuta la tool via MCP
        def make_tool_func(name, sess):
            async def tool_func(**kwargs) -> str:
                result = await sess.call_tool(name, kwargs)
                return result.content[0].text
            tool_func.__name__ = name
            return tool_func

        # StructuredTool combina la función con el esquema Pydantic
        # Esto garantiza que el modelo siempre pase los argumentos correctos
        structured_tool = StructuredTool(
            name=tool_name,
            description=tool_description,
            args_schema=ArgsModel,
            coroutine=make_tool_func(tool_name, session),
        )
        langchain_tools.append(structured_tool)

    return langchain_tools

# ─────────────────────────────────────────────────────────────────
# AGENTES ESPECIALIZADOS
# ─────────────────────────────────────────────────────────────────
def make_agent_node(llm, session, mcp_tools_list, tool_names: list[str], system_prompt: str):
    async def agent_node(state: AgentState):
        # Construir StructuredTools solo con las permitidas para este agente
        lc_tools = mcp_to_structured_tools(mcp_tools_list, session, allowed_names=tool_names)
        agent_llm = llm.bind_tools(lc_tools)

        messages_with_system = [SystemMessage(content=system_prompt)] + state["messages"]
        response = await agent_llm.ainvoke(messages_with_system)

        result_messages = [response]
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                print(f"    🔧 [{tc['name']}] args: {tc['args']}")
                try:
                    res = await session.call_tool(tc["name"], tc["args"])
                    result_text = res.content[0].text
                except Exception as e:
                    result_text = f"Error ejecutando tool: {str(e)}"

                print(f"    📦 Resultado: {result_text[:100]}...")
                result_messages.append(
                    ToolMessage(content=result_text, tool_call_id=tc["id"])
                )

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
Eres un orquestador de agentes. Responde SOLO con una de estas palabras exactas:
- investigador  → búsquedas web, noticias, información actual
- calculador    → matemáticas, fechas, cálculos
- archivero     → leer o escribir archivos
- responder     → respuesta directa sin tools
No expliques tu decisión. Solo escribe la palabra.
        """)
        response = await llm.ainvoke([system] + state["messages"])
        decision = response.content.strip().lower()
        if decision not in ["investigador", "calculador", "archivero", "responder"]:
            decision = "responder"
        print(f"\n  🧠 Orquestador delegando a: [{decision}]")
        return {"next": decision}
    return orchestrator

# ─────────────────────────────────────────────────────────────────
# RESPUESTA DIRECTA
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
# GRAFO
# ─────────────────────────────────────────────────────────────────
def build_graph(session, mcp_tools_list, checkpointer):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    graph = StateGraph(AgentState)
    graph.add_node("orquestador", make_orchestrator(llm))
    graph.add_node("investigador", make_agent_node(
        llm, session, mcp_tools_list, ["web_search"],
        "Eres un agente investigador. Busca información actualizada y presenta resultados claros."))
    graph.add_node("calculador", make_agent_node(
        llm, session, mcp_tools_list, ["calculate", "get_current_date"],
        "Eres un agente calculador. Resuelves matemáticas y consultas de fechas con precisión."))
    graph.add_node("archivero", make_agent_node(
        llm, session, mcp_tools_list, ["read_file", "write_file"],
        "Eres un agente archivero. Gestionas archivos de forma organizada."))
    graph.add_node("responder", make_direct_response(llm))

    graph.set_entry_point("orquestador")
    graph.add_conditional_edges("orquestador", router, {
        "investigador": "investigador",
        "calculador":   "calculador",
        "archivero":    "archivero",
        "responder":    "responder",
    })
    graph.add_edge("investigador", END)
    graph.add_edge("calculador",   END)
    graph.add_edge("archivero",    END)
    graph.add_edge("responder",    END)

    return graph.compile(checkpointer=checkpointer)

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
async def main():
    server_url = "http://localhost:8000/sse"
    print(f"🔌 Conectando al servidor MCP en {server_url}...")

    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Obtener tools del servidor y guardarlas para construir StructuredTools
            tools_result = await session.list_tools()
            mcp_tools_list = tools_result.tools
            print(f"✅ Tools disponibles: {[t.name for t in mcp_tools_list]}\n")

            async with AsyncSqliteSaver.from_conn_string("memoria_http.db") as checkpointer:
                graph = build_graph(session, mcp_tools_list, checkpointer)

                print("🤖 Sistema Multi-Agente HTTP iniciado")
                print("   Escribe 'salir' para terminar\n")

                config = {"configurable": {"thread_id": "sesion_http"}}

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

if __name__ == "__main__":
    asyncio.run(main())