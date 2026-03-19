# client_agent.py
import asyncio
import json
import openai
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# ─────────────────────────────────────────────
# PASO A: Conectarse al servidor MCP
# ─────────────────────────────────────────────
# StdioServerParameters le dice al cliente CÓMO arrancar el servidor:
# en este caso, ejecutando "python server.py" como subproceso
server_params = StdioServerParameters(
    command="python",
    args=["server.py"],
)

async def main():
    # stdio_client arranca el servidor como subproceso y crea
    # los canales de comunicación (read/write) via stdio
    async with stdio_client(server_params) as (read, write):
        
        # ClientSession es la sesión MCP — maneja el protocolo
        # de handshake, listado de tools, y llamadas a tools
        async with ClientSession(read, write) as session:
            
            # Inicializar la sesión (handshake MCP)
            await session.initialize()
            
            # ─────────────────────────────────────────────
            # PASO B: Descubrir las tools disponibles
            # ─────────────────────────────────────────────
            tools_result = await session.list_tools()
            
            # Convertir las tools MCP al formato que espera la API de OpenAI
            # OpenAI espera: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
            openai_tools = []
            for tool in tools_result.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        # inputSchema describe los parámetros de la tool en JSON Schema
                        "parameters": tool.inputSchema,
                    }
                })
            
            print(f"✅ Tools disponibles: {[t['function']['name'] for t in openai_tools]}\n")
            
            # ─────────────────────────────────────────────
            # PASO C: Llamar al agente OpenAI (sin SDK de agentes)
            # ─────────────────────────────────────────────
            client = openai.AsyncOpenAI()
            
            messages = [
            {"role": "user", "content": "¿Cuánto es la raíz cuadrada de 256 y cuál es la fecha de hoy?"}
            ]       
            
            print(f"👤 Usuario: {messages[0]['content']}\n")
            
            # Primera llamada al modelo — puede devolver un tool_call
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",  # el modelo decide si usar una tool o no
            )
            
            assistant_message = response.choices[0].message
            
            # ─────────────────────────────────────────────
            # PASO D: Ejecutar la tool si el modelo la pidió
            # ─────────────────────────────────────────────
            if assistant_message.tool_calls:
                # Agregar la respuesta del asistente (con el tool_call) al historial
                messages.append(assistant_message)
                
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"🔧 El modelo quiere usar la tool: '{tool_name}' con args: {tool_args}")
                    
                    # Llamar a la tool VIA EL CLIENTE MCP — aquí está la magia
                    # session.call_tool se comunica con server.py y ejecuta la función
                    tool_result = await session.call_tool(tool_name, tool_args)
                    
                    result_text = tool_result.content[0].text
                    print(f"📅 Resultado de la tool: {result_text}\n")
                    
                    # Agregar el resultado de la tool al historial de mensajes
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    })
                
                # Segunda llamada al modelo — ahora con el resultado de la tool
                # El modelo genera la respuesta final en lenguaje natural
                final_response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                )
                
                print(f"🤖 Agente: {final_response.choices[0].message.content}")
            
            else:
                # El modelo respondió directamente sin usar tools
                print(f"🤖 Agente: {assistant_message.content}")

if __name__ == "__main__":
    asyncio.run(main())