# server_http.py
from mcp.server.fastmcp import FastMCP
from datetime import datetime
from ddgs import DDGS
import math
import os
import uvicorn

PORT = int(os.environ.get("PORT", 8000))

mcp = FastMCP("multi_tool_server_http")

# ── Tool 1: Fecha actual ──────────────────────────────────────────
@mcp.tool()
def get_current_date() -> str:
    """Devuelve la fecha y hora actual del sistema."""
    now = datetime.now()
    return now.strftime("Hoy es %A, %d de %B de %Y. Hora: %H:%M:%S")

# ── Tool 2: Calculadora ───────────────────────────────────────────
@mcp.tool()
def calculate(expression: str) -> str:
    """
    Evalúa una expresión matemática y devuelve el resultado.
    Soporta: +, -, *, /, **, sqrt(), sin(), cos(), log(), pi, e
    """
    safe_namespace = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "log10": math.log10,
        "abs": abs, "round": round, "pi": math.pi, "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, safe_namespace)
        return f"Resultado de '{expression}' = {result}"
    except ZeroDivisionError:
        return "Error: división entre cero"
    except Exception as ex:
        return f"Error al evaluar '{expression}': {str(ex)}"

# ── Tool 3: Búsqueda web ──────────────────────────────────────────
@mcp.tool()
def web_search(query: str, max_results: int = 3) -> str:
    """Busca información actualizada en la web usando DuckDuckGo."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"📰 {r['title']}\n   🔗 {r['href']}\n   {r['body']}\n")
        return "\n".join(results) if results else "No se encontraron resultados."
    except Exception as ex:
        return f"Error en la búsqueda: {str(ex)}"

# ── Tool 4: Leer archivo ──────────────────────────────────────────
@mcp.tool()
def read_file(filename: str) -> str:
    """Lee el contenido de un archivo de texto."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        return f"Contenido de '{filename}':\n{content}" if content.strip() else f"'{filename}' está vacío."
    except FileNotFoundError:
        return f"Error: '{filename}' no existe."
    except Exception as ex:
        return f"Error al leer '{filename}': {str(ex)}"

# ── Tool 5: Escribir archivo ──────────────────────────────────────
@mcp.tool()
def write_file(filename: str, content: str, mode: str = "w") -> str:
    """Escribe o agrega contenido en un archivo. mode: 'w' sobreescribir, 'a' agregar."""
    try:
        with open(filename, mode, encoding="utf-8") as f:
            f.write(content + "\n")
        return f"Archivo '{filename}' {'sobreescrito' if mode == 'w' else 'actualizado'} correctamente."
    except Exception as ex:
        return f"Error al escribir '{filename}': {str(ex)}"

if __name__ == "__main__":
    print(f"🚀 Servidor MCP corriendo en http://0.0.0.0:{PORT}/sse")
    # ✅ Usamos uvicorn directamente con la app SSE de FastMCP
    # Esto garantiza que Railway detecte correctamente el puerto
    app = mcp.sse_app()
    uvicorn.run(app, host="0.0.0.0", port=PORT)