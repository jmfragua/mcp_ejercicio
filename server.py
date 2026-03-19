# server.py
from mcp.server.fastmcp import FastMCP
from datetime import datetime
from ddgs import DDGS
import math

mcp = FastMCP("multi_tool_server")

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
    Ejemplos: '2 + 2', 'sqrt(144)', 'pi * 5**2'
    """
    safe_namespace = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e,
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
    """
    Busca información actualizada en la web usando DuckDuckGo.
    Útil para noticias recientes, precios, eventos actuales.
    Parámetros:
        query: término de búsqueda
        max_results: número de resultados (default: 3, max recomendado: 5)
    """
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"📰 {r['title']}\n"
                    f"   🔗 {r['href']}\n"
                    f"   {r['body']}\n"
                )
        if not results:
            return "No se encontraron resultados para esa búsqueda."
        return "\n".join(results)
    except Exception as ex:
        return f"Error en la búsqueda: {str(ex)}"

# ── Tool 4: Leer archivo ──────────────────────────────────────────
@mcp.tool()
def read_file(filename: str) -> str:
    """
    Lee el contenido de un archivo de texto.
    El archivo debe estar en la carpeta del proyecto.
    Parámetros:
        filename: nombre del archivo (ej: 'notas.txt')
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            return f"El archivo '{filename}' existe pero está vacío."
        return f"Contenido de '{filename}':\n{content}"
    except FileNotFoundError:
        return f"Error: el archivo '{filename}' no existe."
    except Exception as ex:
        return f"Error al leer '{filename}': {str(ex)}"

# ── Tool 5: Escribir archivo ──────────────────────────────────────
@mcp.tool()
def write_file(filename: str, content: str, mode: str = "w") -> str:
    """
    Escribe o agrega contenido en un archivo de texto.
    Parámetros:
        filename: nombre del archivo (ej: 'notas.txt')
        content: texto a escribir
        mode: 'w' para sobreescribir, 'a' para agregar al final
    """
    try:
        with open(filename, mode, encoding="utf-8") as f:
            f.write(content + "\n")
        action = "sobreescrito" if mode == "w" else "actualizado"
        return f"Archivo '{filename}' {action} correctamente."
    except Exception as ex:
        return f"Error al escribir '{filename}': {str(ex)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")