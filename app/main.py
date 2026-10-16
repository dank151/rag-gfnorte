"""Servicio web del RAG (FastAPI). Envuelve el grafo de LangGraph del cuaderno sin modificar su lógica."""
import hmac
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from .schemas import AskRequest

load_dotenv()   # solo tiene efecto en tu computadora (lee el archivo .env); en Render no existe y se usan las variables del panel
log = logging.getLogger("rag")

ESTADO = {"listo": False, "error": None, "segundos_carga": None}
NUCLEO = {}     # aquí queda el módulo rag_core cuando termina de cargar


def _cargar_rag():
    """Carga el RAG en segundo plano para que el servicio responda /health de inmediato."""
    t0 = time.perf_counter()
    try:
        from . import rag_core
        rag_core.iniciar_modelo(int(os.environ.get("MAX_LLAMADAS", "20")))
        NUCLEO["rag"] = rag_core
        ESTADO["listo"] = True
    except Exception as exc:                        # el detalle completo va solo a los logs, no al cliente
        log.exception("No se pudo cargar el RAG")
        ESTADO["error"] = type(exc).__name__
    ESTADO["segundos_carga"] = round(time.perf_counter() - t0, 1)


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_cargar_rag, daemon=True).start()
    yield


app = FastAPI(title="RAG de auditoría de cartera · GFNorte 2T26 vs Anexo 33", version="1.0.0", lifespan=lifespan)
PAGINA = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def inicio():
    return FileResponse(PAGINA)


@app.get("/health")
def health():
    """Disponibilidad: el proceso está vivo. No consulta la base ni el modelo."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Preparación: el RAG ya cargó (corpus, índices, grafo). Responde 503 mientras carga."""
    if ESTADO["error"]:
        raise HTTPException(status_code=500, detail=f"El RAG no pudo iniciar ({ESTADO['error']}). Revisa los logs.")
    if not ESTADO["listo"]:
        raise HTTPException(status_code=503, detail="El RAG está cargando; reintenta en unos segundos.")
    return {"status": "ready", "segundos_carga": ESTADO["segundos_carga"]}


@app.post("/ask")
def ask(req: AskRequest, x_app_token: str | None = Header(default=None)):
    if ESTADO["error"]:
        raise HTTPException(status_code=500, detail="El RAG no pudo iniciar. Revisa los logs.")
    if not ESTADO["listo"]:
        raise HTTPException(status_code=503, detail="El RAG está cargando; reintenta en unos segundos.")
    if req.usar_modelo:                             # el modelo gasta crédito del curso: solo con clave de la aplicación
        esperado = os.environ.get("APP_TOKEN")
        if not esperado or not x_app_token or not hmac.compare_digest(x_app_token, esperado):
            raise HTTPException(status_code=403, detail="Para usar el modelo se requiere el encabezado X-App-Token correcto.")
    try:
        return NUCLEO["rag"].responder(req.pregunta, usar_modelo=req.usar_modelo)
    except Exception:
        log.exception("Error al responder")
        raise HTTPException(status_code=500, detail="Error interno al procesar la pregunta.")
