# RAG de auditoría de cartera de crédito (GFNorte 2T26 vs Anexo 33, Criterio B-6)

Servicio web (FastAPI) que envuelve el RAG del cuaderno `Proyecto_Integrador_CarteraCredito_V3.ipynb`:
recuperación híbrida sobre PostgreSQL (Neon + pgvector), grafo de LangGraph con tres rutas
(abstención, solo evidencia y generado verificado) y telemetría en la tabla `ingesta_bitacora_v1`.

## Endpoints
| Método | Ruta | Para qué sirve |
|---|---|---|
| GET | `/` | Página con caja de consulta |
| GET | `/health` | El proceso está vivo (responde de inmediato) |
| GET | `/ready` | El RAG terminó de cargar (503 mientras carga) |
| POST | `/ask` | `{"pregunta": "...", "usar_modelo": false}` |
| GET | `/docs` | Documentación interactiva (OpenAPI) |

`usar_modelo: true` gasta crédito del curso y exige el encabezado `X-App-Token` (variable `APP_TOKEN`).

## Variables de entorno (nunca se suben a GitHub)
`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `BASE_URL`, `API_KEY`, `APP_TOKEN`, `MAX_LLAMADAS` (opcional, 20).
Local: copia `.env.example` como `.env`. En Render: pestaña *Environment*.

## Ejecutar en local
```
python -m venv .venv
source .venv/Scripts/activate        # Git Bash en Windows (Mac/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Render
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health Check Path: `/health`

## Estructura
```
app/main.py        rutas HTTP y arranque en segundo plano
app/schemas.py     contrato de entrada (Pydantic)
app/rag_core.py    núcleo del RAG: celdas 20 a 58 del cuaderno, sin cambios de lógica
app/static/        página de consulta
data/              PDF del Anexo 33 (se lee al arrancar para reconocer incisos)
scripts/smoke_test.py   prueba de humo del servicio desplegado
```
