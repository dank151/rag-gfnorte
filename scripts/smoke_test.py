"""Prueba de humo del servicio desplegado. Uso:  python scripts/smoke_test.py https://TU-SERVICIO.onrender.com"""
import sys
import time

import requests

URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
ok = True


def paso(nombre, condicion, detalle=""):
    global ok
    ok = ok and condicion
    print(("  OK   " if condicion else "  FALLA"), nombre, detalle)


print("Servicio:", URL, "(si estaba dormido, la primera respuesta puede tardar cerca de un minuto)")
r = requests.get(f"{URL}/health", timeout=120)
paso("GET /health devuelve 200 y status ok", r.status_code == 200 and r.json().get("status") == "ok")

for _ in range(40):                                 # espera hasta 3 minutos a que el RAG termine de cargar
    r = requests.get(f"{URL}/ready", timeout=60)
    if r.status_code != 503:
        break
    time.sleep(5)
paso("GET /ready indica que el RAG está listo", r.status_code == 200, r.text[:120])

r = requests.post(f"{URL}/ask", json={"pregunta": "Receta tradicional del mole poblano con chocolate y chiles"}, timeout=120)
d = r.json() if r.ok else {}
paso("Pregunta fuera de tema → abstención, sin modelo", r.ok and d.get("ruta") == "abstencion" and d.get("modelo") is None, f"(ruta: {d.get('ruta')})")

r = requests.post(f"{URL}/ask", json={"pregunta": "¿Cuál es la cartera Etapa 3 del sector Servicios?"}, timeout=120)
d = r.json() if r.ok else {}
paso("Pregunta de cifra → solo evidencia con 4,824.00", r.ok and d.get("ruta") == "solo_evidencia" and "4,824.00" in d.get("respuesta", ""), f"(ruta: {d.get('ruta')})")

r = requests.post(f"{URL}/ask", json={"pregunta": "¿Cuál es la cartera Etapa 3 del sector Servicios?", "usar_modelo": True}, timeout=120)
paso("Usar el modelo sin clave de la aplicación → 403", r.status_code == 403, f"(HTTP {r.status_code})")

r = requests.post(f"{URL}/ask", json={"pregunta": "a"}, timeout=60)
paso("Pregunta inválida → 422 (contrato validado)", r.status_code == 422, f"(HTTP {r.status_code})")

print("\nRESULTADO:", "todo en orden" if ok else "hay fallas: revisa los logs en Render")
sys.exit(0 if ok else 1)
