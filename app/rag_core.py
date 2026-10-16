# -*- coding: utf-8 -*-
"""
rag_core.py · Núcleo del RAG (generado a partir del cuaderno Proyecto_Integrador_CarteraCredito_V3.ipynb).

Es el mismo código de las celdas 20 a 58 del cuaderno, en el mismo orden, en un solo módulo.
Cambios respecto del cuaderno (solo en esta cabecera y al final):
  - La configuración se lee de variables de entorno (en Render) en lugar de credenciales.env en Drive.
  - Se omiten las celdas de Colab (montaje de Drive, instalación con pip, widgets) y las de construcción de la base.
"""
import os

# ---- Configuración (reemplaza las celdas 5 y 7 del cuaderno) ---------------------------------------------------
# Carpeta de archivos de trabajo: el PDF del Anexo 33 y la caché de respuestas. En Render es parte del repositorio.
RUTA_PROYECTO = os.environ.get("DATA_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
BASE_URL = os.environ.get("BASE_URL", "https://api.artesaniadigital.org")
API_KEY = os.environ.get("API_KEY")

# Falla rápido: si falta una variable, el servicio lo dice por su nombre (nunca imprime los valores).
_faltantes = [n for n, v in (("DB_HOST", DB_HOST), ("DB_NAME", DB_NAME), ("DB_USER", DB_USER),
                              ("DB_PASSWORD", DB_PASSWORD), ("API_KEY", API_KEY)) if not v]
if _faltantes:
    raise ValueError("Faltan variables de entorno: " + ", ".join(_faltantes))

import psycopg2


def conectar():
    return psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER,
                            password=DB_PASSWORD, sslmode="require", connect_timeout=15)


ESQUEMA = "daniel_rojas_private"
SUFIJO = "_v1"
ESQUEMA_ORIGEN = ESQUEMA
DOCUMENT_ID = "GFNORTE_2T26_RIESGOS"
MODELO = "text-embedding-3-small"
DIMENSION = 1536
CHUNK_DESDE, CHUNK_HASTA = 0, 17
PAG_IMPRESA_MAX = 16

print("rag_core: configuración cargada (host y clave no se imprimen).")



# ====================================================================================================
# >>> CELDA 20 DEL CUADERNO
# ====================================================================================================
# Celda 20 · código
# ==============================================================================
# PARTE 4 · PARÁMETROS DEL SISTEMA DE RECUPERACIÓN
# Dependencias, perillas del RAG y utilidades de acceso a la base.
# ==============================================================================
# Librerías nuevas de esta parte:
#   rank_bm25 = puntaje de relevancia entre una pregunta y un texto | nltk = herramientas de lenguaje (reduce palabras a su raíz)
#   pdfplumber = lee el texto del PDF del Anexo 33 para reconocer sus incisos
# re = expresiones regulares (patrones de texto); json = datos estructurados; math = matemáticas; unicodedata = manejo de acentos.
import os, re, json, math, time, hashlib, unicodedata
from collections import defaultdict
from datetime import datetime
from psycopg2.extras import Json
from rank_bm25 import BM25Okapi
from nltk.stem.snowball import SnowballStemmer

try:
    # Comprobación: si faltan variables de las Partes 1 a 3, se avisa con un mensaje claro en lugar de fallar con un error técnico.
    (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, ESQUEMA, RUTA_PROYECTO, conectar)
except NameError as e:
    raise RuntimeError(f"Falta {e}: ejecuta antes las Partes 1 a 3.")

# Parámetros del proyecto definidos en la Parte 1 (se usan valores por defecto si faltaran).
# Parámetros del proyecto: si ya existen se respetan; si no, se usan estos valores por defecto.
DOCUMENT_ID = globals().get("DOCUMENT_ID", "GFNORTE_2T26_RIESGOS")
MODELO = globals().get("MODELO", "text-embedding-3-small")
DIMENSION = globals().get("DIMENSION", 1536)
# El texto de cada fragmento trae una frase fija que apunta a la tabla SQL. Se elimina antes de buscar para que no 'contamine' el ranking.
if "PATRON_PUNTERO" not in globals():
    PATRON_PUNTERO = re.compile(r"^Para consultas cuantitativas exactas: consultar la tabla relacional [^\n]*\n?", re.M)
if "texto_para_embedding" not in globals():
    def texto_para_embedding(chunk_text):
        """Texto del fragmento sin la frase fija del puntero SQL (misma regla con la que se calcularon los vectores)."""
        return re.sub(r"\n{3,}", "\n\n", PATRON_PUNTERO.sub("", chunk_text)).strip()
if "vector_literal" not in globals():
    # Convierte la lista de 1,536 números al formato de texto que entiende PostgreSQL: [0.1,0.2,...].
    def vector_literal(vec):
        return "[" + ",".join(format(x, ".9g") for x in vec) + "]"

# ---- Parámetros del sistema ------------------------------------------------------------------------
# ---- Perillas del sistema (aquí se ajusta el comportamiento sin tocar la lógica) ----
ALCANCE_NOMBRE = "Versión de entrega: pp. 7 a 16 (fragmentos 0 a 17)"
# False = no se piden vectores nuevos al servicio (así no se gasta crédito). Los vectores de las consultas solo salen de la caché.
USAR_SERVICIO_EMBEDDINGS = False    # se mantiene en False: esta versión no genera embeddings (el crédito es solo para consultas)
# False = la pregunta se usa tal cual se escribe, sin agregarle palabras 'de ayuda'.
EXPANSION_LEXICA = False            # el Anexo 33 se escribe tal cual (decisión del proyecto); True activa EXPANSIONES_MINIMAS
RUTA_ANEXO33 = os.path.join(RUTA_PROYECTO, "Anexo_33_Cartera_de_Credito.pdf")     # solo para reconocer incisos y precalentar la caché
# TOP_LEXICO/TOP_DENSO = cuántos candidatos trae cada canal; TOP_FINAL = cuántos pasan a la respuesta; RRF_K = constante de la fusión (60).
TOP_LEXICO, TOP_DENSO, TOP_FINAL, RRF_K = 20, 5, 5, 60
# Cada fragmento se lee en 'ventanas' de 160 palabras que avanzan de 110 en 110 (se traslapan para no cortar una idea a la mitad).
VENTANA_PALABRAS, PASO_VENTANA = 160, 110
# Qué tan parecido debe ser lo escrito a un inciso del Anexo 33 para reconocerlo (0 a 1; 0.60 = 60% de términos en común).
SIM_MINIMA_INCISO = 0.60            # similitud (Jaccard de términos) para reconocer que lo escrito es un inciso del Anexo 33
CORRIDA_F4 = datetime.now().strftime("f4-%Y%m%d-%H%M%S")

# Utilidad: huella SHA-256 de un texto.
sha = lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()
# Estimador rápido de tokens: en español, 1 token ≈ 3.5 caracteres.
tokens_est = lambda t: math.ceil(len(t) / 3.5)


# Escribe un renglón en la tabla bitácora (qué etapa corrió y con qué resultado).
def bitacora_f4(conn, etapa, **detalle):
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO {ESQUEMA}.ingesta_bitacora_v1 (corrida, etapa, detalle) VALUES (%s, %s, %s)",
                    (CORRIDA_F4, etapa, Json(detalle, dumps=lambda o: json.dumps(o, default=str))))


# Utilidad: ejecuta una consulta SQL y devuelve una sola fila.
def sql_uno(sql, params=()):
    """Ejecuta una consulta y devuelve la primera fila (cierra la conexión)."""
    conn = conectar()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            fila = cur.fetchone()
        conn.rollback()
        return fila
    finally:
        conn.close()


# ====================================================================================================
# >>> CELDA 22 DEL CUADERNO
# ====================================================================================================
# Celda 22 · código
# ==============================================================================
# ALCANCE COMO DATO
# Define qué fragmentos forman el corpus activo (0 a 17) y la vista v_observaciones_alcance (las cifras de esos fragmentos).
# ==============================================================================
# El alcance no está 'escondido' en el código: es un dato en una tabla. Cambiar el tramo es actualizar una fila.
DDL_ALCANCE = f"""
CREATE TABLE IF NOT EXISTS {ESQUEMA}.alcance_corpus_v1 (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES {ESQUEMA}.documents_v1(id) ON DELETE RESTRICT,
    nombre TEXT NOT NULL,
    chunk_index_desde INT NOT NULL CHECK (chunk_index_desde >= 0),
    chunk_index_hasta INT NOT NULL,
    activo BOOLEAN NOT NULL DEFAULT TRUE,
    nota TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_alcance_rango CHECK (chunk_index_hasta >= chunk_index_desde));
-- Solo puede haber un alcance activo por documento.
CREATE UNIQUE INDEX IF NOT EXISTS uq_alcance_activo_v1 ON {ESQUEMA}.alcance_corpus_v1 (document_id) WHERE activo;
-- Una vista es una 'consulta guardada': se comporta como tabla pero solo muestra las cifras de los fragmentos dentro del alcance.
CREATE OR REPLACE VIEW {ESQUEMA}.v_observaciones_alcance_v1 AS
  SELECT o.* FROM {ESQUEMA}.observaciones_cuantitativas_v1 o
  JOIN {ESQUEMA}.banorte_credito_chunks_v1 c ON c.id = o.chunk_id
  JOIN {ESQUEMA}.alcance_corpus_v1 a ON a.document_id = c.document_id AND a.activo
   AND c.chunk_index BETWEEN a.chunk_index_desde AND a.chunk_index_hasta;
-- Limpieza: la versión anterior creaba también esta vista de fragmentos; ya no se usa (si no existe, no pasa nada).
DROP VIEW IF EXISTS {ESQUEMA}.v_chunks_alcance_v1;
-- Caché de vectores de preguntas. En esta versión no se generan vectores nuevos, así que queda vacía y el canal denso permanece inactivo.
CREATE TABLE IF NOT EXISTS {ESQUEMA}.consulta_embeddings_cache_v1 (
    huella TEXT PRIMARY KEY,
    texto TEXT NOT NULL,
    modelo TEXT NOT NULL,
    embedding VECTOR({DIMENSION}) NOT NULL,
    tokens INT,
    origen TEXT NOT NULL DEFAULT 'servicio',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW());"""


conn = conectar()
try:
    with conn.cursor() as cur:
        cur.execute(DDL_ALCANCE)
        cur.execute(f"SELECT id, chunk_index_desde, chunk_index_hasta FROM {ESQUEMA}.alcance_corpus_v1 WHERE document_id = %s AND activo", (DOCUMENT_ID,))
        # ¿Ya hay un alcance activo registrado? Si no, se crea uno automáticamente con el tramo contiguo de fragmentos con vector.
        fila = cur.fetchone()
        if fila is None:
            cur.execute(f"SELECT chunk_index FROM {ESQUEMA}.banorte_credito_chunks_v1 WHERE document_id = %s AND embedding IS NOT NULL ORDER BY chunk_index", (DOCUMENT_ID,))
            con_vector = [r[0] for r in cur.fetchall()]
            hasta = -1
            # Se avanza mientras los fragmentos con vector sean consecutivos desde el 0 (0, 1, 2...). Se detiene al primer hueco.
            for i in con_vector:                       # prefijo contiguo desde 0 de fragmentos con vector
                if i == hasta + 1:
                    hasta = i
                else:
                    break
            assert hasta >= 0, "El fragmento 0 no tiene vector: no se puede definir un alcance contiguo."
            cur.execute(f"""INSERT INTO {ESQUEMA}.alcance_corpus_v1 (document_id, nombre, chunk_index_desde, chunk_index_hasta, nota)
                            VALUES (%s, %s, 0, %s, 'Fragmentos 0 a 17 con vector (páginas impresas 7 a 16)')""", (DOCUMENT_ID, ALCANCE_NOMBRE, hasta))
            creado = True
        else:
            creado = False
        # Resumen y pruebas de consistencia: cuántos fragmentos y cifras hay dentro/fuera del alcance, y si alguna cifra quedó sin fragmento ('huérfana').
        # Resumen y comprobaciones de consistencia
        cur.execute(f"""SELECT COUNT(*), MIN(c.chunk_index), MAX(c.chunk_index), MIN(c.page_printed_start), MAX(c.page_printed_end), COUNT(*) FILTER (WHERE c.embedding IS NULL)
                       FROM {ESQUEMA}.banorte_credito_chunks_v1 c
                       JOIN {ESQUEMA}.alcance_corpus_v1 a ON a.document_id = c.document_id AND a.activo
                        AND c.chunk_index BETWEEN a.chunk_index_desde AND a.chunk_index_hasta""")
        n_alc, i0, i1, p0, p1, sin_vec = cur.fetchone()
        cur.execute(f"SELECT (SELECT COUNT(*) FROM {ESQUEMA}.v_observaciones_alcance_v1), (SELECT COUNT(*) FROM {ESQUEMA}.observaciones_cuantitativas_v1 WHERE document_id = %s), (SELECT COUNT(*) FROM {ESQUEMA}.observaciones_cuantitativas_v1 WHERE chunk_id IS NULL)", (DOCUMENT_ID,))
        n_oa, n_ot, sin_enlace = cur.fetchone()
        cur.execute(f"""SELECT COUNT(*) FROM {ESQUEMA}.v_observaciones_alcance_v1 o JOIN {ESQUEMA}.banorte_credito_chunks_v1 c ON c.id = o.chunk_id WHERE c.embedding IS NULL""")
        huerfanas = cur.fetchone()[0]
        cur.execute(f"SELECT tabla_origen, COUNT(*) FROM {ESQUEMA}.v_observaciones_alcance_v1 GROUP BY 1 ORDER BY 1")
        dentro = cur.fetchall()
        cur.execute(f"""SELECT tabla_origen, COUNT(*) FROM {ESQUEMA}.observaciones_cuantitativas_v1 o WHERE document_id = %s
                        AND NOT EXISTS (SELECT 1 FROM {ESQUEMA}.v_observaciones_alcance_v1 a WHERE a.id = o.id) GROUP BY 1 ORDER BY 1""", (DOCUMENT_ID,))
        fuera = cur.fetchall()
    bitacora_f4(conn, "2_alcance", fragmentos=n_alc, desde=i0, hasta=i1, observaciones_alcance=n_oa, observaciones_total=n_ot, creado=creado)
    conn.commit()
finally:
    conn.close()
# Se guarda un resumen del alcance en una variable; otras partes del sistema lo consultan (por ejemplo, para citar 'págs. 7 a 16').
ALCANCE = dict(fragmentos=n_alc, desde=i0, hasta=i1, pag_ini=p0, pag_fin=p1, obs_alcance=n_oa, obs_total=n_ot,
               sin_vector_en_alcance=sin_vec, huerfanas=huerfanas, sin_enlace=sin_enlace,
               grupos_dentro=[t for t, _ in dentro], grupos_fuera=[t for t, _ in fuera])


# ====================================================================================================
# >>> CELDA 24 DEL CUADERNO
# ====================================================================================================
# Celda 24 · código
# ==============================================================================
# CORPUS DE RECUPERACIÓN
# Carga los fragmentos del alcance, sus ventanas de lectura y la tokenización con raíces de palabras.
# ==============================================================================
# Stemmer: reduce palabras a su raíz (créditos, crédito, acreditar -> 'credit'), para que la búsqueda no dependa de la forma exacta.
STEMMER = SnowballStemmer("spanish")
# 'Palabras vacías': artículos, preposiciones y similares que no aportan significado a la búsqueda y se ignoran.
STOP_F4 = set("""de la el los las en y a que del por con para una un al se su sus es o como cual cuales cuanto cuanta son lo le les
si no mas pero sin sobre entre ser fue han hay este esta estos estas ese esa otros otras asi tambien muy desde hasta""".split())


# Normaliza: minúsculas y sin acentos ('Estimación' -> 'estimacion'), para comparar sin importar cómo se escribió.
def norm_texto(t):
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(ch for ch in t if not unicodedata.combining(ch))


# Tokenizar = partir el texto en palabras y quedarse con la raíz de cada una (sin vacías). Es la 'materia prima' del ranking BM25.
def tok(texto):
    """Términos con raíz (snowball español), sin acentos ni palabras vacías."""
    return [STEMMER.stem(w) for w in re.findall(r"[a-z0-9]+", norm_texto(texto)) if len(w) >= 2 and w not in STOP_F4]


# Parte un fragmento largo en 'ventanas' de ~160 palabras con traslape. Sirve para citar el pasaje exacto y no todo el fragmento.
def ventanas_de(texto, tam=None, paso=None):
    tam, paso = tam or VENTANA_PALABRAS, paso or PASO_VENTANA
    pal = texto.split()
    if len(pal) <= tam:
        return [texto]
    out, i = [], 0
    while i < len(pal):
        out.append(" ".join(pal[i:i + tam]))
        if i + tam >= len(pal):
            break
        i += paso
    return out


# Lee de la base los fragmentos y los deja listos en memoria (texto limpio, términos, ventanas, página, subsección...).
def cargar_corpus():
    conn = conectar()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""SELECT c.id, c.chunk_index, c.sub_section, c.page_printed_start, c.page_printed_end, c.chunk_text, c.metadata,
                                   c.embedding IS NOT NULL, (a.id IS NOT NULL)
                            FROM {ESQUEMA}.banorte_credito_chunks_v1 c
                            LEFT JOIN {ESQUEMA}.alcance_corpus_v1 a ON a.document_id = c.document_id AND a.activo
                                 AND c.chunk_index BETWEEN a.chunk_index_desde AND a.chunk_index_hasta
                            WHERE c.document_id = %s ORDER BY c.chunk_index""", (DOCUMENT_ID,))
            filas = cur.fetchall()
    finally:
        conn.close()
    corpus = []
    for id_, idx, sub, p0, p1, texto, meta, tv, ea in filas:
        # Se quita la frase fija del puntero SQL, igual que cuando se calcularon los vectores (misma regla = comparación justa).
        limpio = texto_para_embedding(texto)
        corpus.append(dict(id=id_, idx=idx, sub=sub or "(inicio)", p_ini=p0, p_fin=p1, texto=limpio, texto_original=texto,
                           tablas=(meta or {}).get("tablas", []), tiene_vector=tv, en_alcance=ea,
                           ventanas=ventanas_de(limpio), terminos=tok(limpio)))
    return corpus


# Se ejecuta la carga una vez. De aquí en adelante todo se busca en memoria, sin volver a la base por cada palabra.
CORPUS = cargar_corpus()
# Acceso rápido por número de fragmento (como buscar por clave primaria) y lista de los que están dentro del alcance.
POR_IDX = {f["idx"]: f for f in CORPUS}
IDX_ALCANCE = [f["idx"] for f in CORPUS if f["en_alcance"]]


# ====================================================================================================
# >>> CELDA 26 DEL CUADERNO
# ====================================================================================================
# Celda 26 · código
# ==============================================================================
# NORMALIZACIÓN DE LA CONSULTA
# La consulta del Anexo 33 se usa tal cual; solo se limpia el enumerador y los espacios.
# ==============================================================================
# Lista de 'ayudas' opcionales: si la pregunta menciona, p. ej., 'garantía', se le agregarían términos relacionados.
# Está apagada (EXPANSION_LEXICA=False) por decisión del proyecto; se deja documentada por si se desea activar.
EXPANSIONES_MINIMAS = [          # (patrón sobre la consulta normalizada, términos que se agregan)
    (r"garant", "reales financieras no financieras personales pari passu primeras perdidas cartera cubierta"),
    (r"concentraci", "principales sectores entidad federativa cartera total"),
    (r"etapa", "etapa 1 y 2 etapa 3 cartera total reservas"),
    (r"sector|entidad", "sector entidad federativa cartera total"),
]
# Patrón que reconoce el inicio de un inciso: «a) », «h) », «jj) ». Se elimina para quedarnos con el contenido de la pregunta.
ENUMERADOR = re.compile(r"^\s*\(?[a-z]{1,2}\)\s+", re.I)


# Devuelve un diccionario con la pregunta original, la limpia y los pasos aplicados (queda registro de lo que se tocó).
def normalizar_consulta(q, expandir=None):
    """Devuelve dict(original, normalizada, pasos, expandida, terminos_agregados). La expansión solo ocurre si se pide."""
    expandir = EXPANSION_LEXICA if expandir is None else expandir
    pasos = []
    # Paso 1: unificar caracteres y colapsar espacios múltiples.
    t = unicodedata.normalize("NFC", q)
    t2 = re.sub(r"\s+", " ", t).strip()
    if t2 != t:
        pasos.append("espacios")
    t = t2
    # Paso 2: quitar el «h) » del inicio.
    t2 = ENUMERADOR.sub("", t)
    if t2 != t:
        pasos.append("enumerador de inciso")
    t = t2
    # Paso 3: quitar signos de puntuación al final.
    t2 = t.rstrip(" ;.,:")
    if t2 != t:
        pasos.append("puntuación final")
    t = t2
    agregados = []
    if expandir:
        tn = norm_texto(t)
        for patron, extra in EXPANSIONES_MINIMAS:
            if re.search(patron, tn):
                agregados.append(extra)
    expandida = (t + " " + " ".join(agregados)).strip() if agregados else t
    if agregados:
        pasos.append("expansión léxica mínima")
    return dict(original=q, normalizada=t, pasos=pasos, expandida=expandida, terminos_agregados=agregados)


# ====================================================================================================
# >>> CELDA 28 DEL CUADERNO
# ====================================================================================================
# Celda 28 · código
# ==============================================================================
# MAPEO NORMATIVO Y VOCABULARIO FUERA DE ALCANCE
# Reconoce los incisos del Anexo 33 y decide, con el mapeo del Plan de Trabajo, si su evidencia está en el tramo cargado.
# ==============================================================================
# Mapa inciso -> dónde está su evidencia en el informe (sección, página, subsección y tablas).
# Es la 'tabla de equivalencias' entre la norma (Anexo 33) y el informe de GFNorte. Los incisos a, e, f, h, m están dentro; r, s, t, w, jj, fuera.
MAPEO_NORMATIVO = {   # Fuente: Plan de Trabajo GFNorte 2T26, p. 8 (ubicación) y firma de evidencia para las pruebas
    "a":  dict(ubicacion="Secc. 2, pp. 7-8 (políticas, NCIR Banorte)", subs=("(inicio)", "2.1"), tablas=()),
    "e":  dict(ubicacion="Secc. 2.1.2, p. 8 (modelo de Merton, límites)", subs=("2.1.2",), tablas=()),
    "f":  dict(ubicacion="Secc. 2.2.1, p. 11 (etapas 1, 2 y 3)", subs=("2.2.1",), tablas=("cartera_por_subsidiaria", "cartera_por_sector", "cartera_por_producto")),
    "h":  dict(ubicacion="Secc. 2.2.1, p. 16 (garantías)", subs=("Garantía",), tablas=("garantias_por_tipo",)),
    "m":  dict(ubicacion="Secc. 2.2.1, pp. 11-12 (sector y entidad)", subs=(), tablas=("cartera_por_sector", "cartera_por_entidad_federativa", "cartera_sector_subsidiaria")),
    "r":  dict(ubicacion="Secc. 2.4, pp. 16-17 (adopción de NIF C-16)", subs=("2.4",), tablas=()),
    "s":  dict(ubicacion="Secc. 2.4.1 a 2.4.5, pp. 17-23 (modelos internos)", subs=(), tablas=("modelo_tarjeta_credito", "modelo_automotriz_pf", "modelo_hipotecaria", "empresas_total", "pyme_total")),
    "t":  dict(ubicacion="Secc. 2.4.1 a 2.4.5, pp. 17-23 (modelos internos)", subs=(), tablas=("modelo_tarjeta_credito", "modelo_automotriz_pf", "modelo_hipotecaria", "empresas_total", "pyme_total")),
    "w":  dict(ubicacion="Secc. 2.3, p. 16 (pérdida esperada y no esperada)", subs=("2.3",), tablas=()),
    "jj": dict(ubicacion="Secc. 2.4.1, p. 18 (compromisos no dispuestos)", subs=("2.4.1",), tablas=("modelo_tarjeta_credito",)),
}


# Lee el PDF del Anexo 33, detecta los renglones que empiezan con 'a)', 'b)'... y guarda el texto de cada inciso.
def cargar_incisos(ruta=None):
    """Lee del PDF del Anexo 33 el texto de cada inciso (a, b, …, ll). Devuelve {} si el archivo no está."""
    ruta = ruta or RUTA_ANEXO33
    if not os.path.exists(ruta):
        return {}
    import pdfplumber
    with pdfplumber.open(ruta) as pdf:
        texto = "\n".join((pg.extract_text() or "") for pg in pdf.pages)
    lineas = texto.splitlines()
    marcas = [(k, re.match(r"^\s*([a-z]{1,2})\)\s+(.*)", l)) for k, l in enumerate(lineas)]
    marcas = [(k, m) for k, m in marcas if m]
    incisos = {}
    for j, (k, m) in enumerate(marcas):
        fin = marcas[j + 1][0] if j + 1 < len(marcas) else len(lineas)
        incisos[m.group(1)] = " ".join(x.strip() for x in [m.group(2)] + lineas[k + 1:fin]).strip()
    return incisos


# Se carga una vez; _TERM_INCISOS guarda los términos de cada inciso para comparar rápido.
INCISOS = cargar_incisos()
_TERM_INCISOS = {k: set(tok(v)) for k, v in INCISOS.items()}


# ¿Lo escrito se parece a un inciso del Anexo 33? Se mide con el índice de Jaccard: términos en común / términos totales (de 0 a 1).
def reconocer_inciso(q):
    """Devuelve (letra, similitud) si lo escrito es un inciso del Anexo 33 (similitud de términos ≥ SIM_MINIMA_INCISO)."""
    if not INCISOS:
        return None, 0.0
    tq = set(tok(ENUMERADOR.sub("", q)))
    if len(tq) < 3:
        return None, 0.0
    mejor, sim = None, 0.0
    for k, t in _TERM_INCISOS.items():
        u = len(tq | t)
        s = len(tq & t) / u if u else 0.0
        if s > sim:
            mejor, sim = k, s
    return (mejor, sim) if sim >= SIM_MINIMA_INCISO else (None, sim)


# Un fragmento 'cumple' si su subsección o sus tablas coinciden con la firma del inciso.
def _cumple(f, firma):
    return any(f["sub"].startswith(p) for p in firma["subs"]) or bool(set(f["tablas"]) & set(firma["tablas"]))


# Dice si hay fragmentos DENTRO del alcance que respondan al inciso, y cuáles quedan fuera (con su ubicación en el informe).
def alcance_de_inciso(letra):
    """Con el mapeo del plan: ¿hay fragmentos dentro del alcance que contengan la evidencia de este inciso?"""
    firma = MAPEO_NORMATIVO.get(letra)
    if firma is None:
        return None
    dentro = [f["idx"] for f in CORPUS if f["en_alcance"] and _cumple(f, firma)]
    fuera = [f["idx"] for f in CORPUS if not f["en_alcance"] and _cumple(f, firma)]
    return dict(en_alcance=bool(dentro), fragmentos_dentro=dentro, fragmentos_fuera=fuera, ubicacion=firma["ubicacion"])


# Detecta temas que solo existen en partes del informe no cargadas (por ejemplo, métricas que solo aparecen fuera del tramo).
def frases_fuera_de_alcance():
    """Frases (nombre de métrica) que solo existen en observaciones de fragmentos fuera del alcance."""
    conn = conectar()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""SELECT o.metrica_codigo, BOOL_OR(a.id IS NOT NULL) FROM {ESQUEMA}.observaciones_cuantitativas_v1 o
                            LEFT JOIN {ESQUEMA}.v_observaciones_alcance_v1 a ON a.id = o.id WHERE o.document_id = %s GROUP BY 1""", (DOCUMENT_ID,))
            solo_fuera = [m for m, en in cur.fetchall() if not en]
            cur.execute(f"SELECT codigo, nombre FROM {ESQUEMA}.catalogo_metricas_v1 WHERE codigo = ANY(%s)", (solo_fuera,))
            nombres = dict(cur.fetchall())
    finally:
        conn.close()
    frases = {norm_texto(n).replace("(", " ").replace(")", " ") for n in nombres.values()}
    frases |= {"backtesting", "prueba retrospectiva", "capital basico", "grandes exposiciones", "probabilidad de incumplimiento"}
    return sorted(re.sub(r"\s+", " ", f).strip() for f in frases), sorted(solo_fuera)


# Se calcula una vez.
FRASES_FUERA = frases_fuera_de_alcance()[0]


# Revisa si la pregunta contiene alguna de esas frases 'fuera de alcance'.
def indicio_vocabulario_fuera(q):
    qn = " " + re.sub(r"\s+", " ", norm_texto(q)) + " "
    return [f for f in FRASES_FUERA if f" {f} " in qn or f" {f}" in qn and len(f) > 6 and f in qn]


# ====================================================================================================
# >>> CELDA 30 DEL CUADERNO
# ====================================================================================================
# Celda 30 · código
# ==============================================================================
# CANAL LÉXICO
# BM25 con raíces de palabras.
# ==============================================================================
# BM25 = fórmula clásica de relevancia: pesa más las palabras poco frecuentes y descuenta la longitud del texto. Se arma una vez.
def _construir_bm25(indices):
    docs = [POR_IDX[i]["terminos"] for i in indices]
    return BM25Okapi(docs), list(indices)


# Dos índices: uno solo con los fragmentos del alcance (el que se usa) y otro con todos (para diagnósticos de fuera de alcance).
BM25_ALCANCE = _construir_bm25(IDX_ALCANCE)
BM25_TODOS = _construir_bm25([f["idx"] for f in CORPUS])


# Calcula el puntaje de cada fragmento para la pregunta y los ordena de mayor a menor; solo se conservan puntajes positivos.
def puntajes_bm25(texto, universo="alcance"):
    """Lista [(idx, puntaje)] ordenada, solo con puntaje positivo."""
    bm, indices = BM25_ALCANCE if universo == "alcance" else BM25_TODOS
    t = tok(texto)
    if not t:
        return []
    sc = bm.get_scores(t)
    orden = sorted(range(len(indices)), key=lambda j: -sc[j])
    return [(indices[j], float(sc[j])) for j in orden if sc[j] > 0]


# Devuelve los mejores TOP_LEXICO (20) fragmentos con su posición (rango 1, 2, 3...).
def buscar_lexico(texto, universo="alcance", k=None):
    """Ranking léxico: lista de dict(idx, puntaje, rango)."""
    k = k or TOP_LEXICO
    base = puntajes_bm25(texto, universo)
    return [dict(idx=i, puntaje=s, rango=r) for r, (i, s) in enumerate(base[:k], 1)]


# ====================================================================================================
# >>> CELDA 32 DEL CUADERNO
# ====================================================================================================
# Celda 32 · código
# ==============================================================================
# CANAL DENSO
# Vector de la consulta desde la caché (sin llamadas al servicio de embeddings) y búsqueda por similitud coseno con pgvector.
# Si la consulta no tiene vector en la caché, el canal denso queda inactivo y la recuperación sigue con el canal léxico.
# ==============================================================================
# Contadores de uso del servicio de vectores (cuántas veces se usó la caché, cuántas se pidió crédito, errores).
ESTADISTICAS_SERVICIO = dict(solicitudes=0, aciertos_cache=0, sin_credito=0, errores=0)


# Identificador único de una pregunta (huella SHA-256 del texto normalizado + modelo). Sirve como 'clave' en la caché.
def huella_consulta(texto):
    return sha(re.sub(r"\s+", " ", norm_texto(texto)).strip() + "|" + MODELO)


# Una clase agrupa datos y funciones relacionadas. Este objeto entrega el vector de una pregunta si ya está guardado en la caché.
class VectorizadorConsulta:
    def __init__(self):
        self._mem = None            # copia en memoria de la caché (evita una consulta a la base por cada pregunta)

    # Trae la caché de la base a memoria para no consultar la base en cada pregunta.
    def recargar_cache(self):
        conn = conectar()
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT huella, embedding::text FROM {ESQUEMA}.consulta_embeddings_cache_v1 WHERE modelo = %s", (MODELO,))
                self._mem = {h: json.loads(v) for h, v in cur.fetchall()}
        finally:
            conn.close()
        return len(self._mem)

    def _cache_get(self, texto):
        if self._mem is None:
            self.recargar_cache()
        return self._mem.get(huella_consulta(texto))


    # Devuelve (vector, estado). Sin vector en caché no se pide uno nuevo: el canal denso simplemente no participa.
    def obtener(self, texto):
        """Devuelve (vector o None, estado). estado: cache | servicio | servicio_apagado | sin_credito | error:…"""
        v = self._cache_get(texto)
        if v is not None:
            ESTADISTICAS_SERVICIO["aciertos_cache"] += 1
            return v, "cache"
        return None, "servicio_apagado"          # esta versión no genera embeddings: sin vector en caché, el canal denso queda inactivo


# Se crea el objeto que se usará durante toda la sesión.
VECTORIZADOR = VectorizadorConsulta()


# Búsqueda por vecinos cercanos: '<=>' es la distancia coseno de pgvector; 1 - distancia = similitud (1 = idéntico).
def buscar_denso(vec, k=None):
    """Ranking denso en el alcance: lista de dict(idx, similitud, rango). Cast explícito ::vector."""
    k = k or TOP_DENSO
    lit = vector_literal(vec)
    conn = conectar()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""SELECT c.chunk_index, 1 - (c.embedding <=> %s::vector) AS sim FROM {ESQUEMA}.banorte_credito_chunks_v1 c
                            JOIN {ESQUEMA}.alcance_corpus_v1 a ON a.document_id = c.document_id AND a.activo
                             AND c.chunk_index BETWEEN a.chunk_index_desde AND a.chunk_index_hasta
                            WHERE c.embedding IS NOT NULL ORDER BY c.embedding <=> %s::vector LIMIT %s""", (lit, lit, k))
            return [dict(idx=i, similitud=float(s), rango=r) for r, (i, s) in enumerate(cur.fetchall(), 1)]
    finally:
        conn.close()


# ====================================================================================================
# >>> CELDA 34 DEL CUADERNO
# ====================================================================================================
# Celda 34 · código
# ==============================================================================
# FUSIÓN POR RANGOS (RRF)
# Combina los rankings de los canales con k = 60.
# ==============================================================================
# RRF: un fragmento bien posicionado en varias listas sube. La constante k=60 suaviza la diferencia entre posiciones altas.
def rrf(listas, k=None, pesos=None):
    """listas: {canal: [idx ordenados]}; devuelve [dict(idx, puntaje, rangos)] ordenado por puntaje y luego por idx."""
    k = RRF_K if k is None else k
    pesos = pesos or {}
    acum, rangos = defaultdict(float), defaultdict(dict)
    for canal, lista in listas.items():
        w = pesos.get(canal, 1.0)
        for r, idx in enumerate(lista, 1):
            # Aporte de este canal: peso / (60 + posición). La posición 1 aporta más que la 20, pero sin saltos bruscos.
            acum[idx] += w / (k + r)
            rangos[idx][canal] = r
    return [dict(idx=i, puntaje=acum[i], rangos=rangos[i]) for i in sorted(acum, key=lambda i: (-acum[i], i))]


# Punto de entrada: si no hay canal denso, se devuelve el ranking léxico tal cual; si lo hay, se aplica RRF.
def fusionar(lexico, denso, k=None):
    """lexico: lista de dict(idx…) ordenada; denso: lista de dict(idx…) o None/[] (canal inactivo). Devuelve (fusion, modo)."""
    if not denso:
        return [dict(idx=x["idx"], puntaje=x["puntaje"], rangos={"lexico": x["rango"]}) for x in lexico], "solo_lexico (canal denso inactivo)"
    return rrf({"lexico": [x["idx"] for x in lexico], "denso": [x["idx"] for x in denso[:TOP_DENSO]]}, k), "rrf"


# ====================================================================================================
# >>> CELDA 36 DEL CUADERNO
# ====================================================================================================
# Celda 36 · código
# ==============================================================================
# ENRUTADOR SQL HACIA LAS CIFRAS
# Consultas parametrizadas sobre las observaciones del alcance (tablas completas del tramo).
# ==============================================================================
# Diccionario de reglas: si la pregunta contiene ciertas palabras (p. ej. 'reserva'), se busca la métrica correspondiente (reserva_total).
# El último campo indica si la regla es específica (True) o genérica (False); las específicas tienen prioridad.
METRICAS_SINONIMOS = [   # (patrón sobre la consulta normalizada, métricas, específica)
    (r"reserva|estimacion(es)? preventiva", ["reserva_total"], True),
    (r"castig", ["castigos"], True),
    (r"dias vencid", ["dias_vencidos"], True),
    (r"garantia|cubiert", ["cartera_cubierta"], True),
    (r"valor razonable", ["valor_razonable_positivo"], True),
    (r"neteo", ["efecto_neteo"], True),
    (r"valor de mercado|\bmtm\b", ["mtm", "mtm_por_calificacion"], True),
    (r"riesgo potencial|riesgo actual", ["exposicion_derivados"], True),
    (r"inversion(es)? en valores", ["exposicion_inversiones_valores"], True),
    (r"exposicion bruta", ["exposicion_bruta"], True),
    (r"variacion|crecimiento", ["variacion_saldo_trimestral_monto", "variacion_saldo_anual_monto", "variacion_reserva_trimestral"], True),
    (r"severidad", ["severidad_perdida"], True),
    (r"probabilidad de incumplimiento", ["pi_ponderada_ei"], True),
    (r"exposicion al incumplimiento", ["exposicion_incumplimiento"], True),
    (r"saldo contable", ["saldo_contable"], True),
    (r"compromisos no dispuestos|no dispuest", ["compromisos_no_dispuestos"], True),
    (r"perdida no esperada", ["perdida_no_esperada_pct"], True),
    (r"perdida esperada", ["perdida_esperada_pct", "perdida_esperada_monto", "perdida_esperada_backtesting"], True),
    (r"perdida observada|backtesting", ["perdida_observada", "perdida_esperada_backtesting"], True),
    (r"capital basico", ["capital_basico"], True),
    (r"grandes exposiciones", ["grandes_exposiciones_monto", "grandes_exposiciones_numero", "grandes_exposiciones_pct_capital"], True),
    (r"saldo|cartera|credito|monto", ["saldo_cartera"], False),
]
# Sinónimos de uso común que no están en la tabla dim_alias (por ejemplo, 'cdmx' = Ciudad de México).
ALIAS_EXTRA = {   # sinónimos de uso común que no están en dim_alias
    "con bap y pd": [("perimetro", "gfnorte_con_bap_pd")], "con bap": [("perimetro", "gfnorte_con_bap_pd")],
    "sin bap": [("perimetro", "gfnorte_sin_bap_pd")], "sin bap ni pd": [("perimetro", "gfnorte_sin_bap_pd")],
    "casa de bolsa": [("perimetro", "casa_bolsa")], "cdmx": [("entidad_federativa", "ciudad_de_mexico")],
    "gfnorte": [("perimetro", "gfnorte_con_bap_pd"), ("perimetro", "gfnorte_sin_bap_pd")],
    "tarjeta de credito": [("producto", "tarjeta")], "hipotecaria": [("producto", "hipotecario")],
}
# El perímetro solo suma puntos (no descarta filas): si el usuario no lo menciona, aún se pueden mostrar ambas versiones.
TIPOS_SUAVES = {"perimetro"}          # viven en una columna, no en dimensiones: solo suman puntaje
# Palabras que delatan una pregunta numérica ('cuánto', 'monto', 'saldo', 'total'...).
PALABRAS_NUMERICAS = re.compile(r"cuanto|cual es el|monto|saldo|importe|total|cifra|millones|porcentaje|cuantos|valor")


# Carga el diccionario de alias de la base (cómo suele escribirse cada valor) y le suma los alias extra.
def cargar_alias():
    conn = conectar()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT tipo_codigo, alias_norm, valor_codigo FROM {ESQUEMA}.dim_alias_v1")
            filas = cur.fetchall()
    finally:
        conn.close()
    alias = defaultdict(list)
    for t, a, v in filas:
        if len(a.split()) <= 6 and "(" not in a:
            alias[a].append((t, v))
    for a, lst in ALIAS_EXTRA.items():
        alias[a] += [x for x in lst if x not in alias[a]]
    return dict(alias)


# Se ejecuta una vez. PERIODO_DOC es el periodo del informe (2T26); MAX_N es la frase más larga de alias, en palabras.
ALIAS = cargar_alias()
PERIODO_DOC = sql_uno(f"SELECT reporting_period FROM {ESQUEMA}.documents_v1 WHERE id = %s", (DOCUMENT_ID,))[0]
MAX_N = max(len(a.split()) for a in ALIAS)


# Lee la pregunta y extrae: qué métricas pide, qué cortes menciona (sector, entidad...) y qué periodos.
def analizar_cifras(q):
    """Devuelve dict(metricas, menciones[(texto, [(tipo, codigo)])], especificas, numerica)."""
    qn = re.sub(r"\s+", " ", norm_texto(q))
    espec, gener = [], []
    for patron, mets, es in METRICAS_SINONIMOS:
        if re.search(patron, qn):
            (espec if es else gener).extend(m for m in mets if m not in espec and m not in gener)
    metricas = espec or gener
    palabras = re.findall(r"[a-z0-9]+", qn)
    menciones, i = [], 0
    # Busca alias de la frase más larga a la más corta ('casa de bolsa' antes que 'bolsa') para no confundir términos.
    while i < len(palabras):
        hallado = None
        for n in range(min(MAX_N, len(palabras) - i), 0, -1):
            cand = " ".join(palabras[i:i + n])
            if cand in ALIAS:
                hallado = (cand, ALIAS[cand], n)
                break
        if hallado:
            menciones.append((hallado[0], hallado[1]))
            i += hallado[2]
        else:
            i += 1
    periodos = [p.upper() for p in re.findall(r"\b([1-4]t\d\d)\b", qn)]
    return dict(metricas=metricas, menciones=menciones, especificas=bool(espec), numerica=bool(PALABRAS_NUMERICAS.search(qn)), periodos=periodos)


# Cortes 'duros': deben cumplirse para que una fila sea válida. Los demás (producto, subsidiaria...) solo suben el puntaje.
TIPOS_FUERTES = {"sector", "etapa", "entidad_federativa", "tipo_garantia", "plazo_remanente", "calificacion_rango", "instrumento_derivado",
                 "contraparte_tipo", "medida_exposicion", "grado_riesgo"}   # cortes que deben cumplirse; los demás (producto, subsidiaria, modelo…) solo suman puntaje


# Agrupa las menciones: dentro de un grupo vale 'o' (cualquiera), entre grupos vale 'y' (todos deben cumplirse).
def _grupos(an):
    """Agrupa las menciones: OR dentro de un mismo conjunto de tipos candidatos, AND entre grupos. Devuelve (duros, debiles, suaves)."""
    por_clave = defaultdict(list)
    suaves = []
    for texto, cands in an["menciones"]:
        duros_c = [(t, v) for t, v in cands if t not in TIPOS_SUAVES]
        suaves += [v for t, v in cands if t in TIPOS_SUAVES]
        if duros_c:
            por_clave[frozenset(t for t, _ in duros_c)].extend(duros_c)
    duros, debiles = [], []
    for tipos, cands in por_clave.items():
        grupo = list(dict.fromkeys(cands))
        (duros if tipos & TIPOS_FUERTES else debiles).append(grupo)
    return duros, debiles, suaves


def _cumple_grupo(dims, grupo):
    return any(dims.get(t) == v for t, v in grupo)


# Una sola consulta SQL por métrica; después los cortes se aplican en Python.
def _consultar(vista, an, limite=3000):
    """Una sola consulta por métricas; los cortes se aplican en Python (los débiles solo suman puntaje)."""
    conn = conectar()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""SELECT id, tabla_origen, page_printed, etiqueta_fila_reportada, etiqueta_columna_reportada, periodo, base_temporal, perimetro,
                                   dimensiones::text, tipo_fila, metrica_codigo, unidad, valor_numerico, texto_reportado, control_calidad, nota_calidad
                            FROM {ESQUEMA}.{vista} WHERE metrica_codigo = ANY(%s) LIMIT {limite}""", (an["metricas"],))
            filas = cur.fetchall()
    finally:
        conn.close()
    claves = ("id", "tabla", "pagina", "fila", "columna", "periodo", "base", "perimetro", "dims", "tipo_fila", "metrica", "unidad", "valor", "texto", "calidad", "nota")
    duros, debiles, suaves = _grupos(an)
    salida = []
    for f in (dict(zip(claves, x)) for x in filas):
        dims = json.loads(f["dims"])
        if not all(_cumple_grupo(dims, g) for g in duros):
            continue
        f["_debiles"] = sum(1 for g in debiles if _cumple_grupo(dims, g))
        salida.append(f)
    return salida, duros, debiles, suaves


# Sistema de puntos para ordenar las filas candidatas: suma puntos por cortes que coinciden, perímetro, periodo, calidad OK, etc.; resta si es subtotal.
def _puntuar(fila, duros, debiles, suaves, an, tablas_preferidas):
    dims = json.loads(fila["dims"])
    p = 10.0 * len(duros) + 10.0 * fila["_debiles"]
    if fila["perimetro"] in suaves:
        p += 3
    cortes_esp = [g for g in duros if not all(t == "etapa" for t, _ in g)]
    if not cortes_esp and not debiles and fila["tipo_fila"] == "total":
        p += 2
    if not any(t == "etapa" for g in duros for t, _ in g) and dims.get("etapa") == "todas":
        p += 2
    if fila["base"] == "cierre":
        p += 1
    if an["periodos"]:
        p += 4 if fila["periodo"] in an["periodos"] else 0
    elif fila["periodo"] == PERIODO_DOC:
        p += 2
    if fila["calidad"] == "OK":
        p += 1
    if tablas_preferidas and fila["tabla"] in tablas_preferidas:
        p += 4
    if fila["tipo_fila"] == "subtotal":
        p -= 2
    return p


# Función principal: devuelve hasta 6 filas de cifras ordenadas por puntaje, sin duplicados.
# Estados: ok | fuera_de_alcance (la cifra existe pero en otra parte del informe) | sin_cifras | no_numerica.
def buscar_cifras(q, tablas_preferidas=(), max_filas=6):
    """Devuelve dict(estado, filas, metricas, menciones, donde). estado: ok | fuera_de_alcance | sin_cifras | no_numerica."""
    an = analizar_cifras(q)
    base = dict(metricas=an["metricas"], menciones=an["menciones"], donde=[])
    if not an["metricas"] or not (an["especificas"] or an["numerica"] or any(t not in TIPOS_SUAVES for _, c in an["menciones"] for t, _ in c)):
        return dict(estado="no_numerica", filas=[], **base)
    # Primero se busca solo dentro del alcance (vista); si no hay nada, se averigua si existe fuera para avisar al usuario.
    filas, duros, debiles, suaves = _consultar("v_observaciones_alcance_v1", an)
    if filas:
        for f in filas:
            f["puntaje"] = _puntuar(f, duros, debiles, suaves, an, set(tablas_preferidas))
            f["fuente"] = f"p. {f['pagina']}, {f['tabla']}, fila «{f['fila']}», columna «{f['columna']}»"
        filas.sort(key=lambda f: (-f["puntaje"], f["tabla"], f["id"]))
        vistos, finales = set(), []
        for f in filas:
            clave = (f["metrica"], f["valor"], f["perimetro"], f["periodo"])
            if clave not in vistos:
                vistos.add(clave)
                finales.append(f)
        return dict(estado="ok", filas=finales[:max_filas], **base)
    todas, _, _, _ = _consultar("observaciones_cuantitativas_v1", an)
    if todas:
        return dict(estado="fuera_de_alcance", filas=[], **{**base, "donde": sorted({(f["tabla"], f["pagina"]) for f in todas})[:6]})
    return dict(estado="sin_cifras", filas=[], **base)


# ====================================================================================================
# >>> CELDA 38 DEL CUADERNO
# ====================================================================================================
# Celda 38 · código
# ==============================================================================
# RECUPERACIÓN INTEGRADA Y GUARDIA DE ALCANCE
# Encadena normalización, canales, fusión, enrutador de cifras y guardia de alcance.
# ==============================================================================
# Umbral de 'soporte': si ni el mejor fragmento alcanza este puntaje BM25, se considera que el informe no habla del tema (calibrado con preguntas de prueba).
MIN_SOPORTE_LEXICO = 4.92    # puntaje BM25 máximo en los 18 fragmentos por debajo del cual no hay soporte (calibrado con el banco de consultas)
# Lista en memoria donde se anota cada consulta (estado y tiempo) para diagnóstico.
TRAZA = []
# Mensajes estándar de la guardia. Se redactan con cuidado para no afirmar que el informe 'no lo documenta' cuando solo está fuera del tramo cargado.
MSG_FUERA = ("FUERA DEL ALCANCE CARGADO: la evidencia de este inciso está en {ubicacion}, que no forma parte del corpus de esta versión "
             "(fragmentos {desde} a {hasta}, pp. {p0} a {p1}, hasta garantías). Esto NO significa que el informe no lo documente.")
MSG_SIN_SOPORTE = "SIN SOPORTE: ninguno de los 18 fragmentos del informe tiene texto relacionado con la pregunta. La información no se encuentra documentada en el informe de riesgos del 2T26."
MSG_POSIBLE = "AVISO: la pregunta menciona temas ({frases}) que solo aparecen en las secciones fuera del alcance cargado (pp. 17 a 24); los resultados pueden no cubrirla."


# Guardia de alcance. Decide ANTES de gastar crédito si la pregunta puede responderse con el tramo cargado:
#   1) si es un inciso conocido del Anexo 33, se usa el mapeo normativo; 2) si no, se mide el soporte léxico; 3) se revisan frases de temas fuera del tramo.
def guardia_alcance(q, texto_busqueda):
    letra, sim = reconocer_inciso(q)
    info = dict(inciso=letra, similitud_inciso=round(sim, 2))
    if letra and letra in MAPEO_NORMATIVO:
        d = alcance_de_inciso(letra)
        if not d["en_alcance"]:
            return dict(estado="fuera_de_alcance", fuente="mapeo_normativo", ubicacion=d["ubicacion"], **info,
                        mensaje=MSG_FUERA.format(ubicacion=d["ubicacion"], desde=ALCANCE["desde"], hasta=ALCANCE["hasta"], p0=ALCANCE["pag_ini"], p1=ALCANCE["pag_fin"]))
        return dict(estado="en_alcance", fuente="mapeo_normativo", ubicacion=d["ubicacion"], **info, mensaje="")
    # Se busca en los 18 fragmentos del corpus. Un puntaje muy bajo significa que ningún fragmento trata el tema.
    pt = puntajes_bm25(texto_busqueda, "todos")
    mejor = pt[0][1] if pt else 0.0
    if mejor < MIN_SOPORTE_LEXICO:
        return dict(estado="sin_soporte", fuente="lexico", puntaje=round(mejor, 2), **info, mensaje=MSG_SIN_SOPORTE)
    ind = indicio_vocabulario_fuera(q)
    if ind:
        return dict(estado="posible_fuera_de_alcance", fuente="vocabulario", frases=ind, puntaje=round(mejor, 2), **info, mensaje=MSG_POSIBLE.format(frases=", ".join(ind[:3])))
    return dict(estado="en_alcance", fuente="lexico", puntaje=round(mejor, 2), **info, mensaje="")


# Elige la ventana (pasaje de ~160 palabras) con más términos en común con la pregunta.
def mejor_ventana(consulta, f):
    tq = set(tok(consulta))
    return max(f["ventanas"], key=lambda w: len(tq & set(tok(w))))


# Flujo completo de recuperación. Cada paso mide su tiempo en res['tiempos'] para poder auditar el desempeño.
def recuperar(q, expandir=None, con_sql=True):
    t_ini = time.perf_counter()
    n = normalizar_consulta(q, expandir)
    res = dict(consulta=q, normalizada=n["normalizada"], pasos=n["pasos"], expansion=n["terminos_agregados"], tiempos={})
    # Paso 1: guardia. Si la pregunta es fuera de alcance o sin soporte, se corta aquí: no se busca ni se gasta nada más.
    g = guardia_alcance(q, n["expandida"])
    res["guardia"] = g
    if g["estado"] in ("fuera_de_alcance", "sin_soporte"):
        res.update(estado=g["estado"], mensaje=g["mensaje"], evidencias=[], cifras=dict(estado="omitida", filas=[]),
                   canal_denso=dict(activo=False, motivo="no aplica"), fusion="no aplica")
        res["tiempos"]["total_ms"] = round((time.perf_counter() - t_ini) * 1000, 1)
        TRAZA.append(dict(consulta=q, estado=res["estado"], total_ms=res["tiempos"]["total_ms"], denso=False))
        return res
    t = time.perf_counter()
    # Paso 2: canal léxico (BM25).
    lexico = buscar_lexico(n["expandida"], "alcance", TOP_LEXICO)
    res["tiempos"]["lexico_ms"] = round((time.perf_counter() - t) * 1000, 1)
    t = time.perf_counter()
    # Paso 3: canal denso. Solo participa si la caché tiene el vector de esta pregunta; si no, queda inactivo.
    vec, estado_vec = VECTORIZADOR.obtener(n["normalizada"])
    denso = buscar_denso(vec, TOP_DENSO) if vec is not None else None
    res["canal_denso"] = dict(activo=vec is not None, motivo=estado_vec)
    res["tiempos"]["denso_ms"] = round((time.perf_counter() - t) * 1000, 1)
    # Paso 4: fusión de rankings (RRF) y selección de candidatos.
    fusion, modo = fusionar(lexico, denso)
    res["fusion"] = modo
    candidatos = [x["idx"] for x in fusion][:max(TOP_FINAL * 2, 8)]
    por_f = {x["idx"]: x for x in fusion}
    lex_p = {x["idx"]: x["puntaje"] for x in lexico}
    den_s = {x["idx"]: x["similitud"] for x in (denso or [])}
    orden = candidatos      # el orden de las ventanas viene directo de la fusión (la ventana se elige por coincidencia de palabras)
    evidencias = []
    # Paso 5: se quedan los 5 mejores fragmentos y, de cada uno, la ventana más relevante.
    for idx in orden[:TOP_FINAL]:
        f = POR_IDX[idx]
        vent = mejor_ventana(n["normalizada"], f)
        evidencias.append(dict(idx=idx, sub=f["sub"], p_ini=f["p_ini"], p_fin=f["p_fin"], en_alcance=f["en_alcance"], tiene_vector=f["tiene_vector"],
                               tablas=f["tablas"], rangos=por_f[idx]["rangos"], puntaje_lexico=lex_p.get(idx), similitud_densa=den_s.get(idx),
                               ventana=vent, tokens_ventana=tokens_est(vent)))
    res["evidencias"] = evidencias
    t = time.perf_counter()
    prefer = {tb for e in evidencias for tb in e["tablas"]}
    # Paso 6: cifras exactas desde SQL, dando preferencia a las tablas mencionadas por los fragmentos elegidos.
    res["cifras"] = buscar_cifras(q, prefer) if con_sql else dict(estado="omitida", filas=[])
    res["tiempos"]["sql_ms"] = round((time.perf_counter() - t) * 1000, 1)
    # Estado final de la recuperación: sin_evidencia, posible_fuera_de_alcance u ok.
    res["estado"] = "sin_evidencia" if not evidencias else ("posible_fuera_de_alcance" if g["estado"] == "posible_fuera_de_alcance" else "ok")
    res["mensaje"] = g.get("mensaje", "")
    res["tiempos"]["total_ms"] = round((time.perf_counter() - t_ini) * 1000, 1)
    TRAZA.append(dict(consulta=q, estado=res["estado"], total_ms=res["tiempos"]["total_ms"], denso=res["canal_denso"]["activo"], n_evid=len(evidencias)))
    return res


# ====================================================================================================
# >>> CELDA 41 DEL CUADERNO
# ====================================================================================================
# Celda 41 · código
# ==============================================================================
# PARÁMETROS, PRESUPUESTO DE TOKENS Y REGLAS DE CALIDAD
# Contexto ≤ 600 tokens y total por consulta ≤ 1,250; la generación se activa en la Parte 6.
# ==============================================================================
# Funciones auxiliares: Decimal sirve para manejar cifras sin errores de redondeo binario (clave en cifras financieras).
from decimal import Decimal, InvalidOperation   # os, re, json, math, time y hashlib ya se importaron en la Parte 4

try:
    # Verifica que la Parte 4 ya se ejecutó; si no, avisa con un mensaje claro.
    (recuperar, tokens_est, tok, norm_texto, POR_IDX, CORPUS, ALCANCE, sql_uno, conectar, ESQUEMA, DOCUMENT_ID, RUTA_PROYECTO)
except NameError as e:
    raise RuntimeError(f"Falta {e}. Ejecuta antes la Parte 4.")

# ---- Presupuesto de tokens (el modelo cobra por token: se limita lo que se envía y lo que se recibe; criterios C1 y C1b) -------------------------------------------------------------------
PRESUPUESTO_CONTEXTO_F5 = 600   # C1: máximo de tokens del bloque de contexto (evidencias + cifras)
TOPE_TOTAL_F5 = 1250            # C1b: instrucciones + contexto + pregunta + respuesta máxima
# Tope de longitud de la respuesta del modelo.
MAX_TOKENS_RESPUESTA = 350      # lo más largo que se le permite escribir al modelo
# Margen de seguridad: como el conteo es aproximado, se suma 10% de colchón.
MARGEN_ESTIMADOR = 1.10         # la cuenta de tokens es una estimación (letras ÷ 3.5): se suma 10 % de colchón
# Máximo de fragmentos de texto y de filas de cifras que entran al contexto.
MAX_EVIDENCIAS_F5 = 3           # cuántos fragmentos de texto como máximo entran al contexto
MAX_CIFRAS_F5 = 6               # cuántas filas de cifras exactas como máximo

# ---- Generación con el modelo (APAGADA hasta que la última celda la active tras una prueba mínima) ------------------------------------------------------------------
USAR_GENERACION = False         # True solo cuando haya crédito y se haya hecho la prueba mínima (última celda)
MODELO_GENERACION = None        # None = el cliente lo lee de /v1/models (consulta gratuita)
# Temperatura 0 = el modelo elige siempre la opción más probable: respuestas repetibles, sin 'creatividad'.
TEMPERATURA = 0.0               # 0 = respuestas lo más repetibles posible (algunos modelos no la aceptan: ver el cliente)
# Segundos de espera máxima por respuesta.
TIMEOUT_S = 60                  # el servicio tarda 35 a 45 s en "despertar" la primera vez (máximo observado: 44.6 s)
REINTENTOS = 2                  # solo para fallas transitorias del servidor (5xx); nunca para falta de crédito
REGISTRAR_BITACORA = True       # guarda un renglón por consulta en ingesta_bitacora (C7); si falla, no detiene nada
# Modo estricto: una cifra solo es válida si aparece junto al concepto al que pertenece (evita atribuir un número a la fila equivocada).
ESTRICTO_TERNA = True           # exige que cada cifra de la respuesta aparezca junto al concepto al que pertenece
# Si se repite la misma pregunta con el mismo contexto, se reutiliza la respuesta guardada y no se paga otra vez.
ARCHIVO_CACHE_F5 = os.path.join(RUTA_PROYECTO, "cache_respuestas_f5.json")   # misma pregunta + mismo contexto = cero costo
BASE_URL_F5 = globals().get("BASE_URL") or "https://api.artesaniadigital.org"
API_KEY_F5 = globals().get("API_KEY")

# Tokens estimados con margen.
tokens_f5 = lambda t: math.ceil(tokens_est(t) * MARGEN_ESTIMADOR)   # tokens estimados con colchón

# ---- Páginas: el informe tiene página FÍSICA (del PDF) y página IMPRESA (la que ve el lector) -----------------------
try:
    # Desfase entre página impresa y física (impresa = física + 6). Se lee de la base; si falla, se usa 6.
    OFFSET_PAGINAS = int(sql_uno(f"SELECT page_offset FROM {ESQUEMA}.documents_v1 WHERE id = %s", (DOCUMENT_ID,))[0])
except Exception:
    OFFSET_PAGINAS = 6    # valor documentado en la Fase 2: impresa = física + 6


# Convierte página impresa en página física del PDF.
def pag_fisica(impresa):
    """Convierte una página impresa en la página física del PDF (impresa − desfase)."""
    return None if impresa is None else int(impresa) - OFFSET_PAGINAS


# ---- Estados de calidad de las cifras: cómo se rotulan y cómo se reconoce que la respuesta los avisó -----------------
# Cómo se rotula cada estado de calidad de una cifra cuando se entrega al modelo.
ETIQUETA_CALIDAD = {
    "OK": "",
    "SIN_VALIDAR": "[ESTADO: NO_VALIDADA_CONTABLEMENTE]",
    "DISCREPANCIA_INTERNA": "[ESTADO: DISCREPANCIA_INTERNA]",
    "ERRATA_FUENTE": "[ESTADO: ERRATA_FUENTE]",
}
# Palabras que la respuesta debe usar para avisar que una cifra tiene un estado dudoso (si no, el verificador la rechaza).
PATRON_ADVERTENCIA = {   # palabras que la respuesta debe usar cuando cita una cifra con ese estado
    "SIN_VALIDAR": re.compile(r"no (fue |pudo |ha sido )?(validad|verificad|cuadr)|sin validar|no validad|no verificad|no pudo verificarse", re.I),
    "DISCREPANCIA_INTERNA": re.compile(r"discrepancia|no cuadra|no concilia|inconsisten|difiere", re.I),
    "ERRATA_FUENTE": re.compile(r"errata|error (de|en) (la )?fuente|error del informe|error tipogr", re.I),
}
# Frase que el modelo debe decir cuando el contexto no contiene la respuesta (en lugar de inventar).
FRASE_SIN_DATO = "No está documentado en el contexto entregado."


# ====================================================================================================
# >>> CELDA 43 DEL CUADERNO
# ====================================================================================================
# Celda 43 · código
# ==============================================================================
# GUARDIA PREVIA
# ==============================================================================
# Mensajes fijos de abstención y de aviso. Siempre aclaran qué parte del informe está cargada y evitan afirmar algo que no se verificó.
MSG_ALCANCE_F5 = ("AVISO DE ALCANCE: la información sobre este tema ({ubicacion}) está en secciones del informe que esta versión no carga "
                  "(fragmentos {a} a {b}, págs. impresas {p0} a {p1}). Esto NO significa que el informe no la documente.")
MSG_NO_DOCUMENTADO_F5 = "La información solicitada no se encuentra documentada en el informe de administración de riesgos de GFNorte al 2T26."
MSG_SIN_EVIDENCIA_F5 = ("No se encontró evidencia suficiente dentro del alcance cargado (págs. impresas {p0} a {p1}) para responder esta pregunta. "
                        "Reformula la consulta con términos del informe.")
MSG_POSIBLE_F5 = "SUPUESTO DECLARADO: la pregunta menciona temas ({frases}) que podrían estar en secciones aún no cargadas; la respuesta se limita a lo que el alcance cargado documenta."
MSG_CIFRAS_FUERA_F5 = "NOTA: las cifras solicitadas existen en el informe pero en tablas fuera del alcance cargado ({donde}); esta respuesta no las incluye."


# Calcula, desde los datos, qué fragmentos y páginas están fuera del alcance (si todo el corpus está dentro, devuelve None).
def rango_pendiente():
    """Fragmentos y páginas impresas que hoy están FUERA del alcance (se calcula de los datos, no se escribe a mano)."""
    fuera = [f for f in CORPUS if not f["en_alcance"]]
    if not fuera:
        return None
    return dict(a=min(f["idx"] for f in fuera), b=max(f["idx"] for f in fuera),
                p0=min(f["p_ini"] for f in fuera), p1=max(f["p_fin"] for f in fuera))


# Reglas de decisión, en orden: fuera de alcance -> abstener; sin soporte -> abstener; sin evidencia -> abstener;
# en otro caso -> generar, con avisos si hay temas posiblemente fuera del tramo o cifras que existen en otra parte del informe.
def evaluar_guardia_f5(res):
    """Convierte el resultado de recuperar() en una acción: 'abstener' (sin llamar al modelo) o 'generar'."""
    est = res.get("estado")
    g = res.get("guardia", {})
    pend = rango_pendiente()
    avisos = []
    if est == "fuera_de_alcance":
        ubic = g.get("ubicacion", "secciones pendientes")
        msg = (MSG_ALCANCE_F5.format(ubicacion=ubic, **pend) if pend else res.get("mensaje", ""))
        return dict(accion="abstener", estado=est, mensaje=msg, avisos=[])
    if est == "sin_soporte":
        return dict(accion="abstener", estado=est, mensaje=MSG_NO_DOCUMENTADO_F5, avisos=[])
    if est == "sin_evidencia" or not res.get("evidencias"):
        return dict(accion="abstener", estado="sin_evidencia", mensaje=MSG_SIN_EVIDENCIA_F5.format(p0=ALCANCE["pag_ini"], p1=ALCANCE["pag_fin"]), avisos=[])
    if est == "posible_fuera_de_alcance":
        avisos.append(MSG_POSIBLE_F5.format(frases=", ".join(g.get("frases", [])[:3]) or "no cubiertos"))
    c = res.get("cifras", {})
    if c.get("estado") == "fuera_de_alcance":
        avisos.append(MSG_CIFRAS_FUERA_F5.format(donde=", ".join(f"{t} (p. {p})" for t, p in c.get("donde", [])[:3])))
    return dict(accion="generar", estado=est, mensaje="", avisos=avisos)


# ====================================================================================================
# >>> CELDA 45 DEL CUADERNO
# ====================================================================================================
# Celda 45 · código
# ==============================================================================
# CONSTRUCTOR DE CONTEXTO Y PROMPT
# ==============================================================================
# Memoria para no reconstruir el índice de ventanas en cada pregunta.
_BM25_VENT = {}


# Índice BM25 a nivel de ventana (no de fragmento completo), para elegir el pasaje más relevante dentro de cada fragmento.
def _indice_ventanas(por_idx):
    """Índice BM25 sobre TODAS las ventanas del corpus (se arma una vez). BM25 = puntaje clásico de coincidencia de términos."""
    clave = id(por_idx)
    if clave not in _BM25_VENT:
        from rank_bm25 import BM25Okapi
        docs, ref = [], []
        for i, f in por_idx.items():
            for k, v in enumerate(f["ventanas"]):
                docs.append(tok(v) or ["_"])
                ref.append((i, k))
        _BM25_VENT[clave] = (BM25Okapi(docs), ref)
    return _BM25_VENT[clave]


# La ventana de cada fragmento se elige por BM25. Si hay empate, gana la más temprana (resultado reproducible).
def elegir_ventana(consulta, idx, por_idx):
    """Devuelve (texto de la ventana, método). Desempata siempre a favor de la ventana más temprana."""
    f = por_idx[idx]
    if len(f["ventanas"]) == 1:
        return f["ventanas"][0], "ventana_unica"
    bm, ref = _indice_ventanas(por_idx)
    tq = tok(consulta)
    pts = bm.get_scores(tq) if tq else [0.0] * len(ref)
    propios = [(k, pts[j]) for j, (i, k) in enumerate(ref) if i == idx]
    k_mejor = max(propios, key=lambda x: (x[1], -x[0]))[0]
    return f["ventanas"][k_mejor], "bm25_fallback"


# Formatea cifras con separador de miles, sin redondear (para que el verificador pueda comparar literalmente).
def fmt_valor(v):
    """Formatea un número sin ocultar decimales (no se redondea nada)."""
    d = Decimal(str(v))
    return f"{d:,.2f}" if d == d.quantize(Decimal("0.01")) else f"{d:,}"


# Convierte una fila de cifra en una línea de texto: concepto | columna | valor y unidad | periodo | perímetro | estado de calidad.
def _linea_cifra(f):
    valor = fmt_valor(f["valor"]) if f.get("valor") is not None else (f.get("texto") or "s/d")
    partes = [str(f.get("fila") or "").strip(), str(f.get("columna") or "").strip(), f"{valor} {f.get('unidad') or ''}".strip(), str(f.get("periodo") or "")]
    if f.get("perimetro"):
        partes.append(str(f["perimetro"]))
    linea = "- " + " | ".join(p for p in partes if p)
    etiqueta = ETIQUETA_CALIDAD.get(f.get("calidad"), f"[ESTADO: {f.get('calidad')}]")
    if etiqueta:
        linea += " " + etiqueta
    if f.get("calidad") in ("ERRATA_FUENTE", "DISCREPANCIA_INTERNA") and f.get("nota"):
        linea += " nota: " + str(f["nota"])[:90]
    return linea


# Arma los bloques [E#]: primero los de texto y luego los de tablas que no estén cubiertos por un texto.
def _armar_entradas(res, consulta, por_idx, max_ev, max_ci, recorte):
    """Arma la lista de bloques [E#]: primero los de texto y luego los de tabla que no estén ya cubiertos por uno de texto."""
    entradas = []
    for e in res["evidencias"][:max_ev]:
        texto, metodo = elegir_ventana(consulta, e["idx"], por_idx)
        if recorte < 1.0:
            pal = texto.split()
            texto = " ".join(pal[:max(25, int(len(pal) * recorte))]) + (" …" if len(pal) > 25 else "")
        entradas.append(dict(tipo="texto", idx=e["idx"], sub=e["sub"], p_ini=e["p_ini"], p_fin=e["p_fin"], tablas=list(e.get("tablas") or []),
                             metodo_ventana=metodo, texto=texto, filas=[]))
    filas = list((res.get("cifras") or {}).get("filas") or [])[:max_ci]
    extra = {}
    for f in filas:
        destino = next((en for en in entradas if en["tipo"] == "texto" and f["tabla"] in en["tablas"]), None)
        if destino is None:
            destino = extra.setdefault(f["tabla"], dict(tipo="tabla", idx=None, sub=f"Tabla {f['tabla']}", p_ini=f["pagina"], p_fin=f["pagina"],
                                                        tablas=[f["tabla"]], metodo_ventana="no_aplica", texto="", filas=[]))
            destino["p_ini"] = min(destino["p_ini"], f["pagina"]); destino["p_fin"] = max(destino["p_fin"], f["pagina"])
        destino["filas"].append(f)
    entradas += list(extra.values())
    for k, en in enumerate(entradas, 1):
        en["tag"] = f"E{k}"
        en["fis_ini"], en["fis_fin"] = pag_fisica(en["p_ini"]), pag_fisica(en["p_fin"])
    return entradas


# El informe tiene dos numeraciones; se citan ambas, 'pág. impresa 11 (física 5)', para que se pueda ubicar el dato sin ambigüedad.
def foliacion(en):
    """Texto de la doble foliación de un bloque: página impresa (la oficial) y física (la del PDF)."""
    imp = f"{en['p_ini']}" if en["p_ini"] == en["p_fin"] else f"{en['p_ini']}-{en['p_fin']}"
    fis = f"{en['fis_ini']}" if en["fis_ini"] == en["fis_fin"] else f"{en['fis_ini']}-{en['fis_fin']}"
    return f"pág. impresa {imp} (física {fis})"


# Pasa los bloques a texto plano: encabezado [E#] y cuerpo.
def _render_contexto(entradas):
    bloques = []
    for en in entradas:
        cab = f"[{en['tag']}] {en['sub']} · {foliacion(en)}"
        cuerpo = [en["texto"]] if en["texto"] else []
        if en["filas"]:
            cuerpo.append("Cifras exactas:\n" + "\n".join(_linea_cifra(f) for f in en["filas"]))
        bloques.append(cab + "\n" + "\n".join(cuerpo))
    return "\n\n".join(bloques)


# Instrucciones fijas del modelo: responder solo con el contexto, copiar cifras literalmente, citar [E#], respetar perímetro y periodo, máx. 120 palabras.
def construir_system_prompt():
    """Instrucciones fijas para el modelo. El rango de páginas sale de los datos (no está escrito a mano)."""
    return (
        f"Eres un auditor bancario de riesgo de crédito. Respondes solo con el CONTEXTO, tomado del informe de riesgos de GFNorte al 2T26 "
        f"(págs. impresas {ALCANCE['pag_ini']} a {ALCANCE['pag_fin']}). No uses conocimiento externo; si el contexto no contiene la respuesta di: «{FRASE_SIN_DATO}»\n"
        "Reglas:\n"
        "1. Copia cada cifra exactamente como aparece; no redondees, conviertas, sumes ni restes.\n"
        "2. Cierra cada afirmación con su etiqueta [E#]. No escribas números de página.\n"
        "3. Si una cifra trae [ESTADO: NO_VALIDADA_CONTABLEMENTE], aclara que no pudo verificarse con las identidades contables del informe. "
        "Si trae ERRATA_FUENTE o DISCREPANCIA_INTERNA, declara la anomalía de la fuente.\n"
        "4. Respeta el perímetro (con o sin BAP y PD) y el periodo de cada cifra.\n"
        "5. Español, máximo 120 palabras.")


# Arma el paquete completo (prompt + fichas + medidas de tokens). Aplica el recorte en cascada hasta cumplir los dos topes.
def construir_prompt(res, por_idx=None, presupuesto=None, avisos=()):
    """Devuelve un paquete con el prompt listo, la ficha de cada [E#] y las medidas de tokens (C1 y C1b)."""
    por_idx = por_idx or POR_IDX
    consulta = res["consulta"]
    system = construir_system_prompt()
    nota = ("\nNota: la pregunta podría requerir información que no está en el contexto; si falta, dilo." if avisos else "")
    pres = presupuesto or PRESUPUESTO_CONTEXTO_F5
    while True:
        max_ev, max_ci, recorte, vueltas = MAX_EVIDENCIAS_F5, MAX_CIFRAS_F5, 1.0, 0
        # Orden de recorte: fragmentos de texto -> cifras -> longitud de ventanas. Las cifras exactas son lo más valioso y se sacrifican al final.
        while True:   # reduce en este orden: fragmentos hasta 1 → cifras hasta 3 → recorte de ventanas → cifras hasta 1
            entradas = _armar_entradas(res, consulta, por_idx, max_ev, max_ci, recorte)
            ctx = _render_contexto(entradas)
            if tokens_f5(ctx) <= pres or vueltas > 80:
                break
            vueltas += 1
            if max_ev > 1:          # primero se sacrifican fragmentos de texto (las cifras exactas son lo más valioso)
                max_ev -= 1
            elif max_ci > 3:
                max_ci -= 1
            elif recorte > 0.45:
                recorte = round(recorte * 0.85, 3)
            elif max_ci > 1:
                max_ci -= 1
            elif recorte > 0.2:
                recorte = round(recorte * 0.85, 3)
            else:
                break
        prompt = f"CONTEXTO:\n{ctx}\n\nPREGUNTA: {consulta}{nota}\nRESPUESTA:"
        total = tokens_f5(system) + tokens_f5(prompt) + MAX_TOKENS_RESPUESTA
        if total <= TOPE_TOTAL_F5 or pres <= 150:
            break
        pres = int(pres * 0.92)   # si el total no cabe (p. ej. pregunta larga), se baja el presupuesto del contexto y se rearma
    # Se devuelve el paquete con las medidas de tokens y la lista de páginas válidas (la usa el verificador para detectar páginas inventadas).
    t_ctx = tokens_f5(ctx)
    return dict(system=system, prompt=prompt, contexto=ctx, entradas=entradas,
                filas=[f for en in entradas for f in en["filas"]],
                tokens=dict(system=tokens_f5(system), contexto=t_ctx, pregunta_y_formato=tokens_f5(prompt) - t_ctx, respuesta_max=MAX_TOKENS_RESPUESTA, total=total),
                c1_ok=t_ctx <= PRESUPUESTO_CONTEXTO_F5, c1b_ok=total <= TOPE_TOTAL_F5,
                presupuesto_usado=pres, metodos_ventana=[en["metodo_ventana"] for en in entradas if en["tipo"] == "texto"],
                paginas_validas=sorted({p for en in entradas for p in list(range(en["p_ini"], en["p_fin"] + 1)) + list(range(en["fis_ini"], en["fis_fin"] + 1))}))


# ====================================================================================================
# >>> CELDA 47 DEL CUADERNO
# ====================================================================================================
# Celda 47 · código
# ==============================================================================
# CLIENTE DE /v1/completions
# ==============================================================================
# Contadores de uso del servicio en esta sesión (llamadas, aciertos de caché, tokens cobrados).
ESTADISTICAS_F5 = dict(solicitudes=0, aciertos_cache=0, sin_credito=0, errores=0, tokens_prompt=0, tokens_completion=0)


# Lector tolerante: el servicio puede devolver el texto en distintos campos; esta función lo encuentra sea cual sea el formato.
def extraer_texto(js):
    """Lector tolerante: devuelve el texto generado sin importar en qué campo lo ponga el servicio."""
    if isinstance(js, str):
        return js.strip()
    if not isinstance(js, dict):
        return ""
    ch = js.get("choices")
    if isinstance(ch, list) and ch:
        c0 = ch[0] if isinstance(ch[0], dict) else {}
        if isinstance(c0.get("text"), str):
            return c0["text"].strip()
        m = c0.get("message")
        if isinstance(m, dict) and isinstance(m.get("content"), str):
            return m["content"].strip()
        if isinstance(m, dict) and isinstance(m.get("content"), list):
            return " ".join(x.get("text", "") for x in m["content"] if isinstance(x, dict)).strip()
    for clave in ("text", "response", "output", "completion", "content", "answer", "message"):
        v = js.get(clave)
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, dict):
            t = extraer_texto(v)
            if t:
                return t
    return ""


# Lee cuántos tokens cobró el servicio (de entrada y de salida).
def extraer_tokens(js):
    u = js.get("usage") if isinstance(js, dict) else None
    u = u if isinstance(u, dict) else {}
    p, c = u.get("prompt_tokens") or u.get("input_tokens"), u.get("completion_tokens") or u.get("output_tokens")
    return dict(prompt=p, completion=c, total=u.get("total_tokens") or ((p or 0) + (c or 0) if (p or c) else None))


# Empaqueta todas las salidas del cliente en un formato único (ok, estado, texto, tokens, intentos, latencia...).
def resultado_cliente(ok, estado, texto="", tokens=None, intentos=0, latencia_s=0.0, modelo=None, desde_cache=False, mensaje="", crudo=None):
    return dict(ok=ok, estado=estado, texto=texto, tokens=tokens or dict(prompt=None, completion=None, total=None), intentos=intentos,
                latencia_s=round(latencia_s, 2), modelo=modelo, desde_cache=desde_cache, mensaje=mensaje, crudo=crudo)


# El cliente: agrupa los datos de conexión (dirección, llave, modelo) y las operaciones disponibles.
class ClienteCompletions:
    def __init__(self, base_url, api_key, modelo=None, temperatura=0.0, max_tokens=350, timeout=60, reintentos=2, archivo_cache=None):
        import requests
        self.rq = requests
        self.base, self.modelo, self.temperatura = base_url.rstrip("/"), modelo, temperatura
        self.max_tokens, self.timeout, self.reintentos = max_tokens, timeout, reintentos
        self.usar_temperatura = temperatura is not None
        self.esquema = "prompt"                 # cambia a "messages" si el servicio rechaza prompt + system_prompt
        self.bloqueado = False                  # True tras un 402/403: ya no se intenta ninguna llamada
        self.sesion = requests.Session()
        # La llave de acceso se envía en el encabezado 'x-student-key' (así lo exige el servicio del curso).
        self.sesion.headers.update({"x-student-key": api_key or "", "Content-Type": "application/json"})
        self.archivo_cache = archivo_cache
        self.cache = {}
        try:
            if archivo_cache and os.path.exists(archivo_cache):
                self.cache = json.load(open(archivo_cache, encoding="utf-8"))
        except Exception:
            self.cache = {}

    # -- consultas gratuitas --------------------------------------------------------------------------------------
    # Consulta gratuita /health: el servicio 'duerme' y tarda unos 35-45 s en despertar; así la primera pregunta real no se pierde por tiempo.
    def precalentar(self):
        """GET /health (sin llave y sin costo): despierta el servicio para que la primera llamada real no sufra el arranque en frío."""
        try:
            t = time.perf_counter()
            r = self.rq.get(f"{self.base}/health", timeout=self.timeout)
            return dict(ok=r.status_code == 200, estado=r.status_code, segundos=round(time.perf_counter() - t, 1))
        except Exception as e:
            return dict(ok=False, estado=type(e).__name__, segundos=None)

    # Consulta gratuita /v1/models: lista los modelos disponibles.
    def listar_modelos(self):
        """GET /v1/models (sin llave y sin costo): devuelve los identificadores de modelos disponibles."""
        try:
            r = self.rq.get(f"{self.base}/v1/models", timeout=self.timeout)
            js = r.json()
            filas = js.get("data", js) if isinstance(js, dict) else js
            return [(m.get("id") if isinstance(m, dict) else str(m)) for m in filas]
        except Exception:
            return []

    # -- armado de la solicitud ---------------------------------------------------------------------------------------
    # Arma el contenido de la solicitud. Si el servicio rechaza el formato, el cliente prueba una variante (sin temperatura, o con formato de chat).
    def _cuerpo(self, system, prompt):
        c = dict(max_tokens=self.max_tokens)
        if self.modelo:
            c["model"] = self.modelo
        if self.esquema == "prompt":
            c.update(prompt=prompt, system_prompt=system)
        else:
            c["messages"] = [dict(role="system", content=system), dict(role="user", content=prompt)]
        if self.usar_temperatura:
            c["temperature"] = self.temperatura
        return c

    # Huella de la solicitud (modelo + instrucciones + pregunta + parámetros): misma huella = misma respuesta en caché.
    def _clave(self, system, prompt):
        return hashlib.sha256(json.dumps([self.modelo, system, prompt, self.temperatura, self.max_tokens], ensure_ascii=False).encode()).hexdigest()

    def _guardar_cache(self):
        try:
            if self.archivo_cache:
                json.dump(self.cache, open(self.archivo_cache, "w", encoding="utf-8"), ensure_ascii=False)
        except Exception:
            pass

    # -- llamada ----------------------------------------------------------------------------------------------------------
    # La llamada real. Orden: 1) caché (gratis) 2) ¿bloqueado por falta de crédito? 3) enviar y manejar la respuesta.
    def completar(self, system, prompt):
        clave = self._clave(system, prompt)
        if clave in self.cache:
            ESTADISTICAS_F5["aciertos_cache"] += 1
            return resultado_cliente(True, "ok", **self.cache[clave], desde_cache=True)
        if self.bloqueado:
            return resultado_cliente(False, "sin_credito", mensaje="El cliente ya recibió un 402/403: no se intenta otra llamada (cero solicitudes).")
        if not self.modelo:
            ids = self.listar_modelos()
            self.modelo = next((m for m in ids if "gpt" in str(m).lower()), ids[0] if ids else None)
        intentos, ajustes, fallas, t0 = 0, 0, 0, time.perf_counter()
        while True:
            intentos += 1
            ESTADISTICAS_F5["solicitudes"] += 1
            try:
                r = self.sesion.post(f"{self.base}/v1/completions", json=self._cuerpo(system, prompt), timeout=self.timeout)
            except (self.rq.Timeout, self.rq.ConnectionError) as e:
                fallas += 1
                if fallas > (1 if isinstance(e, self.rq.Timeout) else self.reintentos):   # un timeout se repite una sola vez (cada intento puede costar)
                    ESTADISTICAS_F5["errores"] += 1
                    return resultado_cliente(False, "sin_respuesta", intentos=intentos, latencia_s=time.perf_counter() - t0, modelo=self.modelo, mensaje=type(e).__name__)
                time.sleep(2 * fallas)
                continue
            code = r.status_code
            # 402/403 = sin crédito o llave rechazada. Se bloquea el cliente: reintentar no sirve y podría gastar más.
            if code in (402, 403):
                self.bloqueado = True
                ESTADISTICAS_F5["sin_credito"] += 1
                est = "sin_credito" if "credit" in r.text.lower() else "acceso_denegado"
                return resultado_cliente(False, est, intentos=intentos, latencia_s=time.perf_counter() - t0, modelo=self.modelo, mensaje=r.text[:200], crudo=r.text[:500])
            # 400/422 = el servicio no entendió el formato. Se ajusta una vez (sin temperatura) y otra vez (formato de chat).
            if code in (400, 422):
                ajustes += 1
                if ajustes <= 2:
                    if self.usar_temperatura:
                        self.usar_temperatura = False        # 1.er ajuste: probar sin temperatura
                    elif self.esquema == "prompt":
                        self.esquema = "messages"            # 2.º ajuste: probar con el formato de chat
                    else:
                        ajustes = 3
                    if ajustes <= 2:
                        continue
                ESTADISTICAS_F5["errores"] += 1
                return resultado_cliente(False, "solicitud_rechazada", intentos=intentos, latencia_s=time.perf_counter() - t0, modelo=self.modelo, mensaje=r.text[:300], crudo=r.text[:500])
            # 429 = demasiadas solicitudes. Se espera el tiempo que indique el servicio y se reintenta una sola vez.
            if code == 429:
                fallas += 1
                if fallas > 1:
                    ESTADISTICAS_F5["errores"] += 1
                    return resultado_cliente(False, "limite_de_tasa", intentos=intentos, latencia_s=time.perf_counter() - t0, modelo=self.modelo, mensaje="429")
                time.sleep(min(float(r.headers.get("Retry-After", 5) or 5), 20))
                continue
            # 5xx = falla del servidor. Se reintenta un número limitado de veces con pausa creciente.
            if code >= 500:
                fallas += 1
                if fallas > self.reintentos:
                    ESTADISTICAS_F5["errores"] += 1
                    # Si se agotan los reintentos, se conserva el texto que devolvió el servidor para poder diagnosticar la causa.
                    return resultado_cliente(False, "error_servidor", intentos=intentos, latencia_s=time.perf_counter() - t0, modelo=self.modelo, mensaje=f"HTTP {code}: {r.text[:300]}", crudo=r.text[:500])
                time.sleep(2 * fallas)
                continue
            try:
                js = r.json()
            except Exception:
                js = r.text
            texto, tokens = extraer_texto(js), extraer_tokens(js)
            ESTADISTICAS_F5["tokens_prompt"] += tokens["prompt"] or 0
            ESTADISTICAS_F5["tokens_completion"] += tokens["completion"] or 0
            if not texto:
                return resultado_cliente(False, "respuesta_vacia", tokens=tokens, intentos=intentos, latencia_s=time.perf_counter() - t0, modelo=self.modelo,
                                         mensaje="Texto vacío: revisa el campo «crudo» (formato) o sube max_tokens (algunos modelos gastan tokens razonando).", crudo=str(js)[:800])
            # Respuesta exitosa: se guarda en la caché (memoria y archivo) para que repetirla no cueste.
            self.cache[clave] = dict(texto=texto, tokens=tokens, intentos=intentos, latencia_s=round(time.perf_counter() - t0, 2), modelo=self.modelo)
            self._guardar_cache()
            return resultado_cliente(True, "ok", texto=texto, tokens=tokens, intentos=intentos, latencia_s=time.perf_counter() - t0, modelo=self.modelo)


# ====================================================================================================
# >>> CELDA 49 DEL CUADERNO
# ====================================================================================================
# Celda 49 · código
# ==============================================================================
# VERIFICADOR POSTERIOR
# ==============================================================================
# Patrones de texto (expresiones regulares) para reconocer números como 1,234.56 o 12.3 %, y para ignorar números que no son cifras (p. ej. 'NIF C-16', 'Anexo 33').
_NUM = re.compile(r"(?<![\w.,])(?P<signo>[-−–]?)(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?P<pct>\s?%)?")
_PERIODO = re.compile(r"\b([1-4])T(\d{2})\b", re.I)
_NO_CIFRAS = [   # patrones cuyos dígitos NO son cifras (se borran antes de buscar números)
    re.compile(r"\[E\d+\]"), re.compile(r"\b\d+(?:\.\d+){2,}\b"),
    re.compile(r"\bNIF\s*C-?\d+\b", re.I), re.compile(r"\bCriterio\s+B-?\d+\b", re.I), re.compile(r"\bAnexo\s+\d+\b", re.I), re.compile(r"\bIFRS\s*\d+\b", re.I),
    re.compile(r"\bEtapas?\s+[123](?:\s*(?:,|y|e)\s*[123])*", re.I), re.compile(r"(?m)^\s*\d+[.)]\s"),
]
_PAGINAS = re.compile(r"\b(?:p[áa]gs?\.?|p[áa]ginas?|pp?\.)\s*(?:impresas?|f[ií]sicas?)?\s*(\d+)(?:\s*[-–]\s*(\d+))?", re.I)
_UNIDAD_DESPUES = re.compile(r"^\s*(millones|miles|mil\b|por ciento|puntos)", re.I)


# Quita del texto lo que parece número pero no es cifra (páginas, periodos, normas) y devuelve las páginas y años que se mencionaron.
def _limpiar(texto):
    """Quita lo que no es cifra y devuelve (texto limpio, páginas mencionadas, años/periodos vistos)."""
    paginas = set()
    for m in _PAGINAS.finditer(texto):
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        paginas.update(range(a, b + 1) if b >= a and b - a < 60 else [a])
    t = _PAGINAS.sub(" ", texto)
    anios = {2000 + int(m.group(2)) for m in _PERIODO.finditer(t)}
    t = _PERIODO.sub(" ", t)
    for p in _NO_CIFRAS:
        t = p.sub(" ", t)
    return t, paginas, anios


# Convierte texto a número Decimal exacto (sin los errores de redondeo del punto flotante).
def _a_decimal(s):
    try:
        return Decimal(s.replace(",", ""))
    except InvalidOperation:
        return None


# Lista los números de un texto y los clasifica: cifra, año o entero pequeño.
def extraer_numeros(texto):
    """Lista de dicts (valor, firmado, es_cifra, es_anio, pos, fin). 'es_cifra' = decimal, con coma, con %, ≥ 100 o seguido de «millones»/«miles»."""
    return _numeros_limpios(_limpiar(texto)[0])


def _numeros_limpios(t):
    out = []
    for m in _NUM.finditer(t):
        raw = m.group("num")
        v = _a_decimal(raw)
        if v is None:
            continue
        firmado = -v if m.group("signo") else v
        es_anio = bool(re.fullmatch(r"(?:19|20)\d{2}", raw))
        es_cifra = ("." in raw) or ("," in raw) or bool(m.group("pct")) or v >= 100 or bool(_UNIDAD_DESPUES.match(t[m.end():m.end() + 20]))
        out.append(dict(valor=v, firmado=firmado, es_cifra=es_cifra and not es_anio, es_anio=es_anio, pos=m.start(), fin=m.end()))
    return out


_FIN_ORACION = re.compile(r"(?<!\bPs)(?<!\bpág)(?<!\bpágs)(?<!\bpp)(?<!\bp)(?<!\bvs)(?<=[.!?])\s+|\n+")


# Parte el texto en oraciones, cuidando no cortar en abreviaturas como 'pág.' o 'Ps.'.
def _oraciones(texto):
    """Parte el texto en oraciones (sin cortar en «Ps.», «pág.», «p.»)."""
    return [x for x in _FIN_ORACION.split(texto) if x and x.strip()]


# 'Palabras ancla' de una cifra: el nombre de su concepto. Se usan para comprobar que el número está junto a su concepto en la respuesta.
def _anclas(fila):
    """Palabras ancla de una fila de cifras: el nombre del concepto (p. ej. «Servicios») y, si no hay, el de la columna."""
    a = {w for w in tok(str(fila.get("fila") or "")) if not w.isdigit()} - {"total"}
    if not a:
        a = {w for w in tok(str(fila.get("columna") or "")) if not w.isdigit()}
    return a


# Devuelve ok=True solo si NO hay errores. Las advertencias no bloquean, pero quedan registradas. La fidelidad es la proporción de cifras verificadas.
def verificar_respuesta(texto, paq):
    """Devuelve dict(ok, errores, advertencias, fidelidad, cifras_total, cifras_verificadas, citas). ok=True solo si NO hay errores."""
    errores, avisos = [], []
    texto = (texto or "").strip()
    if not texto:
        return dict(ok=False, errores=[dict(tipo="respuesta_vacia", detalle="El modelo no devolvió texto.")], advertencias=[], fidelidad=0.0, cifras_total=0, cifras_verificadas=0, citas=[])
    es_abstencion = FRASE_SIN_DATO.lower()[:30] in texto.lower()
    # Comprobación de citas: cada [E#] debe existir, y la respuesta debe citar alguna (salvo que diga que no hay datos).
    # ---- 3) citas -------------------------------------------------------------------------------------------------------
    etiquetas_validas = {en["tag"] for en in paq["entradas"]}
    citas = sorted(set(re.findall(r"\[(E\d+)\]", texto)), key=lambda x: int(x[1:]))
    for c in citas:
        if c not in etiquetas_validas:
            errores.append(dict(tipo="cita_inexistente", detalle=f"[{c}] no está en el contexto entregado ({sorted(etiquetas_validas)})."))
    if not citas and not es_abstencion:
        errores.append(dict(tipo="sin_citas", detalle="La respuesta no cita ninguna evidencia [E#]."))
    _, paginas_dichas, _ = _limpiar(texto)
    # El modelo tiene prohibido escribir páginas; si las escribe y no pertenecen a la evidencia, es error.
    malas = sorted(p for p in paginas_dichas if p not in set(paq["paginas_validas"]))
    if malas:
        errores.append(dict(tipo="pagina_no_respaldada", detalle=f"Páginas mencionadas que no son de la evidencia: {malas}."))
    # Comprobación de cifras: todo número 'importante' de la respuesta debe estar literalmente en el contexto.
    # ---- 1) cifras --------------------------------------------------------------------------------------------------------
    ctx_nums = extraer_numeros(paq["contexto"])
    permitidos = {n["valor"] for n in ctx_nums}
    firmados = {n["firmado"] for n in ctx_nums}
    _, _, anios_ctx = _limpiar(paq["contexto"])
    anios_ctx |= {int(n["valor"]) for n in ctx_nums if n["es_anio"]}
    filas = paq["filas"]
    dichas = extraer_numeros(texto)
    texto_narrativo = {n["valor"] for n in extraer_numeros("\n".join(en["texto"] for en in paq["entradas"]))}
    verificadas, total, usadas = 0, 0, []
    for o in _oraciones(texto):
        limpia = _limpiar(o)[0]
        for n in _numeros_limpios(limpia):
            if not n["es_cifra"]:
                continue
            total += 1
            v = n["valor"]
            # Cifra no encontrada en el contexto = inventada o redondeada -> error.
            if v not in permitidos:
                errores.append(dict(tipo="cifra_no_respaldada", detalle=f"{v:,} no aparece en el contexto (¿cifra inventada o redondeada?)."))
                continue
            if n["firmado"] not in firmados:
                avisos.append(dict(tipo="signo_ambiguo", detalle=f"{v:,}: el contexto la trae con signo distinto; revisar si se pierde un «menos»."))
            coinc = [f for f in filas if f.get("valor") is not None and Decimal(str(f["valor"])) == v]
            usadas += coinc
            # Verificación estricta: la cifra debe estar cerca del nombre de su concepto (evita 'mezclar' cifras de filas distintas).
            if coinc and ESTRICTO_TERNA and v not in texto_narrativo:
                cerca = set(tok(limpia[max(0, n["pos"] - 70):n["fin"] + 20]))      # palabras a 70 letras antes y 20 después de la cifra
                if not any(_anclas(f) & cerca for f in coinc):
                    errores.append(dict(tipo="terna_no_concordante",
                                        detalle=f"{v:,} existe en el contexto, pero no aparece junto a su concepto ({', '.join(sorted(set().union(*[_anclas(f) for f in coinc])))})."))
                    continue
            verificadas += 1
    for n in dichas:     # años: solo se aceptan los que aparecen en el contexto
        if n["es_anio"] and int(n["valor"]) not in anios_ctx:
            errores.append(dict(tipo="anio_no_respaldado", detalle=f"El año {int(n['valor'])} no aparece en el contexto."))
    enteros = [n for n in dichas if not n["es_cifra"] and not n["es_anio"] and n["valor"] not in permitidos]
    if enteros:
        avisos.append(dict(tipo="enteros_no_verificados", detalle=f"Números pequeños sin respaldo literal (conteos u ordinales): {sorted({int(n['valor']) for n in enteros})}."))
    # ---- 3b) cifras de calidad dudosa que el paso 1 no contó (corrección de la Fase 6, hallazgo de la prueba C6) ------------
    # Antes, un valor ENTERO pequeño (p. ej. 35, sin coma ni decimales) no se consideraba «cifra»; si venía de una fila con
    # DISCREPANCIA_INTERNA, la respuesta podía citarlo SIN avisar y pasaba. Ahora se busca cada número de la respuesta entre las
    # filas de calidad dudosa: cuenta si es una cifra «grande» o si, siendo pequeño, aparece junto al nombre de su concepto.
    tl = _limpiar(texto)[0]
    for n in _numeros_limpios(tl):
        if n["es_anio"]:
            continue
        for f in filas:
            if f.get("valor") is None or f.get("calidad") in (None, "OK") or Decimal(str(f["valor"])) != n["valor"]:
                continue
            if n["es_cifra"] or (_anclas(f) & set(tok(tl[max(0, n["pos"] - 70):n["fin"] + 20]))):
                usadas.append(f)
    # Si se usó una cifra con calidad dudosa (sin validar, discrepancia, errata), la respuesta debe avisarlo.
    # ---- 4) advertencias de calidad ------------------------------------------------------------------------------------
    for f in {id(x): x for x in usadas}.values():
        pat = PATRON_ADVERTENCIA.get(f.get("calidad"))
        if pat and not pat.search(texto):
            errores.append(dict(tipo="falta_advertencia", detalle=f"Usa una cifra {f['calidad']} ({f.get('fila')} | {f.get('columna')}) sin avisarlo."))
    return dict(ok=not errores, errores=errores, advertencias=avisos, fidelidad=(verificadas / total if total else 1.0),
                cifras_total=total, cifras_verificadas=verificadas, citas=citas)


# Agrega al final de la respuesta qué significa cada [E#]: su sección y su página (impresa y física).
def traducir_citas(texto, paq):
    """Pie de la respuesta: cada [E#] citada se explica con su sección y su doble foliación oficial."""
    mapa = {en["tag"]: en for en in paq["entradas"]}
    citadas = sorted(set(re.findall(r"\[(E\d+)\]", texto)), key=lambda x: int(x[1:]))
    if not citadas:
        return ""
    return "Evidencias citadas:\n" + "\n".join(f"  [{c}] {mapa[c]['sub']} · {foliacion(mapa[c])}" for c in citadas if c in mapa)


# ====================================================================================================
# >>> CELDA 51 DEL CUADERNO
# ====================================================================================================
# Celda 51 · código
# ==============================================================================
# RESPUESTA «SOLO EVIDENCIA» Y PRESENTACIÓN DEL RESULTADO
# ==============================================================================
# Traducción de cada código de error a una frase entendible para el usuario.
MOTIVOS_FALLBACK = {
    "generacion_apagada": "la generación con el modelo está desactivada en esta sesión",
    "sin_credito": "el servicio rechazó la llamada por falta de crédito (HTTP 402/403)",
    "acceso_denegado": "el servicio rechazó la llave de acceso",
    "error_servidor": "el servicio devolvió errores del servidor",
    "sin_respuesta": "el servicio no respondió a tiempo",
    "solicitud_rechazada": "el servicio no aceptó la solicitud",
    "respuesta_vacia": "el modelo devolvió una respuesta vacía",
    "limite_de_tasa": "el servicio pidió esperar (límite de solicitudes)",
    "verificacion_fallida": "el verificador encontró cifras, citas o advertencias no respaldadas en el texto del modelo",
    "sin_cliente": "no hay cliente de generación configurado",
}


# Respuesta 100 % determinista: lista las evidencias [E#] con su página y las cifras con valor, unidad, periodo, perímetro, calidad y fuente. Nada la inventa.
def responder_solo_evidencia(res, paq, motivo="generacion_apagada", avisos=()):
    """Respuesta 100 % determinista: evidencia citada + cifras exactas con su estado. No interviene ningún modelo."""
    L = [f"MODO «SOLO EVIDENCIA»: {MOTIVOS_FALLBACK.get(motivo, motivo)}.", f"Pregunta: {res['consulta'][:400]}"]
    L += [a for a in avisos]
    L.append("")
    for en in paq["entradas"]:
        if en["tipo"] == "texto":
            extracto = en["texto"][:420].replace("\n", " ") + ("…" if len(en["texto"]) > 420 else "")
            L.append(f"[{en['tag']}] {en['sub']} · {foliacion(en)}\n    «{extracto}»")
        else:
            L.append(f"[{en['tag']}] {en['sub']} · {foliacion(en)}")
        for f in en["filas"]:
            valor = fmt_valor(f["valor"]) if f.get("valor") is not None else (f.get("texto") or "s/d")
            et = ETIQUETA_CALIDAD.get(f.get("calidad"), "")
            L.append(f"    • {f.get('fila')} | {f.get('columna')} = {valor} {f.get('unidad') or ''} ({f.get('periodo')}; {f.get('perimetro') or 'sin perímetro'}) "
                     f"· calidad: {f.get('calidad')} {et}\n      fuente: {f.get('fuente') or ''}")
    # Si alguna cifra no está plenamente validada, se agrega una nota explicando qué significa cada estado.
    nq = sorted({f["calidad"] for f in paq["filas"] if f.get("calidad") != "OK"})
    if nq:
        L.append("\nNota de calidad: hay cifras con estado " + ", ".join(nq) + ". SIN_VALIDAR = no pudo verificarse con las identidades contables del informe; "
                 "ERRATA_FUENTE y DISCREPANCIA_INTERNA = anomalías del propio informe.")
    return "\n".join(L)


# Imprime el resultado: consulta, ruta seguida (generado / solo evidencia / abstención), tokens estimados, verificación y respuesta.
def mostrar_resultado(r):
    print("=" * 100)
    print("CONSULTA:", r["consulta"][:300])
    print(f"RUTA: {r['ruta']} | estado de alcance: {r['estado']}" + (f" | motivo: {r['motivo_fallback']}" if r.get("motivo_fallback") else ""))
    if r.get("tokens"):
        t = r["tokens"]
        print(f"TOKENS (estimados): contexto {t['contexto']} | total con respuesta {t['total']} | C1 {'sí' if r['c1_ok'] else 'NO'} | C1b {'sí' if r['c1b_ok'] else 'NO'} | ventanas: {r['metodos_ventana']}")
    if r.get("verificacion"):
        v = r["verificacion"]
        print(f"VERIFICACIÓN: {'aprobada' if v['ok'] else 'RECHAZADA'} | fidelidad {v['fidelidad']:.2f} ({v['cifras_verificadas']}/{v['cifras_total']}) | errores: {[e['tipo'] for e in v['errores']]}")
    print("-" * 100)
    print(r["respuesta"])


# ====================================================================================================
# >>> CELDA 54 DEL CUADERNO
# ====================================================================================================
# Celda 54 · código
# ==============================================================================
# INSTALACIÓN, PARÁMETROS Y ESTADO DEL AGENTE
# ==============================================================================
# LangGraph: librería para modelar el proceso como un grafo de pasos (nodos) con decisiones entre ellos.
import operator   # time, json y os ya se importaron en la Parte 4
from typing import TypedDict, Annotated, Optional, Any, Dict, List

try:
    # Verifica que las Partes 4 y 5 ya se ejecutaron.
    (recuperar, buscar_cifras, evaluar_guardia_f5, construir_prompt, ClienteCompletions, verificar_respuesta,
     traducir_citas, responder_solo_evidencia, mostrar_resultado, MOTIVOS_FALLBACK, ESTADISTICAS_SERVICIO, ESTADISTICAS_F5,
     PRESUPUESTO_CONTEXTO_F5, TOPE_TOTAL_F5, sql_uno, conectar, ESQUEMA, RUTA_PROYECTO, CORPUS)
except NameError as e:
    raise RuntimeError(f"Falta {e}. Ejecuta antes las Partes 4 y 5.")

from langgraph.graph import StateGraph, START, END

CORRIDA_F6 = time.strftime("f6-%Y%m%d-%H%M%S")
ETAPA_BITACORA_F6 = "6_consulta"
# Parámetros del agente: tope de pasos por consulta, tope de generaciones por sesión (protege el crédito) y tope de intentos de red.
RECURSION_LIMIT_F6 = 25          # máximo de pasos que LangGraph permite por consulta (la ruta más larga usa 6)
MAX_GENERACIONES_SESION = 20     # tope de llamadas al modelo por sesión (protege el crédito cuando regrese)
MAX_INTENTOS_HTTP = 3            # intentos de red máximos por consulta (incluye reintentos del cliente)


# Regla para combinar los tiempos de cada paso: se van sumando entradas al mismo diccionario.
def _unir_dicts(a, b):
    """Regla para combinar los tiempos de cada paso: se van sumando entradas al mismo diccionario."""
    return {**(a or {}), **(b or {})}


# El 'estado' es la hoja de trabajo del agente: cada paso lee lo que necesita y devuelve solo lo que cambia.
# Los campos marcados con 'operator.add' o '_unir_dicts' se acumulan (avisos, pasos recorridos y tiempos) en lugar de reemplazarse.
class EstadoAgente(TypedDict, total=False):
    """La hoja de trabajo del agente. Cada paso devuelve SOLO lo que cambia; las listas y el diccionario marcados se acumulan."""
    consulta: str                                  # la pregunta tal como se escribió
    res: Dict[str, Any]                            # lo que devolvió la recuperación (Fase 4)
    guardia: Dict[str, Any]                        # decisión previa: abstenerse o generar (Fase 5)
    estado_guardia: str                            # fuera_de_alcance / sin_soporte / sin_evidencia / posible_fuera_de_alcance / ok
    avisos: Annotated[List[str], operator.add]     # avisos que se anteponen a la respuesta
    paquete: Dict[str, Any]                        # prompt armado, fichas [E#] y medidas de tokens (Fase 5)
    presupuesto_ok: bool                           # ¿cabe en 600 tokens de contexto y 1,250 en total?
    salida_cliente: Dict[str, Any]                 # lo que informó el cliente de /v1/completions
    respuesta_cruda: str                           # texto del modelo (antes de verificarlo)
    verificacion: Dict[str, Any]                   # detalle de la verificación (errores, fidelidad)
    texto_rechazado: str                           # texto del modelo que NO pasó la verificación (solo auditoría)
    respuesta_final: str                           # lo único que se muestra al usuario
    modo_salida: str                               # generacion / solo_evidencia / abstencion
    motivo_fallback: Optional[str]                 # por qué se usó «solo evidencia» (None si no hubo plan B)
    pasos_recorridos: Annotated[List[str], operator.add]          # nodos visitados, en orden
    tiempos_ms: Annotated[Dict[str, float], _unir_dicts]          # milisegundos por nodo (y red del modelo por separado)
    telemetria: Dict[str, Any]                     # renglón que se guardó en la bitácora


# ====================================================================================================
# >>> CELDA 56 DEL CUADERNO
# ====================================================================================================
# Celda 56 · código
# ==============================================================================
# RECUPERACIÓN CON RESCATE NUMÉRICO Y CONTROL DE PRESUPUESTO
# ==============================================================================
# Se guarda la función original para poder envolverla sin perderla.
_RECUPERAR_F4 = recuperar          # guarda la función de la Fase 4 (Parte 4); esta celda no la reemplaza


# Recuperación con 'rescate numérico': evita rechazar preguntas de cifras cuyo texto no tiene suficientes palabras en común, pero cuyos datos sí están en SQL.
def recuperar_f6(q, **kw):
    """Recuperación de la Fase 4 + rescate numérico + orden de cifras validado."""
    # 'global' permite modificar el umbral de la Parte 4 solo durante esta llamada.
    global MIN_SOPORTE_LEXICO                          # el umbral de «sin soporte» vive en la Parte 4
    res = _RECUPERAR_F4(q, **kw)
    g = res.get("guardia", {})
    if res.get("estado") == "sin_soporte" and g.get("fuente") == "lexico":
        previa = buscar_cifras(q)                      # ¿el enrutador SQL encuentra cifras para esta pregunta?
        if previa.get("estado") == "ok" and previa.get("filas"):
            umbral_original = MIN_SOPORTE_LEXICO
            # Se baja el umbral a cero únicamente en esta llamada; el 'finally' lo restaura SIEMPRE, aunque algo falle.
            MIN_SOPORTE_LEXICO = 0.0                   # solo durante esta llamada: deja pasar la guardia de texto
            try:
                res2 = _RECUPERAR_F4(q, **kw)
            finally:
                MIN_SOPORTE_LEXICO = umbral_original   # se restaura SIEMPRE, aunque algo falle
            if res2.get("estado") in ("ok", "posible_fuera_de_alcance"):
                res = res2
                res["guardia"] = dict(res2.get("guardia", {}), rescate_numerico=True, puntaje_original=g.get("puntaje"), umbral=umbral_original)
                res["rescate_numerico"] = True
    if res.get("cifras", {}).get("estado") == "ok":
        validada = buscar_cifras(q)                    # mismo orden que el banco numérico de la Fase 4
        if validada.get("estado") == "ok":
            res["cifras"] = validada
            res["cifras_ranking"] = "fase4_validado"
    return res


# Se agrega un motivo nuevo al catálogo: tope de generaciones de la sesión alcanzado.
MOTIVOS_FALLBACK.setdefault("limite_sesion", "se alcanzó el tope de generaciones de la sesión (control de presupuesto)")


# El guardián de gastos: lleva la cuenta de llamadas, intentos y tokens, y decide si se puede seguir llamando al modelo.
class ControlPresupuesto:
    """El guardián de gastos. Una instancia (CONTROL_F6) vigila toda la sesión."""

    def __init__(self, max_generaciones=None, max_intentos=None):
        self.max_generaciones = MAX_GENERACIONES_SESION if max_generaciones is None else max_generaciones
        self.max_intentos = MAX_INTENTOS_HTTP if max_intentos is None else max_intentos
        self.generaciones = 0
        self.intentos_http = 0
        self.tokens_reales = 0
        self.alertas = []
        self.base_embeddings = ESTADISTICAS_SERVICIO["solicitudes"]    # solicitudes a embeddings al empezar

    # Devuelve (permitido, motivo). No se puede consultar el saldo: la falta de crédito solo se descubre al llamar.
    def puede_generar(self, cliente):
        """(True, None) si se puede llamar al modelo; si no, (False, motivo). No se puede consultar el saldo: el 403 solo aparece al llamar."""
        if cliente is None:
            return False, "sin_cliente"
        if getattr(cliente, "bloqueado", False):
            return False, "sin_credito"
        if self.generaciones >= self.max_generaciones:
            return False, "limite_sesion"
        return True, None

    # Contabiliza una llamada al modelo. Una respuesta tomada de la caché no cuesta y no se cuenta.
    def registrar_llamada(self, salida):
        if salida.get("desde_cache"):
            return                                    # una respuesta guardada no cuesta nada
        self.generaciones += 1
        self.intentos_http += salida.get("intentos") or 0
        self.tokens_reales += (salida.get("tokens") or {}).get("total") or 0
        if (salida.get("intentos") or 0) > self.max_intentos:
            self.alertas.append(f"intentos de red {salida.get('intentos')} > {self.max_intentos}")

    # Lista de comprobaciones de seguridad (no se pidieron embeddings, se respetó el presupuesto de tokens, no se excedieron los intentos).
    def invariantes(self, paquete=None, salida=None):
        """Revisa los topes de una consulta. Devuelve un dict de verdadero/falso (todo verdadero = cumple)."""
        inv = dict(embeddings_ok=ESTADISTICAS_SERVICIO["solicitudes"] == self.base_embeddings)
        if paquete:
            inv.update(c1=paquete["tokens"]["contexto"] <= PRESUPUESTO_CONTEXTO_F5, c1b=paquete["tokens"]["total"] <= TOPE_TOTAL_F5)
        if salida:
            inv["intentos_ok"] = (salida.get("intentos") or 0) <= self.max_intentos
        return inv

    def resumen(self):
        return dict(generaciones=self.generaciones, tope=self.max_generaciones, intentos_http=self.intentos_http, tokens_reales=self.tokens_reales,
                    solicitudes_embeddings_nuevas=ESTADISTICAS_SERVICIO["solicitudes"] - self.base_embeddings, alertas=list(self.alertas))


# Un solo guardián vigila toda la sesión.
CONTROL_F6 = ControlPresupuesto()


# ====================================================================================================
# >>> CELDA 58 DEL CUADERNO
# ====================================================================================================
# Celda 58 · código
# ==============================================================================
# NODOS, ARISTAS Y GRAFO
# ==============================================================================
# Decorador: envuelve cada paso para medir su tiempo y anotar que pasó por él. Un decorador agrega comportamiento a una función sin cambiarla.
def nodo(nombre):
    """Decorador: mide el tiempo del nodo, anota que pasó por él y le entrega la configuración. (No usa functools.wraps a propósito:
    LangGraph reconoce el parámetro llamado «config» por el nombre en la firma de la función.)"""
    def deco(fn):
        def envuelto(state, config=None):
            t = time.perf_counter()
            cfg = (config or {}).get("configurable", {}) if config else {}
            cambios = dict(fn(state, cfg) or {})
            tiempos = dict(cambios.pop("tiempos_ms", {}))
            tiempos[nombre] = round((time.perf_counter() - t) * 1000, 2)
            cambios["tiempos_ms"] = tiempos
            cambios["pasos_recorridos"] = [nombre]
            return cambios
        envuelto.__name__ = nombre
        return envuelto
    return deco


# ¿Está encendida la generación con el modelo? Se decide por consulta o, en su defecto, por el parámetro general.
def _generar_activo(cfg):
    g = cfg.get("generar")
    return USAR_GENERACION if g is None else g


# PASO 1 · RECUPERAR: busca evidencia y cifras, y decide si hay que abstenerse (guardia).
@nodo("recuperar")
def n_recuperar(s, cfg):
    """Paso 1: toda la recuperación de la Fase 4 (normaliza, guardia, búsqueda, cifras) y la decisión previa de la Fase 5."""
    res = cfg["res"] if cfg.get("res") is not None else recuperar_f6(s["consulta"])
    g = evaluar_guardia_f5(res)
    return dict(res=res, guardia=g, estado_guardia=g["estado"], avisos=list(g["avisos"]))


# Decisión 1: abstenerse o seguir hacia el armado del contexto.
def ruta_guardia(s):
    return "abstener" if s["guardia"]["accion"] == "abstener" else "construir_contexto"


# SALIDA A · ABSTENER: mensaje fijo, sin costo y sin modelo.
@nodo("abstener")
def n_abstener(s, cfg):
    """Salida A: mensaje fijo de abstención. No se consulta nada más ni se llama al modelo."""
    return dict(respuesta_final=s["guardia"]["mensaje"], modo_salida="abstencion", motivo_fallback=None)


# PASO 2 · CONSTRUIR CONTEXTO: arma el prompt [E#] y decide si se puede llamar al modelo (generación encendida, crédito disponible, tope de sesión, presupuesto de tokens).
@nodo("construir_contexto")
def n_construir_contexto(s, cfg):
    """Paso 2: arma el prompt con [E#] y revisa el presupuesto; además decide si se puede llamar al modelo (compuerta estática)."""
    paq = construir_prompt(s["res"], por_idx=cfg.get("por_idx"), avisos=s["avisos"])
    presupuesto_ok = bool(paq["c1_ok"] and paq["c1b_ok"])
    motivo = None
    if not _generar_activo(cfg):
        motivo = "generacion_apagada"
    else:
        puede, m = (cfg.get("control") or CONTROL_F6).puede_generar(cfg.get("cliente"))
        if not puede:
            motivo = m
        elif not presupuesto_ok:
            motivo = "verificacion_fallida"      # el prompt excede el presupuesto: no se gasta crédito en él
    return dict(paquete=paq, presupuesto_ok=presupuesto_ok, motivo_fallback=motivo)


# Decisión 2: si hay motivo para no llamar al modelo, se va directo a 'solo evidencia'.
def ruta_contexto(s):
    return "solo_evidencia" if s.get("motivo_fallback") else "generar"


# PASO 3 · GENERAR: la única llamada al modelo de todo el flujo.
@nodo("generar")
def n_generar(s, cfg):
    """Paso 3: la única llamada al modelo (/v1/completions). Si falla (403, 5xx…), se anota el motivo y se sigue por «solo evidencia»."""
    control = cfg.get("control") or CONTROL_F6
    paq = s["paquete"]
    out = cfg["cliente"].completar(paq["system"], paq["prompt"])
    control.registrar_llamada(out)
    cambios = dict(salida_cliente=dict(estado=out["estado"], latencia_s=out["latencia_s"], tokens=out["tokens"], intentos=out["intentos"],
                                      modelo=out["modelo"], desde_cache=out["desde_cache"], mensaje=out["mensaje"]),
                   respuesta_cruda=out.get("texto") or "",
                   tiempos_ms={"generar_red": round((out["latencia_s"] or 0) * 1000, 2)})     # tiempo de red, aparte del cómputo local
    if not out["ok"]:
        cambios["motivo_fallback"] = out["estado"]
    return cambios


# Decisión 3: si la llamada falló, plan B; si no, a verificar.
def ruta_generacion(s):
    return "solo_evidencia" if s.get("motivo_fallback") else "verificar"


# PASO 4 · VERIFICAR: revisa cifras, citas y advertencias del texto del modelo.
@nodo("verificar")
def n_verificar(s, cfg):
    """Paso 4: revisa cifras, concepto, citas y advertencias. Si algo falla, el texto del modelo NO se muestra."""
    v = verificar_respuesta(s["respuesta_cruda"], s["paquete"])
    cambios = dict(verificacion=v)
    if not v["ok"]:
        cambios.update(motivo_fallback="verificacion_fallida", texto_rechazado=s["respuesta_cruda"])
    return cambios


# Decisión 4: si no pasó la verificación, el texto se descarta y se va a 'solo evidencia'.
def ruta_verificacion(s):
    return "solo_evidencia" if s.get("motivo_fallback") else "finalizar"


# SALIDA B · FINALIZAR: respuesta del modelo ya verificada, con avisos al inicio y la explicación de las citas al final.
@nodo("finalizar")
def n_finalizar(s, cfg):
    """Salida B: respuesta del modelo verificada, con los avisos al principio y las citas explicadas al final."""
    prefijo = ("\n".join(s["avisos"]) + "\n\n") if s["avisos"] else ""
    pie = traducir_citas(s["respuesta_cruda"], s["paquete"])
    return dict(respuesta_final=prefijo + s["respuesta_cruda"] + ("\n\n" + pie if pie else ""), modo_salida="generacion", motivo_fallback=None)


# SALIDA C · SOLO EVIDENCIA: plan B determinista, sin modelo.
@nodo("solo_evidencia")
def n_solo_evidencia(s, cfg):
    """Salida C (plan B): citas y cifras exactas tal como están en la base, sin intervención del modelo."""
    return dict(respuesta_final=responder_solo_evidencia(s["res"], s["paquete"], s.get("motivo_fallback") or "generacion_apagada", s["avisos"]),
                modo_salida="solo_evidencia")


# Guarda una fila de telemetría en la tabla bitácora; si la base falla, no se detiene la consulta.
def registrar_bitacora_f6(fila):
    """Guarda el renglón en ingesta_bitacora (solo se insertan renglones; si la base falla, no se detiene nada)."""
    try:
        from psycopg2.extras import Json
        conn = conectar()
        try:
            with conn.cursor() as cur:
                cur.execute(f"INSERT INTO {ESQUEMA}.ingesta_bitacora_v1 (corrida, etapa, detalle) VALUES (%s, %s, %s)",
                            (CORRIDA_F6, ETAPA_BITACORA_F6, Json(fila, dumps=lambda o: json.dumps(o, default=str))))
            conn.commit()
        finally:
            conn.close()
        return True
    except Exception as e:
        return f"no ({type(e).__name__})"


TRAZA_F6 = []
# Traduce el modo de salida del grafo al nombre de ruta que se muestra al usuario.
RUTA_DE_MODO = {"abstencion": "abstencion", "generacion": "generado", "solo_evidencia": "solo_evidencia"}


# PASO FINAL · TELEMETRÍA: toda consulta, sea cual sea su ruta, termina aquí dejando una fila con ruta, pasos, tiempos y verificación de topes.
@nodo("telemetria")
def n_telemetria(s, cfg):
    """Último paso de TODAS las rutas: deja un renglón por consulta con la ruta, los pasos, los tiempos y los topes."""
    control = cfg.get("control") or CONTROL_F6
    paq, salida = s.get("paquete"), s.get("salida_cliente")
    pasos = list(s["pasos_recorridos"]) + ["telemetria"]
    fila = dict(consulta=s["consulta"][:200], ruta=RUTA_DE_MODO.get(s.get("modo_salida"), s.get("modo_salida")), estado=s["estado_guardia"],
                motivo=s.get("motivo_fallback"), pasos=pasos, n_pasos=len(pasos), tiempos_ms=dict(s["tiempos_ms"]),
                tokens_prompt_est=(paq or {}).get("tokens", {}).get("total"), tokens_reales=(salida or {}).get("tokens"),
                llamo_al_modelo=bool(salida), fidelidad=(s.get("verificacion") or {}).get("fidelidad"),
                errores_verificador=[e["tipo"] for e in (s.get("verificacion") or {}).get("errores", [])],
                invariantes=control.invariantes(paq, salida))
    fila["en_bitacora"] = registrar_bitacora_f6(fila) if REGISTRAR_BITACORA else "desactivada"
    TRAZA_F6.append(fila)
    return dict(telemetria=fila)


# ---- Topología definida UNA vez como datos --------------------------------------------------------------------------------------
# ---- Mapa del grafo definido como datos: nodos, aristas fijas (siempre se siguen) y aristas condicionales (dependen de una decisión) ----
NODOS_F6 = {"recuperar": n_recuperar, "abstener": n_abstener, "construir_contexto": n_construir_contexto, "generar": n_generar,
            "verificar": n_verificar, "finalizar": n_finalizar, "solo_evidencia": n_solo_evidencia, "telemetria": n_telemetria}
ARISTAS_FIJAS_F6 = [(START, "recuperar"), ("abstener", "telemetria"), ("finalizar", "telemetria"), ("solo_evidencia", "telemetria"), ("telemetria", END)]
ARISTAS_COND_F6 = {
    "recuperar": (ruta_guardia, {"abstener": "abstener", "construir_contexto": "construir_contexto"}),
    "construir_contexto": (ruta_contexto, {"solo_evidencia": "solo_evidencia", "generar": "generar"}),
    "generar": (ruta_generacion, {"solo_evidencia": "solo_evidencia", "verificar": "verificar"}),
    "verificar": (ruta_verificacion, {"solo_evidencia": "solo_evidencia", "finalizar": "finalizar"}),
}


# Arma el grafo de LangGraph a partir de las tablas de arriba y lo compila.
def construir_grafo():
    """Arma el grafo de LangGraph a partir de los datos de arriba."""
    g = StateGraph(EstadoAgente)
    for nombre, fn in NODOS_F6.items():
        g.add_node(nombre, fn)
    for a, b in ARISTAS_FIJAS_F6:
        g.add_edge(a, b)
    for origen, (router, mapa) in ARISTAS_COND_F6.items():
        g.add_conditional_edges(origen, router, mapa)
    return g.compile()


# Se compila el grafo una sola vez.
GRAFO_F6 = construir_grafo()


# Ejecuta el agente completo para una pregunta y devuelve el estado final con la ruta recorrida y los tiempos.
def ejecutar_grafo(q, cliente=None, generar=None, res=None, por_idx=None, control=None, limite=None):
    """Ejecuta el agente completo para una pregunta y devuelve el estado final."""
    inicial = {"consulta": q, "avisos": [], "pasos_recorridos": [], "tiempos_ms": {}}
    config = {"recursion_limit": limite or RECURSION_LIMIT_F6,
              "configurable": {"cliente": cliente, "generar": generar, "res": res, "por_idx": por_idx, "control": control}}
    t = time.perf_counter()
    final = GRAFO_F6.invoke(inicial, config)
    final["tiempos_ms"]["total"] = round((time.perf_counter() - t) * 1000, 2)
    return final


# Convierte el estado final del grafo al formato que entiende mostrar_resultado.
def a_resultado(f):
    """Pasa el estado final al mismo formato de resultado de la Fase 5 (para reutilizar mostrar_resultado y las comparaciones)."""
    paq = f.get("paquete") or {}
    return dict(consulta=f["consulta"], ruta=RUTA_DE_MODO[f["modo_salida"]], estado=f["estado_guardia"], respuesta=f["respuesta_final"],
                motivo_fallback=f.get("motivo_fallback"), tokens=paq.get("tokens"), c1_ok=paq.get("c1_ok"), c1b_ok=paq.get("c1b_ok"),
                metodos_ventana=paq.get("metodos_ventana"), verificacion=f.get("verificacion"), texto_rechazado=f.get("texto_rechazado"),
                tiempos=f["tiempos_ms"], pasos=f["pasos_recorridos"], paquete=paq)


# Consulta completa: ejecuta el grafo, imprime el resultado y la ruta recorrida.
def consultar_f6(q, cliente=None, generar=None, ver_pasos=True, **kw):
    f = ejecutar_grafo(q, cliente=cliente, generar=generar, **kw)
    r = a_resultado(f)
    mostrar_resultado(r)
    if ver_pasos:
        print("-" * 100)
        print("RUTA DEL GRAFO:", " → ".join(f["pasos_recorridos"]), f"({len(f['pasos_recorridos'])} pasos)")
        print("TIEMPOS (ms):", f["tiempos_ms"])
    return f


# ====================================================================================================
# >>> DESPLIEGUE (reemplaza la celda 61 del cuaderno: sin widgets ni llamadas de prueba que gasten crédito)
# ====================================================================================================
import threading
_CERROJO = threading.Lock()      # el núcleo usa variables globales: se atiende una pregunta a la vez
CLIENTE_F6 = None


def iniciar_modelo(max_llamadas=20):
    """Crea el cliente del servicio de IA y el control de presupuesto. No hace llamadas que cuesten (solo prepara)."""
    global USAR_GENERACION, CLIENTE_F6, CONTROL_F6
    CLIENTE_F6 = ClienteCompletions(BASE_URL_F5, API_KEY_F5, modelo=MODELO_GENERACION, temperatura=TEMPERATURA,
                                    max_tokens=MAX_TOKENS_RESPUESTA, timeout=TIMEOUT_S, reintentos=REINTENTOS,
                                    archivo_cache=ARCHIVO_CACHE_F5)
    CONTROL_F6 = ControlPresupuesto(max_generaciones=max_llamadas)
    USAR_GENERACION = True
    return CLIENTE_F6


def responder(pregunta, usar_modelo=False):
    """Ejecuta el grafo de LangGraph para una pregunta y devuelve un diccionario listo para convertir a JSON."""
    with _CERROJO:
        f = ejecutar_grafo(pregunta, cliente=CLIENTE_F6 if usar_modelo else None, generar=bool(usar_modelo))
        r = a_resultado(f)
        sc = f.get("salida_cliente") or {}
        ver = r.get("verificacion") or {}
        return dict(
            consulta=r["consulta"], ruta=r["ruta"], estado=r["estado"], motivo=r.get("motivo_fallback"),
            respuesta=r["respuesta"], pasos=r["pasos"], tiempos_ms=r["tiempos"], tokens_prompt_est=(r.get("tokens") or {}),
            modelo=dict(nombre=sc.get("modelo"), tokens=sc.get("tokens"), latencia_s=sc.get("latencia_s"),
                        desde_cache=sc.get("desde_cache")) if sc else None,
            verificacion=dict(aprobada=ver.get("ok"), fidelidad=ver.get("fidelidad"),
                              errores=[e.get("tipo") for e in ver.get("errores", [])]) if ver else None,
            sesion=CONTROL_F6.resumen())
