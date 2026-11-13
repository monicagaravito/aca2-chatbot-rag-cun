"""
rag_core.py - Núcleo del chatbot RAG (ACA 2 - Computación en la Nube, CUN)
==========================================================================
Pilar: Educación - normatividad institucional y permanencia estudiantil.

Este módulo lo usan tanto el notebook del ACA 2 como la app de Streamlit:

  PASO 1  Conexión con la base vectorial del ACA 1 (Qdrant Cloud).
  PASO 2  Configuración del LLM (Google Gemini vía API, credenciales seguras).
  PASO 3  Pipeline RAG: recuperación Top-k + prompt fundamentado + citas.
  PASO 4  Funciones de gestión documental que consume la interfaz gráfica.
  PASO 5  Utilidades de evaluación (latencia, citas válidas, abstención).

Las credenciales NUNCA se escriben en el código: se leen de variables de
entorno (.env), de los secrets de Streamlit o de los secrets de Colab.
"""
from __future__ import annotations

import os
import re
import time
import uuid

import pandas as pd
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

# ---------------------------------------------------------------------------
# Configuración general (debe coincidir con la indexación del ACA 1)
# ---------------------------------------------------------------------------
COLLECTION_NAME = "cun_educacion_permanencia_rag"
EMBED_MODEL = "gemini-embedding-001"   # mismo modelo de embeddings del ACA 1
EMBED_DIM = 768                        # misma dimensión del ACA 1
TIPO_NORMATIVO = "documento_normativo_institucional"
TIPO_PERFIL = "registro_tabular_uci_697"

# Modelos LLM candidatos, en orden de preferencia. Si Google retira uno,
# el sistema pasa automáticamente al siguiente disponible.
LLM_CANDIDATOS = [
    "gemini-3.1-flash-lite",   # mayor cuota en la capa gratuita y menor latencia (verificado en pruebas)
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]

# Hiperparámetros de generación (justificación en el notebook, sección 3)
HIPERPARAMETROS = {
    "temperature": 0.1,        # casi determinista: prioriza fidelidad sobre creatividad
    "top_p": 0.9,              # recorta tokens poco probables (menos "inventos")
    "max_output_tokens": 1024, # respuestas completas pero acotadas
}
TOP_K = 4                 # fragmentos que se inyectan en el prompt
UMBRAL_SIMILITUD = 0.45   # por debajo de este coseno el fragmento se descarta
TURNOS_HISTORIAL = 3      # turnos previos que se envían como memoria

SYSTEM_PROMPT = (
    "Eres el Asistente Normativo de la Corporación Unificada Nacional de Educación Superior (CUN), "
    "especializado en reglamento estudiantil, bienestar institucional, becas y permanencia estudiantil.\n"
    "REGLAS ESTRICTAS DE GENERACIÓN FUNDAMENTADA:\n"
    "1. Responde ÚNICAMENTE con la información explícita de los FRAGMENTOS DE EVIDENCIA suministrados.\n"
    "2. NO inventes ni extrapoles requisitos, promedios, porcentajes, montos, plazos ni conclusiones "
    "que no aparezcan textualmente en los fragmentos.\n"
    "3. Si la evidencia no contiene la respuesta, dilo expresamente con la frase: "
    "'La evidencia recuperada no contiene información sobre esa consulta.' y no agregues datos externos.\n"
    "4. Si la pregunta no tiene relación con la CUN, su normatividad o la permanencia estudiantil, "
    "indica amablemente que está fuera del alcance del asistente.\n"
    "5. Cita la fuente de cada afirmación o cifra con el identificador exacto entre corchetes, "
    "por ejemplo [DOC-reglamento-estudiantil-012] o [EST-0042]. Usa un corchete por cada ID.\n"
    "6. El HISTORIAL sirve solo para entender a qué se refiere el usuario; la evidencia manda.\n"
    "7. Redacta en español claro y breve, con viñetas cuando haya varios requisitos."
)

PATRON_CITA = re.compile(r"\[((?:DOC|EST)-[^\]]+)\]")
FRASE_ABSTENCION = "no contiene información"


# ---------------------------------------------------------------------------
# Credenciales (gestión segura)
# ---------------------------------------------------------------------------
def leer_secreto(nombre: str) -> str | None:
    """Busca una credencial en: variables de entorno/.env -> secrets de Streamlit -> secrets de Colab."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    valor = os.getenv(nombre)
    if valor:
        return valor
    try:
        import streamlit as st
        if nombre in st.secrets:
            return st.secrets[nombre]
    except Exception:
        pass
    try:
        from google.colab import userdata
        return userdata.get(nombre)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PASO 1 - Conexión con la base vectorial
# ---------------------------------------------------------------------------
def conectar(qdrant_url: str, qdrant_api_key: str, gemini_api_key: str):
    """Crea los clientes de Qdrant Cloud y Gemini."""
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
    gemini = genai.Client(api_key=gemini_api_key)
    return qdrant, gemini


def estado_coleccion(qdrant: QdrantClient) -> dict:
    """Verifica que la colección del ACA 1 exista y devuelve sus características."""
    if not qdrant.collection_exists(COLLECTION_NAME):
        return {"existe": False, "fragmentos": 0, "dimension": None, "metrica": None}
    info = qdrant.get_collection(COLLECTION_NAME)
    vectores = info.config.params.vectors
    return {
        "existe": True,
        "fragmentos": qdrant.count(COLLECTION_NAME, exact=True).count,
        "dimension": getattr(vectores, "size", None),
        "metrica": str(getattr(vectores, "distance", "")).replace("Distance.", ""),
        "estado": str(info.status).replace("CollectionStatus.", ""),
    }


def crear_coleccion_si_no_existe(qdrant: QdrantClient) -> None:
    """Crea la colección (coseno, 768 dimensiones) y los índices de metadatos para filtros."""
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
    for campo in ("tipo_fuente", "documento", "documento_origen", "programa_academico", "nivel_riesgo_alerta"):
        try:
            qdrant.create_payload_index(COLLECTION_NAME, field_name=campo, field_schema="keyword")
        except Exception:
            pass  # el índice ya existe


def _con_reintento(funcion, max_reintentos: int = 4, espera_defecto: int = 30):
    """Ejecuta una llamada a Gemini reintentando si se agota la cuota gratuita (HTTP 429)."""
    for intento in range(1, max_reintentos + 1):
        try:
            return funcion()
        except Exception as err:
            msg = str(err)
            cuota_diaria = re.search(r"per ?day|PerDay|daily", msg, re.IGNORECASE)
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and intento < max_reintentos and not cuota_diaria:
                m = re.search(r"retry in ([0-9.]+)s", msg, re.IGNORECASE)
                time.sleep(int(float(m.group(1))) + 2 if m else espera_defecto)
            else:
                raise


def vectorizar(gemini, textos: list[str], task_type: str) -> list[list[float]]:
    """Convierte textos en vectores de 768 dimensiones (lotes de 50 por límite de la capa gratuita)."""
    vectores = []
    for i in range(0, len(textos), 50):
        lote = textos[i:i + 50]
        resp = _con_reintento(lambda: gemini.models.embed_content(
            model=EMBED_MODEL,
            contents=lote,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM),
        ), espera_defecto=62)
        vectores.extend(e.values for e in resp.embeddings)
    return vectores


def _filtro_tipo(tipo_fuente: str | None):
    if not tipo_fuente:
        return None
    return Filter(must=[FieldCondition(key="tipo_fuente", match=MatchValue(value=tipo_fuente))])


def recuperar(qdrant, gemini, pregunta: str, top_k: int = TOP_K, tipo_fuente: str | None = None,
              umbral: float = 0.0) -> list[dict]:
    """Recupera los Top-k fragmentos más similares (coseno) a la pregunta."""
    # Mismo formato de consulta que en el ACA 1 (RETRIEVAL_QUERY + prefijo de tarea)
    q_vec = vectorizar(gemini, [f"task: search result | query: {pregunta}"], "RETRIEVAL_QUERY")[0]
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        query_filter=_filtro_tipo(tipo_fuente),
        limit=top_k,
        with_payload=True,
    ).points
    fragmentos = []
    for rank, h in enumerate(hits, start=1):
        p = h.payload or {}
        es_doc = p.get("tipo_fuente") == TIPO_NORMATIVO
        fragmentos.append({
            "rank": rank,
            "similitud": round(float(h.score), 4),
            "id": p.get("id_documento", str(h.id)),
            "tipo_fuente": p.get("tipo_fuente", ""),
            "fuente": p.get("documento") if es_doc else f"Perfil UCI - {p.get('programa_academico', '')}",
            "ubicacion": (
                (" | ".join(x for x in (p.get("capitulo"), p.get("articulo")) if x) or "Disposición general")
                if es_doc else f"Riesgo: {p.get('nivel_riesgo_alerta', '')}"
            ),
            "texto": p.get("texto_semantico_rag", ""),
            "supera_umbral": float(h.score) >= umbral,
        })
    return fragmentos


# ---------------------------------------------------------------------------
# PASO 2 - Configuración del LLM
# ---------------------------------------------------------------------------
def modelos_disponibles(gemini) -> list[str]:
    """Devuelve los modelos candidatos que la API key realmente puede usar hoy."""
    try:
        nombres = {m.name.replace("models/", "") for m in gemini.models.list()}
        disponibles = [m for m in LLM_CANDIDATOS if m in nombres]
        return disponibles or LLM_CANDIDATOS
    except Exception:
        return LLM_CANDIDATOS


_MODELO_OK = None


def _generar(gemini, modelos: list[str], contenido: str, system: str, params: dict) -> tuple[str, str]:
    """Llama al LLM probando los modelos en orden. Devuelve (texto, modelo_usado)."""
    global _MODELO_OK
    ultimo_error = None
    # El último modelo que funcionó se prueba primero (evita repetir esperas en cada pregunta)
    orden = ([_MODELO_OK] if _MODELO_OK in modelos else []) + [m for m in modelos if m != _MODELO_OK]
    for modelo in orden:
        try:
            resp = _con_reintento(lambda: gemini.models.generate_content(
                model=modelo,
                contents=contenido,
                config=types.GenerateContentConfig(system_instruction=system, **params),
            ), max_reintentos=2, espera_defecto=15)
            texto = (resp.text or "").strip()
            if texto:
                _MODELO_OK = modelo
                return texto, modelo
        except Exception as err:  # modelo retirado (404) o cuota agotada: se prueba el siguiente
            ultimo_error = err
    raise RuntimeError(f"Ningún modelo LLM respondió. Último error: {ultimo_error}")


# ---------------------------------------------------------------------------
# PASO 3 - Pipeline RAG con historial conversacional
# ---------------------------------------------------------------------------
def _historial_a_texto(historial: list[dict], turnos: int = TURNOS_HISTORIAL) -> str:
    recientes = historial[-2 * turnos:]
    return "\n".join(
        f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content'][:600]}" for m in recientes
    )


def reformular_pregunta(gemini, modelos, pregunta: str, historial: list[dict]) -> str:
    """Convierte una pregunta de seguimiento ('¿y si no cumplo eso?') en una pregunta autónoma
    para que la búsqueda vectorial encuentre los fragmentos correctos."""
    if not historial:
        return pregunta
    instruccion = (
        "Reescribe la ÚLTIMA pregunta del usuario como una pregunta autónoma y completa, "
        "usando el historial solo para resolver referencias (eso, ese, y si...). "
        "Devuelve únicamente la pregunta reescrita, sin explicaciones."
    )
    contenido = f"HISTORIAL:\n{_historial_a_texto(historial)}\n\nÚLTIMA PREGUNTA: {pregunta}"
    try:
        texto, _ = _generar(gemini, modelos, contenido, instruccion,
                            {"temperature": 0.0, "max_output_tokens": 200})
        return texto.splitlines()[0].strip() or pregunta
    except Exception:
        return pregunta


def construir_prompt(pregunta: str, fragmentos: list[dict], historial: list[dict]) -> str:
    """Plantilla de prompt: historial + fragmentos numerados con su ID + pregunta."""
    bloques = [
        f"[{f['id']}] Fuente: {f['fuente']} | Ubicación: {f['ubicacion']} | Similitud: {f['similitud']}\n"
        f"Contenido: {f['texto']}"
        for f in fragmentos
    ]
    evidencia = "\n\n---\n\n".join(bloques) if bloques else "(No se recuperó evidencia relevante.)"
    hist = _historial_a_texto(historial) if historial else "(sin historial)"
    return (
        f"HISTORIAL DE LA CONVERSACIÓN:\n{hist}\n\n"
        f"FRAGMENTOS DE EVIDENCIA (TOP-{len(fragmentos)}):\n{evidencia}\n\n"
        f"PREGUNTA DEL USUARIO:\n{pregunta}\n\n"
        "Responde siguiendo las reglas, citando [ID] en cada afirmación o cifra."
    )


def extraer_citas(texto: str) -> list[str]:
    citas = []
    for grupo in PATRON_CITA.findall(texto):
        for c in re.split(r"[,;]\s*", grupo):
            c = c.strip()
            if c and c not in citas:
                citas.append(c)
    return citas


def responder(qdrant, gemini, pregunta: str, historial: list[dict] | None = None,
              modelos: list[str] | None = None, top_k: int = TOP_K, tipo_fuente: str | None = None,
              umbral: float = UMBRAL_SIMILITUD, params: dict | None = None,
              usar_reformulacion: bool = True) -> dict:
    """Flujo RAG completo: (reformulación) -> recuperación Top-k -> filtro por umbral -> generación."""
    historial = historial or []
    modelos = modelos or LLM_CANDIDATOS
    params = {**HIPERPARAMETROS, **(params or {})}

    t0 = time.perf_counter()
    pregunta_busqueda = (
        reformular_pregunta(gemini, modelos, pregunta, historial) if usar_reformulacion else pregunta
    )
    t1 = time.perf_counter()
    todos = recuperar(qdrant, gemini, pregunta_busqueda, top_k, tipo_fuente, umbral)
    utiles = [f for f in todos if f["supera_umbral"]]
    t2 = time.perf_counter()

    if not utiles:
        # Ningún fragmento es suficientemente parecido: no se llama al LLM (evita alucinaciones)
        respuesta = ("La evidencia recuperada no contiene información sobre esa consulta. "
                     "Puedo ayudarte con temas del reglamento estudiantil, bienestar, becas y "
                     "permanencia en la CUN.")
        modelo_usado = "sin llamada (bajo umbral)"
    else:
        respuesta, modelo_usado = _generar(
            gemini, modelos, construir_prompt(pregunta, utiles, historial), SYSTEM_PROMPT, params
        )
    t3 = time.perf_counter()

    citas = extraer_citas(respuesta)
    ids_validos = {f["id"] for f in utiles}
    return {
        "pregunta": pregunta,
        "pregunta_busqueda": pregunta_busqueda,
        "respuesta": respuesta,
        "fragmentos": todos,
        "modelo": modelo_usado,
        "citas": citas,
        "citas_validas": [c for c in citas if c in ids_validos],
        "se_abstuvo": FRASE_ABSTENCION in respuesta.lower() or "fuera del alcance" in respuesta.lower(),
        "tiempos": {
            "reformulacion_s": round(t1 - t0, 2),
            "recuperacion_s": round(t2 - t1, 2),
            "generacion_s": round(t3 - t2, 2),
            "total_s": round(t3 - t0, 2),
        },
    }


# ---------------------------------------------------------------------------
# PASO 4 - Gestión documental (usado por la interfaz gráfica)
# ---------------------------------------------------------------------------
def todos_los_payloads(qdrant) -> pd.DataFrame:
    """Lee los metadatos de todos los fragmentos de la colección (sin vectores)."""
    filas, offset = [], None
    while True:
        puntos, offset = qdrant.scroll(COLLECTION_NAME, limit=256, offset=offset,
                                       with_payload=True, with_vectors=False)
        for p in puntos:
            filas.append({"point_id": str(p.id), **(p.payload or {})})
        if offset is None:
            break
    return pd.DataFrame(filas)


def resumen_documentos(df_payloads: pd.DataFrame) -> pd.DataFrame:
    """Agrupa los fragmentos por documento de origen."""
    if df_payloads.empty:
        return pd.DataFrame()
    df = df_payloads.copy()
    for col in ("documento_origen", "documento", "tipo_fuente", "fecha", "tema"):
        if col not in df:
            df[col] = None
    df["origen"] = df["documento_origen"].fillna("Perfiles estudiantiles (UCI #697)")
    df["documento"] = df["documento"].fillna(df["origen"])
    agg = (
        df.groupby("origen")
        .agg(
            documento=("documento", "first"),
            tipo_fuente=("tipo_fuente", "first"),
            fragmentos=("point_id", "count"),
            fecha=("fecha", "first"),
            tema=("tema", "first"),
        )
        .reset_index()
        .rename(columns={"origen": "archivo"})
        .sort_values("fragmentos", ascending=False)
    )
    return agg.fillna("")


# --- Fragmentación (misma lógica del ACA 1: por artículo y luego por tamaño) ---
_ORDINALES = r"(?:Primero|Segundo|Tercero|Cuarto|Quinto|Sexto|S[ée]ptimo|Octavo|Noveno|D[ée]cimo)"
RE_ART = re.compile(
    rf"^(?:#{{1,6}}\s*)?(?:\*\*)?\s*(?:ART[ÍI]CULO|Art)\.?\s*(\d+|{_ORDINALES})\s*[ºo°]?\s*[\.:\-–]",
    re.I,
)
RE_CAP = re.compile(r"^##\s+(?:LIBRO|CAP[ÍI]TULO)\b", re.I)


def _frontmatter(texto: str):
    metadatos, cuerpo = {}, texto
    if texto.startswith("---\n"):
        partes = texto.split("---\n", 2)
        if len(partes) == 3:
            for linea in partes[1].strip().splitlines():
                if ":" in linea:
                    k, v = linea.split(":", 1)
                    metadatos[k.strip()] = v.strip()
            cuerpo = partes[2].strip()
    return metadatos, cuerpo


def _limpiar(cuerpo: str) -> str:
    lineas = [l for l in cuerpo.splitlines() if not l.lstrip().startswith(">")]
    return "\n".join(l if RE_ART.match(l) else re.sub(r"^#{3,6}\s+", "", l) for l in lineas).strip()


def _dividir_por_articulos(cuerpo: str) -> list[dict]:
    secciones, capitulo, etiqueta, buffer = [], "", "Preámbulo", []
    lineas, i = cuerpo.splitlines(), 0

    def cerrar():
        txt = "\n".join(buffer).strip()
        if txt:
            secciones.append({"capitulo": capitulo, "articulo": etiqueta, "texto": txt})

    while i < len(lineas):
        l = lineas[i]
        if RE_CAP.match(l):
            cerrar(); buffer.clear()
            capitulo = l.lstrip("# ").strip()
            j = i + 1
            while j < len(lineas) and not lineas[j].strip():
                j += 1
            if (j < len(lineas) and len(lineas[j]) < 120
                    and not lineas[j].startswith(("**", "#")) and not RE_ART.match(lineas[j])):
                capitulo += " — " + lineas[j].strip()
                i = j
            etiqueta = "Preámbulo"
        elif RE_ART.match(l):
            cerrar(); buffer.clear()
            etiqueta = "Artículo " + RE_ART.match(l).group(1).capitalize()
            buffer.append(l)
        else:
            buffer.append(l)
        i += 1
    cerrar()
    return secciones


def fragmentar_documento(nombre_archivo: str, texto: str, chunk_size: int = 800,
                         chunk_overlap: int = 100) -> list[dict]:
    """Divide un documento .md/.txt en fragmentos con metadatos (configuración 800/100 del ACA 1)."""
    texto = texto.replace("\r\n", "\n")
    meta, cuerpo = _frontmatter(texto)
    stem = re.sub(r"\.(md|txt)$", "", nombre_archivo, flags=re.I)
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )
    fragmentos, n = [], 0
    for s in _dividir_por_articulos(_limpiar(cuerpo)):
        for frag in divisor.split_text(s["texto"]):
            n += 1
            contexto = " | ".join(x for x in (s["capitulo"], s["articulo"]) if x)
            fragmentos.append({
                "id_documento": f"DOC-{stem}-{n:03d}",
                "documento": meta.get("documento", stem),
                "institucion": meta.get("institucion", "CUN"),
                "tipo_fuente": meta.get("tipo_fuente", TIPO_NORMATIVO),
                "tema": meta.get("tema", "normatividad"),
                "fecha": meta.get("fecha", ""),
                "capitulo": s["capitulo"],
                "articulo": s["articulo"],
                "pilar": "Educación",
                "documento_origen": nombre_archivo,
                "texto_semantico_rag": f"{contexto}\n\n{frag}" if contexto else frag,
            })
    return fragmentos


def indexar_fragmentos(qdrant, gemini, fragmentos: list[dict]) -> int:
    """Vectoriza (RETRIEVAL_DOCUMENT) e inserta los fragmentos en Qdrant Cloud."""
    crear_coleccion_si_no_existe(qdrant)
    textos = [
        f"title: {f.get('documento', f.get('programa_academico', 'none'))} | text: {f['texto_semantico_rag']}"
        for f in fragmentos
    ]
    vectores = vectorizar(gemini, textos, "RETRIEVAL_DOCUMENT")
    puntos = [
        PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, f["id_documento"])), vector=v, payload=f)
        for f, v in zip(fragmentos, vectores)
    ]
    for i in range(0, len(puntos), 64):
        qdrant.upsert(COLLECTION_NAME, points=puntos[i:i + 64])
    return len(puntos)


_CURSOS = {
    33: "Tecnologías de Producción de Biocombustibles", 171: "Diseño de Animación y Multimedia",
    8014: "Trabajo Social (Jornada nocturna)", 9003: "Agronomía", 9070: "Diseño de Comunicación",
    9085: "Enfermería Veterinaria", 9119: "Ingeniería Informática", 9130: "Equinocultura",
    9147: "Administración (Management)", 9238: "Trabajo Social", 9254: "Turismo", 9500: "Enfermería",
    9556: "Higiene Oral", 9670: "Gestión de Publicidad y Marketing", 9773: "Periodismo y Comunicación",
    9853: "Educación Básica", 9991: "Administración (Jornada nocturna)",
}
_ESTADO_CIVIL = {1: "soltero(a)", 2: "casado(a)", 3: "viudo(a)", 4: "divorciado(a)",
                 5: "unión libre (facto union)", 6: "separado(a) legalmente"}
_SI_NO = {1: "Sí", 0: "No"}
_RESULTADO = {"Dropout": "Deserción", "Enrolled": "Matriculado", "Graduate": "Graduado"}


def generar_perfiles_uci(df_uci: pd.DataFrame, muestra: int = 60, semilla: int = 42) -> list[dict]:
    """Recrea las fichas narrativas de estudiantes del ACA 1 (dataset UCI #697) con la misma
    regla de alerta temprana y una muestra estratificada por resultado académico."""
    filas = []
    for i, r in enumerate(df_uci.itertuples(index=False), start=1):
        g = dict(zip(df_uci.columns, r))
        ins1, apr1 = g["Curricular units 1st sem (enrolled)"], g["Curricular units 1st sem (approved)"]
        ins2, apr2 = g["Curricular units 2nd sem (enrolled)"], g["Curricular units 2nd sem (approved)"]
        tasa1 = round(apr1 / ins1 * 100, 1) if ins1 > 0 else 0.0
        tasa2 = round(apr2 / ins2 * 100, 1) if ins2 > 0 else 0.0
        al_dia, deudor = _SI_NO[g["Tuition fees up to date"]], _SI_NO[g["Debtor"]]
        factores = sum([al_dia == "No", deudor == "Sí", ins1 > 0 and tasa1 < 50,
                        ins2 > 0 and tasa2 < 50, ins1 == 0 and ins2 == 0])
        riesgo = "Alto" if factores >= 2 or al_dia == "No" else ("Medio" if factores == 1 else "Bajo")
        programa = _CURSOS.get(g["Course"], str(g["Course"]))
        jornada = "Diurna" if g["Daytime/evening attendance"] == 1 else "Nocturna"
        resultado = _RESULTADO.get(g["Target"], g["Target"])
        id_est = f"EST-{i:04d}"
        texto = (
            f"Ficha de seguimiento académico {id_est}: Estudiante de género "
            f"{'masculino' if g['Gender'] == 1 else 'femenino'}, {g['Age at enrollment']} años al ingreso, "
            f"estado civil {_ESTADO_CIVIL.get(g['Marital Status'], 'no registrado')}, matriculado en el programa de "
            f"{programa} (jornada {jornada.lower()}). Perfil socioeconómico y financiero: "
            f"desplazado={_SI_NO[g['Displaced']]}, becado={_SI_NO[g['Scholarship holder']]}, deudor={deudor}, "
            f"matrícula al día={al_dia}. Desempeño 1er semestre: {apr1} de {ins1} unidades aprobadas ({tasa1}%) "
            f"con promedio {round(g['Curricular units 1st sem (grade)'], 2)}/20. Desempeño 2do semestre: {apr2} de "
            f"{ins2} unidades aprobadas ({tasa2}%) con promedio {round(g['Curricular units 2nd sem (grade)'], 2)}/20. "
            f"Nivel de alerta temprana: {riesgo}. Resultado académico final registrado: {resultado}."
        )
        filas.append({
            "id_documento": id_est, "tipo_fuente": TIPO_PERFIL, "pilar": "Educación",
            "programa_academico": programa, "jornada": jornada, "matricula_al_dia": al_dia,
            "deudor": deudor, "becado": _SI_NO[g["Scholarship holder"]], "nivel_riesgo_alerta": riesgo,
            "estado_estudiante": resultado, "texto_semantico_rag": texto,
        })
    df = pd.DataFrame(filas)
    df = (df.groupby("estado_estudiante", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), muestra // 3), random_state=semilla))
            .reset_index(drop=True))
    return df.to_dict(orient="records")


def eliminar_documento(qdrant, nombre_archivo: str) -> None:
    """Elimina de la colección todos los fragmentos de un documento."""
    crear_coleccion_si_no_existe(qdrant)  # asegura el índice del campo documento_origen
    qdrant.delete(
        COLLECTION_NAME,
        points_selector=FilterSelector(filter=Filter(must=[
            FieldCondition(key="documento_origen", match=MatchValue(value=nombre_archivo))
        ])),
    )


# ---------------------------------------------------------------------------
# PASO 5 - Casos de prueba para la evaluación
# ---------------------------------------------------------------------------
CASOS_PRUEBA = [
    {"caso": "C1 Becas", "tipo": "normativa",
     "pregunta": "¿Qué promedio necesito para obtener una beca por excelencia académica y de qué porcentaje es?"},
    {"caso": "C2 Calidad de estudiante", "tipo": "normativa",
     "pregunta": "¿En qué casos se pierde la calidad de estudiante en la CUN?"},
    {"caso": "C3 Seguimiento (historial)", "tipo": "seguimiento",
     "pregunta": "¿Y qué pasa si mi promedio queda por debajo de la nota mínima aprobatoria?"},
    {"caso": "C4 Bienestar", "tipo": "normativa",
     "pregunta": "¿Qué servicios o dimensiones ofrece Bienestar Institucional para apoyar la permanencia?"},
    {"caso": "C5 Perfiles UCI", "tipo": "perfiles",
     "pregunta": "Perfil de estudiante de Ingeniería Informática con matrícula en mora y baja aprobación en primer semestre"},
    {"caso": "C6 Dato inexistente", "tipo": "fuera_de_contexto",
     "pregunta": "¿Cuánto paga la CUN de auxilio mensual de transporte en helicóptero?"},
    {"caso": "C7 Tema ajeno", "tipo": "fuera_de_contexto",
     "pregunta": "¿Quién ganó el mundial de fútbol de 2022?"},
]


def evaluar(qdrant, gemini, modelos=None, casos=None, pausa_s: float = 2.0, **kwargs):
    """Ejecuta los casos de prueba y mide precisión contextual, fidelidad de citas,
    abstención ante consultas fuera de contexto y latencia.
    El caso de 'seguimiento' reutiliza la pregunta y respuesta anterior como historial."""
    casos = casos or CASOS_PRUEBA
    filas, resultados, previo = [], [], None
    for caso in casos:
        historial = []
        if caso["tipo"] == "seguimiento" and previo:
            historial = [{"role": "user", "content": previo["pregunta"]},
                         {"role": "assistant", "content": previo["respuesta"]}]
        tipo_fuente = TIPO_PERFIL if caso["tipo"] == "perfiles" else None
        r = responder(qdrant, gemini, caso["pregunta"], historial=historial, modelos=modelos,
                      tipo_fuente=tipo_fuente, **kwargs)
        esperado_abstener = caso["tipo"] == "fuera_de_contexto"
        n_citas = len(r["citas"])
        filas.append({
            "Caso": caso["caso"],
            "Tipo": caso["tipo"],
            "Similitud Top-1": r["fragmentos"][0]["similitud"] if r["fragmentos"] else None,
            "Fragmentos útiles": sum(f["supera_umbral"] for f in r["fragmentos"]),
            "Citas": n_citas,
            "Citas válidas (%)": round(100 * len(r["citas_validas"]) / n_citas, 1) if n_citas else None,
            "Se abstuvo": r["se_abstuvo"],
            "Comportamiento correcto": r["se_abstuvo"] if esperado_abstener else (not r["se_abstuvo"] and n_citas > 0),
            "Recuperación (s)": r["tiempos"]["recuperacion_s"],
            "Generación (s)": r["tiempos"]["generacion_s"],
            "Total (s)": r["tiempos"]["total_s"],
            "Modelo": r["modelo"],
        })
        resultados.append(r)
        previo = r
        time.sleep(pausa_s)  # respeta la cuota gratuita de Gemini
    return pd.DataFrame(filas), resultados
