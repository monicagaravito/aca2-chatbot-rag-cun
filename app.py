# =====================================================================
# ACA 2 - Computación en la Nube (CUN) - Pilar Educación
# Chatbot RAG: Qdrant Cloud (base vectorial ACA 1) + Google Gemini (LLM)
# Ejecutar localmente:  streamlit run app.py
# =====================================================================
from pathlib import Path

import pandas as pd
import streamlit as st

import rag_core as rag

st.set_page_config(page_title="Asistente Normativo CUN - RAG", page_icon="🎓", layout="wide")


# ---------- Credenciales (secrets de Streamlit o archivo .env) ----------
QDRANT_URL = rag.leer_secreto("QDRANT_URL")
QDRANT_API_KEY = rag.leer_secreto("QDRANT_API_KEY")
GEMINI_API_KEY = rag.leer_secreto("GEMINI_API_KEY")

if not all([QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY]):
    st.error(
        "Faltan credenciales. Configura `QDRANT_URL`, `QDRANT_API_KEY` y `GEMINI_API_KEY` en "
        "**Settings → Secrets** (Streamlit Cloud) o en un archivo `.env` local. Ver README."
    )
    st.stop()


@st.cache_resource(show_spinner="Conectando con Qdrant Cloud y Gemini...")
def clientes():
    return rag.conectar(QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY)


@st.cache_data(ttl=3600, show_spinner=False)
def lista_modelos(_gemini):
    return rag.modelos_disponibles(_gemini)


@st.cache_data(ttl=120, show_spinner="Leyendo la base vectorial...")
def payloads(_qdrant, version):
    return rag.todos_los_payloads(_qdrant)


qdrant, gemini = clientes()
if "version_bd" not in st.session_state:
    st.session_state.version_bd = 0
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

try:
    estado = rag.estado_coleccion(qdrant)
except Exception as e:
    st.error(f"No fue posible conectar con Qdrant Cloud: {e}")
    st.stop()

# ---------- Barra lateral: configuración ----------
with st.sidebar:
    st.header("⚙️ Configuración")
    if estado["existe"]:
        st.success(f"Qdrant Cloud conectado\n\n**{estado['fragmentos']}** fragmentos · "
                   f"{estado['dimension']} dim · {estado['metrica']}")
    else:
        st.warning("La colección no existe todavía. Ve a **Gestión documental** para crearla.")

    modelos = lista_modelos(gemini)
    modelo = st.selectbox("Modelo LLM (Gemini API)", modelos, help="Si un modelo falla, se usa el siguiente.")
    temperatura = st.slider("Temperatura", 0.0, 1.0, rag.HIPERPARAMETROS["temperature"], 0.05,
                            help="Baja = respuestas más fieles a los documentos.")
    top_k = st.slider("Fragmentos a recuperar (Top-k)", 2, 8, rag.TOP_K)
    umbral = st.slider("Umbral mínimo de similitud", 0.0, 0.9, rag.UMBRAL_SIMILITUD, 0.05,
                       help="Fragmentos con similitud menor se descartan.")
    fuente = st.radio("Buscar en", ["Todo", "Solo normatividad CUN", "Solo perfiles estudiantiles"])
    reformular = st.toggle("Usar historial para entender preguntas de seguimiento", value=True)
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()

FILTRO = {"Todo": None, "Solo normatividad CUN": rag.TIPO_NORMATIVO,
          "Solo perfiles estudiantiles": rag.TIPO_PERFIL}[fuente]
MODELOS_ORDEN = [modelo] + [m for m in modelos if m != modelo]
PARAMS = {"temperature": temperatura}

# ---------- Encabezado ----------
st.title("🎓 Asistente Normativo CUN")
st.caption("Chatbot RAG · Reglamento Estudiantil, Bienestar y Becas de la CUN + perfiles UCI #697 · "
           "Qdrant Cloud + Google Gemini · Grupo ACA 2 Computación en la Nube")

tab_chat, tab_docs, tab_eval, tab_arq = st.tabs(
    ["💬 Chat", "📚 Gestión documental", "🧪 Evaluación", "🏗️ Arquitectura"]
)


def mostrar_fuentes(r):
    """Tabla y texto de los fragmentos recuperados para una respuesta."""
    t = r["tiempos"]
    st.caption(f"⏱️ {t['total_s']} s (búsqueda {t['recuperacion_s']} s · generación {t['generacion_s']} s) · "
               f"modelo: {r['modelo']} · citas válidas: {len(r['citas_validas'])}/{len(r['citas'])}")
    with st.expander(f"📎 Fuentes recuperadas ({len(r['fragmentos'])})"):
        if r["pregunta_busqueda"] != r["pregunta"]:
            st.write(f"**Búsqueda reformulada:** {r['pregunta_busqueda']}")
        for f in r["fragmentos"]:
            citado = "✅ citado" if f["id"] in r["citas"] else ("⚪ no usado" if f["supera_umbral"] else "⛔ bajo umbral")
            st.markdown(f"**[{f['id']}]** · {f['fuente']} · *{f['ubicacion']}* · "
                        f"similitud **{f['similitud']}** · {citado}")
            st.text(f["texto"][:900])


# ---------- 1. Chat ----------
with tab_chat:
    if not st.session_state.mensajes:
        st.info("Pregunta sobre el reglamento estudiantil, becas, bienestar o permanencia en la CUN. Ejemplos:")
        ejemplos = [c["pregunta"] for c in rag.CASOS_PRUEBA[:2]] + [rag.CASOS_PRUEBA[3]["pregunta"]]
        cols = st.columns(len(ejemplos))
        for col, ej in zip(cols, ejemplos):
            if col.button(ej, use_container_width=True):
                st.session_state.pregunta_ejemplo = ej
                st.rerun()

    for m in st.session_state.mensajes:
        with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🎓"):
            st.markdown(m["content"])
            if m.get("resultado"):
                mostrar_fuentes(m["resultado"])

    pregunta = st.chat_input("Escribe tu pregunta...") or st.session_state.pop("pregunta_ejemplo", None)
    if pregunta:
        historial = [{"role": m["role"], "content": m["content"]} for m in st.session_state.mensajes]
        st.session_state.mensajes.append({"role": "user", "content": pregunta})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(pregunta)
        with st.chat_message("assistant", avatar="🎓"):
            with st.spinner("Buscando en la base vectorial y redactando..."):
                try:
                    r = rag.responder(qdrant, gemini, pregunta, historial=historial, modelos=MODELOS_ORDEN,
                                      top_k=top_k, tipo_fuente=FILTRO, umbral=umbral, params=PARAMS,
                                      usar_reformulacion=reformular)
                except Exception as e:
                    st.error(f"Error al generar la respuesta: {e}")
                    st.stop()
            st.markdown(r["respuesta"])
            mostrar_fuentes(r)
        st.session_state.mensajes.append({"role": "assistant", "content": r["respuesta"], "resultado": r})

# ---------- 2. Gestión documental ----------
with tab_docs:
    if not estado["existe"] or estado["fragmentos"] == 0:
        st.warning("La colección está vacía o no existe. Puedes reconstruirla con los 3 documentos "
                   "normativos de la carpeta `data/docs` (se vectorizan con Gemini, ~2-4 minutos).")
        if st.button("🔄 Reconstruir colección desde data/docs", type="primary"):
            total = 0
            with st.status("Indexando documentos...") as s:
                for ruta in sorted(Path("data/docs").glob("*.md")):
                    frags = rag.fragmentar_documento(ruta.name, ruta.read_text(encoding="utf-8"))
                    s.write(f"{ruta.name}: {len(frags)} fragmentos")
                    total += rag.indexar_fragmentos(qdrant, gemini, frags)
                s.update(label=f"Listo: {total} fragmentos indexados", state="complete")
            st.session_state.version_bd += 1
            st.rerun()
    else:
        df = payloads(qdrant, st.session_state.version_bd)
        resumen = rag.resumen_documentos(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Documentos / fuentes", len(resumen))
        c2.metric("Fragmentos indexados", estado["fragmentos"])
        c3.metric("Dimensión del vector", estado["dimension"])
        c4.metric("Métrica", estado["metrica"])

        st.subheader("Documentos cargados en la base vectorial")
        st.dataframe(resumen, use_container_width=True, hide_index=True)
        st.bar_chart(resumen.set_index("archivo")["fragmentos"], horizontal=True)

        st.subheader("Explorar fragmentos y metadatos")
        archivo = st.selectbox("Documento", resumen["archivo"].tolist())
        if archivo.startswith("Perfiles"):
            sub = df[df.get("documento_origen").isna()] if "documento_origen" in df else df
        else:
            sub = df[df["documento_origen"] == archivo]
        texto_busqueda = st.text_input("Filtrar por palabra (opcional)")
        if texto_busqueda:
            sub = sub[sub["texto_semantico_rag"].str.contains(texto_busqueda, case=False, na=False)]
        columnas = [c for c in ["id_documento", "articulo", "capitulo", "programa_academico",
                                "nivel_riesgo_alerta", "estado_estudiante", "texto_semantico_rag"] if c in sub]
        st.dataframe(sub[columnas], use_container_width=True, hide_index=True, height=350)

        st.subheader("Probar la recuperación (sin LLM)")
        consulta = st.text_input("Consulta de prueba", "requisitos para la beca de excelencia académica")
        if st.button("🔎 Buscar Top-k"):
            frags = rag.recuperar(qdrant, gemini, consulta, top_k, FILTRO, umbral)
            st.dataframe(pd.DataFrame(frags)[["rank", "similitud", "id", "fuente", "ubicacion", "supera_umbral", "texto"]],
                         use_container_width=True, hide_index=True)

        st.subheader("Agregar o eliminar documentos")
        col_a, col_b = st.columns(2)
        with col_a:
            subido = st.file_uploader("Nuevo documento (.md o .txt)", type=["md", "txt"])
            if subido:
                texto = subido.read().decode("utf-8", errors="ignore")
                frags = rag.fragmentar_documento(subido.name, texto)
                st.write(f"Se generarán **{len(frags)} fragmentos** (800 caracteres, solapamiento 100).")
                if st.button("⬆️ Vectorizar e indexar", type="primary"):
                    with st.spinner("Generando embeddings e insertando en Qdrant..."):
                        n = rag.indexar_fragmentos(qdrant, gemini, frags)
                    st.success(f"{n} fragmentos indexados.")
                    st.session_state.version_bd += 1
                    st.rerun()
        with col_b:
            normativos = [a for a in resumen["archivo"] if not a.startswith("Perfiles")]
            borrar = st.selectbox("Documento a eliminar", normativos)
            confirmar = st.checkbox("Confirmo que quiero eliminar este documento de la base vectorial")
            if st.button("🗑️ Eliminar", disabled=not confirmar):
                rag.eliminar_documento(qdrant, borrar)
                st.success(f"Se eliminaron los fragmentos de {borrar}.")
                st.session_state.version_bd += 1
                st.rerun()

# ---------- 3. Evaluación ----------
with tab_eval:
    st.subheader("Validación integral del sistema (Paso 5)")
    st.write("Se ejecutan 7 casos de prueba: consultas normativas, una pregunta de seguimiento que usa el "
             "historial, una consulta sobre perfiles estudiantiles y dos consultas fuera de contexto que el "
             "asistente debe rechazar sin inventar información.")
    st.dataframe(pd.DataFrame(rag.CASOS_PRUEBA), use_container_width=True, hide_index=True)
    if st.button("▶️ Ejecutar evaluación (≈1 minuto)", type="primary"):
        with st.spinner("Ejecutando casos de prueba..."):
            tabla, resultados = rag.evaluar(qdrant, gemini, modelos=MODELOS_ORDEN, top_k=top_k,
                                            umbral=umbral, params=PARAMS)
        st.session_state.evaluacion = (tabla, resultados)
    if "evaluacion" in st.session_state:
        tabla, resultados = st.session_state.evaluacion
        ok = tabla["Comportamiento correcto"].mean() * 100
        validas = tabla["Citas válidas (%)"].dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("Casos con comportamiento correcto", f"{ok:.0f}%")
        c2.metric("Citas válidas (promedio)", f"{validas.mean():.0f}%" if len(validas) else "N/A")
        c3.metric("Latencia promedio", f"{tabla['Total (s)'].mean():.2f} s")
        st.dataframe(tabla, use_container_width=True, hide_index=True)
        st.bar_chart(tabla.set_index("Caso")[["Recuperación (s)", "Generación (s)"]], stack=True)
        for r in resultados:
            with st.expander(r["pregunta"]):
                st.markdown(r["respuesta"])

# ---------- 4. Arquitectura ----------
with tab_arq:
    st.subheader("Arquitectura RAG end-to-end")
    st.graphviz_chart("""
    digraph {
      rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];
      U [label="Usuario\\n(Streamlit)"]; R [label="Reformulación\\ncon historial"];
      E [label="Embedding consulta\\ngemini-embedding-001\\n(768 dim)"];
      Q [label="Qdrant Cloud\\ncolección ACA 1\\n(coseno, Top-k)", fillcolor="#fde9d9"];
      F [label="Filtro por\\numbral"]; P [label="Prompt fundamentado\\n(system + evidencia\\n+ historial)"];
      L [label="LLM Gemini\\n(API, temp 0.1)", fillcolor="#e2f0d9"]; A [label="Respuesta\\ncon citas [ID]"];
      D [label="Documentos CUN\\n+ perfiles UCI #697"];
      U -> R -> E -> Q -> F -> P -> L -> A -> U;
      D -> Q [style=dashed, label="indexación"];
    }
    """)
    st.markdown(
        "- **Base vectorial:** Qdrant Cloud (capa gratuita), colección `cun_educacion_permanencia_rag` creada en el ACA 1.\n"
        "- **Embeddings:** `gemini-embedding-001`, 768 dimensiones, `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY`.\n"
        "- **LLM:** Google Gemini vía API (capa gratuita), temperatura 0.1, con modelos de respaldo.\n"
        "- **Interfaz:** Streamlit Community Cloud.\n"
        "- **Credenciales:** `st.secrets` / `.env`; nunca se suben al repositorio."
    )
