# 🎓 Asistente Normativo CUN: Chatbot RAG (ACA 2, Computación en la Nube)

**Corporación Unificada Nacional de Educación Superior (CUN)**, Especialización en Inteligencia Artificial
**Asignatura:** Computación en la Nube | **Docente:** Dr. Hamilton Rivera Flor
**Pilar:** Educación (normatividad institucional y permanencia estudiantil)

**Integrantes:** Jhoan Ezequiel Hurtado Ramos · Melania Martinez · Anderson Rojas · Camila Arteaga · Monica Garavito

**🔗 Aplicación en línea:** https://asistente-cun-rag.streamlit.app/
**📂 Repositorio:** https://github.com/monicagaravito/aca2-chatbot-rag-cun

---

## ¿Qué hace?

Es un chatbot que responde preguntas sobre el **Reglamento Estudiantil (Acuerdo 019 de 2023)**, el **Reglamento de Bienestar (Acuerdo 022 de 2018)** y la **Resolución 007 de 2024 de Becas** de la CUN. También consulta perfiles de estudiantes del dataset **UCI #697 (deserción estudiantil)**.

El chatbot usa la técnica **RAG (Retrieval-Augmented Generation)**:

1. La pregunta se convierte en un vector con `gemini-embedding-001` (768 dimensiones).
2. **Qdrant Cloud** (la base vectorial construida en el ACA 1) devuelve los *Top-k* fragmentos más similares por similitud coseno.
3. Los fragmentos que superan un umbral de similitud se insertan en un *prompt* con reglas estrictas.
4. **Google Gemini** redacta la respuesta usando solo esa evidencia y cita cada dato con su identificador, por ejemplo `[DOC-reglamento-estudiantil-012]`.
5. Si no hay evidencia suficiente, el asistente lo dice y no inventa información.

```
Usuario ─► Streamlit ─► (reformulación con historial) ─► Embedding Gemini ─► Qdrant Cloud (Top-k)
                                                                              │
Usuario ◄── respuesta con citas [ID] ◄── LLM Gemini ◄── prompt fundamentado ◄─┘
```

## Estructura del repositorio

| Archivo | Descripción |
|---|---|
| `app.py` | Interfaz Streamlit con 4 pestañas: chat con historial, gestión documental, evaluación y arquitectura. |
| `rag_core.py` | Núcleo RAG: conexión con Qdrant, LLM, prompt, recuperación, gestión documental y evaluación. |
| `ACA2_Chatbot_RAG_Educacion.ipynb` | Notebook técnico reproducible (Google Colab). |
| `data/docs/` | Corpus normativo de la CUN en Markdown con metadatos YAML. |
| `requirements.txt` | Dependencias. |
| `.env.example`, `.streamlit/secrets.toml.example` | Plantillas de credenciales, sin valores reales. |

## Ejecución local

```bash
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>
pip install -r requirements.txt
cp .env.example .env        # y escribe tus credenciales en .env
streamlit run app.py
```

## Despliegue en Streamlit Community Cloud

1. Sube este repositorio a GitHub. **No subas `.env` ni `secrets.toml`.**
2. En [share.streamlit.io](https://share.streamlit.io) elige *Create app*, selecciona el repositorio y el archivo `app.py`.
3. En **Advanced settings → Secrets**, pega:
   ```toml
   QDRANT_URL = "https://...cloud.qdrant.io:6333"
   QDRANT_API_KEY = "..."
   GEMINI_API_KEY = "..."
   ```
4. Pulsa *Deploy*.

Si la colección de Qdrant no existe, por ejemplo porque el clúster gratuito se reinició, la pestaña **Gestión documental** ofrece el botón *Reconstruir colección desde data/docs*.

## Configuración del LLM

| Parámetro | Valor | Justificación |
|---|---|---|
| Modelo | `gemini-3.1-flash-lite`, con modelos de respaldo | Mayor cuota en la capa gratuita, baja latencia y buen español. Si un modelo falla o se retira, se usa el siguiente. |
| `temperature` | 0.1 | Respuestas casi deterministas que priorizan la fidelidad al documento. |
| `top_p` | 0.9 | Descarta tokens poco probables y reduce las invenciones. |
| `max_output_tokens` | 1024 | Respuestas completas pero acotadas. |
| Top-k | 4 | Equilibrio entre cobertura de evidencia y ruido en el prompt. |
| Umbral de similitud | 0.45 | Por debajo de este valor no se llama al LLM, lo que evita alucinaciones. |

## Seguridad de credenciales

Las claves se leen con `rag_core.leer_secreto()` en este orden:

1. Variables de entorno o archivo `.env`.
2. `st.secrets`.
3. Secrets de Google Colab.

El `.gitignore` excluye `.env` y `.streamlit/secrets.toml`.

## Referencias (APA 7)

- Google. (2025). *Gemini API documentation*. https://ai.google.dev/gemini-api/docs
- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459–9474. https://arxiv.org/abs/2005.11401
- Qdrant Solutions GmbH. (2024). *Qdrant documentation*. https://qdrant.tech/documentation/
- Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict students' dropout and academic success* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
- Streamlit Inc. (2024). *Streamlit documentation*. https://docs.streamlit.io/
