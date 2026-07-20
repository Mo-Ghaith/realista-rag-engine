# Realista RAG Engine

An evidence-bounded retrieval-augmented generation system for traceable question answering. The project turns source-labelled documents into cleaned text, overlapping chunks, deterministic vector representations, a Chroma vector index, retrieved context, and cited answers through a Streamlit interface.

## Pipeline

```text
documents
  -> preprocessing
  -> chunking
  -> vector representation
  -> Chroma vector store
  -> context retrieval
  -> evidence-bounded prompting
  -> Streamlit UI
```

Each stage is intentionally separated into an executable Python module:

- `01_documents.py` - source-labelled document loading
- `02_preprocessing.py` - UTF-8 and Arabic-safe normalization
- `03_chunking.py` - overlapping, traceable chunks
- `04_vector_representation.py` - deterministic offline vectors
- `05_create_chroma_store.py` - Chroma indexing
- `06_retrieve_context.py` - similarity retrieval and citation labels
- `07_prompting.py` - retrieved-context-only answering
- `streamlit_app.py` - interactive application

## Trust and evidence rules

- Answers use retrieved chunks, not unrestricted model knowledge.
- Retrieved chunks receive stable citation labels such as `[S1]`.
- Missing evidence produces an explicit insufficient-context response.
- Arabic UTF-8 text and source metadata remain intact across the pipeline.
- OpenRouter is optional; the system has a cited local extractive fallback.
- No API key is stored in source code.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The built-in documents make the application runnable immediately. UTF-8 `.txt` and `.md` documents can also be uploaded through the UI.

## Optional OpenRouter configuration

For local Streamlit development, create `.streamlit/secrets.toml` without committing it:

```toml
OPENROUTER_API_KEY = "your_openrouter_key_here"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
```

When no key is configured, the application answers with its deterministic extractive fallback and still cites the retrieved sources.

## Security

`.env`, `.streamlit/secrets.toml`, Python caches, and local Chroma data are excluded by `.gitignore`.
