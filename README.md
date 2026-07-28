# Realista RAG Engine

An evidence-bounded hybrid retrieval-augmented generation system for traceable
question answering over a frozen Nawy scrape. The app combines deterministic
record-level filtering and statistics with Chroma retrieval and an optional
OpenRouter LLM. If an entity, field, or fact is absent from the packaged scrape,
the app says so instead of filling the gap from model knowledge.

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

Exact market questions take a structured path:

```text
question
  -> bilingual entity and unit-type resolution
  -> filters over the packaged latest-unit JSONL release
  -> deterministic count/list/statistic
  -> evidence item with release id, cutoff, filters, row count, and source ids
  -> cited LLM wording or deterministic fallback
```

Each stage is intentionally separated into an executable Python module:

- `01_documents.py` - source-labelled document loading
- `02_preprocessing.py` - UTF-8 and Arabic-safe normalization
- `03_chunking.py` - overlapping, traceable chunks
- `04_vector_representation.py` - deterministic offline vectors
- `05_create_chroma_store.py` - Chroma indexing
- `06_retrieve_context.py` - similarity retrieval and citation labels
- `07_prompting.py` - retrieved-context-only answering
- `08_market_query.py` - exact record-level market filtering and calculations
- `streamlit_app.py` - interactive application
- `nawy_release.py` - versioned release export, hashing, loading, and validation

## Trust and evidence rules

- Answers use retrieved chunks, not unrestricted model knowledge.
- Retrieved chunks receive stable citation labels such as `[S1]`.
- Missing evidence produces an explicit insufficient-context response.
- Missing entities, zero matching cohorts, missing fields, and unsupported
  transaction/ROI/rent questions have distinct abstention responses.
- Counts and market statistics are calculated from matching listing records,
  never by the LLM.
- Generated citations and numeric claims are checked against retrieved evidence;
  an invalid generated answer falls back to the deterministic answer.
- Arabic UTF-8 text and source metadata remain intact across the pipeline.
- OpenRouter is optional; the system has a cited local extractive fallback.
- No API key is stored in source code.
- Deployed builds include compact Realista evidence exports under `data/processed/`.
- The market export contains every validated compact Nawy location, developer, and project rollup available at generation time; raw pages and quarantined observations are excluded.
- Review-required comment labels are cited as model/committee evidence awaiting human validation, not as final market truth.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The built-in documents make the application runnable immediately. UTF-8 `.txt` and `.md` documents can also be uploaded through the UI.

To refresh the deployment evidence files from the full Realista workspace and its validated Mongo rollups, run:

```powershell
python build_realista_rag_exports.py --market-source mongo
```

This writes:

- `data/processed/evidence_capsules.jsonl`
- `data/processed/fact_packs.jsonl`
- `data/processed/market_facts.jsonl`
- `data/processed/nawy_listings.jsonl`
- `data/processed/nawy_release_manifest.json`

The packaged release is `nawy_2026-07-26`: 13,079 deduplicated validated latest
units, 42 locations, 147 developers, 537 projects, and 29 unit types. The
manifest records hashes, field coverage, crawl batches, and the exact capture
cutoff. These are crawl-coverage figures, not claims that the export represents
every developer or listing in Egypt.

The normalized release contains asking price for all 13,079 units and area /
price-per-m² for 13,076. Bedrooms, bathrooms, delivery, availability, payment
plans, and finishing are absent from this normalized release. Questions about
those fields correctly return “not present in the scrape.”

Example questions:

- `What is the average price of apartments in New Cairo?`
- `How many apartments in New Cairo are under 10 million?`
- `Who are the developers in New Cairo?`
- `What is the price of unit 104837?`
- `من هم المطورين في القاهرة الجديدة؟`
- `What is the delivery date for apartments in New Cairo?` (abstains)
- `What is the average transaction price in New Cairo?` (abstains)

## Optional OpenRouter configuration

For local Streamlit development, create `.streamlit/secrets.toml` without committing it:

```toml
OPENROUTER_API_KEY = "your_openrouter_key_here"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
```

When no key is configured, the application answers with its deterministic extractive fallback and still cites the retrieved sources.

The Streamlit sidebar exposes the main RAG controls: chunk size, overlap, retrieved chunk count, rerank-pool depth, OpenRouter model, temperature, and timeout. This lets users trade precision, context breadth, and generation style without changing code.

The base evidence index and record-level release are cached by Streamlit.
Refreshing the release id invalidates the cache automatically.

## Tests

```powershell
python -m pytest -q
```

The suite covers release integrity, exact record-level statistics, Arabic entity
resolution, unknown entities, missing fields, unsupported transaction-price
questions, unit lookup, price filters, aggregate retrieval, and stale-index
cleanup.

## Security

`.env`, `.streamlit/secrets.toml`, Python caches, and local Chroma data are excluded by `.gitignore`.
