"""Streamlit interface for the complete Realista RAG pipeline."""

from __future__ import annotations

import base64
import importlib
from pathlib import Path
import sys

import streamlit as st


APP_DIRECTORY = Path(__file__).resolve().parent
HERO_IMAGE = APP_DIRECTORY / "assets" / "realista-hero.png"
if str(APP_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(APP_DIRECTORY))


importlib.invalidate_caches()
stage_modules = {
    name: importlib.import_module(name)
    for name in (
        "01_documents",
        "02_preprocessing",
        "03_chunking",
        "04_vector_representation",
        "05_create_chroma_store",
        "06_retrieve_context",
        "07_prompting",
        "08_market_query",
    )
}
documents_stage = stage_modules["01_documents"]
store_stage = stage_modules["05_create_chroma_store"]
retrieval_stage = stage_modules["06_retrieve_context"]
rag = stage_modules["07_prompting"]
market_query = stage_modules["08_market_query"]

try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass


@st.cache_resource(show_spinner="Loading the complete Nawy release…")
def load_market_state() -> dict:
    return market_query.load_market_state()


@st.cache_resource(show_spinner="Building the evidence index…")
def load_base_rag_index(
    chunk_size: int,
    overlap: int,
    release_id: str,
) -> tuple[list[dict], object]:
    del release_id  # Included in the cache key so a refreshed release rebuilds.
    documents = documents_stage.load_documents()
    _, collection = store_stage.build_store_from_documents(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return documents, collection


def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def inject_styles(hero_uri: str) -> None:
    st.markdown(
        f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {{
        --ink: #10201f;
        --muted: #667b78;
        --line: #dce8e5;
        --teal: #0f766e;
        --teal-dark: #0a4f4a;
        --gold: #c89b3c;
        --paper: #f7faf8;
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background:
            linear-gradient(180deg, rgba(247, 250, 248, 0.74), rgba(247, 250, 248, 0.98) 420px),
            #f7faf8;
        color: var(--ink);
    }}

    .block-container {{
        max-width: 1180px;
        padding-top: 28px;
        padding-bottom: 56px;
    }}

    header[data-testid="stHeader"] {{
        background: rgba(247, 250, 248, 0.76);
        backdrop-filter: blur(14px);
    }}

    .realista-hero {{
        min-height: 390px;
        border-radius: 8px;
        padding: 38px;
        overflow: hidden;
        position: relative;
        background:
            linear-gradient(90deg, rgba(8, 24, 25, 0.96), rgba(8, 24, 25, 0.72) 42%, rgba(8, 24, 25, 0.18)),
            url("{hero_uri}");
        background-size: cover;
        background-position: center;
        box-shadow: 0 26px 70px rgba(13, 33, 34, 0.24);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}

    .realista-hero::before {{
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(90deg, rgba(5, 18, 18, 0.28), rgba(5, 18, 18, 0.16) 52%, rgba(5, 18, 18, 0.04));
        pointer-events: none;
    }}

    .hero-copy {{
        width: min(650px, 100%);
        position: relative;
        z-index: 1;
    }}

    .eyebrow {{
        display: inline-flex;
        gap: 10px;
        align-items: center;
        padding: 8px 12px;
        border: 1px solid rgba(151, 220, 209, 0.36);
        background: rgba(9, 68, 64, 0.34);
        color: #bcece2;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0;
    }}

    .hero-title {{
        color: white;
        font-size: clamp(42px, 5vw, 76px);
        line-height: 0.95;
        font-weight: 800;
        letter-spacing: 0;
        margin: 28px 0 18px;
        text-shadow: 0 3px 18px rgba(0, 0, 0, 0.74);
    }}

    .hero-subtitle {{
        color: rgba(238, 250, 247, 0.86);
        font-size: 18px;
        line-height: 1.6;
        max-width: 610px;
        margin-bottom: 26px;
    }}

    .hero-tags {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }}

    .hero-tags span {{
        color: #f6fbfa;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 9px 12px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
    }}

    .metric-row {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 18px 0 24px;
    }}

    .metric {{
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 10px 28px rgba(16, 32, 31, 0.07);
    }}

    .metric-value {{
        font-size: 26px;
        font-weight: 800;
        color: var(--teal-dark);
        line-height: 1;
    }}

    .metric-label {{
        margin-top: 8px;
        color: var(--muted);
        font-size: 13px;
        font-weight: 600;
    }}

    .coverage-strip {{
        color: var(--muted);
        font-size: 14px;
        margin: -6px 0 26px;
    }}

    .coverage-strip strong {{
        color: var(--teal-dark);
    }}

    .section-title {{
        font-size: 22px;
        font-weight: 800;
        color: var(--ink);
        margin: 4px 0 6px;
    }}

    .section-note {{
        color: var(--muted);
        margin-bottom: 16px;
    }}

    div[data-testid="stFileUploader"] section {{
        background: rgba(255, 255, 255, 0.96);
        border: 1px dashed rgba(15, 118, 110, 0.72);
        border-radius: 8px;
    }}

    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] p,
    div[data-testid="stFileUploader"] small {{
        color: var(--ink);
        opacity: 1;
    }}

    div[data-testid="stFileUploader"] button {{
        background: #0f211f;
        color: #ffffff;
        border: 1px solid #0f211f;
        opacity: 1;
    }}

    div[data-testid="stFileUploader"] button:hover {{
        background: var(--teal-dark);
        color: #ffffff;
        border-color: var(--teal-dark);
    }}

    div[data-testid="stTextInput"] input {{
        border-radius: 8px;
        border: 1px solid var(--line);
        background: white;
        min-height: 52px;
        font-size: 16px;
    }}

    .stButton > button {{
        min-height: 48px;
        border-radius: 8px;
        border: 0;
        background: linear-gradient(135deg, #0f766e, #c89b3c);
        color: white;
        font-weight: 800;
        box-shadow: 0 14px 28px rgba(15, 118, 110, 0.23);
    }}

    .stButton > button:hover {{
        border: 0;
        filter: brightness(1.03);
        color: white;
    }}

    div[data-testid="stSlider"] [role="slider"] {{
        background: var(--teal);
    }}

    .source-card {{
        background: white;
        border: 1px solid var(--line);
        border-left: 4px solid var(--teal);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 8px 20px rgba(16, 32, 31, 0.06);
    }}

    .source-card strong {{
        color: var(--teal-dark);
    }}

    .st-key-answer_shell {{
        background: #0f211f;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 8px;
        color: #ecfffb;
        padding: 20px;
        box-shadow: 0 18px 42px rgba(15, 33, 31, 0.18);
    }}

    .st-key-answer_shell p,
    .st-key-answer_shell li {{
        color: #ecfffb;
        font-size: 16px;
        line-height: 1.65;
        overflow-wrap: anywhere;
    }}

    .st-key-answer_shell p:first-child {{
        margin-top: 0;
    }}

    .st-key-answer_shell p:last-child {{
        margin-bottom: 0;
    }}

    .st-key-answer_shell code {{
        background: rgba(255, 255, 255, 0.11);
        color: #baf8ed;
        white-space: normal;
        overflow-wrap: anywhere;
    }}

    @media (max-width: 760px) {{
        .block-container {{
            padding-top: 16px;
        }}
        .realista-hero {{
            min-height: 430px;
            padding: 26px;
            background-position: 58% center;
        }}
        .metric-row {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
    }}
</style>
        """,
        unsafe_allow_html=True,
    )


def metric_row(document_count: int, chunk_count: int, market_count: int, mode: str) -> None:
    st.markdown(
        f"""
<div class="metric-row">
    <div class="metric"><div class="metric-value">{document_count}</div><div class="metric-label">Evidence sources</div></div>
    <div class="metric"><div class="metric-value">{chunk_count}</div><div class="metric-label">Searchable chunks</div></div>
    <div class="metric"><div class="metric-value">{market_count}</div><div class="metric-label">Market rollups</div></div>
    <div class="metric"><div class="metric-value">{mode}</div><div class="metric-label">Answer mode</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="Realista RAG Engine", page_icon="Search", layout="wide")
inject_styles(image_data_uri(HERO_IMAGE))

st.markdown(
    """
<section class="realista-hero">
    <div class="hero-copy">
        <div class="eyebrow">Evidence-bounded real-estate intelligence</div>
        <h1 class="hero-title">Realista RAG Engine</h1>
        <p class="hero-subtitle">
            Ask exact questions across the complete packaged Nawy release and its
            evidence packs. Structured calculations use record-level rows; generated
            answers stay grounded in retrieved evidence and traceable citations.
        </p>
        <div class="hero-tags">
            <span>Egyptian market context</span>
            <span>Comment IDs</span>
            <span>Fact packs</span>
            <span>Source citations</span>
        </div>
    </div>
</section>
    """,
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Add UTF-8 text documents",
    type=["txt", "md"],
    accept_multiple_files=True,
)

with st.sidebar:
    st.header("RAG Settings")
    chunk_size = st.slider(
        "Chunk size",
        min_value=50,
        max_value=220,
        value=90,
        step=10,
        help="Words per chunk before embedding. Smaller chunks are precise; larger chunks carry more context.",
    )
    max_overlap = max(0, chunk_size - 10)
    overlap_default = min(20, max_overlap)
    overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=max_overlap,
        value=overlap_default,
        step=5,
        help="Repeated words between neighboring chunks.",
    )
    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=12,
        value=4,
        help="Final number of chunks sent to the answer step.",
    )
    candidate_multiplier = st.slider(
        "Rerank pool",
        min_value=2,
        max_value=20,
        value=8,
        help="How many vector matches are considered before Realista-specific reranking.",
    )

    st.header("LLM Settings")
    llm_enabled = st.toggle(
        "Use OpenRouter LLM",
        value=bool(rag.OPENROUTER_API_KEY),
        disabled=not bool(rag.OPENROUTER_API_KEY),
        help="Configure OPENROUTER_API_KEY in Streamlit secrets or the environment to enable generated responses.",
    )
    llm_model = st.text_input(
        "Model",
        value=rag.OPENROUTER_MODEL,
        disabled=not llm_enabled,
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        disabled=not llm_enabled,
    )
    timeout_seconds = st.slider(
        "LLM timeout",
        min_value=10,
        max_value=90,
        value=45,
        step=5,
        disabled=not llm_enabled,
    )

uploads = []
for uploaded in uploaded_files:
    try:
        uploads.append((uploaded.name, uploaded.getvalue().decode("utf-8")))
    except UnicodeDecodeError:
        st.warning(f"Skipped {uploaded.name}: the file is not valid UTF-8.")

market_state = load_market_state()
release_manifest = market_state.get("manifest") or {}
release_id = str(release_manifest.get("release_id") or "no_release")
base_documents, base_collection = load_base_rag_index(
    chunk_size,
    overlap,
    release_id,
)
if uploads:
    documents = list(base_documents)
    documents.extend(documents_stage.documents_from_uploads(uploads))
    _, collection = store_stage.build_store_from_documents(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )
else:
    documents = base_documents
    collection = base_collection
market_counts = {
    entity_type: sum(
        document.get("document_type") == "market_fact"
        and document.get("entity_type") == entity_type
        for document in documents
    )
    for entity_type in ("location", "developer", "project")
}
metric_row(
    document_count=len(documents),
    chunk_count=collection.count(),
    market_count=sum(market_counts.values()),
    mode="OpenRouter" if rag.OPENROUTER_API_KEY else "Local",
)
release_status = str(release_manifest.get("status") or "missing")
release_cutoff = str(release_manifest.get("capture_cutoff") or "unavailable").replace("T", " ").split(".", 1)[0]
listing_count = int(release_manifest.get("listing_count") or 0)
st.markdown(
    f"""
<div class="coverage-strip">
    Validated Nawy coverage: <strong>{market_counts['location']} locations</strong> &middot;
    <strong>{market_counts['developer']} developers</strong> &middot;
    <strong>{market_counts['project']} projects</strong> &middot;
    <strong>{listing_count:,} latest units</strong><br>
    Active release: <strong>{release_id}</strong> &middot;
    cutoff <strong>{release_cutoff}</strong> &middot;
    status <strong>{release_status}</strong>
</div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.12, 0.88], gap="large")
with left:
    st.markdown('<div class="section-title">Ask The Evidence</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Ask about scraped prices, areas, units, developers, projects, locations, or coverage.</div>',
        unsafe_allow_html=True,
    )
    question = st.text_input(
        "Question",
        placeholder="What is the average price of apartments in New Cairo?",
        label_visibility="collapsed",
    )
    run_query = st.button("Retrieve and answer", type="primary", use_container_width=True)

with right:
    st.markdown('<div class="section-title">Suggested Tests</div>', unsafe_allow_html=True)
    st.markdown(
        """
<div class="source-card"><strong>Market price</strong><br>What is the mean price of apartments in New Cairo?</div>
<div class="source-card"><strong>Area developers</strong><br>Who are the developers in New Cairo?</div>
<div class="source-card"><strong>Exact filter</strong><br>How many apartments in New Cairo are under 10 million?</div>
<div class="source-card"><strong>Missing evidence</strong><br>What is the delivery date for apartments in New Cairo?</div>
        """,
        unsafe_allow_html=True,
    )

if run_query:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        structured = market_query.query_market(question, market_state)
        if structured.get("status") in {"answered", "insufficient"}:
            retrieved = structured["retrieved"]
        else:
            retrieved = retrieval_stage.retrieve_context(
                collection,
                question,
                top_k=top_k,
                candidate_multiplier=candidate_multiplier,
            )
        try:
            force_local = structured.get("status") == "insufficient"
            if not llm_enabled or force_local:
                original_key = rag.OPENROUTER_API_KEY
                rag.OPENROUTER_API_KEY = ""
            result = rag.answer_question(
                question,
                retrieved,
                timeout_seconds=timeout_seconds,
                model=llm_model,
                temperature=temperature,
            )
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)
            with st.container(key="answer_shell"):
                st.markdown(str(result["answer"]))
            st.caption(
                f"Mode: {result['mode']} | "
                f"Retrieved context used: {result['used_retrieved_context']} | "
                f"Release: {release_id}"
            )
            st.markdown('<div class="section-title">Sources</div>', unsafe_allow_html=True)
            for source in result["sources"]:
                st.markdown(
                    f"""
<div class="source-card">
    <strong>[{source['citation']}] {source['source_name']}</strong><br>
    Chunk <code>{source['chunk_id']}</code>
</div>
                    """,
                    unsafe_allow_html=True,
                )
            with st.expander("Retrieved chunks"):
                for item in retrieved:
                    st.markdown(f"**[{item['citation']}] {item['source_name']}**")
                    st.write(item["text"])
        finally:
            if not llm_enabled or force_local:
                rag.OPENROUTER_API_KEY = original_key
