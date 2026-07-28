"""Streamlit interface for the complete Realista RAG pipeline."""

from __future__ import annotations

import base64
import html
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
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {{
        --ink: #0b2825;
        --muted: #4d625e;
        --line: #cddbd7;
        --teal: #0d6f67;
        --teal-dark: #073d39;
        --mint: #8de3d4;
        --gold: #e3a93b;
        --gold-soft: #f5c86b;
        --paper: #f2f1ea;
        --white: #ffffff;
        --shadow: 0 24px 70px rgba(11, 40, 37, 0.10);
    }}

    html, body, [class*="css"] {{
        font-family: 'Manrope', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
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

    /* 2026 quiet-tech visual system */
    .stApp {{
        background:
            radial-gradient(circle at 78% 3%, rgba(141, 227, 212, 0.18), transparent 26rem),
            radial-gradient(circle at 8% 28%, rgba(245, 200, 107, 0.11), transparent 24rem),
            var(--paper);
    }}

    .block-container {{
        max-width: 1240px;
        padding-top: 22px;
        padding-bottom: 80px;
    }}

    header[data-testid="stHeader"] {{
        background: rgba(242, 241, 234, 0.82);
        border-bottom: 1px solid rgba(11, 40, 37, 0.08);
        backdrop-filter: blur(20px) saturate(150%);
    }}

    #MainMenu, footer {{
        visibility: hidden;
    }}

    .realista-hero {{
        min-height: 440px;
        border-radius: 30px;
        padding: 52px;
        background:
            linear-gradient(90deg, rgba(4, 25, 24, 0.98) 0%, rgba(4, 25, 24, 0.91) 37%, rgba(4, 25, 24, 0.48) 68%, rgba(4, 25, 24, 0.18) 100%),
            url("{hero_uri}");
        background-size: cover;
        background-position: center;
        border: 1px solid rgba(255, 255, 255, 0.16);
        box-shadow: 0 32px 90px rgba(5, 30, 28, 0.24);
        isolation: isolate;
    }}

    .realista-hero::after {{
        content: "";
        position: absolute;
        width: 380px;
        height: 380px;
        right: -120px;
        top: -160px;
        border-radius: 50%;
        border: 1px solid rgba(141, 227, 212, 0.22);
        box-shadow:
            0 0 0 42px rgba(141, 227, 212, 0.04),
            0 0 0 92px rgba(141, 227, 212, 0.025);
        pointer-events: none;
    }}

    .hero-copy {{
        width: min(720px, 100%);
    }}

    .eyebrow {{
        padding: 9px 14px;
        border-color: rgba(141, 227, 212, 0.32);
        background: rgba(5, 50, 46, 0.68);
        color: #d9fff7;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        backdrop-filter: blur(12px);
    }}

    .live-dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #70f3d7;
        box-shadow: 0 0 0 5px rgba(112, 243, 215, 0.12);
    }}

    .hero-title {{
        margin: 32px 0 20px;
        max-width: 760px;
        font-size: clamp(46px, 6.2vw, 84px);
        line-height: 0.96;
        letter-spacing: -0.055em;
        text-wrap: balance;
    }}

    .hero-title span {{
        color: var(--gold-soft);
    }}

    .hero-subtitle {{
        color: #dcebe7;
        font-size: 17px;
        line-height: 1.7;
        max-width: 650px;
        margin-bottom: 28px;
    }}

    .hero-proof {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        max-width: 650px;
    }}

    .hero-proof > div {{
        display: flex;
        flex-direction: column;
        gap: 5px;
        padding: 14px 16px;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.085);
        border: 1px solid rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(16px);
    }}

    .hero-proof strong {{
        color: #ffffff;
        font-size: 13px;
    }}

    .hero-proof span {{
        color: #c6d9d5;
        font-size: 11px;
        line-height: 1.45;
    }}

    .metric-row {{
        gap: 14px;
        margin: 20px 0 16px;
    }}

    .metric {{
        position: relative;
        min-height: 132px;
        border-radius: 22px;
        padding: 22px;
        background: rgba(255, 255, 255, 0.90);
        border-color: rgba(11, 40, 37, 0.12);
        box-shadow: 0 14px 38px rgba(11, 40, 37, 0.07);
        overflow: hidden;
    }}

    .metric::after {{
        content: "";
        position: absolute;
        width: 64px;
        height: 64px;
        right: -24px;
        bottom: -24px;
        border-radius: 50%;
        background: rgba(13, 111, 103, 0.08);
    }}

    .metric-primary {{
        background: var(--teal-dark);
        border-color: var(--teal-dark);
    }}

    .metric-primary .metric-value,
    .metric-primary .metric-label,
    .metric-primary .metric-index {{
        color: #f3fffc;
    }}

    .metric-index {{
        margin-bottom: 22px;
        color: #607a75;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.08em;
    }}

    .metric-value {{
        color: var(--ink);
        font-size: 29px;
        letter-spacing: -0.04em;
    }}

    .metric-mode {{
        font-size: 21px;
    }}

    .metric-label {{
        color: var(--muted);
        font-size: 12px;
    }}

    .release-panel {{
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 0 0 22px;
        padding: 15px 18px;
        border: 1px solid rgba(13, 111, 103, 0.20);
        border-radius: 18px;
        background: rgba(232, 248, 243, 0.88);
        color: var(--ink);
    }}

    .release-icon {{
        display: grid;
        place-items: center;
        width: 34px;
        height: 34px;
        flex: 0 0 auto;
        border-radius: 11px;
        background: var(--teal-dark);
        color: #ffffff;
        font-weight: 800;
    }}

    .release-copy {{
        display: flex;
        flex: 1;
        flex-direction: column;
        gap: 3px;
    }}

    .release-copy strong {{
        color: var(--ink);
        font-size: 13px;
    }}

    .release-copy span {{
        color: #3f5b56;
        font-size: 12px;
    }}

    .release-fields {{
        display: flex;
        gap: 8px;
    }}

    .release-fields span {{
        padding: 7px 10px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.72);
        color: #3f5b56;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
    }}

    .release-fields b {{
        color: var(--ink);
    }}

    .st-key-query_workspace {{
        padding: 30px;
        border: 1px solid rgba(11, 40, 37, 0.12);
        border-radius: 28px;
        background: rgba(255, 255, 255, 0.90);
        box-shadow: var(--shadow);
    }}

    .workspace-heading,
    .answer-heading {{
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 30px;
        margin-bottom: 24px;
    }}

    .workspace-heading > div > span,
    .answer-heading > div > span,
    .evidence-title > span,
    .starter-label {{
        color: var(--teal);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.10em;
    }}

    .workspace-heading h2,
    .answer-heading h2 {{
        margin: 5px 0 0;
        color: var(--ink);
        font-size: clamp(25px, 3vw, 38px);
        letter-spacing: -0.045em;
    }}

    .workspace-heading p {{
        max-width: 430px;
        margin: 0;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.6;
    }}

    div[data-testid="stTextInput"] input {{
        min-height: 62px;
        border: 1px solid #9bb0ab;
        border-radius: 16px;
        background: #ffffff;
        color: var(--ink);
        font-size: 16px;
        font-weight: 550;
        box-shadow: inset 0 1px 0 rgba(11, 40, 37, 0.04);
    }}

    div[data-testid="stTextInput"] input::placeholder {{
        color: #667b76;
        opacity: 1;
    }}

    div[data-testid="stTextInput"] input:focus {{
        border-color: var(--teal);
        box-shadow: 0 0 0 4px rgba(13, 111, 103, 0.14);
    }}

    .st-key-ask_button button {{
        min-height: 56px;
        border-radius: 15px;
        background: linear-gradient(135deg, #0b5d57, #0d786f);
        color: #ffffff;
        font-size: 14px;
        letter-spacing: 0.01em;
        box-shadow: 0 14px 26px rgba(13, 111, 103, 0.24);
    }}

    .st-key-ask_button button:hover {{
        background: linear-gradient(135deg, #084d48, #0b6961);
        transform: translateY(-1px);
    }}

    .query-hint {{
        margin: 10px 2px 0;
        color: #5a706b;
        font-size: 11px;
    }}

    .starter-label {{
        margin: 2px 0 9px;
    }}

    .st-key-starter_1 button,
    .st-key-starter_2 button,
    .st-key-starter_3 button,
    .st-key-starter_4 button {{
        justify-content: flex-start;
        min-height: 42px;
        margin-bottom: -2px;
        padding: 9px 12px;
        border: 1px solid #d1dedb;
        border-radius: 12px;
        background: #f8faf8;
        color: #183c38;
        font-size: 11px;
        font-weight: 650;
        text-align: left;
        box-shadow: none;
    }}

    .st-key-starter_1 button:hover,
    .st-key-starter_2 button:hover,
    .st-key-starter_3 button:hover,
    .st-key-starter_4 button:hover {{
        border-color: var(--teal);
        background: #eaf7f3;
        color: #073d39;
        transform: translateX(2px);
    }}

    .st-key-answer_section {{
        margin-top: 24px;
        padding: 30px;
        border: 1px solid rgba(11, 40, 37, 0.12);
        border-radius: 28px;
        background: rgba(255, 255, 255, 0.90);
        box-shadow: var(--shadow);
    }}

    .answer-badge {{
        padding: 8px 11px;
        border-radius: 999px;
        background: #e4f5f1;
        color: #075149;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        font-weight: 600;
    }}

    .st-key-answer_shell {{
        border-radius: 20px;
        padding: 26px;
        background:
            linear-gradient(135deg, rgba(12, 58, 54, 0.98), rgba(6, 37, 35, 1)),
            #082a27;
        border-color: rgba(141, 227, 212, 0.18);
        box-shadow: 0 24px 52px rgba(5, 37, 34, 0.20);
    }}

    .st-key-answer_shell p,
    .st-key-answer_shell li {{
        color: #f3fffc;
        font-size: 16px;
        line-height: 1.72;
    }}

    .response-meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 12px 0 30px;
    }}

    .response-meta span {{
        padding: 7px 10px;
        border: 1px solid #d5e1de;
        border-radius: 10px;
        background: #f8faf8;
        color: #506762;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
    }}

    .response-meta strong {{
        color: var(--ink);
    }}

    .evidence-title {{
        margin: 0 0 12px;
    }}

    .evidence-title h3 {{
        margin: 4px 0 0;
        color: var(--ink);
        font-size: 21px;
        letter-spacing: -0.035em;
    }}

    .source-card {{
        display: flex;
        align-items: center;
        gap: 13px;
        min-height: 68px;
        margin-bottom: 8px;
        padding: 12px 14px;
        border: 1px solid #d1dedb;
        border-left: 1px solid #d1dedb;
        border-radius: 15px;
        background: #fbfcfa;
        box-shadow: none;
    }}

    .source-id {{
        display: grid;
        place-items: center;
        width: 40px;
        height: 40px;
        flex: 0 0 auto;
        border-radius: 12px;
        background: #0c4e49;
        color: #ffffff;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        font-weight: 600;
    }}

    .source-body {{
        display: flex;
        flex: 1;
        flex-direction: column;
        gap: 3px;
        min-width: 0;
    }}

    .source-body strong {{
        color: var(--ink);
        font-size: 13px;
    }}

    .source-body small {{
        overflow: hidden;
        color: #5e746f;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}

    .source-action a,
    .source-action span {{
        color: #0a625b;
        font-size: 11px;
        font-weight: 700;
        text-decoration: none;
    }}

    .source-action a:hover {{
        text-decoration: underline;
    }}

    [data-testid="stSidebar"] {{
        background:
            radial-gradient(circle at 15% 0%, rgba(141, 227, 212, 0.12), transparent 18rem),
            #082522;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }}

    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 22px;
    }}

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small {{
        color: #dceae6;
    }}

    [data-testid="stSidebar"] [data-testid="stExpander"] {{
        margin-bottom: 8px;
        border: 1px solid rgba(255, 255, 255, 0.13);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.045);
    }}

    [data-testid="stSidebar"] [data-testid="stExpander"] summary {{
        color: #f4fffc;
        font-weight: 700;
    }}

    [data-testid="stSidebar"] div[data-testid="stTextInput"] input {{
        min-height: 44px;
        border-color: rgba(255, 255, 255, 0.22);
        background: rgba(255, 255, 255, 0.10);
        color: #ffffff;
        font-size: 12px;
    }}

    [data-testid="stSidebar"] div[data-testid="stTextInput"] input:disabled {{
        color: #d6e3df;
        -webkit-text-fill-color: #d6e3df;
        opacity: 0.78;
    }}

    .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 11px;
        margin-bottom: 18px;
    }}

    .brand-mark {{
        display: grid;
        place-items: center;
        width: 38px;
        height: 38px;
        border-radius: 12px;
        background: var(--gold-soft);
        color: #17332f;
        font-size: 17px;
        font-weight: 800;
    }}

    .sidebar-brand > div:last-child {{
        display: flex;
        flex-direction: column;
        gap: 1px;
    }}

    .sidebar-brand strong {{
        color: #ffffff;
        font-size: 15px;
    }}

    .sidebar-brand span {{
        color: #9fb7b2 !important;
        font-size: 10px;
    }}

    .sidebar-release {{
        display: flex;
        flex-direction: column;
        gap: 5px;
        margin-bottom: 16px;
        padding: 14px;
        border: 1px solid rgba(141, 227, 212, 0.17);
        border-radius: 15px;
        background: rgba(141, 227, 212, 0.07);
    }}

    .sidebar-release strong {{
        color: #ffffff;
        font-size: 12px;
    }}

    .sidebar-release small {{
        color: #a9c0bb !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
    }}

    .status-pill {{
        display: inline-flex;
        align-items: center;
        align-self: flex-start;
        gap: 6px;
        color: #a9f6e5 !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}

    .status-pill i {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #70f3d7;
    }}

    .sidebar-footnote {{
        margin-top: 18px;
        color: #8fa8a3 !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 8px !important;
        letter-spacing: 0.04em;
        text-align: center;
        text-transform: uppercase;
    }}

    div[data-testid="stFileUploader"] section {{
        border-color: rgba(141, 227, 212, 0.52);
        background: rgba(255, 255, 255, 0.07);
    }}

    div[data-testid="stFileUploader"] button {{
        border-color: var(--gold-soft);
        background: var(--gold-soft);
        color: #17332f;
    }}

    button:focus-visible,
    input:focus-visible,
    summary:focus-visible,
    a:focus-visible {{
        outline: 3px solid var(--gold-soft) !important;
        outline-offset: 2px;
    }}

    @media (prefers-reduced-motion: no-preference) {{
        .stButton > button,
        .source-card,
        .metric {{
            transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
        }}
    }}

    @media (max-width: 760px) {{
        .block-container {{
            padding: 12px 14px 50px;
        }}
        .realista-hero {{
            min-height: 520px;
            padding: 28px 24px;
            border-radius: 22px;
            background-position: 62% center;
        }}
        .hero-title {{
            font-size: 48px;
        }}
        .hero-proof {{
            grid-template-columns: 1fr;
        }}
        .metric-row {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }}
        .metric {{
            min-height: 120px;
            padding: 18px;
        }}
        .metric-value {{
            font-size: 24px;
        }}
        .release-panel,
        .workspace-heading,
        .answer-heading {{
            align-items: flex-start;
            flex-direction: column;
        }}
        .release-fields {{
            width: 100%;
        }}
        .release-fields span {{
            flex: 1;
        }}
        .st-key-query_workspace,
        .st-key-answer_section {{
            padding: 22px 18px;
            border-radius: 22px;
        }}
        .source-action {{
            display: none;
        }}
    }}

    @media (max-width: 430px) {{
        .metric-row {{
            grid-template-columns: 1fr;
        }}
        .realista-hero {{
            min-height: 550px;
        }}
    }}
</style>
        """,
        unsafe_allow_html=True,
    )


def metric_row(
    listing_count: int,
    entity_count: int,
    chunk_count: int,
    mode: str,
) -> None:
    st.markdown(
        f"""
<div class="metric-row">
    <div class="metric metric-primary"><div class="metric-index">01</div><div class="metric-value">{listing_count:,}</div><div class="metric-label">Validated latest units</div></div>
    <div class="metric"><div class="metric-index">02</div><div class="metric-value">{entity_count:,}</div><div class="metric-label">Market entities</div></div>
    <div class="metric"><div class="metric-index">03</div><div class="metric-value">{chunk_count:,}</div><div class="metric-label">Semantic evidence chunks</div></div>
    <div class="metric"><div class="metric-index">04</div><div class="metric-value metric-mode">{mode}</div><div class="metric-label">Grounded answer mode</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Realista · Nawy Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles(image_data_uri(HERO_IMAGE))

st.markdown(
    """
<section class="realista-hero">
    <div class="hero-copy">
        <div class="eyebrow"><span class="live-dot"></span> Nawy evidence layer · online</div>
        <h1 class="hero-title">Ask the market.<br><span>Trace the answer.</span></h1>
        <p class="hero-subtitle">
            A citation-first workspace for Egypt's real-estate market. Explore the
            packaged Nawy release with deterministic calculations, bilingual entity
            resolution, and evidence-bounded generation.
        </p>
        <div class="hero-proof">
            <div><strong>Structured first</strong><span>Counts and statistics are calculated—not guessed.</span></div>
            <div><strong>Source visible</strong><span>Every response carries inspectable evidence.</span></div>
        </div>
    </div>
</section>
    """,
    unsafe_allow_html=True,
)

market_state = load_market_state()
release_manifest = market_state.get("manifest") or {}
release_id = str(release_manifest.get("release_id") or "no_release")
release_status = str(release_manifest.get("status") or "missing")
release_cutoff = (
    str(release_manifest.get("capture_cutoff") or "unavailable")
    .replace("T", " ")
    .split(".", 1)[0]
)
listing_count = int(release_manifest.get("listing_count") or 0)
field_coverage = release_manifest.get("field_coverage") or {}

with st.sidebar:
    st.markdown(
        """
<div class="sidebar-brand">
    <div class="brand-mark">R</div>
    <div><strong>Realista</strong><span>Evidence workspace</span></div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="sidebar-release">
    <span class="status-pill"><i></i>{html.escape(release_status.title())}</span>
    <strong>{html.escape(release_id)}</strong>
    <small>Cutoff · {html.escape(release_cutoff)}</small>
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Retrieval tuning", expanded=False):
        chunk_size = st.slider(
            "Context window",
            min_value=50,
            max_value=220,
            value=90,
            step=10,
            help="Words per semantic chunk. Structured market queries do not depend on this setting.",
        )
        max_overlap = max(0, chunk_size - 10)
        overlap_default = min(20, max_overlap)
        overlap = st.slider(
            "Context overlap",
            min_value=0,
            max_value=max_overlap,
            value=overlap_default,
            step=5,
            help="Words repeated between neighboring evidence chunks.",
        )
        top_k = st.slider(
            "Evidence depth",
            min_value=1,
            max_value=12,
            value=4,
            help="Final evidence chunks supplied to the answer layer.",
        )
        candidate_multiplier = st.slider(
            "Candidate breadth",
            min_value=2,
            max_value=20,
            value=8,
            help="How broadly semantic retrieval searches before reranking.",
        )

    with st.expander("Generation", expanded=True):
        llm_enabled = st.toggle(
            "Use OpenRouter generation",
            value=bool(rag.OPENROUTER_API_KEY),
            disabled=not bool(rag.OPENROUTER_API_KEY),
            help="Without a configured key, Realista returns the deterministic cited answer.",
        )
        llm_model = st.text_input(
            "Model",
            value=rag.OPENROUTER_MODEL,
            disabled=not llm_enabled,
        )
        temperature = st.slider(
            "Creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            disabled=not llm_enabled,
            help="Keep this near zero for analytical answers.",
        )
        timeout_seconds = st.slider(
            "Response timeout",
            min_value=10,
            max_value=90,
            value=45,
            step=5,
            disabled=not llm_enabled,
        )

    with st.expander("Add private context", expanded=False):
        st.caption("Optional UTF-8 notes are indexed only for this session.")
        uploaded_files = st.file_uploader(
            "Upload text or Markdown",
            type=["txt", "md"],
            accept_multiple_files=True,
        )

    st.markdown(
        '<p class="sidebar-footnote">Evidence-bound · Arabic-safe · Asking prices only</p>',
        unsafe_allow_html=True,
    )

uploads = []
for uploaded in uploaded_files:
    try:
        uploads.append((uploaded.name, uploaded.getvalue().decode("utf-8")))
    except UnicodeDecodeError:
        st.warning(f"Skipped {uploaded.name}: the file is not valid UTF-8.")

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
    listing_count=listing_count,
    entity_count=sum(market_counts.values()),
    chunk_count=collection.count(),
    mode="OpenRouter" if rag.OPENROUTER_API_KEY else "Local",
)
st.markdown(
    f"""
<div class="release-panel">
    <div class="release-icon">✓</div>
    <div class="release-copy">
        <strong>Release verified and ready</strong>
        <span>{market_counts['location']} locations · {market_counts['developer']} developers · {market_counts['project']} projects · cutoff {html.escape(release_cutoff)}</span>
    </div>
    <div class="release-fields">
        <span>Price <b>{int(field_coverage.get('total_price_egp') or 0):,}</b></span>
        <span>Area <b>{int(field_coverage.get('area_sqm') or 0):,}</b></span>
    </div>
</div>
    """,
    unsafe_allow_html=True,
)

def use_starter_question(value: str) -> None:
    st.session_state["question_input"] = value


with st.container(key="query_workspace"):
    st.markdown(
        """
<div class="workspace-heading">
    <div><span>01 · QUERY</span><h2>What do you want to know?</h2></div>
    <p>Ask naturally in English or Arabic. Exact market questions use deterministic calculations before generation.</p>
</div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.18, 0.82], gap="large")
    with left:
        question = st.text_input(
            "Market question",
            placeholder="e.g. What is the average apartment price in New Cairo?",
            label_visibility="collapsed",
            key="question_input",
        )
        run_query = st.button(
            "Run evidence query  →",
            type="primary",
            use_container_width=True,
            key="ask_button",
        )
        st.markdown(
            '<p class="query-hint">Try entities, prices, areas, unit types, comparisons, or a Nawy unit ID.</p>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="starter-label">START WITH A PROVEN QUERY</div>', unsafe_allow_html=True)
        starter_questions = [
            ("Average apartment price · New Cairo", "What is the average price of apartments in New Cairo?"),
            ("Developers active · New Cairo", "Who are the developers in New Cairo?"),
            ("Apartments below EGP 10M", "How many apartments in New Cairo are under 10 million?"),
            ("Test a missing field · Delivery", "What is the delivery date for apartments in New Cairo?"),
        ]
        for index, (label, value) in enumerate(starter_questions, start=1):
            st.button(
                label,
                key=f"starter_{index}",
                use_container_width=True,
                on_click=use_starter_question,
                args=(value,),
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
            st.error(
                "The generation provider did not respond. Your evidence query is "
                f"safe; retry or switch to local mode. Details: {exc}"
            )
        else:
            with st.container(key="answer_section"):
                st.markdown(
                    f"""
<div class="answer-heading">
    <div><span>02 · GROUNDED RESPONSE</span><h2>Answer from the evidence</h2></div>
    <div class="answer-badge">● {html.escape(str(result['mode']).replace('_', ' ').title())}</div>
</div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.container(key="answer_shell"):
                    st.markdown(str(result["answer"]))
                st.markdown(
                    f"""
<div class="response-meta">
    <span>Release <strong>{html.escape(release_id)}</strong></span>
    <span>Evidence used <strong>{'Yes' if result['used_retrieved_context'] else 'No'}</strong></span>
    <span>Citations <strong>{len(result['sources'])}</strong></span>
</div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="evidence-title"><span>03 · SOURCES</span><h3>Evidence trail</h3></div>',
                    unsafe_allow_html=True,
                )
                for source in result["sources"]:
                    safe_name = html.escape(str(source["source_name"]))
                    safe_chunk = html.escape(str(source["chunk_id"]))
                    safe_url = html.escape(str(source.get("source_url") or ""), quote=True)
                    source_action = (
                        f'<a href="{safe_url}" target="_blank" rel="noopener">Open source ↗</a>'
                        if safe_url
                        else "<span>Packaged evidence</span>"
                    )
                    st.markdown(
                        f"""
<div class="source-card">
    <div class="source-id">[{html.escape(str(source['citation']))}]</div>
    <div class="source-body"><strong>{safe_name}</strong><small>Evidence chunk · {safe_chunk}</small></div>
    <div class="source-action">{source_action}</div>
</div>
                        """,
                        unsafe_allow_html=True,
                    )
                with st.expander("Inspect retrieved evidence"):
                    for item in retrieved:
                        st.markdown(f"**[{item['citation']}] {item['source_name']}**")
                        st.write(item["text"])
        finally:
            if not llm_enabled or force_local:
                rag.OPENROUTER_API_KEY = original_key
