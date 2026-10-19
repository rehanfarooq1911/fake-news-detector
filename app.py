import streamlit as st
import joblib
from pathlib import Path

import numpy as np

from text_utils import clean_text

APP_NAME = "NewsLens"
MIN_WORDS = 20

# Two original example texts written by us (fictional event, same facts, different writing style).
# They are used to demonstrate that the model reacts to WRITING STYLE, not to truth.
EXAMPLE_NARRATIVE = (
    'A local man pulled three children out of the river near the old bridge on Friday evening, calling it '
    '"just something anyone would do". The children were taken to hospital and are reported to be well. '
    '"I don\'t know what I was thinking, I just ran in and grabbed them," he said. Neighbours described him as '
    'a quiet man who has lived in the area for years. The mayor praised his bravery and thanked the emergency '
    'teams for their quick response.'
)
EXAMPLE_WIRE = (
    'A man rescued three children from a river near the Old Bridge on Friday, police said. The children were '
    'taken to a hospital and were in stable condition, a hospital spokesman said. The man, who was not identified, '
    'told reporters he acted on instinct, according to a statement from the city police department. The mayor said '
    'on Friday the city would review safety measures along the river, officials said.'
)

TIPS = [
    "Check who published it and whether other trusted outlets report the same story.",
    "Look for a named author, a date and links to original sources.",
    "Be careful with headlines that make you angry or scared. That is often the goal.",
    "Search for the exact claim on a fact-checking website before you share.",
    "Read beyond the headline. Many posts are shared without being read.",
]


def explain(model, cleaned, k=6):
    """Words/phrases that pushed the score towards wire-style (+) or sensational-style (-)."""
    tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    contrib = tfidf.transform([cleaned]).multiply(clf.coef_[0]).toarray()[0]
    names = tfidf.get_feature_names_out()
    order = np.argsort(contrib)
    sens = [names[i] for i in order[:k] if contrib[i] < 0]
    wire = [names[i] for i in order[::-1][:k] if contrib[i] > 0]
    return wire, sens


@st.cache_resource
def load_stats():
    """Training-data word statistics (made by build_word_stats.py). Optional."""
    p = Path(__file__).parent / "model" / "word_stats.joblib"
    if not p.exists():
        return None
    s = joblib.load(p)
    s["index"] = {w: i for i, w in enumerate(s["names"])}
    return s


def word_note(stats, w):
    """Small text: in what % of wire / sensational training articles this word appeared."""
    if stats is None or w not in stats["index"]:
        return ""
    i = stats["index"][w]
    return (f' <small>(wire {stats["real_pct"][i]:.1f}% · '
            f'sensational {stats["fake_pct"][i]:.1f}%)</small>')


def use_example(text):
    st.session_state["news_text"] = text

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="NewsLens - Think Before You Share",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== THEME & STATE INITIALIZATION ======================
PAGES = ["🏠 Home", "🔍 Analyze News", "ℹ️ About Project"]

# Detect system or context theme if available
ctx_theme = getattr(st.context, "theme", None)
ctx_type = getattr(ctx_theme, "type", None) if ctx_theme else None

if "theme_mode" not in st.session_state:
    if ctx_type == "dark":
        st.session_state.theme_mode = "🌙 Dark"
    else:
        st.session_state.theme_mode = "☀️ Light"

if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Home"

def set_page(page_name):
    st.session_state.current_page = page_name

def toggle_theme():
    if st.session_state.get("theme_mode") == "☀️ Light":
        st.session_state.theme_mode = "🌙 Dark"
    else:
        st.session_state.theme_mode = "☀️ Light"

# Determine active theme styling
is_light = (st.session_state.get("theme_mode", "☀️ Light") == "☀️ Light")

# Define Theme Variables
light_theme_vars = """
    --bg-page: linear-gradient(135deg, #e0f2fe 0%, #e8edff 35%, #ede9fe 70%, #f0fdf4 100%);
    --title-gradient: linear-gradient(90deg, #4338ca 0%, #2563eb 50%, #0284c7 100%);
    --text-primary: #1e1b4b;
    --text-secondary: #3730a3;
    --text-muted: #6366f1;
    --card-bg: rgba(255, 255, 255, 0.88);
    --card-border: rgba(199, 210, 254, 0.85);
    --card-shadow: 0 14px 38px rgba(99, 102, 241, 0.12), 0 4px 14px rgba(14, 165, 233, 0.08);
    --input-bg: rgba(255, 255, 255, 0.95);
    --input-border: #a5b4fc;
    --input-text: #1e1b4b;
    --sidebar-bg: linear-gradient(180deg, rgba(238, 242, 255, 0.97) 0%, rgba(245, 243, 255, 0.97) 50%, rgba(240, 253, 250, 0.97) 100%);
    --sidebar-border: rgba(199, 210, 254, 0.8);
    --btn-primary-bg: linear-gradient(90deg, #4f46e5, #0284c7);
    --btn-primary-border: rgba(79, 70, 229, 0.5);
    --btn-primary-shadow: 0 6px 20px rgba(79, 70, 229, 0.35);
    --btn-primary-hover-shadow: 0 10px 28px rgba(79, 70, 229, 0.5);
    --btn-sec-bg: rgba(255, 255, 255, 0.9);
    --btn-sec-color: #4338ca;
    --btn-sec-border: rgba(199, 210, 254, 0.9);
    --btn-sec-hover-bg: #ffffff;
    --btn-sec-hover-color: #312e81;
    --btn-sec-hover-border: #6366f1;
    --divider: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.35), transparent);
    --footer-color: #6366f1;
"""

dark_theme_vars = """
    --bg-page: linear-gradient(135deg, #09081a 0%, #171438 40%, #282152 75%, #152240 100%);
    --title-gradient: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 50%, #c084fc 100%);
    --text-primary: #ffffff;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --card-bg: rgba(255, 255, 255, 0.05);
    --card-border: rgba(255, 255, 255, 0.12);
    --card-shadow: 0 14px 40px rgba(0, 0, 0, 0.45);
    --input-bg: rgba(255, 255, 255, 0.07);
    --input-border: rgba(255, 255, 255, 0.18);
    --input-text: #ffffff;
    --sidebar-bg: rgba(12, 10, 29, 0.97);
    --sidebar-border: rgba(255, 255, 255, 0.08);
    --btn-primary-bg: linear-gradient(90deg, #00d2ff, #3a7bd5);
    --btn-primary-border: rgba(0, 210, 255, 0.5);
    --btn-primary-shadow: 0 6px 20px rgba(0, 210, 255, 0.35);
    --btn-primary-hover-shadow: 0 10px 28px rgba(0, 210, 255, 0.55);
    --btn-sec-bg: rgba(255, 255, 255, 0.08);
    --btn-sec-color: #e2e8f0;
    --btn-sec-border: rgba(255, 255, 255, 0.15);
    --btn-sec-hover-bg: rgba(255, 255, 255, 0.18);
    --btn-sec-hover-color: #ffffff;
    --btn-sec-hover-border: rgba(0, 210, 255, 0.5);
    --divider: linear-gradient(90deg, transparent, rgba(0, 210, 255, 0.35), transparent);
    --footer-color: #94a3b8;
"""

active_vars = light_theme_vars if is_light else dark_theme_vars

# ====================== CUSTOM CSS (Theme Responsive) ======================
st.markdown(f"""
<style>
    :root, .stApp {{
        {active_vars}
    }}

    /* Completely remove Streamlit's top header white background & decoration */
    header[data-testid="stHeader"],
    header[data-testid="stHeader"] > div {{
        background: transparent !important;
        background-color: transparent !important;
    }}

    div[data-testid="stDecoration"] {{
        display: none !important;
    }}

    div[data-testid="stToolbar"] {{
        background: transparent !important;
    }}

    div[data-testid="stStatusWidget"] {{
        background: transparent !important;
    }}

    /* Reduce excessive top gap */
    .main .block-container {{
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1200px;
    }}

    /* Main background & base typography */
    .stApp {{
        background: var(--bg-page) !important;
        color: var(--text-primary) !important;
        font-size: 1.15rem;
    }}

    /* General text elements */
    .stApp p,
    .stApp li,
    .stApp label,
    div[data-testid="stMarkdownContainer"] > p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] label {{
        color: var(--text-primary) !important;
        font-size: 1.2rem !important;
        line-height: 1.7;
    }}

    /* Headings */
    h1,
    .main-hero-title,
    .stApp h1,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] .main-hero-title,
    div[data-testid="stMarkdownContainer"] > h1 {{
        background: var(--title-gradient) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-weight: 900 !important;
        font-size: 5.2rem !important;
        text-align: center !important;
        letter-spacing: -1.5px !important;
        line-height: 1.15 !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0.5rem !important;
        display: block !important;
    }}

    @media (max-width: 992px) {{
        h1,
        .main-hero-title,
        .stApp h1,
        div[data-testid="stMarkdownContainer"] h1,
        div[data-testid="stMarkdownContainer"] .main-hero-title,
        div[data-testid="stMarkdownContainer"] > h1 {{
            font-size: 3.8rem !important;
        }}
    }}

    @media (max-width: 768px) {{
        h1,
        .main-hero-title,
        .stApp h1,
        div[data-testid="stMarkdownContainer"] h1,
        div[data-testid="stMarkdownContainer"] .main-hero-title,
        div[data-testid="stMarkdownContainer"] > h1 {{
            font-size: 2.8rem !important;
        }}
    }}

    h2,
    .stApp h2,
    div[data-testid="stMarkdownContainer"] h2 {{
        color: var(--text-primary) !important;
        font-size: 2.2rem !important;
        font-weight: 800 !important;
    }}

    h3,
    .stApp h3,
    div[data-testid="stMarkdownContainer"] h3 {{
        color: var(--text-primary) !important;
        font-size: 1.7rem !important;
        font-weight: 700 !important;
    }}

    /* Subtitle */
    .subtitle {{
        text-align: center;
        color: var(--text-muted) !important;
        font-size: 1.45rem !important;
        font-weight: 500;
        margin-bottom: 2.2rem;
    }}

    /* Cards */
    .card {{
        background: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 18px;
        padding: 2.2rem;
        backdrop-filter: blur(14px);
        box-shadow: var(--card-shadow) !important;
        margin-bottom: 1.8rem;
        font-size: 1.2rem;
        color: var(--text-primary) !important;
        position: relative;
        overflow: hidden;
    }}

    .card::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--title-gradient);
    }}

    .card h3 {{
        color: var(--text-primary) !important;
        margin-top: 0;
    }}

    .card p, .card li, .card span, .card ol, .card ul {{
        color: var(--text-secondary) !important;
        font-size: 1.2rem !important;
    }}

    .card b, .card strong {{
        color: var(--text-primary) !important;
    }}

    /* Warning box */
    .warning-box {{
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        color: #ffffff !important;
        padding: 1.2rem 1.8rem;
        border-radius: 14px;
        font-weight: 500;
        font-size: 1.25rem;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(255, 65, 108, 0.25);
    }}

    .warning-box * {{
        color: #ffffff !important;
    }}

    /* Result cards */
    .result-real,
    div[data-testid="stMarkdownContainer"] .result-real {{
        background: linear-gradient(135deg, #11998e, #38ef7d) !important;
        color: #ffffff !important;
        padding: 1.6rem !important;
        border-radius: 16px !important;
        text-align: center !important;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        box-shadow: 0 10px 30px rgba(56, 239, 125, 0.3) !important;
    }}

    .result-fake,
    div[data-testid="stMarkdownContainer"] .result-fake {{
        background: linear-gradient(135deg, #ff416c, #ff4b2b) !important;
        color: #ffffff !important;
        padding: 1.6rem !important;
        border-radius: 16px !important;
        text-align: center !important;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        box-shadow: 0 10px 30px rgba(255, 65, 108, 0.3) !important;
    }}

    /* Result: mixed + probability bar + word chips */
    .result-mixed {{
        background: linear-gradient(135deg, #f59e0b, #f97316) !important;
        color: #ffffff !important;
        padding: 1.6rem !important;
        border-radius: 16px !important;
        text-align: center !important;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        box-shadow: 0 10px 30px rgba(245, 158, 11, 0.3) !important;
    }}

    .result-sub {{
        display: block;
        font-size: 1.2rem !important;
        font-weight: 500 !important;
        margin-top: 0.4rem;
        opacity: 0.95;
    }}

    .bar-wrap {{
        display: flex;
        height: 22px;
        border-radius: 999px;
        overflow: hidden;
        margin: 1.4rem 0 0.4rem 0;
        border: 1px solid var(--card-border);
    }}

    .bar-real {{ background: linear-gradient(90deg, #11998e, #38ef7d); }}
    .bar-fake {{ background: linear-gradient(90deg, #ff4b2b, #ff416c); }}

    .bar-labels {{
        display: flex;
        justify-content: space-between;
        font-size: 1.1rem !important;
        font-weight: 600;
        color: var(--text-primary) !important;
        margin-bottom: 1.2rem;
    }}

    .chip {{
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-size: 1.05rem !important;
        font-weight: 600;
        margin: 0.25rem 0.3rem 0.25rem 0;
        border: 1px solid transparent;
    }}

    .chip-real {{ background: rgba(17, 153, 142, 0.14); color: #0f766e !important; border-color: rgba(17, 153, 142, 0.4); }}
    .chip-fake {{ background: rgba(255, 65, 108, 0.14); color: #e11d48 !important; border-color: rgba(255, 65, 108, 0.4); }}

    /* Button styling */
    div.stButton > button {{
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: 1.15rem !important;
        padding: 0.65rem 1.2rem !important;
        height: 52px !important;
        min-height: 52px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}

    /* Primary button (Active navbar button or main action button) */
    div.stButton > button[data-testid="baseButton-primary"],
    div.stButton > button[kind="primary"] {{
        background: var(--btn-primary-bg) !important;
        color: #ffffff !important;
        border: 1px solid var(--btn-primary-border) !important;
        box-shadow: var(--btn-primary-shadow) !important;
    }}

    div.stButton > button[data-testid="baseButton-primary"]:hover,
    div.stButton > button[kind="primary"]:hover {{
        transform: translateY(-2px);
        box-shadow: var(--btn-primary-hover-shadow) !important;
    }}

    /* Secondary button (Inactive navbar buttons) */
    div.stButton > button[data-testid="baseButton-secondary"],
    div.stButton > button[kind="secondary"] {{
        background: var(--btn-sec-bg) !important;
        color: var(--btn-sec-color) !important;
        border: 1px solid var(--btn-sec-border) !important;
        backdrop-filter: blur(8px) !important;
    }}

    div.stButton > button[data-testid="baseButton-secondary"]:hover,
    div.stButton > button[kind="secondary"]:hover {{
        background: var(--btn-sec-hover-bg) !important;
        color: var(--btn-sec-hover-color) !important;
        border-color: var(--btn-sec-hover-border) !important;
        transform: translateY(-2px);
    }}

    /* Top Navbar */
    .nav-brand,
    div[data-testid="stMarkdownContainer"] .nav-brand {{
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        font-size: 2.35rem !important;
        font-weight: 900 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.8px !important;
        height: 52px !important;
        line-height: 52px !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    .nav-brand span,
    .nav-brand b,
    div[data-testid="stMarkdownContainer"] .nav-brand span,
    div[data-testid="stMarkdownContainer"] .nav-brand b {{
        font-size: 2.35rem !important;
        font-weight: 900 !important;
    }}

    .nav-brand span.nav-icon,
    div[data-testid="stMarkdownContainer"] .nav-brand span.nav-icon {{
        font-size: 2.8rem !important;
        display: inline-flex !important;
        align-items: center !important;
    }}

    .nav-brand b,
    div[data-testid="stMarkdownContainer"] .nav-brand b {{
        background: var(--title-gradient) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }}

    .nav-divider {{
        border: none;
        height: 1px;
        background: var(--divider) !important;
        margin: 0.8rem 0 2rem 0;
    }}

    /* Text area */
    .stTextArea textarea {{
        background-color: var(--input-bg) !important;
        border: 1.5px solid var(--input-border) !important;
        border-radius: 14px !important;
        color: var(--input-text) !important;
        font-size: 1.25rem !important;
        line-height: 1.6 !important;
        transition: all 0.2s ease !important;
    }}

    /* Make the text box wrapper follow our theme too (fixes unreadable text in Dark mode) */
    .stTextArea div:has(> textarea),
    .stTextArea div[data-baseweb="textarea"],
    .stTextArea div[data-baseweb="base-input"] {{
        background-color: var(--input-bg) !important;
        border-radius: 14px !important;
    }}

    .stTextArea textarea:focus {{
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
    }}

    .stTextArea label, label[data-testid="stWidgetLabel"] p {{
        color: var(--text-primary) !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }}

    /* Streamlit Alert boxes (info, warning, error) */
    div[data-testid="stAlert"] * {{
        font-size: 1.2rem !important;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--sidebar-border) !important;
    }}

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] label {{
        color: var(--text-primary) !important;
        font-size: 1.12rem !important;
        line-height: 1.6;
    }}

    /* Sidebar Brand Title */
    .sidebar-brand-title,
    section[data-testid="stSidebar"] .sidebar-brand-title,
    div[data-testid="stMarkdownContainer"] .sidebar-brand-title {{
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        line-height: 1.2 !important;
        letter-spacing: -0.5px !important;
        background: var(--title-gradient) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin: 0 !important;
        text-align: center !important;
        display: block !important;
    }}

    .sidebar-brand-icon,
    section[data-testid="stSidebar"] .sidebar-brand-icon,
    div[data-testid="stMarkdownContainer"] .sidebar-brand-icon {{
        font-size: 3.8rem !important;
        text-align: center !important;
        margin-bottom: 0.3rem !important;
        display: block !important;
    }}

    .sidebar-intro-box {{
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        box-shadow: var(--card-shadow);
        backdrop-filter: blur(12px);
    }}

    .sidebar-intro-title {{
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.5rem 0 !important;
        color: var(--text-primary) !important;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .sidebar-badge {{
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        font-size: 0.88rem !important;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.15);
        color: var(--text-muted) !important;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin: 0.2rem 0.2rem 0.2rem 0;
    }}

    .stCaption, [data-testid="stCaptionContainer"] {{
        color: var(--text-muted) !important;
        font-size: 1.05rem !important;
    }}

    /* Footer */
    .footer {{
        text-align: center;
        color: var(--footer-color) !important;
        font-size: 1.1rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--sidebar-border) !important;
    }}
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 0.6rem 0 1.2rem 0;">
        <div class="sidebar-brand-icon">📰</div>
        <div class="sidebar-brand-title">NewsLens</div>
        <p style="font-size: 1.15rem !important; color: var(--text-muted) !important; margin: 0.3rem 0 0 0; font-weight: 600;">
            Think before you share
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-intro-box">
        <div class="sidebar-intro-title">📖 About This Project</div>
        <p style="font-size: 1.05rem !important; margin: 0; color: var(--text-secondary) !important;">
            <b>NewsLens</b> helps readers think critically. It shows whether an article's <b>writing style</b> looks like wire-agency reporting or like sensational content, and <b>which words</b> drove that result. It does <b>not</b> decide what is true.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-intro-box">
        <div class="sidebar-intro-title">⚡ How To Use</div>
        <p style="font-size: 1.02rem !important; margin: 0; color: var(--text-secondary) !important;">
            1. Use the <b>top navbar</b> to switch pages.<br>
            2. Go to <b>🔍 Analyze News</b>.<br>
            3. Paste an article (20+ words) and analyze.<br>
            4. Toggle Light/Dark mode via the navbar.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-intro-box">
        <div class="sidebar-intro-title">🛠️ Core Technology</div>
        <div style="margin-top: 0.4rem;">
            <span class="sidebar-badge">NLP</span>
            <span class="sidebar-badge">TF-IDF</span>
            <span class="sidebar-badge">Logistic Regression</span>
            <span class="sidebar-badge">Streamlit</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-intro-box" style="border-left: 4px solid #f59e0b;">
        <div class="sidebar-intro-title" style="color: #f59e0b !important;">⚠️ Disclaimer</div>
        <p style="font-size: 0.95rem !important; margin: 0; color: var(--text-muted) !important;">
            Educational research prototype. Predictions reflect learned dataset patterns and do not replace professional journalistic fact-checking.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Media-literacy prototype • v3.0")

# ====================== TOP NAVBAR ======================
page = st.session_state.current_page

nav_brand, nav_home, nav_analyze, nav_about, nav_theme = st.columns(
    [2.7, 1.1, 1.35, 1.35, 1.05], vertical_alignment="center"
)

with nav_brand:
    st.markdown("""
    <div class="nav-brand">
        <span class="nav-icon">📰</span>
        <span>News<b>Lens</b></span>
    </div>
    """, unsafe_allow_html=True)

with nav_home:
    st.button("🏠 Home", key="nav_btn_home", on_click=set_page, args=("🏠 Home",), use_container_width=True, type="primary" if page == "🏠 Home" else "secondary")

with nav_analyze:
    st.button("🔍 Analyze News", key="nav_btn_analyze", on_click=set_page, args=("🔍 Analyze News",), use_container_width=True, type="primary" if page == "🔍 Analyze News" else "secondary")

with nav_about:
    st.button("ℹ️ About Project", key="nav_btn_about", on_click=set_page, args=("ℹ️ About Project",), use_container_width=True, type="primary" if page == "ℹ️ About Project" else "secondary")

with nav_theme:
    # Quick one-click toggle in top navbar
    toggle_icon = "☀️ Light" if not is_light else "🌙 Dark"
    st.button(toggle_icon, key="nav_btn_theme", on_click=toggle_theme, use_container_width=True)

st.markdown('<hr class="nav-divider">', unsafe_allow_html=True)

# ====================== MAIN CONTENT ======================

if page == "🏠 Home":
    st.markdown('<h1 class="main-hero-title">NewsLens</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Think before you share. See how an article is written, and why.</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3 style="margin-top:0;">The problem 🧭</h3>
        <p>Every day people share news they have only skimmed, and it is hard to tell which stories deserve a second look. Fact-checkers cannot keep up with the volume.</p>
        <br>
        <h3>What NewsLens does</h3>
        <ol>
            <li>You paste a news article.</li>
            <li>It compares the <b>writing style</b> with two kinds of news in its training data: wire-agency reporting and sensational / low-credibility content.</li>
            <li>It shows a percentage, the <b>words that drove the result</b>, and simple tips for checking the story yourself.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
        ⚠️ NewsLens is <b>not</b> a fact-checker. It analyses writing style only, and it can be wrong. See the limitations on the About page.
    </div>
    """, unsafe_allow_html=True)

elif page == "🔍 Analyze News":
    st.markdown('<h1 class="main-hero-title">Analyze an Article</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Paste the full article text (English). We show the writing style, not the truth.</p>', unsafe_allow_html=True)

    # Model loading
    MODEL_PATH = Path(__file__).parent / "model" / "fake_news_model.joblib"

    @st.cache_resource
    def load_model():
        return joblib.load(MODEL_PATH)

    try:
        model = load_model()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        st.error(f"⚠️ Model load nahi hua: {e}. Pehle terminal mein `python train_model.py` chalayein.")

    st.markdown("**Try an example** (two short texts we wrote about the same fictional event, in different styles):")
    ex1, ex2 = st.columns(2)
    with ex1:
        st.button("📖 Example A: story-style report", on_click=use_example, args=(EXAMPLE_NARRATIVE,), use_container_width=True)
    with ex2:
        st.button("📰 Example B: wire-agency style", on_click=use_example, args=(EXAMPLE_WIRE,), use_container_width=True)

    news_text = st.text_area(
        "News Article Text",
        height=260,
        key="news_text",
        placeholder="Paste the complete news article here (at least 20 words)..."
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("🔍 Analyze Article", use_container_width=True, type="primary")

    if analyze_btn:
        cleaned = clean_text(news_text)
        if not news_text.strip():
            st.warning("Please paste some news text first.")
        elif not model_loaded:
            st.error("Model is not loaded yet.")
        elif len(cleaned.split()) < MIN_WORDS:
            st.warning(f"Please paste at least {MIN_WORDS} words. Very short texts give unreliable results.")
        else:
            p_fake, p_real = model.predict_proba([cleaned])[0]   # classes: 0 = sensational/unreliable, 1 = wire/real
            real_pct, fake_pct = round(p_real * 100), round(p_fake * 100)

            if p_real >= 0.65:
                box, title, sub = "result-real", "📰 Wire-agency style", "Reads like the professional wire reporting in our training data"
            elif p_real <= 0.35:
                box, title, sub = "result-fake", "⚠️ Sensational style", "Reads like the sensational / low-credibility content in our training data"
            else:
                box, title, sub = "result-mixed", "❓ Mixed style", "The model is not sure. Treat this result with extra caution"

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="{box}">
                {title}
                <span class="result-sub">{sub}</span>
            </div>
            <div class="bar-wrap">
                <div class="bar-real" style="width:{p_real*100:.1f}%"></div>
                <div class="bar-fake" style="width:{p_fake*100:.1f}%"></div>
            </div>
            <div class="bar-labels"><span>📰 Wire style {real_pct}%</span><span>Sensational style {fake_pct}% ⚠️</span></div>
            """, unsafe_allow_html=True)

            wire_words, sens_words = explain(model, cleaned)
            stats = load_stats()
            chips_w = "".join(f'<span class="chip chip-real">{w}{word_note(stats, w)}</span>' for w in wire_words) or "<i>none</i>"
            chips_s = "".join(f'<span class="chip chip-fake">{w}{word_note(stats, w)}</span>' for w in sens_words) or "<i>none</i>"
            stats_note = (
                "<p><small>Numbers = percentage of training articles of each type "
                "(wire / sensational) that contained this word or phrase. They show a "
                "pattern in the training data, not whether a word is true or false.</small></p>"
                if stats is not None else
                "<p><small>Run <code>python build_word_stats.py</code> to also see how often "
                "each word appeared in the training data.</small></p>"
            )
            st.markdown(f"""
            <div class="card">
                <h3>🔎 Why this result? Words that influenced it</h3>
                <p><b>Pushed towards wire style:</b><br>{chips_w}</p>
                <p><b>Pushed towards sensational style:</b><br>{chips_s}</p>
                {stats_note}
            </div>
            """, unsafe_allow_html=True)

            tips_html = "".join(f"<li>{t}</li>" for t in TIPS)
            st.markdown(f"""
            <div class="card">
                <h3>✅ Check it yourself</h3>
                <ul>{tips_html}</ul>
            </div>
            """, unsafe_allow_html=True)

            st.info("ℹ️ This is a writing-style analysis, not a fact-check. A calm, accurate article can still look 'sensational' to the model, and a false story written in a formal style can look like wire reporting.")

elif page == "ℹ️ About Project":
    st.markdown('<h1 class="main-hero-title">About This Project</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">A Data Science project powered by NLP & Machine Learning</p>', unsafe_allow_html=True)

    # Project Details card
    st.markdown("""
    <div class="card">
        <h3>📋 Project Details</h3>
        <ul>
            <li><b>Domain:</b> Data Science &amp; Natural Language Processing (NLP)</li>
            <li><b>Task:</b> Writing-style analysis of news text (two classes: wire-agency style vs sensational / low-credibility style)</li>
            <li><b>Model:</b> TF-IDF Vectorizer + Logistic Regression</li>
            <li><b>Explainability:</b> the words that influenced each result come from the model's own weights (Logistic Regression coefficients)</li>
            <li><b>Interface:</b> Streamlit</li>
            <li><b>Training:</b> Trained by our team on a local machine using <code>train_model.py</code> (Python, scikit-learn)</li>
            <li><b>Dataset:</b> &ldquo;Fake and Real News Dataset&rdquo; (Fake.csv &amp; True.csv) from Kaggle &mdash; English news, 2015&ndash;2018, mostly US politics. About 39,000 articles used after removing duplicates. Credit to the original dataset creators.</li>
            <li><b>Acknowledgement:</b> Our first baseline was inspired by a public Kaggle notebook; the cleaning pipeline, saved model and this app were built by us for this hackathon.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Team section using Streamlit columns (safe rendering)
    st.markdown('<div class="card"><h3>👥 Project Team</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
<div style="border:1px solid rgba(79,70,229,0.35); border-top:4px solid #4f46e5; border-radius:14px; padding:1.6rem 1.2rem; text-align:center; background:rgba(79,70,229,0.06);">
  <p style="font-size:2.8rem; margin:0 0 0.5rem 0;">&#128104;&#8205;&#128187;</p>
  <p style="font-size:1.2rem; font-weight:800; margin:0 0 0.4rem 0; color:var(--text-primary);">Muhammad Rehan Farooq</p>
  <p style="display:inline-block; font-size:0.75rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; margin:0 0 0.6rem 0; padding:0.2rem 0.7rem; border-radius:999px; background:#4f46e5; color:#fff;">&#11088; Team Leader</p>
  <p style="font-size:1rem; margin:0; color:var(--text-secondary);">&#129302; AI &amp; Data Science</p>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div style="border:1px solid rgba(219,39,119,0.35); border-top:4px solid #db2777; border-radius:14px; padding:1.6rem 1.2rem; text-align:center; background:rgba(219,39,119,0.06);">
  <p style="font-size:2.8rem; margin:0 0 0.5rem 0;">&#128105;&#8205;&#128187;</p>
  <p style="font-size:1.2rem; font-weight:800; margin:0 0 0.4rem 0; color:var(--text-primary);">Habiba Siddique</p>
  <p style="font-size:1rem; margin:0; color:var(--text-secondary);">&#129504; Machine Learning &amp; Web Development</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Limitations card
    st.markdown("""
    <div class="card">
        <h3>🧪 Known limitations (we tested these)</h3>
        <ul>
            <li><b>Style, not truth.</b> The model learns how articles are written, not whether the facts are correct.</li>
            <li><b>One kind of "reliable" source.</b> In the training data, almost all wire-style articles (99.8%) come from a single news agency, Reuters. Other trusted outlets that write differently can be marked as sensational. In our own testing, a genuine mainstream-news article was labelled sensational-style.</li>
            <li><b>Same facts, different result.</b> On the Analyze page, Example A and Example B describe the same fictional event. A calm story-style version scores as sensational, the wire-style version as wire. Try it yourself.</li>
            <li><b>Narrow data.</b> English only; US political news from about 2015 to 2018. Other topics, languages and newer events may not work well.</li>
            <li><b>Accuracy on the dataset's own test split is high (about 98%)</b>, but that does not mean it works on any article from the web.</li>
        </ul>
        <p><b>Future work:</b> train on news from many publishers and regions, add Urdu support, and connect to fact-checking sources.</p>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer card
    st.markdown("""
    <div class="card">
        <h3>⚠️ Disclaimer</h3>
        <p>This application is an <b>educational research prototype</b>. A machine learning model cannot conclusively determine whether a news story is true or false — it only predicts based on patterns learned from historical training data. Always verify news through credible sources.</p>
    </div>
    """, unsafe_allow_html=True)

# ====================== FOOTER ======================
st.markdown("""
<div class="footer">
    NewsLens &bull; Educational prototype &bull; Style analysis only &bull; Not a fact-checking service
</div>
""", unsafe_allow_html=True)