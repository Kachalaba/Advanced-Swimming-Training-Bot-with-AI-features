"""
🏊 SPRINT AI - Професійний аналіз спортсменів
Плавання • Суходіл • AI-біомеханіка
"""

import logging
import streamlit as st
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="SPRINT AI • Аналіз спортсменів",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# THEME & LANGUAGE STATE
# ============================================================================
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
if "lang" not in st.session_state:
    st.session_state["lang"] = "uk"

_theme = st.session_state["theme"]

# ============================================================================
# PREMIUM CSS — THEME-AWARE
# ============================================================================
if _theme == "dark":
    _theme_vars = """
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a24;
        --bg-glass: rgba(26,26,36,0.85);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border-color: rgba(255,255,255,0.07);
        --shadow-card: 0 4px 24px rgba(0,0,0,0.4);
        --bg-app-start: #0a0a0f;
        --bg-app-end: #12121a;
    """
else:
    _theme_vars = """
        --bg-primary: #f8fafc;
        --bg-secondary: #f1f5f9;
        --bg-card: #ffffff;
        --bg-glass: rgba(255,255,255,0.92);
        --text-primary: #0f172a;
        --text-secondary: #64748b;
        --border-color: rgba(0,0,0,0.07);
        --shadow-card: 0 4px 24px rgba(0,0,0,0.08);
        --bg-app-start: #f8fafc;
        --bg-app-end: #e8f0fe;
    """

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {{
        {_theme_vars}
        --accent-blue: #3b82f6;
        --accent-purple: #8b5cf6;
        --accent-cyan: #06b6d4;
        --accent-green: #10b981;
        --accent-orange: #f59e0b;
    }}

    .stApp {{
        background: linear-gradient(135deg, var(--bg-app-start) 0%, var(--bg-app-end) 100%);
        color: var(--text-primary);
    }}

    /* === HEADER === */
    .premium-header {{
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        margin-bottom: 0.5rem;
    }}
    .logo-text {{
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -2px;
        margin-bottom: 0.25rem;
    }}
    .tagline {{
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: var(--text-secondary);
        font-weight: 400;
        letter-spacing: 3px;
        text-transform: uppercase;
    }}

    /* === TAB NAVIGATION === */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: var(--bg-card);
        padding: 6px;
        border-radius: 16px;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-card);
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 12px;
        color: var(--text-secondary);
        font-weight: 600;
        padding: 10px 28px;
        font-size: 0.95rem;
        transition: all 0.2s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
        color: white !important;
        box-shadow: 0 4px 12px rgba(59,130,246,0.35);
    }}

    /* === CARDS === */
    .glass-card {{
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-card);
    }}

    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }}
    .metric-item {{
        background: linear-gradient(135deg, rgba(59,130,246,0.08) 0%, rgba(139,92,246,0.08) 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-card);
    }}
    .metric-item:hover {{
        transform: translateY(-3px);
        border-color: var(--accent-blue);
        box-shadow: 0 12px 32px rgba(59,130,246,0.18);
    }}
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
    }}
    .metric-label {{
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }}

    /* === BUTTONS === */
    .stButton>button {{
        background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.9rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
        box-shadow: 0 4px 16px rgba(59,130,246,0.28);
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(59,130,246,0.45);
    }}

    /* === STATUS BOXES === */
    .status-success {{
        background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(6,182,212,0.12) 100%);
        border: 1px solid rgba(16,185,129,0.5);
        border-radius: 10px;
        padding: 0.85rem 1.25rem;
        margin: 0.4rem 0;
        color: var(--accent-green);
        font-weight: 500;
    }}
    .status-info {{
        background: linear-gradient(135deg, rgba(59,130,246,0.1) 0%, rgba(139,92,246,0.1) 100%);
        border: 1px solid rgba(59,130,246,0.4);
        border-radius: 10px;
        padding: 0.85rem 1.25rem;
        margin: 0.4rem 0;
        color: var(--accent-blue);
        font-weight: 500;
    }}
    .status-warning {{
        background: linear-gradient(135deg, rgba(245,158,11,0.12) 0%, rgba(249,115,22,0.12) 100%);
        border: 1px solid rgba(245,158,11,0.5);
        border-radius: 10px;
        padding: 0.85rem 1.25rem;
        margin: 0.4rem 0;
        color: var(--accent-orange);
        font-weight: 500;
    }}

    /* === UPLOAD AREA === */
    .upload-zone {{
        border: 2px dashed var(--border-color);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: var(--bg-glass);
        transition: all 0.2s ease;
    }}
    .upload-zone:hover {{
        border-color: var(--accent-blue);
        background: rgba(59,130,246,0.04);
    }}

    /* === SECTION HEADERS === */
    .section-title {{
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 1.75rem 0 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}
    .section-title::before {{
        content: '';
        width: 3px;
        height: 22px;
        background: linear-gradient(180deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
        border-radius: 2px;
        flex-shrink: 0;
    }}

    /* === SIDEBAR === */
    [data-testid="stSidebar"] {{
        background: var(--bg-card);
        border-right: 1px solid var(--border-color);
    }}

    /* === INPUTS === */
    .stTextInput>div>div>input, .stSelectbox>div>div, .stSlider {{
        background: var(--bg-card) !important;
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
    }}

    /* === EXPANDER === */
    .streamlit-expanderHeader {{
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
    }}

    /* === HIDE STREAMLIT BRANDING === */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* === LEGACY COMPAT === */
    .success-box {{
        background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(6,182,212,0.12) 100%);
        border: 1px solid rgba(16,185,129,0.5);
        border-radius: 10px;
        padding: 0.85rem 1.25rem;
        margin: 0.4rem 0;
    }}
    .warning-box {{
        background: linear-gradient(135deg, rgba(245,158,11,0.12) 0%, rgba(249,115,22,0.12) 100%);
        border: 1px solid rgba(245,158,11,0.5);
        border-radius: 10px;
        padding: 0.85rem 1.25rem;
        margin: 0.4rem 0;
    }}
    .info-box {{
        background: linear-gradient(135deg, rgba(59,130,246,0.1) 0%, rgba(139,92,246,0.1) 100%);
        border: 1px solid rgba(59,130,246,0.4);
        border-radius: 10px;
        padding: 0.85rem 1.25rem;
        margin: 0.4rem 0;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HEADER with theme + language toggles
# ============================================================================
from i18n.translations import t  # noqa: E402

_col_header, _col_lang, _col_toggle = st.columns([13, 1, 1])
with _col_header:
    st.markdown(f"""
    <div class="premium-header">
        <div class="logo-text">⚡ SPRINT AI</div>
        <div class="tagline">{t('tagline')}</div>
    </div>
    """, unsafe_allow_html=True)
with _col_lang:
    st.markdown("<div style='margin-top:1.8rem'></div>", unsafe_allow_html=True)
    _lang = st.session_state["lang"]
    _lang_icon = "🇬🇧" if _lang == "uk" else "🇺🇦"
    if st.button(_lang_icon, key="lang_toggle", help="Switch language / Змінити мову", use_container_width=True):
        st.session_state["lang"] = "en" if _lang == "uk" else "uk"
        st.rerun()
with _col_toggle:
    st.markdown("<div style='margin-top:1.8rem'></div>", unsafe_allow_html=True)
    _toggle_icon = "☀️" if _theme == "dark" else "🌙"
    if st.button(_toggle_icon, key="theme_toggle", help="Переключити тему / Switch theme", use_container_width=True):
        st.session_state["theme"] = "light" if _theme == "dark" else "dark"
        st.rerun()

# ============================================================================
# PAGE MODULE IMPORTS
# ============================================================================
from pages.swimming import render_swimming_tab  # noqa: E402
from pages.running import render_running_tab  # noqa: E402
from pages.cycling import render_cycling_tab  # noqa: E402
from pages.dryland import render_dryland_tab  # noqa: E402
from pages.history import render_history_tab  # noqa: E402
from pages.ai_assistant import render_ai_tab  # noqa: E402
from pages.tools import render_tools_tab  # noqa: E402


def main():
    """Main Streamlit app with tabs."""

    # ========================================================================
    # MAIN TABS
    # ========================================================================
    tab_swimming, tab_running, tab_cycling, tab_dryland, tab_history, tab_ai, tab_tools = st.tabs([
        t("tab_swimming"),
        t("tab_running"),
        t("tab_cycling"),
        t("tab_dryland"),
        t("tab_history"),
        t("tab_ai"),
        t("tab_tools"),
    ])

    # ========================================================================
    # TAB 1: SWIMMING
    # ========================================================================
    with tab_swimming:
        render_swimming_tab()

    # ========================================================================
    # TAB 2: RUNNING
    # ========================================================================
    with tab_running:
        render_running_tab()

    # ========================================================================
    # TAB 3: CYCLING
    # ========================================================================
    with tab_cycling:
        render_cycling_tab()

    # ========================================================================
    # TAB 4: DRYLAND
    # ========================================================================
    with tab_dryland:
        render_dryland_tab()

    # ========================================================================
    # TAB 5: HISTORY
    # ========================================================================
    with tab_history:
        render_history_tab()

    # ========================================================================
    # TAB 6: AI ASSISTANT
    # ========================================================================
    with tab_ai:
        render_ai_tab()

    # ========================================================================
    # TAB 7: VIDEO TOOLS
    # ========================================================================
    with tab_tools:
        render_tools_tab()


if __name__ == "__main__":
    main()
