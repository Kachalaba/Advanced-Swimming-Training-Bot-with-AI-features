"""
⚡ SPRINT AI — професійний відеоаналіз спортсменів.

Streamlit-оболонка з вкладками для плавання, бігу, велоспорту, суходолу,
реабілітації, історії атлета, AI-асистента та відеоінструментів.
"""

import logging
import sys
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from video_analysis.monitoring import init_sentry  # noqa: E402

init_sentry()


def _load_static_css() -> str:
    """Return the static premium stylesheet, or an empty string if missing."""
    css_path = Path(__file__).parent / "assets" / "styles.css"
    try:
        return css_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - cosmetic fallback
        logger.warning("Could not load styles.css: %s", exc)
        return ""


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

# Theme-specific custom properties are dynamic (light/dark); the rest of the
# premium stylesheet lives in assets/styles.css and is loaded once below.
st.markdown(f"<style>:root {{{_theme_vars}}}</style>", unsafe_allow_html=True)
st.markdown(f"<style>{_load_static_css()}</style>", unsafe_allow_html=True)


# ============================================================================
# HEADER with theme + language toggles
# ============================================================================
from i18n.translations import t  # noqa: E402

_col_header, _col_lang, _col_toggle = st.columns([13, 1, 1])
with _col_header:
    st.markdown(
        f"""
    <div class="premium-header">
        <div class="logo-text">⚡ SPRINT AI</div>
        <div class="tagline">{t('tagline')}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )
with _col_lang:
    st.markdown("<div style='margin-top:1.8rem'></div>", unsafe_allow_html=True)
    _lang = st.session_state["lang"]
    _lang_icon = "🇬🇧" if _lang == "uk" else "🇺🇦"
    if st.button(
        _lang_icon,
        key="lang_toggle",
        help="Switch language / Змінити мову",
        use_container_width=True,
    ):
        st.session_state["lang"] = "en" if _lang == "uk" else "uk"
        st.rerun()
with _col_toggle:
    st.markdown("<div style='margin-top:1.8rem'></div>", unsafe_allow_html=True)
    _toggle_icon = "☀️" if _theme == "dark" else "🌙"
    if st.button(
        _toggle_icon,
        key="theme_toggle",
        help="Переключити тему / Switch theme",
        use_container_width=True,
    ):
        st.session_state["theme"] = "light" if _theme == "dark" else "dark"
        st.rerun()

from pages.ai_assistant import render_ai_tab  # noqa: E402
from pages.cycling import render_cycling_tab  # noqa: E402
from pages.dryland import render_dryland_tab  # noqa: E402
from pages.history import render_history_tab  # noqa: E402
from pages.rehabilitation import render_rehabilitation_tab  # noqa: E402
from pages.running import render_running_tab  # noqa: E402

# ============================================================================
# PAGE MODULE IMPORTS
# ============================================================================
from pages.swimming import render_swimming_tab  # noqa: E402
from pages.tools import render_tools_tab  # noqa: E402


def main():
    """Main Streamlit app with tabs."""

    # ========================================================================
    # MAIN TABS
    # ========================================================================
    (
        tab_swimming,
        tab_running,
        tab_cycling,
        tab_dryland,
        tab_rehab,
        tab_history,
        tab_ai,
        tab_tools,
    ) = st.tabs(
        [
            t("tab_swimming"),
            t("tab_running"),
            t("tab_cycling"),
            t("tab_dryland"),
            t("tab_rehab"),
            t("tab_history"),
            t("tab_ai"),
            t("tab_tools"),
        ]
    )

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
    # TAB 5: REHABILITATION
    # ========================================================================
    with tab_rehab:
        render_rehabilitation_tab()

    # ========================================================================
    # TAB 6: HISTORY
    # ========================================================================
    with tab_history:
        render_history_tab()

    # ========================================================================
    # TAB 7: AI ASSISTANT
    # ========================================================================
    with tab_ai:
        render_ai_tab()

    # ========================================================================
    # TAB 8: VIDEO TOOLS
    # ========================================================================
    with tab_tools:
        render_tools_tab()


if __name__ == "__main__":
    main()
