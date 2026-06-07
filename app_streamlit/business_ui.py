"""Business-friendly UI helpers for the Streamlit app."""

from __future__ import annotations

import base64
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import TARGET, UNIFIED_PARQUET
from src.inference import demand_type_label
from src.regions import REGION_MAP

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_BACKGROUND_CANDIDATES = (
    _ASSETS_DIR / "chicago_background.jpg",
    _ASSETS_DIR / "chicago_background.jpeg",
    _ASSETS_DIR / "chicago_background.png",
)
_LOGO_PATH = _ASSETS_DIR / "logo.png"
_LOGO_DISPLAY_W = 180
_LOGO_DISPLAY_H = 132
_LOGO_SLOT_W = 196

BUSINESS_CATEGORY_LABELS = {
    "Coffe": "Coffee Shop",
    "Restaurent": "Restaurant",
    "ghost kichen": "Ghost Kitchen",
    "Bakery": "Bakery",
}

REGION_LABELS = {
    "DOWNTOWN": "Downtown",
    "NORTH_SIDE": "North Side",
    "WEST_SIDE": "West Side",
    "SOUTH_SIDE": "South Side",
    "FAR_SOUTH": "Far South",
    "OTHER": "Other",
}

DEMAND_BANDS = [
    ("VERY LOW", "A quiet day — lighter foot traffic than usual.", "#cbd5e1", "#334155"),
    ("LOW", "Below-average demand — plan for a slower shift.", "#e2e8f0", "#3b4f68"),
    ("MODERATE", "A typical day — steady customer flow.", "#f1f5f9", "#1e3a5f"),
    ("HIGH", "A busy day — more customers than average.", "#f8fafc", "#1d4ed8"),
    ("PEAK", "Peak demand — prepare for a major rush.", "#ffffff", "#1e40af"),
]

_THEME_CSS = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0,0&display=swap');

        :root {
            --brand: #1d4ed8;
            --brand-deep: #0f172a;
            --accent: #d4bc8e;
            --accent-soft: rgba(180, 155, 115, 0.22);
            --gold: #c9b896;
            --gold-light: #ddd0b4;
            --gold-muted: #a89878;
            --gold-dark: #8a7a62;
            --panel: #0f172a;
            --panel-raised: #1a2438;
            --panel-surface: #1f2d42;
            --border: rgba(185, 165, 128, 0.45);
            --border-strong: rgba(201, 184, 150, 0.72);
            --inner-border: 1px solid rgba(185, 165, 128, 0.5);
            --frame-border: 3px solid rgba(201, 184, 150, 0.78);
            --frame-glow: 0 0 0 1px rgba(221, 208, 180, 0.18), 0 0 0 6px rgba(138, 122, 98, 0.14);
            --text-primary: #f5f0e8;
            --text-muted: #b0a898;
            --positive: #7dcea8;
            --dropdown-bg: #1e293b;
            --dropdown-text: #f5f0e8;
            --dropdown-hover: #2d3a50;
            --glass-overlay: linear-gradient(
                165deg,
                rgba(8, 14, 24, 0.78) 0%,
                rgba(15, 23, 42, 0.62) 50%,
                rgba(8, 14, 24, 0.74) 100%
            );
        }

        .stApp {
            background-color: #080e18;
            font-family: 'Inter', sans-serif;
            color: var(--text-primary);
        }
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            background-image: __BG_IMAGE__;
            background-position: center 32%;
            background-size: cover;
            background-repeat: no-repeat;
            z-index: 0;
            pointer-events: none;
        }
        [data-testid="stHeader"], [data-testid="stToolbar"], footer { visibility: hidden; height: 0; }
        [data-testid="stAppViewContainer"] {
            position: relative;
            z-index: 1;
        }
        [data-testid="stAppViewContainer"] > .main,
        .main,
        .main > div,
        [data-testid="stMainBlockContainer"] {
            background: transparent !important;
        }
        section[data-testid="stSidebar"] { display: none; }

        .main .block-container {
            position: relative;
            max-width: 900px;
            margin: 1.25rem auto 2rem;
            padding: 2rem 2.25rem 2rem;
            background-color: rgba(8, 14, 24, 0.55) !important;
            background-image: var(--glass-overlay), __BG_IMAGE__;
            background-position: center, center 32%;
            background-size: cover, cover;
            background-repeat: no-repeat, no-repeat;
            border: var(--frame-border);
            border-radius: 14px;
            box-shadow: var(--frame-glow), 0 28px 72px rgba(0, 0, 0, 0.58);
            overflow: hidden;
        }
        .main .block-container::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--gold-dark), var(--gold), var(--gold-light), var(--gold));
            z-index: 2;
        }
        .main .block-container::after {
            content: "";
            position: absolute;
            inset: 9px;
            border: 1px solid rgba(201, 184, 150, 0.38);
            border-radius: 8px;
            pointer-events: none;
            z-index: 0;
        }
        .main .block-container > div {
            position: relative;
            z-index: 1;
        }

        .app-hero {
            display: grid;
            grid-template-columns: __LOGO_SLOT_W__ 1fr __LOGO_SLOT_W__;
            align-items: center;
            gap: 0.75rem 1rem;
            margin-bottom: 1.5rem;
            padding: 1.25rem 1.5rem 1.15rem;
            background: var(--panel-raised);
            border: var(--inner-border);
            border-radius: 12px;
        }
        .app-hero-logo-wrap {
            justify-self: start;
        }
        .app-hero-logo {
            display: block;
            width: __LOGO_DISPLAY_W__;
            height: __LOGO_DISPLAY_H__;
            box-sizing: border-box;
            object-fit: contain;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.98);
            padding: 6px 8px;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
            image-rendering: -webkit-optimize-contrast;
            image-rendering: high-quality;
        }
        .app-hero-text {
            grid-column: 2;
            text-align: center;
            justify-self: center;
            width: 100%;
        }
        .app-hero-balance {
            grid-column: 3;
            width: __LOGO_SLOT_W__;
            height: 1px;
        }
        .app-hero h1 {
            font-size: 2.35rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 0 0 0.45rem 0;
            letter-spacing: -0.03em;
            line-height: 1.15;
        }
        .app-hero p {
            margin: 0 auto;
            max-width: 34rem;
            color: var(--text-muted);
            font-size: 0.95rem;
            font-weight: 400;
            line-height: 1.55;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel-raised) !important;
            border: 2px solid rgba(201, 184, 150, 0.62) !important;
            border-radius: 12px !important;
            padding: 1rem 1.1rem 0.85rem !important;
            margin-bottom: 0.85rem !important;
            box-shadow: inset 0 0 0 1px rgba(221, 208, 180, 0.08) !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="column"] {
            background: var(--panel-surface) !important;
            border: 1px solid rgba(185, 165, 128, 0.45) !important;
            border-radius: 10px !important;
            padding: 0.65rem 0.55rem 0.5rem !important;
            margin: 0 0.2rem !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stDateInput"],
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSelectbox"] {
            background: #4b5563 !important;
            border-radius: 8px !important;
            padding: 0.45rem 0.5rem 0.5rem !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stDateInput"] [data-testid="stWidgetLabel"] p,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stDateInput"] label p,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSelectbox"] label p {
            color: #e5e7eb !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stDateInput"] > div,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stDateInput"] [data-baseweb="input"],
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stDateInput"] fieldset,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stDateInput"] input {
            background: #374151 !important;
            background-color: #374151 !important;
            border-color: rgba(156, 163, 175, 0.45) !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSelectbox"] div[role="combobox"],
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSelectbox"] div[role="combobox"] > div,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSelectbox"] div[role="combobox"] > div > div {
            background: #374151 !important;
            background-color: #374151 !important;
            color: #f3f4f6 !important;
        }

        [data-testid="stSelectbox"],
        [data-testid="stDateInput"],
        [data-testid="stExpander"],
        [data-testid="stHorizontalBlock"],
        [data-testid="column"],
        .stSelectbox,
        .stDateInput {
            background: transparent !important;
        }

        [data-testid="stWidgetLabel"] p,
        .stSelectbox label, .stDateInput label, .stToggle label, .stSlider label, .stCheckbox label,
        .stSelectbox label p, .stDateInput label p, .stSlider label p, .stCheckbox label p,
        .stCheckbox label span, .stToggle label span, .stToggle label p {
            color: var(--text-primary) !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-testid="stDateInput"] > div,
        div[data-testid="stDateInput"] [data-baseweb="input"],
        div[data-testid="stDateInput"] fieldset,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background: var(--panel-surface) !important;
            background-color: var(--panel-surface) !important;
            color: var(--text-primary) !important;
            border: var(--inner-border) !important;
            border-radius: 8px !important;
            font-family: 'Inter', sans-serif !important;
        }
        div[data-testid="stDateInput"] input,
        input[type="date"] {
            background: var(--panel-surface) !important;
            background-color: var(--panel-surface) !important;
            color: var(--text-primary) !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            border: none !important;
        }
        div[data-baseweb="select"] div[role="combobox"],
        div[data-baseweb="select"] div[role="combobox"] > div,
        div[data-baseweb="select"] div[role="combobox"] > div > div {
            color: var(--text-primary) !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            background: var(--panel-surface) !important;
            background-color: var(--panel-surface) !important;
        }
        div[data-baseweb="select"] svg { fill: var(--text-muted) !important; }

        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        [data-baseweb="menu"],
        ul[role="listbox"] {
            background: var(--dropdown-bg) !important;
            background-color: var(--dropdown-bg) !important;
            border: 1px solid var(--gold-muted) !important;
            border-radius: 8px !important;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5) !important;
        }
        li[role="option"],
        li[role="option"] > div,
        li[role="option"] span,
        [data-baseweb="menu"] li,
        [data-baseweb="menu"] li > div,
        [data-baseweb="menu"] li span,
        ul[role="listbox"] li,
        ul[role="listbox"] li * {
            color: var(--dropdown-text) !important;
            background-color: var(--dropdown-bg) !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
        }
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"],
        [data-baseweb="menu"] li:hover,
        [data-baseweb="menu"] li[aria-selected="true"] {
            background: var(--dropdown-hover) !important;
            background-color: var(--dropdown-hover) !important;
            color: var(--text-primary) !important;
        }
        li[role="option"][aria-selected="true"] *,
        [data-baseweb="menu"] li[aria-selected="true"] * {
            color: var(--text-primary) !important;
        }
        .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
            color: var(--text-muted) !important;
            font-weight: 400 !important;
        }

        div[data-testid="stButton"] > button[kind="primary"] {
            background: var(--brand) !important;
            color: #ffffff !important;
            border: 1px solid var(--gold-muted) !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-family: 'Inter', sans-serif !important;
            padding: 0.7rem 1.25rem !important;
            box-shadow: none !important;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: #2563eb !important;
        }
        div[data-testid="stButton"] > button[kind="secondary"] {
            background: var(--panel-surface) !important;
            color: var(--text-muted) !important;
            border: var(--inner-border) !important;
            font-family: 'Inter', sans-serif !important;
        }

        div[data-testid="stMetric"] {
            background: var(--panel-raised);
            border: var(--inner-border);
            border-radius: 10px;
            padding: 1rem 1.1rem;
            color: var(--text-primary) !important;
        }
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] *,
        div[data-testid="stMetricLabel"] p,
        div[data-testid="stMetricLabel"] label,
        div[data-testid="stMetricLabel"] div,
        [data-testid="stMetric"] [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            color: var(--text-primary) !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] div,
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.75rem !important;
            font-weight: 700 !important;
            color: var(--text-primary) !important;
        }
        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] div {
            color: var(--positive) !important;
            font-weight: 500 !important;
        }
        div[data-testid="stMetricDelta"] svg {
            fill: var(--positive) !important;
        }

        .block-container h4, .block-container h3 {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }

        .forecast-hero {
            background: var(--panel-raised);
            border-radius: 10px;
            padding: 1rem 1.25rem;
            color: var(--text-primary);
            margin: 0 auto 0.85rem auto;
            max-width: 640px;
            border: var(--inner-border);
        }
        .forecast-hero-inner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1.25rem;
            flex-wrap: wrap;
        }
        .forecast-main { min-width: 140px; }
        .forecast-meta { flex: 1; min-width: 200px; }
        .forecast-hero .label {
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-muted);
        }
        .forecast-hero .number {
            font-size: 2.75rem;
            font-weight: 800;
            line-height: 1.05;
            margin: 0.15rem 0 0;
            letter-spacing: -0.03em;
            background: linear-gradient(135deg, #f0e6d0 0%, #d4bc8e 45%, #c9a86c 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .forecast-hero .context {
            font-size: 0.84rem;
            font-weight: 400;
            color: var(--text-muted);
            margin-top: 0;
        }
        .forecast-hero .context strong {
            color: var(--text-primary);
            font-weight: 600;
        }
        .demand-badge {
            display: inline-block;
            padding: 0.28rem 0.65rem;
            border-radius: 6px;
            font-size: 0.62rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
            border: 1px solid var(--gold-muted);
        }

        .summary-callout {
            background: var(--panel-raised);
            border: var(--inner-border);
            border-left: 4px solid var(--gold);
            border-radius: 0 10px 10px 0;
            padding: 0.85rem 1.1rem;
            color: var(--text-primary);
            font-size: 0.88rem;
            font-weight: 400;
            line-height: 1.55;
            margin: 0 auto 1.1rem auto;
            max-width: 640px;
        }

        .insight-card {
            background: var(--panel-raised);
            border: var(--inner-border);
            border-radius: 10px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 0.75rem;
        }
        .insight-card h4 {
            margin: 0 0 0.4rem;
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--gold);
        }
        .insight-card p {
            margin: 0;
            color: var(--text-muted);
            font-size: 0.86rem;
            font-weight: 400;
            line-height: 1.6;
        }
        .insight-card p strong { color: var(--text-primary); font-weight: 600; }

        .welcome-panel {
            text-align: center;
            padding: 3rem 2rem;
            background: var(--panel-raised);
            border: var(--inner-border);
            border-radius: 14px;
        }
        .welcome-panel .focus-label {
            display: inline-block;
            margin-bottom: 0.65rem;
            padding: 0.22rem 0.65rem;
            border: 1px solid var(--gold-muted);
            border-radius: 999px;
            color: var(--gold-light);
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .welcome-panel h2 {
            color: var(--text-primary);
            font-size: 1.35rem;
            margin: 0 0 0.5rem;
        }
        .welcome-panel p {
            color: var(--text-muted);
            max-width: 420px;
            margin: 0 auto 1.5rem;
            line-height: 1.6;
        }
        .value-props-shell {
            background: var(--panel-raised);
            border: var(--inner-border);
            border-radius: 12px;
            padding: 1rem;
            margin-top: 1rem;
        }
        .value-props-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.75rem;
        }
        .value-prop {
            background: var(--panel-surface);
            border: var(--inner-border);
            border-radius: 12px;
            padding: 1.15rem 0.9rem 1.1rem;
            text-align: center;
            height: 100%;
        }
        .value-prop-head {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            margin-bottom: 0.7rem;
        }
        .value-prop-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 2.1rem;
            height: 2.1rem;
            border-radius: 10px;
            background: linear-gradient(145deg, #3d3528, #6b5d45);
            border: 1px solid var(--gold-muted);
            color: var(--gold-light);
            font-family: 'Material Symbols Rounded' !important;
            font-size: 1.2rem !important;
            font-variation-settings: 'FILL' 1, 'wght' 500, 'GRAD' 0, 'opsz' 24;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        }
        .value-prop .step {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 1.55rem;
            height: 1.55rem;
            padding: 0 0.35rem;
            border-radius: 6px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.04em;
        }
        .value-prop h3 {
            font-size: 0.88rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 0 0 0.4rem;
        }
        .value-prop p {
            font-size: 0.8rem;
            color: var(--text-muted);
            margin: 0;
            line-height: 1.5;
        }

        .loading-panel {
            background: var(--panel-raised);
            border: var(--inner-border);
            border-radius: 12px;
            padding: 2.5rem 1.5rem;
            margin: 1.25rem 0;
            text-align: center;
        }
        .loading-panel .loading-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 3rem;
            height: 3rem;
            margin-bottom: 1rem;
            border-radius: 50%;
            background: linear-gradient(145deg, #3d3528, #6b5d45);
            border: 1px solid var(--gold-muted);
            color: var(--gold-light);
            font-family: 'Material Symbols Rounded' !important;
            font-size: 1.6rem !important;
            animation: spin-pulse 1.2s linear infinite;
        }
        .loading-panel h3 {
            margin: 0 0 0.35rem;
            color: var(--text-primary);
            font-size: 1.05rem;
            font-weight: 600;
        }
        .loading-panel p {
            margin: 0;
            color: var(--text-muted);
            font-size: 0.88rem;
        }
        @keyframes spin-pulse {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        [data-testid="stSpinner"],
        [data-testid="stSpinner"] > div,
        [data-testid="stSpinner"] > div > div {
            background: var(--panel) !important;
            background-image: none !important;
        }
        [data-testid="stSpinner"] p,
        [data-testid="stSpinner"] span,
        [data-testid="stSpinner"] label {
            color: var(--text-primary) !important;
            opacity: 1 !important;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }
        .chip {
            background: var(--panel);
            border: var(--inner-border);
            border-radius: 6px;
            padding: 0.2rem 0.6rem;
            font-size: 0.7rem;
            font-weight: 500;
            color: var(--text-muted);
        }

        .app-footer-start { display: none; }
        .main .block-container > div > [data-testid="stVerticalBlock"]:last-child {
            background: var(--panel-raised) !important;
            border: var(--inner-border) !important;
            border-radius: 10px !important;
            padding: 0.85rem 1rem 0.75rem !important;
            margin-top: 1.5rem !important;
        }
        .footer-note {
            margin: 0;
            padding: 0;
            border: none;
            color: var(--text-primary);
            font-size: 0.82rem;
            font-weight: 500;
            line-height: 1.5;
        }
        .footer-note .footer-muted {
            color: var(--text-muted);
            font-weight: 400;
        }

        div[data-testid="stExpander"] {
            background: var(--panel-raised) !important;
            border: var(--inner-border) !important;
            border-radius: 12px !important;
            margin-top: 0.35rem !important;
        }
        div[data-testid="stExpander"] summary [data-testid="stIconMaterial"],
        span[data-testid="stIconMaterial"] {
            font-family: 'Material Symbols Rounded' !important;
            font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24 !important;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary span,
        div[data-testid="stExpander"] summary p,
        div[data-testid="stExpander"] summary [data-testid="stMarkdownContainer"] p {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
        }
        div[data-testid="stExpanderDetails"] label,
        div[data-testid="stExpanderDetails"] p,
        div[data-testid="stExpanderDetails"] span,
        div[data-testid="stExpanderDetails"] strong {
            color: var(--text-muted) !important;
            font-family: 'Inter', sans-serif !important;
        }
        div[data-testid="stExpanderDetails"] strong,
        div[data-testid="stExpanderDetails"] .stMarkdown p {
            color: var(--text-primary) !important;
        }

        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
            background-color: rgba(100, 116, 139, 0.35) !important;
        }
        div[data-testid="stSlider"] [data-baseweb="slider"] > div > div[data-index="0"] {
            background-color: var(--gold-muted) !important;
        }
        div[data-testid="stSlider"] [role="slider"] {
            background-color: var(--gold) !important;
            border: 2px solid var(--gold-light) !important;
        }
        div[data-testid="stSlider"] [data-testid="stThumbValue"] {
            color: var(--text-primary) !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
        }
        div[data-testid="stCheckbox"] label,
        div[data-testid="stCheckbox"] label p,
        div[data-testid="stCheckbox"] label span {
            color: var(--text-muted) !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            text-transform: none !important;
        }
        div[data-testid="stToggle"] [data-testid="stCheckbox"] label,
        div[data-testid="stToggle"] label span,
        div[data-testid="stToggle"] label p {
            color: var(--text-primary) !important;
            text-transform: uppercase !important;
        }

        .stCaption, [data-testid="stCaptionContainer"] p,
        div[data-testid="stExpanderDetails"] .stCaption p {
            color: var(--text-muted) !important;
            font-size: 0.82rem !important;
        }
        div[data-testid="stExpanderDetails"] .stCaption strong {
            color: var(--gold) !important;
        }

        .section-divider {
            border: none;
            border-top: 1px solid rgba(201, 184, 150, 0.35);
            margin: 1.35rem 0 1.1rem;
        }

        .main .block-container .app-hero,
        .main .block-container .forecast-hero,
        .main .block-container .summary-callout,
        .main .block-container .insight-card,
        .main .block-container .welcome-panel,
        .main .block-container .value-props-shell,
        .main .block-container .value-prop,
        .main .block-container .loading-panel,
        .main .block-container [data-testid="stVerticalBlockBorderWrapper"],
        .main .block-container div[data-testid="stMetric"],
        .main .block-container div[data-testid="stExpander"],
        .main .block-container > div > [data-testid="stVerticalBlock"]:last-child {
            border-color: rgba(185, 165, 128, 0.52) !important;
        }

        [data-testid="stMarkdownContainer"],
        [data-testid="stElementContainer"],
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"] > [data-testid="column"] > div {
            background: transparent !important;
        }
        div[data-testid="stExpanderDetails"] {
            background: var(--panel-raised) !important;
        }
        </style>
"""


def display_business_category(value: str) -> str:
    return BUSINESS_CATEGORY_LABELS.get(value, value.replace("_", " ").title())


def display_region(value: str) -> str:
    return REGION_LABELS.get(value, value.replace("_", " ").title())


def format_forecast_date(value: date) -> str:
    return value.strftime("%a, %B %d, %Y").replace(" 0", " ")


def _summary_for_band(band_name: str, band_summary: str) -> str:
    messages = {
        "PEAK": "Expect a major rush — significantly more customers than your area's average.",
        "HIGH": "Busier than usual — plan for above-average traffic through the day.",
        "MODERATE": "Right around a normal day — steady flow, no major surprises expected.",
        "LOW": "Slower than usual — a good day to catch up on prep and reduce waste.",
        "VERY LOW": "Quiet day ahead — lighter foot traffic than you typically see.",
    }
    return messages.get(band_name, band_summary)


def band_style(band: str) -> dict[str, str]:
    for name, _, color, bg in DEMAND_BANDS:
        if name == band:
            return {"name": name, "color": color, "bg": bg}
    return {"name": band, "color": "#f1f5f9", "bg": "#1e3a5f"}


@lru_cache(maxsize=1)
def _reference_traffic_stats() -> pd.DataFrame:
    df = pd.read_parquet(UNIFIED_PARQUET, engine="pyarrow")
    df["region"] = df["COMMUNITY AREA"].map(REGION_MAP).fillna("OTHER")
    return (
        df.groupby(["region", "business_category"])[TARGET]
        .agg(p10=lambda s: s.quantile(0.10), p50="median", p90=lambda s: s.quantile(0.90))
        .reset_index()
    )


def demand_gauge(prediction: int, region: str, business_category: str) -> dict:
    stats = _reference_traffic_stats()
    row = stats.loc[
        (stats["region"] == region) & (stats["business_category"] == business_category)
    ]
    if row.empty:
        p10, p50, p90 = 80.0, 120.0, 160.0
    else:
        p10 = float(row.iloc[0]["p10"])
        p50 = float(row.iloc[0]["p50"])
        p90 = float(row.iloc[0]["p90"])

    span = max(p90 - p10, 1.0)
    score = max(0.0, min(1.0, (prediction - p10) / span))
    band_idx = min(4, int(score * 5) if score < 1 else 4)
    band_name, band_summary, _, _ = DEMAND_BANDS[band_idx]
    summary = _summary_for_band(band_name, band_summary)

    return {
        "band": band_name,
        "summary": summary,
        "score_pct": int(score * 100),
        "reference_median": int(round(p50)),
        "p10": int(round(p10)),
        "p90": int(round(p90)),
    }


def business_insights(
    prediction: int,
    details: dict,
    gauge: dict,
    *,
    business_category: str,
    region: str,
    target_date: date,
) -> dict[str, str]:
    raw = details.get("raw_row", {})
    demand_type = details.get("demand_type", "baseline")
    ref = int(gauge.get("reference_median", details.get("reference_median", prediction)))
    delta = prediction - ref
    delta_pct = round((delta / ref) * 100) if ref else 0
    band = gauge.get("band", "MODERATE")

    temp = float(raw.get("temperature_f", 70) or 70)
    precip = float(raw.get("precipitation_in", 0) or 0)
    snow = float(raw.get("snowfall_in", 0) or 0)
    events = int(float(raw.get("city_special_events", 0) or 0))
    festival = bool(raw.get("is_major_festival", 0))
    cubs = bool(raw.get("cubs_home_game", 0))
    bulls = bool(raw.get("bulls_home_game", 0))
    holiday = bool(raw.get("is_holiday", 0))
    weekend = bool(raw.get("is_weekend", 0))
    category = display_business_category(business_category)
    region_label = display_region(region)
    day_name = target_date.strftime("%A")
    day_pattern = demand_type_label(demand_type).split(" — ")[0]

    if delta_pct >= 15:
        staff_note = "Add 1–2 extra team members"
    elif delta_pct >= 5:
        staff_note = "Add one flexible shift or extend peak coverage by 1 hour"
    elif delta_pct <= -15:
        staff_note = "You can run a leaner schedule"
    elif delta_pct <= -5:
        staff_note = "Keep core staff only and reduce prep waste"
    else:
        staff_note = "Your normal staffing level should be enough"

    staffing_parts = [
        f"Forecast is **{prediction:,} customers** vs your typical **{ref:,}** "
        f"({'+' if delta >= 0 else ''}{delta_pct}%) in **{region_label}** on **{day_name}**. "
        f"{staff_note}."
    ]
    if holiday:
        staffing_parts.append("It's a holiday — expect irregular timing and longer lines at peak.")
    if cubs and region == "DOWNTOWN":
        staffing_parts.append("Cubs home game day nearby — foot traffic may spill over before and after the game.")
    if bulls:
        staffing_parts.append("Bulls home game at United Center — evening demand may lift across the West Side.")
    if snow >= 1:
        staffing_parts.append(
            f"Snow forecast ({snow:.1f} in) — allow extra time for commutes and consider delivery backup."
        )
    elif precip >= 0.5:
        staffing_parts.append(
            f"Rain forecast ({precip:.2f} in) — walk-in traffic may dip; keep pickup/delivery ready."
        )

    inventory_parts = []
    if category == "Coffee Shop":
        if temp >= 78:
            inventory_parts.append(
                f"Warm day ({temp:.0f}°F) — increase cold brew, iced drinks, and chilled pastries by ~{max(10, abs(delta_pct))}%."
            )
        elif temp <= 35:
            inventory_parts.append(f"Cold day ({temp:.0f}°F) — prioritize hot coffee, tea, and warm baked goods.")
        else:
            inventory_parts.append(
                f"Mild weather ({temp:.0f}°F) — balance hot and cold beverage prep for ~{prediction:,} expected visits."
            )
    elif category == "Bakery":
        inventory_parts.append(
            f"Plan batches for **{prediction:,} customers** "
            f"({'above' if delta >= 0 else 'below'} your usual {ref:,}) — prioritize top sellers first."
        )
    elif category == "Ghost Kitchen":
        inventory_parts.append(
            f"Pre-portion for **{prediction:,} orders**; increase packaging and delivery containers if demand is up {max(0, delta_pct)}%."
        )
    else:
        inventory_parts.append(
            f"Stock core menu items for **{prediction:,} customers**, about {abs(delta_pct)}% "
            f"{'above' if delta >= 0 else 'below'} your regional average."
        )

    if festival or events >= 3:
        inventory_parts.append(
            f"City events active ({events} nearby) — add grab-and-go items and bottled drinks for pass-through traffic."
        )
    if weekend and category == "Coffee Shop":
        inventory_parts.append("Weekend brunch traffic — boost pastry and specialty drink inventory.")

    outlook_note = (
        f"{day_name} · {day_pattern}"
        f"{' · Holiday' if holiday else ''}"
        f"{' · Weekend' if weekend else ''}"
    )

    return {
        "staffing": " ".join(staffing_parts),
        "inventory": " ".join(inventory_parts),
        "demand_level": band,
        "day_pattern": day_pattern,
        "profile_note": outlook_note,
        "delta_pct": f"{delta_pct:+d}%",
        "vs_typical": f"{prediction:,} vs {ref:,} typical",
    }


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@lru_cache(maxsize=1)
def _background_data_url() -> str:
    for path in _BACKGROUND_CANDIDATES:
        if path.exists():
            return f"url('{_image_data_url(path)}')"
    return ""


_logo_cache: tuple[float, str, int, int] | None = None


def _logo_asset() -> tuple[str, int, int]:
    global _logo_cache

    if not _LOGO_PATH.exists():
        return "", 0, 0

    mtime = _LOGO_PATH.stat().st_mtime
    if _logo_cache and _logo_cache[0] == mtime:
        return _logo_cache[1], _logo_cache[2], _logo_cache[3]

    from PIL import Image

    with Image.open(_LOGO_PATH) as img:
        intrinsic_w, intrinsic_h = img.size

    data_url = _image_data_url(_LOGO_PATH)
    _logo_cache = (mtime, data_url, intrinsic_w, intrinsic_h)
    return data_url, intrinsic_w, intrinsic_h


def inject_theme_css() -> None:
    import streamlit as st

    bg_image = _background_data_url() or "none"
    css = (
        _THEME_CSS.replace("__BG_IMAGE__", bg_image)
        .replace("__LOGO_DISPLAY_W__", f"{_LOGO_DISPLAY_W}px")
        .replace("__LOGO_DISPLAY_H__", f"{_LOGO_DISPLAY_H}px")
        .replace("__LOGO_SLOT_W__", f"{_LOGO_SLOT_W}px")
    )
    st.markdown(css, unsafe_allow_html=True)


def _rich_text(text: str) -> str:
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def render_footer_marker_html() -> str:
    return '<div class="app-footer-start"></div>'


def render_footer_note_html(refresh_label: str) -> str:
    return f"""
    <div class="footer-note">
        Chicago city &amp; weather data
        <span class="footer-muted">&nbsp;&middot;&nbsp; {refresh_label}</span>
    </div>
    """


def render_header_html() -> str:
    logo_src, logo_w, logo_h = _logo_asset()
    logo_html = (
        f'<div class="app-hero-logo-wrap">'
        f'<img class="app-hero-logo" src="{logo_src}" alt="Berkane Nexus Insights logo" '
        f'width="{logo_w}" height="{logo_h}" decoding="async" />'
        f"</div>"
        if logo_src
        else '<div class="app-hero-logo-wrap"></div>'
    )
    return f"""
    <div class="app-hero">
        {logo_html}
        <div class="app-hero-text">
            <h1>Chicago Demand Insights</h1>
            <p>Data-driven demand forecasting for Chicago food &amp; beverage businesses.</p>
        </div>
        <div class="app-hero-balance" aria-hidden="true"></div>
    </div>
    """


def render_forecast_hero(
    prediction: int,
    gauge: dict,
    *,
    region_label: str,
    category_label: str,
    date_label: str,
) -> str:
    style = band_style(gauge["band"])
    return f"""
    <div class="forecast-hero">
        <div class="forecast-hero-inner">
            <div class="forecast-main">
                <span class="demand-badge" style="background:{style['bg']}; color:{style['color']};">
                    {gauge['band']} demand
                </span>
                <div class="label">Predicted customers</div>
                <div class="number">{prediction:,}</div>
            </div>
            <div class="forecast-meta">
                <div class="context">
                    Typical: <strong>{gauge['reference_median']:,}</strong>
                    &nbsp;·&nbsp; Range {gauge['p10']:,}–{gauge['p90']:,}
                </div>
                <div class="chip-row">
                    <span class="chip">{region_label}</span>
                    <span class="chip">{category_label}</span>
                    <span class="chip">{date_label}</span>
                </div>
            </div>
        </div>
    </div>
    """


def render_welcome_html() -> str:
    return """
    <div class="welcome-panel">
        <div class="focus-label">Restaurant / Food Business Focus</div>
        <h2>Your daily demand briefing</h2>
        <p>
            Pick a date, location, and business type above, then generate a forecast.
            You'll get a clear customer estimate and specific staffing and inventory guidance.
        </p>
    </div>
    """


def render_loading_html() -> str:
    return """
    <div class="loading-panel">
        <div class="loading-icon">autorenew</div>
        <h3>Building your forecast</h3>
        <p>Pulling weather, events, and demand signals for your selection…</p>
    </div>
    """


def render_value_props_html() -> str:
    return """
    <div class="value-props-shell">
        <div class="value-props-grid">
            <div class="value-prop">
                <div class="value-prop-head">
                    <span class="value-prop-icon">insights</span>
                    <span class="step">01</span>
                </div>
                <h3>Demand forecast</h3>
                <p>See expected customer traffic before the day begins.</p>
            </div>
            <div class="value-prop">
                <div class="value-prop-head">
                    <span class="value-prop-icon">groups</span>
                    <span class="step">02</span>
                </div>
                <h3>Smart staffing</h3>
                <p>Match staffing levels to expected demand and avoid over- or under-scheduling.</p>
            </div>
            <div class="value-prop">
                <div class="value-prop-head">
                    <span class="value-prop-icon">inventory_2</span>
                    <span class="step">03</span>
                </div>
                <h3>Inventory planning</h3>
                <p>Stock the right ingredients for the days that matter most.</p>
            </div>
        </div>
    </div>
    """

