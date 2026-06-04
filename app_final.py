#!/usr/bin/env python3
"""
Streamlit PoC app for V16 AF recurrence risk inference.

Run:
    streamlit run streamlit_app_v16_poc.py

Inputs:
    - clinical variables entered manually: type_af and redo
    - patient signal data from one of:
        1. local patient folder path containing the original .mat file
        2. uploaded .zip containing the patient folder
        3. uploaded .mat file
        4. uploaded CSV/XLSX containing one row of already extracted V16 features

This is a research/proof-of-concept interface only. It is not a clinical
decision-support device.
"""

from __future__ import annotations

import base64
import html
import io
import json
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = SCRIPT_DIR / "app_artifacts" / "v16_poc_model_artifact.joblib"
CONFUSION_METRICS_PATH = (
    SCRIPT_DIR
    / "results_comp_modelos"
    / "final_options"
    / "v16_confusion_matrix_metrics_summary.csv"
)
LOOCV_PREDICTIONS_PATH = (
    SCRIPT_DIR
    / "results_comp_modelos"
    / "final_options"
    / "v16_3_final_options_with_blanking_predictions_long.csv"
)
THRESHOLD_OPTIMIZATION_PATHS = [
    SCRIPT_DIR / "results" / "treshold_optimization" / "v16_threshold_optimization_selected_thresholds.csv",
    SCRIPT_DIR.parent / "results" / "treshold_optimization" / "v16_threshold_optimization_selected_thresholds.csv",
]
CORNER_ANIMATION_PATH = SCRIPT_DIR / "app_artifacts" / "corner_loop_transparent.webp"
CORNER_VIDEO_PATH = SCRIPT_DIR / "app_artifacts" / "corner_loop_seamless.mp4"
FALLBACK_CORNER_VIDEO_PATH = Path("/Users/ricardo/Downloads/O meu filme-enhanced.mp4")
HEART_LOADING_PATH = SCRIPT_DIR / "app_artifacts" / "Heart Loading.json"
FALLBACK_HEART_LOADING_PATH = Path("/Users/ricardo/Downloads/Heart Loading.json")

MODEL_VERSION = "V16 Final Power Mean Ensemble"
FINAL_MODEL_LABEL = "Power Mean Ensemble fixed k=3"
BASE_MODEL_ET = "extratrees"
BASE_MODEL_BNB = "bernoulli_tuned"
POWER_K = 3.0
FINAL_THRESHOLD = 0.41697552537145693
ECG_PREVIEW_LEAD_ORDER = ("II", "V1", "AVL", "AVF", "I")
FINAL_METRICS = {
    "LOOCV AUC": 0.7676767676767677,
    "Average Precision": 0.5867966833255265,
    "Brier": 0.18876702057782024,
    "Sensitivity": 0.6538461538461539,
    "Specificity": 0.7676767676767676,
    "F1": 0.6238532110091742,
}
FINAL_CONFUSION = {
    "threshold": FINAL_THRESHOLD,
    "n": 151,
    "tn": 76,
    "fp": 23,
    "fn": 18,
    "tp": 34,
}
MODEL_CHOICES = {
    "power_mean_ensemble": "Power Mean Ensemble fixed k=3",
    "extratrees": "ExtraTrees tuned",
    "bernoulli_tuned": "BernoulliNB tuned",
    "gnb_fixed": "GNB fixed",
}
MODEL_SHORT_LABELS = {
    "power_mean_ensemble": "PM Ensemble (recommended)",
    "extratrees": "ExtraTrees",
    "bernoulli_tuned": "BernoulliNB",
    "gnb_fixed": "GNB",
}
MODEL_HELP = {
    "power_mean_ensemble": "Final V16 model",
    "extratrees": "Base model",
    "bernoulli_tuned": "Base model",
    "gnb_fixed": "Baseline",
}
METRIC_CSV_NAMES = {
    "gnb_fixed": "GNB fixed",
    "bernoulli_tuned": "BernoulliNB tuned",
    "extratrees": "ExtraTrees tuned",
    "power_mean_ensemble": "Power Mean Ensemble fixed k=3",
}
LOOCV_RESULT_KEYS = {
    "gnb_fixed": ("gnb_fixed", 18.0),
    "bernoulli_tuned": ("bernoulli_tuned", 18.0),
    "extratrees": ("extratrees_tuned", 12.0),
    "power_mean_ensemble": ("pm_ensemble", 3.0),
}
THRESHOLD_POLICIES = {
    "max_all": {
        "strategy": "Max-all composite",
        "label": "Max-all",
        "help": "Balanced operational threshold from the threshold-optimization analysis.",
    },
    "more_false_positives": {
        "strategy": "Sensitivity >= 80%, max specificity",
        "label": "Less false negatives",
        "help": "More sensitive: flags more patients as recurrence risk to reduce missed recurrences, accepting more false positives.",
    },
    "more_false_negatives": {
        "strategy": "Specificity >= 90%, max sensitivity",
        "label": "Less false positives",
        "help": "More specific: flags fewer patients as recurrence risk to reduce unnecessary positive alerts, accepting more false negatives.",
    },
}


st.set_page_config(
    page_title="AF Recurrence Risk Predictor",
    page_icon=":material/monitor_heart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #111827;
          --muted: #6b7280;
          --line: #e5e7eb;
          --panel: #ffffff;
          --soft: #f9fafb;
          --blue: #2563eb;
          --blue-soft: #eff6ff;
          --green: #059669;
          --green-soft: #ecfdf5;
          --red: #dc2626;
          --red-soft: #fef2f2;
        }
        .stApp {
          background: #f3f4f6;
          color: var(--ink);
        }
        .main .block-container {
          padding-left: clamp(1rem, 4vw, 3rem);
          padding-right: clamp(1rem, 4vw, 3rem);
        }
        [data-testid="stSidebar"] {
          background: #ffffff;
          border-right: 1px solid var(--line);
          box-shadow: 2px 0 15px rgba(0,0,0,.03);
        }
        [data-testid="stSidebar"] * { color: var(--ink) !important; }
        [data-testid="stSidebar"] .stCaptionContainer,
        [data-testid="stSidebar"] small { color: var(--muted) !important; }
        [data-testid="stSidebar"] .model-chip {
          color: var(--blue) !important;
          background: var(--blue-soft) !important;
          border: 1px solid #bfdbfe;
        }
        header[data-testid="stHeader"] { display: none; }
        .block-container { padding-top: 2rem; max-width: 1360px; }
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,
        [data-testid="stAppViewContainer"] [role="radiogroup"] p {
          color: var(--ink) !important;
          font-weight: 700;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] [role="radiogroup"] p {
          color: var(--ink) !important;
        }
        h1, h2, h3 { letter-spacing: 0; color: var(--ink); }
        .app-header {
          display: flex;
          align-items: center;
          gap: .15rem;
          margin: .15rem 0 1.15rem;
        }
        .brand-mark {
          width: 118px;
          height: 72px;
          border-radius: 0;
          background: transparent;
          border: 0;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          box-shadow: none;
          overflow: visible;
        }
        .brand-heart {
          width: 118px;
          height: auto;
          object-fit: contain;
          filter: drop-shadow(0 8px 12px rgba(15,23,42,.16));
          animation: none;
          transform: none;
        }
        .brand-heart-fallback {
          color: #dc2626;
          font-size: 1.55rem;
          line-height: 1;
        }
        .app-title {
          font-size: 1.85rem;
          line-height: 1.08;
          font-weight: 850;
          margin: 0;
          color: var(--ink);
        }
        .app-subtitle {
          margin-top: .25rem;
          color: var(--muted);
          font-size: .94rem;
          font-weight: 600;
        }
        .corner-animation {
          position: fixed;
          top: 18px;
          right: 34px;
          width: 118px;
          height: auto;
          object-fit: contain;
          filter: drop-shadow(0 10px 16px rgba(15,23,42,.18));
          z-index: 1000;
          pointer-events: none;
        }
        @media (max-width: 1100px) {
          .corner-animation { display: none; }
        }
        .notice {
          border: 1px solid #bfdbfe;
          border-radius: 10px;
          background: var(--blue-soft);
          color: #1e40af;
          padding: .8rem .95rem;
          margin-bottom: 1.1rem;
          font-size: .84rem;
          font-weight: 600;
        }
        .model-panel {
          background: #ffffff;
          border: 1px solid var(--line);
          border-radius: 12px;
          box-shadow: 0 1px 3px rgba(15,23,42,.05);
          padding: 1.05rem;
        }
        div[data-testid="stExpander"] {
          border: 1px solid var(--line) !important;
          border-radius: 10px !important;
          overflow: hidden !important;
          background: #f3f4f6 !important;
        }
        div[data-testid="stExpander"] details summary {
          background: #f3f4f6 !important;
          color: var(--ink) !important;
          font-weight: 750 !important;
        }
        div[data-testid="stExpander"] details summary * {
          color: var(--ink) !important;
        }
        .st-emotion-cache-1r4qj8v,
        [data-testid="stVerticalBlockBorderWrapper"] {
          background: #ffffff !important;
          border: 1px solid var(--line) !important;
          border-radius: 12px !important;
          box-shadow: 0 1px 3px rgba(15,23,42,.05) !important;
          overflow: hidden;
        }
        [data-testid="stVerticalBlockBorderWrapper"] h3 {
          margin-top: 0;
          font-size: 1rem;
          padding-bottom: .45rem;
          border-bottom: 1px solid #f1f5f9;
        }
        .risk-card {
          border-radius: 12px;
          border: 1px solid var(--line);
          background: #ffffff;
          padding: 1.35rem 1.5rem;
          box-shadow: 0 1px 3px rgba(15,23,42,.06);
        }
        .risk-number {
          font-size: clamp(3rem, 8vw, 4.8rem);
          line-height: 1;
          font-weight: 900;
          letter-spacing: 0;
        }
        .risk-label { color: var(--muted); font-weight: 700; margin-top: .35rem; }
        .prob-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: .8rem;
          margin: 1rem 0;
        }
        .prob-card {
          border: 1px solid var(--line);
          border-radius: 10px;
          background: #ffffff;
          padding: .95rem;
          box-shadow: 0 1px 2px rgba(15,23,42,.04);
        }
        .prob-label { color: var(--muted); font-weight: 800; font-size: .72rem; text-transform: uppercase; }
        .prob-value { color: var(--ink); font-weight: 850; font-size: 1.32rem; margin-top: .2rem; }
        .risk-scale {
          position: relative;
          height: 22px;
          border-radius: 999px;
          background: linear-gradient(90deg, #34d399 0%, #a7f3d0 39.6%, #fed7aa 39.6%, #ef4444 100%);
          box-shadow: inset 0 0 0 1px rgba(15,23,42,.11);
          margin: 1rem 0 .55rem;
        }
        .risk-score-label {
          position: absolute;
          top: -31px;
          transform: translateX(-50%);
          background: #111827;
          color: #ffffff;
          border-radius: 999px;
          padding: .28rem .5rem;
          font-size: .68rem;
          font-weight: 850;
          white-space: nowrap;
          box-shadow: 0 8px 16px rgba(15,23,42,.16);
        }
        .risk-score-label::after {
          content: "";
          position: absolute;
          left: 50%;
          bottom: -4px;
          transform: translateX(-50%) rotate(45deg);
          width: 8px;
          height: 8px;
          background: #111827;
        }
        .risk-scale-marker {
          position: absolute;
          top: -7px;
          width: 3px;
          height: 36px;
          background: var(--ink);
          border-radius: 999px;
        }
        .risk-scale-threshold {
          position: absolute;
          top: -6px;
          width: 2px;
          height: 34px;
          background: #ffffff;
          border-radius: 999px;
          box-shadow: 0 0 0 1px rgba(15,23,42,.45);
        }
        .scale-labels {
          display: flex;
          justify-content: space-between;
          color: var(--muted);
          font-size: .72rem;
          font-weight: 800;
        }
        .ecg-panel {
          border: 1px solid var(--line);
          border-radius: 12px;
          background: #ffffff;
          padding: 1rem;
          box-shadow: 0 1px 3px rgba(15,23,42,.05);
          margin: 1rem 0;
        }
        .ecg-meta-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
          gap: .7rem;
          margin: .85rem 0 0;
        }
        .ecg-meta-card {
          background: #f9fafb;
          border: 1px solid var(--line);
          border-radius: 10px;
          padding: .78rem;
        }
        .ecg-meta-label {
          color: #6b7280;
          font-size: .66rem;
          font-weight: 900;
          text-transform: uppercase;
          letter-spacing: .04em;
        }
        .ecg-meta-value {
          color: var(--ink);
          font-size: 1rem;
          font-weight: 850;
          margin-top: .12rem;
          overflow-wrap: anywhere;
        }
        .model-line {
          border-left: 3px solid var(--blue);
          padding: .2rem 0 .2rem .75rem;
          margin: .85rem 0;
          color: var(--ink);
        }
        .sidebar-kicker {
          color: var(--blue);
          font-size: .72rem;
          font-weight: 900;
          letter-spacing: .08em;
          text-transform: uppercase;
          margin-bottom: .6rem;
        }
        .metric-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 1px;
          background: var(--line);
          border: 1px solid var(--line);
          border-radius: 10px;
          overflow: hidden;
          margin: .75rem 0 1rem;
        }
        .metric-tile {
          background: #ffffff;
          padding: .75rem;
        }
        .metric-tile.wide { grid-column: 1 / -1; }
        .metric-label {
          color: #9ca3af;
          font-size: .65rem;
          font-weight: 900;
          letter-spacing: .04em;
          text-transform: uppercase;
        }
        .metric-value {
          color: var(--ink);
          font-size: 1.35rem;
          font-weight: 850;
          margin-top: .12rem;
        }
        .metric-value.muted {
          color: #9ca3af;
          font-size: 1.05rem;
        }
        .confusion-grid {
          display: grid;
          grid-template-columns: 1fr 1fr 1fr;
          border: 1px solid var(--line);
          border-radius: 10px;
          overflow: hidden;
          font-size: .72rem;
          text-align: center;
        }
        .confusion-cell {
          padding: .55rem .35rem;
          background: #ffffff;
          border-right: 1px solid var(--line);
          border-bottom: 1px solid var(--line);
          font-weight: 700;
        }
        .confusion-head { background: #f9fafb; color: var(--muted); }
        .confusion-good { background: var(--blue-soft); color: #1e3a8a; }
        .confusion-bad { background: var(--red-soft); color: #991b1b; }
        .fusion-note {
          border: 1px solid var(--line);
          border-radius: 10px;
          background: #ffffff;
          padding: .85rem .95rem;
          color: #374151;
          font-size: .9rem;
          font-weight: 600;
        }
        .stTextInput input,
        [data-baseweb="select"] > div,
        [data-testid="stFileUploaderDropzone"] {
          background: #ffffff !important;
          border: 1px solid #d1d5db !important;
          color: var(--ink) !important;
          border-radius: 8px !important;
          caret-color: var(--ink) !important;
          box-shadow: 0 1px 2px rgba(15,23,42,.04) !important;
        }
        .stTextInput [data-baseweb="input"] {
          background: #ffffff !important;
          border: 1px solid #d1d5db !important;
          border-radius: 8px !important;
          box-shadow: 0 1px 2px rgba(15,23,42,.04) !important;
        }
        .stTextInput [data-baseweb="input"]:focus-within {
          border-color: var(--blue) !important;
          box-shadow: 0 0 0 2px rgba(37,99,235,.16) !important;
        }
        .stTextInput [data-baseweb="input"] input,
        .stTextInput input[aria-invalid="true"],
        .stTextInput [aria-invalid="true"] {
          border-color: transparent !important;
          box-shadow: none !important;
          outline: none !important;
        }
        .stTextInput input:focus {
          border-color: transparent !important;
          box-shadow: none !important;
        }
        .stTextInput input::placeholder {
          color: #9ca3af !important;
          opacity: 1 !important;
          font-weight: 500 !important;
        }
        ::selection {
          background: rgba(37,99,235,.18);
          color: var(--ink);
        }
        [data-baseweb="tab-list"] {
          gap: .35rem;
          border-bottom: 1px solid var(--line);
          margin-top: .75rem;
        }
        [data-baseweb="tab"] {
          background: #ffffff !important;
          color: #374151 !important;
          border: 1px solid var(--line) !important;
          border-bottom: 0 !important;
          border-radius: 8px 8px 0 0 !important;
          padding: .55rem .8rem !important;
          font-weight: 750 !important;
        }
        [data-baseweb="tab"] p,
        [data-baseweb="tab"] span {
          color: #374151 !important;
        }
        [data-baseweb="tab"][aria-selected="true"] {
          background: var(--blue) !important;
          color: #ffffff !important;
          border-color: var(--blue) !important;
        }
        [data-baseweb="tab"][aria-selected="true"] p,
        [data-baseweb="tab"][aria-selected="true"] span {
          color: #ffffff !important;
        }
        [data-baseweb="tab-highlight"] {
          background: transparent !important;
        }
        [data-testid="stMetric"] * {
          color: var(--ink) !important;
        }
        [data-testid="stMetricLabel"] * {
          color: #374151 !important;
        }
        [data-testid="stFileUploader"] button {
          background: var(--blue) !important;
          color: #ffffff !important;
          border: 1px solid var(--blue) !important;
          box-shadow: none !important;
        }
        [data-testid="stFileUploader"] button * {
          color: #ffffff !important;
        }
        [data-testid="stFileUploader"] button:hover {
          background: #1d4ed8 !important;
          color: #ffffff !important;
        }
        [data-testid="stFileUploaderDropzone"] p,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] span {
          color: #6b7280 !important;
        }
        [data-testid="stFileUploaderFile"] *,
        [data-testid="stFileUploaderFileName"],
        .uploadedFileName {
          color: var(--ink) !important;
        }
        [data-testid="stCheckbox"] label[data-baseweb="checkbox"] > div:first-child {
          width: 32px !important;
          height: 18px !important;
          min-width: 32px !important;
          border-radius: 999px !important;
          background: #cbd5e1 !important;
          border: 1px solid #94a3b8 !important;
          box-shadow: inset 0 0 0 1px rgba(15,23,42,.06) !important;
          display: flex !important;
          align-items: center !important;
          padding: 1px !important;
        }
        [data-testid="stCheckbox"] label[data-baseweb="checkbox"] > div:first-child > div {
          width: 14px !important;
          height: 14px !important;
          border-radius: 999px !important;
          background: #ffffff !important;
          transform: translateX(0) !important;
          box-shadow: 0 1px 2px rgba(15,23,42,.22) !important;
        }
        [data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input[aria-checked="true"]) > div:first-child {
          background: #ff4b4b !important;
          border-color: #ff4b4b !important;
        }
        [data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input[aria-checked="true"]) > div:first-child > div {
          transform: translateX(14px) !important;
        }
        .light-table-wrap {
          overflow: auto;
          max-height: 430px;
          border: 1px solid var(--line);
          border-radius: 10px;
          background: #ffffff;
          margin: .5rem 0 1rem;
        }
        .light-table th {
          position: sticky;
          top: 0;
          z-index: 1;
        }
        .light-table {
          width: 100%;
          border-collapse: collapse;
          font-size: .82rem;
          color: var(--ink);
          background: #ffffff;
        }
        .light-table th {
          background: #f9fafb;
          color: #6b7280;
          font-size: .7rem;
          text-transform: uppercase;
          letter-spacing: .04em;
          text-align: left;
          padding: .72rem .8rem;
          border-bottom: 1px solid var(--line);
        }
        .light-table td {
          padding: .68rem .8rem;
          border-bottom: 1px solid #f1f5f9;
          color: var(--ink);
          vertical-align: top;
          white-space: nowrap;
        }
        .impact-row {
          display: grid;
          grid-template-columns: minmax(170px, 1.2fr) minmax(90px, .55fr) minmax(240px, 1.4fr);
          gap: 1rem;
          align-items: center;
          border: 1px solid #e5e7eb;
          border-radius: 12px;
          background: #ffffff;
          padding: .9rem 1rem;
          margin: .65rem 0;
          color: var(--ink);
          box-shadow: 0 1px 2px rgba(15,23,42,.035);
        }
        .impact-bar {
          position: relative;
          height: 10px;
          background: #eef2f7;
          border-radius: 999px;
          overflow: hidden;
          box-shadow: inset 0 0 0 1px rgba(15,23,42,.06);
        }
        .impact-bar::after {
          content: "";
          position: absolute;
          left: 50%;
          top: -2px;
          bottom: -2px;
          width: 1px;
          background: #cbd5e1;
          z-index: 2;
        }
        .impact-fill {
          position: absolute;
          top: 0;
          bottom: 0;
          left: 50%;
          border-radius: 999px;
          z-index: 1;
        }
        .guidance-card {
          border: 1px solid var(--line);
          border-radius: 12px;
          padding: 1rem;
          background: #ffffff;
          margin-bottom: .85rem;
          box-shadow: 0 1px 3px rgba(15,23,42,.05);
        }
        .guidance-title {
          display: flex;
          align-items: center;
          gap: .45rem;
          font-weight: 850;
          margin-bottom: .35rem;
        }
        .guidance-list {
          margin: .55rem 0 0 0;
          padding: 0;
          list-style: none;
        }
        .guidance-list li {
          margin: .45rem 0;
          padding-left: 1.45rem;
          position: relative;
        }
        .guidance-list li::before {
          content: "✓";
          position: absolute;
          left: 0;
          color: var(--blue);
          font-weight: 900;
        }
        .guidance-card.warning {
          background: #fffbeb;
          border-color: #fcd34d;
        }
        .guidance-card.action {
          background: #eff6ff;
          border-color: #bfdbfe;
        }
        .explanation-grid {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: .8rem;
          margin: .8rem 0 1rem;
        }
        .explanation-card {
          border: 1px solid var(--line);
          border-radius: 12px;
          background: #ffffff;
          padding: .95rem 1rem;
          box-shadow: 0 1px 3px rgba(15,23,42,.045);
        }
        .explanation-label {
          color: var(--muted);
          font-size: .72rem;
          font-weight: 850;
          text-transform: uppercase;
          letter-spacing: .04em;
          margin-bottom: .28rem;
        }
        .explanation-value {
          color: var(--ink);
          font-size: 1.2rem;
          font-weight: 850;
          line-height: 1.2;
        }
        .explanation-body {
          color: #374151;
          font-size: .88rem;
          line-height: 1.5;
          margin-top: .35rem;
        }
        .report-preview {
          border: 1px solid var(--line);
          border-radius: 12px;
          background: #ffffff;
          padding: 1rem;
          color: var(--ink);
          box-shadow: 0 1px 3px rgba(15,23,42,.04);
        }
        @media (max-width: 900px) {
          .explanation-grid { grid-template-columns: 1fr; }
        }
        div[data-testid="stAlert"] {
          border-radius: 10px !important;
          color: #064e3b !important;
        }
        div[data-testid="stAlert"] * {
          color: inherit !important;
        }
        div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
          font-weight: 650 !important;
        }
        .json-box {
          background: #ffffff;
          border: 1px solid var(--line);
          border-radius: 12px;
          color: var(--ink);
          padding: 1rem;
          overflow-x: auto;
          font-size: .82rem;
          line-height: 1.55;
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
          white-space: nowrap;
          box-shadow: 0 1px 3px rgba(15,23,42,.04);
        }
        .json-box .json-key {
          color: #1d4ed8;
          font-weight: 800;
        }
        .json-box .json-string {
          color: #b45309;
        }
        .json-box .json-number {
          color: #047857;
          font-weight: 750;
        }
        [data-baseweb="popover"],
        [role="listbox"] {
          background: #ffffff !important;
          color: var(--ink) !important;
        }
        [role="option"],
        [data-baseweb="menu"] * {
          color: var(--ink) !important;
          background: #ffffff !important;
        }
        @media (max-width: 900px) {
          .prob-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 760px) {
          .app-header { align-items: flex-start; }
          .app-title { font-size: 1.45rem; max-width: calc(100vw - 7rem); }
          .app-subtitle { font-size: .82rem; }
          .brand-mark { width: 78px; height: 52px; }
          .brand-heart { width: 92px; }
          .prob-grid { grid-template-columns: 1fr; }
          .metric-grid { grid-template-columns: 1fr; }
          .metric-tile.wide { grid-column: auto; }
          .confusion-grid { font-size: .64rem; }
          .confusion-cell { padding: .45rem .25rem; }
          [data-testid="stHorizontalBlock"] { gap: .75rem !important; }
        }
        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {
          border-radius: 8px;
          border: 0;
          font-weight: 750;
          background: var(--blue);
          color: #ffffff;
          box-shadow: 0 2px 5px rgba(37,99,235,.18);
        }
        div[data-testid="stButton"] > button:hover,
        div[data-testid="stDownloadButton"] > button:hover {
          background: #1d4ed8;
          color: #ffffff;
        }
        .stAlert { border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_dark_mode_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ink: #f9fafb;
          --muted: #cbd5e1;
          --line: #374151;
          --panel: #111827;
          --soft: #1f2937;
          --blue-soft: #172554;
          --green-soft: #052e2b;
          --red-soft: #450a0a;
        }
        .stApp,
        [data-testid="stAppViewContainer"] {
          background: #111827 !important;
          color: #e5e7eb !important;
        }
        [data-testid="stSidebar"] {
          background: #111827 !important;
          border-right-color: #374151 !important;
          box-shadow: none !important;
        }
        [data-testid="stSidebar"] *,
        .stApp p,
        .stApp span,
        .stApp label,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp strong,
        [data-testid="stWidgetLabel"] p,
        [role="radiogroup"] p,
        [data-testid="stRadio"] * {
          color: #f9fafb !important;
        }
        .app-subtitle,
        .risk-label,
        .scale-labels,
        .metric-label,
        .stCaptionContainer,
        small {
          color: #cbd5e1 !important;
        }
        .notice,
        .risk-card,
        .prob-card,
        .fusion-note,
        [data-testid="stVerticalBlockBorderWrapper"],
        .guidance-card,
        .light-table-wrap,
        .json-box {
          background: #1f2937 !important;
          border-color: #374151 !important;
          color: #f9fafb !important;
        }
        div[data-testid="stExpander"],
        div[data-testid="stExpander"] details summary {
          background: #111827 !important;
          border-color: #374151 !important;
          color: #f9fafb !important;
        }
        .metric-grid { background: #374151 !important; border-color: #374151 !important; }
        .metric-tile,
        .light-table,
        .light-table td {
          background: #111827 !important;
          color: #f9fafb !important;
          border-color: #374151 !important;
        }
        .light-table th { background: #1f2937 !important; color: #cbd5e1 !important; }
        .impact-row { background: #111827 !important; border-color: #374151 !important; color: #f9fafb !important; }
        .impact-bar { background: #374151 !important; box-shadow: inset 0 0 0 1px rgba(255,255,255,.08) !important; }
        .impact-bar::after { background: #9ca3af !important; }
        [data-testid="stCheckbox"] label[data-baseweb="checkbox"] > div:first-child {
          background: #475569 !important;
          border-color: #64748b !important;
        }
        [data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input[aria-checked="true"]) > div:first-child {
          background: #ff4b4b !important;
          border-color: #ff4b4b !important;
        }
        .confusion-cell { background: #111827 !important; border-color: #374151 !important; color: #f9fafb !important; }
        .confusion-head { background: #1f2937 !important; color: #cbd5e1 !important; }
        .confusion-good { background: #172554 !important; color: #bfdbfe !important; }
        .confusion-bad { background: #450a0a !important; color: #fecaca !important; }
        .stTextInput input,
        .stTextInput [data-baseweb="input"],
        [data-baseweb="select"] > div,
        [data-testid="stFileUploaderDropzone"] {
          background: #111827 !important; color: #f9fafb !important; border-color: #4b5563 !important; caret-color: #f9fafb !important;
        }
        .stTextInput [data-baseweb="input"]:focus-within {
          border-color: #60a5fa !important;
          box-shadow: 0 0 0 2px rgba(96,165,250,.2) !important;
        }
        .stTextInput input::placeholder { color: #9ca3af !important; }
        [data-baseweb="select"] *,
        [data-baseweb="popover"],
        [role="listbox"],
        [role="option"],
        [data-baseweb="menu"] * {
          background: #111827 !important;
          color: #f9fafb !important;
        }
        [data-testid="stFileUploaderDropzone"] p,
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzone"] span,
        [data-testid="stFileUploaderFile"] *,
        [data-testid="stFileUploaderFileName"] {
          color: #e5e7eb !important;
        }
        [data-testid="stFileUploader"] button {
          background: #2563eb !important;
          color: #ffffff !important;
          border-color: #2563eb !important;
          opacity: 1 !important;
        }
        [data-testid="stFileUploader"] button *,
        [data-testid="stFileUploader"] button:disabled * {
          color: #ffffff !important;
        }
        [data-testid="stFileUploader"] button:disabled {
          background: #334155 !important;
          border-color: #475569 !important;
          color: #e5e7eb !important;
          opacity: 1 !important;
        }
        [data-testid="stMetric"] * { color: #f9fafb !important; }
        [data-testid="stMetricLabel"] * { color: #cbd5e1 !important; }
        .ecg-panel,
        .ecg-meta-card {
          background: #1f2937 !important;
          border-color: #374151 !important;
          color: #f9fafb !important;
        }
        .ecg-meta-label { color: #cbd5e1 !important; }
        .ecg-meta-value { color: #f9fafb !important; }
        [data-baseweb="tab"] { background: #1f2937 !important; color: #e5e7eb !important; border-color: #374151 !important; }
        [data-baseweb="tab"] p, [data-baseweb="tab"] span { color: #e5e7eb !important; }
        [data-baseweb="tab"][aria-selected="true"] { background: #2563eb !important; }
        [data-baseweb="tab"][aria-selected="true"] p,
        [data-baseweb="tab"][aria-selected="true"] span { color: #ffffff !important; }
        .guidance-card.warning { background: #422006 !important; border-color: #a16207 !important; }
        .guidance-card.action { background: #172554 !important; border-color: #1d4ed8 !important; }
        .explanation-card,
        .report-preview {
          background: #1f2937 !important;
          border-color: #374151 !important;
          color: #f9fafb !important;
        }
        .explanation-label,
        .explanation-body {
          color: #cbd5e1 !important;
        }
        .explanation-value {
          color: #f9fafb !important;
        }
        div[data-testid="stAlert"] {
          background: #064e3b !important;
          color: #d1fae5 !important;
          border-color: #047857 !important;
        }
        div[data-testid="stAlert"] * { color: #d1fae5 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Artifact not found: {ARTIFACT_PATH}. "
            "Run `python ml_v16_build_poc_artifact.py` first."
        )
    return joblib.load(ARTIFACT_PATH)


@st.cache_data(show_spinner=False)
def load_confusion_metrics() -> pd.DataFrame:
    if not CONFUSION_METRICS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CONFUSION_METRICS_PATH)


@st.cache_data(show_spinner=False)
def load_loocv_predictions() -> pd.DataFrame:
    if not LOOCV_PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(LOOCV_PREDICTIONS_PATH)


@st.cache_data(show_spinner=False)
def load_threshold_optimization() -> pd.DataFrame:
    for path in THRESHOLD_OPTIMIZATION_PATHS:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{(100 * float(value)) + 1e-9:.1f}%"


def metric_value(value: float, fmt: str = ".3f") -> str:
    if pd.isna(value):
        return "n/a"
    return format(float(value), fmt)


def probability_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    value = float(value)
    if 0 < value < 0.001:
        return "<0.1%"
    if 0.999 < value < 1:
        return ">99.9%"
    return f"{100 * value:.1f}%"


def loocv_rows_for_model(model_slug: str) -> pd.DataFrame:
    predictions = load_loocv_predictions()
    if predictions.empty:
        return pd.DataFrame()
    result_slug, result_k = LOOCV_RESULT_KEYS.get(model_slug, (model_slug, 18.0))
    rows = predictions[
        (predictions["model_slug"] == result_slug)
        & (predictions["clinical_scope"] == "preop")
        & (np.isclose(predictions["k"].astype(float), float(result_k)))
    ]
    return rows


def brier_for_loocv_model(model_slug: str) -> float:
    rows = loocv_rows_for_model(model_slug)
    if rows.empty:
        return np.nan
    return float(np.mean((rows["score"].astype(float) - rows["y_true"].astype(float)) ** 2))


def threshold_policy_caption(policy: str) -> str:
    if policy == "operational":
        policy = "max_all"
    if policy in THRESHOLD_POLICIES:
        return THRESHOLD_POLICIES[policy]["label"]
    if policy == "fixed_0_5":
        return "Fixed threshold 0.50"
    return str(policy)


def threshold_for_model(model_slug: str, policy: str) -> float:
    if policy == "fixed_0_5":
        return 0.5
    if policy == "operational":
        policy = "max_all"
    strategy = THRESHOLD_POLICIES.get(policy, THRESHOLD_POLICIES["max_all"])["strategy"]

    result_slug, _ = LOOCV_RESULT_KEYS.get(model_slug, (model_slug, 18.0))
    thresholds = load_threshold_optimization()
    if not thresholds.empty:
        rows = thresholds[
            (thresholds["model_slug"] == result_slug)
            & (thresholds["strategy"] == strategy)
        ]
        if not rows.empty:
            return float(rows.iloc[0]["threshold"])

    if model_slug == "power_mean_ensemble":
        return FINAL_THRESHOLD
    return 0.5


def apply_threshold_to_rows(rows: pd.DataFrame, threshold: float) -> dict:
    y_true = rows["y_true"].astype(int).to_numpy()
    scores = rows["score"].astype(float).to_numpy()
    pred = (scores >= threshold).astype(int)
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    tp = int(((y_true == 1) & (pred == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    accuracy = (tp + tn) / len(y_true) if len(y_true) else np.nan
    return {
        "threshold": threshold,
        "n": int(len(y_true)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
    }


def model_metrics(model_slug: str, artifact: dict, threshold_policy: str = "operational") -> dict:
    threshold = threshold_for_model(model_slug, threshold_policy)
    if model_slug == "power_mean_ensemble":
        metrics = {
            "available": True,
            "threshold": threshold,
            "threshold_policy": threshold_policy_caption(threshold_policy),
            "n": FINAL_CONFUSION["n"],
            "tn": FINAL_CONFUSION["tn"],
            "fp": FINAL_CONFUSION["fp"],
            "fn": FINAL_CONFUSION["fn"],
            "tp": FINAL_CONFUSION["tp"],
            "auc": FINAL_METRICS["LOOCV AUC"],
            "ap": FINAL_METRICS["Average Precision"],
            "brier": FINAL_METRICS["Brier"],
            "sensitivity": FINAL_METRICS["Sensitivity"],
            "specificity": FINAL_METRICS["Specificity"],
            "f1": FINAL_METRICS["F1"],
        }
        rows = loocv_rows_for_model(model_slug)
        if not rows.empty:
            metrics.update(apply_threshold_to_rows(rows, threshold))
        return metrics

    metrics_df = load_confusion_metrics()
    model_name = METRIC_CSV_NAMES.get(model_slug)
    if model_name and not metrics_df.empty:
        rows = metrics_df[metrics_df["model"] == model_name]
        if not rows.empty:
            row = rows.iloc[0]
            metrics = {
                "available": True,
                "threshold": threshold,
                "threshold_policy": threshold_policy_caption(threshold_policy),
                "n": int(row["n"]),
                "tn": int(row["tn"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "tp": int(row["tp"]),
                "auc": float(row["auc_loocv"]),
                "ap": float(row["average_precision"]) if "average_precision" in row.index else np.nan,
                "brier": float(row["brier_score"]) if "brier_score" in row.index else brier_for_loocv_model(model_slug),
                "sensitivity": float(row["sensitivity_recall_tpr"]),
                "specificity": float(row["specificity_tnr"]),
                "f1": float(row["f1"]),
            }
            loocv_rows = loocv_rows_for_model(model_slug)
            if not loocv_rows.empty:
                metrics.update(apply_threshold_to_rows(loocv_rows, threshold))
            return metrics

    return {
        "available": False,
        "threshold": threshold,
        "threshold_policy": threshold_policy_caption(threshold_policy),
        "n": artifact.get("n_training_patients", np.nan),
        "tn": np.nan,
        "fp": np.nan,
        "fn": np.nan,
        "tp": np.nan,
        "auc": np.nan,
        "ap": np.nan,
        "brier": np.nan,
        "sensitivity": np.nan,
        "specificity": np.nan,
        "f1": np.nan,
    }


@st.cache_data(show_spinner=False)
def load_corner_media_data_uri(media_path: str, mime_type: str, cache_bust: float = 0.0) -> str:
    path = Path(media_path)
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def media_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def render_corner_video() -> None:
    animation_uri = load_corner_media_data_uri(
        str(CORNER_ANIMATION_PATH),
        "image/webp",
        media_mtime(CORNER_ANIMATION_PATH),
    )
    if animation_uri:
        st.markdown(
            f'<img class="corner-animation" src="{animation_uri}" alt="" aria-hidden="true">',
            unsafe_allow_html=True,
        )
        return

    video_path = CORNER_VIDEO_PATH if CORNER_VIDEO_PATH.exists() else FALLBACK_CORNER_VIDEO_PATH
    video_uri = load_corner_media_data_uri(str(video_path), "video/mp4", media_mtime(video_path))
    if not video_uri:
        return
    st.markdown(
        f"""
        <video class="corner-animation" autoplay muted loop playsinline preload="auto">
          <source src="{video_uri}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True,
    )


def brand_heart_html() -> str:
    animation_uri = load_corner_media_data_uri(
        str(CORNER_ANIMATION_PATH),
        "image/webp",
        media_mtime(CORNER_ANIMATION_PATH),
    )
    if animation_uri:
        return f'<img class="brand-heart" src="{animation_uri}" alt="" aria-hidden="true">'
    return '<span class="brand-heart-fallback" aria-hidden="true">♥</span>'


@st.cache_data(show_spinner=False)
def load_lottie_json(animation_path: str) -> dict:
    path = Path(animation_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def heart_loader_animation() -> dict:
    animation_path = HEART_LOADING_PATH if HEART_LOADING_PATH.exists() else FALLBACK_HEART_LOADING_PATH
    return load_lottie_json(str(animation_path))


def render_heart_loader(message: str = "Loading patient data...") -> None:
    animation_data = heart_loader_animation()
    if not animation_data:
        st.info(message)
        return
    animation_json = json.dumps(animation_data)
    components.html(
        f"""
        <div style="display:flex;align-items:center;gap:16px;padding:8px 2px 18px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
          <div id="heart-loader" style="width:92px;height:92px;flex:0 0 auto;"></div>
          <div>
            <div style="font-weight:800;color:#111827;font-size:14px;">{message}</div>
            <div style="color:#6b7280;font-size:12px;margin-top:3px;">Extracting signal features and preparing inference.</div>
          </div>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js"></script>
        <script>
          const animationData = {animation_json};
          const container = document.getElementById("heart-loader");
          function fallback() {{
            container.innerHTML = '<div style="width:68px;height:68px;border-radius:999px;background:#fee2e2;color:#dc2626;display:flex;align-items:center;justify-content:center;font-size:38px;animation:pulse 1s ease-in-out infinite;">♥</div><style>@keyframes pulse{{0%,100%{{transform:scale(1)}}50%{{transform:scale(1.12)}}}}</style>';
          }}
          if (window.lottie) {{
            window.lottie.loadAnimation({{
              container,
              renderer: "svg",
              loop: true,
              autoplay: true,
              animationData
            }});
          }} else {{
            fallback();
          }}
        </script>
        """,
        height=118,
    )


def power_mean(p_extra_trees: float, p_bernoulli: float, k_power: float = POWER_K) -> float:
    return float(((p_extra_trees**k_power + p_bernoulli**k_power) / 2.0) ** (1.0 / k_power))


def classify_risk(probability: float, threshold: float) -> tuple[int, str, str]:
    if probability >= threshold:
        return 1, "Higher estimated recurrence risk", "#dc2626"
    return 0, "Lower estimated recurrence risk", "#0f766e"


def classify_final_risk(probability: float) -> tuple[int, str, str]:
    return classify_risk(probability, FINAL_THRESHOLD)


def render_probability_cards(p_et: float, p_bnb: float, p_pm: float, threshold: float, risk_band: str) -> None:
    st.markdown(
        f"""
        <div class="prob-grid">
          <div class="prob-card">
            <div class="prob-label">p_ExtraTrees</div>
            <div class="prob-value">{probability_pct(p_et)}</div>
          </div>
          <div class="prob-card">
            <div class="prob-label">p_BernoulliNB</div>
            <div class="prob-value">{probability_pct(p_bnb)}</div>
          </div>
          <div class="prob-card">
            <div class="prob-label">p_PowerMean Ensemble</div>
            <div class="prob-value">{probability_pct(p_pm)}</div>
          </div>
          <div class="prob-card">
            <div class="prob-label">Threshold</div>
            <div class="prob-value">{threshold:.3f}</div>
          </div>
          <div class="prob-card">
            <div class="prob-label">Risk category</div>
            <div class="prob-value" style="font-size:1rem;">{risk_band}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_single_probability_cards(label: str, probability: float, threshold: float, risk_band: str) -> None:
    st.markdown(
        f"""
        <div class="prob-grid">
          <div class="prob-card">
            <div class="prob-label">Selected model</div>
            <div class="prob-value" style="font-size:1rem;">{label}</div>
          </div>
          <div class="prob-card">
            <div class="prob-label">p_selected</div>
            <div class="prob-value">{probability_pct(probability)}</div>
          </div>
          <div class="prob-card">
            <div class="prob-label">Threshold</div>
            <div class="prob-value">{threshold:.3f}</div>
          </div>
          <div class="prob-card">
            <div class="prob-label">Risk category</div>
            <div class="prob-value" style="font-size:1rem;">{risk_band}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_scale(probability: float, threshold_value: float) -> None:
    marker = min(100, max(0, probability * 100))
    threshold = min(100, max(0, threshold_value * 100))
    label_left = min(94, max(6, marker))
    st.markdown(
        f"""
        <div class="risk-scale">
          <div class="risk-score-label" style="left:{label_left:.1f}%;">Patient score {probability:.3f}</div>
          <div class="risk-scale-threshold" style="left:{threshold:.1f}%;"></div>
          <div class="risk-scale-marker" style="left:{marker:.1f}%;"></div>
        </div>
        <div class="scale-labels">
          <span>0.00</span>
          <span>Threshold {threshold_value:.3f}</span>
          <span>1.00</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_panel(artifact: dict) -> tuple[str, str]:
    st.markdown(
        """
        <div class="sidebar-kicker">Model & configuration</div>
        <h3 style="margin:0 0 .5rem; font-size:1.1rem;">Inference model</h3>
        """,
        unsafe_allow_html=True,
    )
    selected_model = st.radio(
        "Model",
        options=list(MODEL_CHOICES.keys()),
        format_func=lambda slug: MODEL_SHORT_LABELS[slug],
        index=0,
        help="Choose which deployed model to use for this patient inference.",
    )
    threshold_policy = st.radio(
        "Operating threshold",
        options=list(THRESHOLD_POLICIES.keys()),
        format_func=threshold_policy_caption,
        index=0,
        help=(
            "Max-all is the default operational threshold. The other options intentionally move the cut-off "
            "toward fewer false negatives or fewer false positives for sensitivity/specificity trade-off review."
        ),
    )
    st.caption(THRESHOLD_POLICIES[threshold_policy]["help"])
    metrics = model_metrics(selected_model, artifact, threshold_policy)
    threshold = metrics["threshold"]
    label = MODEL_CHOICES[selected_model]
    role = MODEL_HELP[selected_model]

    st.markdown(
        f"""
        <div class="model-line">
          <strong>{label}</strong><br>
          {role}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Training cohort: {artifact['n_training_patients']} patients")
    if selected_model == "power_mean_ensemble":
        st.caption(f"{metrics['threshold_policy']}: {threshold:.3f}")
        st.caption("LOOCV metrics from the current V16 final-options outputs.")
    else:
        st.caption(f"{metrics['threshold_policy']}: {threshold:.3f}")

    st.markdown(
        f"""
        <div class="sidebar-kicker" style="margin-top:1.3rem;">LOOCV performance</div>
        <div class="metric-grid">
          <div class="metric-tile">
            <div class="metric-label">AUC</div>
            <div class="metric-value{' muted' if pd.isna(metrics['auc']) else ''}">{metric_value(metrics['auc'])}</div>
          </div>
          <div class="metric-tile">
            <div class="metric-label">AP</div>
            <div class="metric-value{' muted' if pd.isna(metrics['ap']) else ''}">{metric_value(metrics['ap'], '.3f')}</div>
          </div>
          <div class="metric-tile">
            <div class="metric-label">Brier</div>
            <div class="metric-value{' muted' if pd.isna(metrics['brier']) else ''}">{metric_value(metrics['brier'], '.4f')}</div>
          </div>
          <div class="metric-tile">
            <div class="metric-label">F1</div>
            <div class="metric-value{' muted' if pd.isna(metrics['f1']) else ''}">{pct(metrics['f1'])}</div>
          </div>
          <div class="metric-tile">
            <div class="metric-label">Sensitivity</div>
            <div class="metric-value{' muted' if pd.isna(metrics['sensitivity']) else ''}">{pct(metrics['sensitivity'])}</div>
          </div>
          <div class="metric-tile">
            <div class="metric-label">Specificity</div>
            <div class="metric-value{' muted' if pd.isna(metrics['specificity']) else ''}">{pct(metrics['specificity'])}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not metrics["available"]:
        st.caption("LOOCV confusion metrics unavailable for this deployment-only view.")

    with st.expander("Base models", expanded=False):
        st.write("ExtraTrees tuned")
        st.write("BernoulliNB")
        st.caption(f"The final score combines both probabilities using a Power Mean with k={POWER_K:g}.")

    with st.expander("Confusion matrix", expanded=False):
        if metrics["available"]:
            st.markdown(
                f"""
                <div class="confusion-grid">
                  <div class="confusion-cell confusion-head"></div>
                  <div class="confusion-cell confusion-head">Pred. no recurrence</div>
                  <div class="confusion-cell confusion-head">Pred. recurrence</div>
                  <div class="confusion-cell confusion-head">Actual no recurrence</div>
                  <div class="confusion-cell confusion-good">{metrics['tn']} TN</div>
                  <div class="confusion-cell confusion-bad">{metrics['fp']} FP</div>
                  <div class="confusion-cell confusion-head">Actual recurrence</div>
                  <div class="confusion-cell confusion-bad">{metrics['fn']} FN</div>
                  <div class="confusion-cell confusion-good">{metrics['tp']} TP</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.caption("No real LOOCV confusion matrix found for this model.")
    return selected_model, threshold_policy


def render_sidebar_model_metrics(model_slug: str) -> None:
    metric_names = {
        "bernoulli_tuned": "BernoulliNB tuned",
        "gnb_fixed": "GNB fixed",
        "extratrees": "ExtraTrees tuned",
        "power_mean_ensemble": "Power Mean Ensemble fixed k=3",
        "ensemble": "Power Mean Ensemble fixed k=3",
    }
    metrics_df = load_confusion_metrics()
    if metrics_df.empty:
        st.caption("Model performance summary unavailable.")
        return

    rows = metrics_df[metrics_df["model"] == metric_names.get(model_slug, "")]
    if rows.empty:
        st.caption("Deployment model. LOOCV summary pending for this view.")
        return

    row = rows.iloc[0]
    st.divider()
    st.subheader("LOOCV performance")
    st.caption(f"Threshold {row['threshold']:.2f} | n={int(row['n'])}")
    st.metric("AUC", f"{float(row['auc_loocv']):.3f}")

    c1, c2 = st.columns(2)
    c1.metric("Sensitivity", pct(row["sensitivity_recall_tpr"]))
    c2.metric("Specificity", pct(row["specificity_tnr"]))

    c3, c4 = st.columns(2)
    c3.metric("PPV", pct(row["precision_ppv"]))
    c4.metric("NPV", pct(row["npv"]))

    with st.expander("Confusion matrix details", expanded=False):
        st.write(
            {
                "TN": int(row["tn"]),
                "FP": int(row["fp"]),
                "FN": int(row["fn"]),
                "TP": int(row["tp"]),
                "accuracy": pct(row["accuracy"]),
                "balanced_accuracy": pct(row["balanced_accuracy"]),
                "F1": pct(row["f1"]),
            }
        )


def clinical_value(label: str, help_text: str) -> float:
    value = st.selectbox(
        label,
        options=["Missing/unknown", "0", "1"],
        help=help_text,
    )
    if value == "Missing/unknown":
        return np.nan
    return float(value)


def largest_mat_file(folder: str | Path) -> Path | None:
    folder = Path(folder)
    mats = [p for p in folder.rglob("*.mat") if p.is_file()]
    if not mats:
        return None
    return max(mats, key=lambda p: p.stat().st_size)


def is_supported_carto_mat(mat_path: str | Path) -> tuple[bool, str]:
    """Return whether a .mat file has the HDF5 Carto structure used by V16."""
    path = Path(mat_path)
    try:
        import h5py
    except Exception as exc:
        return False, f"h5py is not available: {exc}"

    try:
        if not h5py.is_hdf5(path):
            return False, "not an HDF5/Carto export"
        required = [
            "userdata/electric/egm",
            "userdata/electric/ecg",
            "userdata/electric/voltages/bipolar",
            "userdata/electric/voltages/unipolar",
            "userdata/electric/annotations/mapAnnot",
            "userdata/electric/ecgNames",
        ]
        with h5py.File(path, "r") as f:
            missing = [item for item in required if item not in f]
        if missing:
            return False, "missing required Carto fields: " + ", ".join(missing[:3])
        return True, ""
    except Exception as exc:
        return False, str(exc)


def best_supported_mat_file(folder: str | Path) -> tuple[Path | None, list[tuple[Path, str]]]:
    """Pick the largest supported Carto .mat and report skipped .mat files."""
    folder = Path(folder)
    mats = sorted([p for p in folder.rglob("*.mat") if p.is_file()], key=lambda p: p.stat().st_size, reverse=True)
    skipped: list[tuple[Path, str]] = []
    for mat in mats:
        ok, reason = is_supported_carto_mat(mat)
        if ok:
            return mat, skipped
        skipped.append((mat, reason))
    return None, skipped


def unsupported_mat_message(skipped: list[tuple[Path, str]]) -> str:
    if not skipped:
        return "No .mat file was found."
    examples = "; ".join(f"{path.name} ({reason})" for path, reason in skipped[:4])
    return (
        "Found .mat files, but none had the full HDF5 Carto ECG/EGM structure "
        f"required by the V16 extractor. Examples skipped: {examples}. "
        "Upload the complete patient folder/ZIP containing the *_1-Map.mat or *_AE.mat export, "
        "or upload a CSV/XLSX table with already extracted V16 features."
    )


def clear_signal_preview_state() -> None:
    st.session_state.pop("signal_mat_path", None)
    st.session_state.pop("signal_mat_bytes", None)
    st.session_state.pop("signal_mat_name", None)


def remember_signal_mat_path(mat_path: str | Path) -> None:
    mat = Path(mat_path)
    st.session_state["signal_mat_path"] = str(mat)
    st.session_state["signal_mat_name"] = mat.name
    st.session_state.pop("signal_mat_bytes", None)


def remember_signal_mat_bytes(mat_path: str | Path, display_name: str | None = None) -> None:
    mat = Path(mat_path)
    st.session_state["signal_mat_bytes"] = mat.read_bytes()
    st.session_state["signal_mat_name"] = display_name or mat.name
    st.session_state.pop("signal_mat_path", None)


def _decode_h5_text(value: object) -> str:
    arr = np.asarray(value)
    if arr.dtype.kind in "US":
        return "".join(str(item) for item in arr.ravel()).strip()
    try:
        chars = [chr(int(item)) for item in arr.ravel() if int(item) > 0]
        return "".join(chars).strip()
    except Exception:
        return str(value).strip()


def _read_carto_ecg_names(handle) -> list[str]:
    try:
        refs = np.asarray(handle["userdata/electric/ecgNames"]).ravel()
    except Exception:
        return []

    names: list[str] = []
    for ref in refs:
        try:
            if ref:
                names.append(_decode_h5_text(handle[ref][()]))
            else:
                names.append("")
        except Exception:
            names.append("")
    return names


def _lead_key(name: str) -> str:
    return str(name).split("(")[0].strip().upper().replace(" ", "")


def _select_carto_lead(names: list[str], n_leads: int, requested_lead: str | None = None) -> tuple[int, str]:
    lead_keys = [_lead_key(name) for name in names]
    if requested_lead:
        requested_key = _lead_key(requested_lead)
        if requested_key in lead_keys:
            idx = lead_keys.index(requested_key)
            if idx < n_leads:
                return idx, names[idx] or f"lead {idx + 1}"

    for wanted in ("II", "V1", "I", "AVF", "AVL", "V2", "III"):
        if wanted in lead_keys:
            idx = lead_keys.index(wanted)
            if idx < n_leads:
                return idx, names[idx] or f"lead {idx + 1}"
    fallback = min(1, max(0, n_leads - 1))
    label = names[fallback] if fallback < len(names) and names[fallback] else f"lead {fallback + 1}"
    return fallback, label


def _estimate_pwave_windows(signal: np.ndarray, r_peaks: np.ndarray, fs: float) -> list[dict]:
    """Estimate visually plausible P-wave windows, guarded away from QRS onset."""
    try:
        import ml_v16_extract as extractor
    except Exception:
        return []

    y = np.asarray(signal, dtype=float).ravel()
    windows: list[dict] = []
    for r in np.asarray(r_peaks, dtype=int):
        search_start = int(r - 0.26 * fs)
        search_end = int(r - 0.12 * fs)
        if search_start < 0 or search_end <= search_start + int(0.04 * fs):
            continue

        region = y[search_start:search_end].astype(float)
        finite = np.isfinite(region)
        if finite.sum() < int(0.04 * fs) or np.nanstd(region) <= 1e-9:
            continue

        if not finite.all():
            idx = np.arange(region.size)
            region = np.interp(idx, idx[finite], region[finite])

        p_filt = extractor.safe_lowpass(region - np.nanmedian(region), 15, fs=fs, order=3)
        if np.nanstd(p_filt) <= 1e-9:
            continue

        peak_local = int(np.nanargmax(np.abs(p_filt)))
        peak_amp = float(abs(p_filt[peak_local]))
        if not np.isfinite(peak_amp) or peak_amp <= 1e-9:
            continue

        threshold = max(0.18 * peak_amp, 0.20 * float(np.nanstd(p_filt)))
        onset = peak_local
        while onset > 0 and abs(p_filt[onset]) > threshold:
            onset -= 1
        offset = peak_local
        while offset < len(p_filt) - 1 and abs(p_filt[offset]) > threshold:
            offset += 1

        min_width = int(0.04 * fs)
        max_width = int(0.13 * fs)
        if offset - onset < min_width:
            half = int(0.045 * fs)
            onset = max(0, peak_local - half)
            offset = min(len(p_filt) - 1, peak_local + half)
        if offset - onset > max_width:
            half = max_width // 2
            onset = max(0, peak_local - half)
            offset = min(len(p_filt) - 1, peak_local + half)

        start = max(search_start, int(search_start + onset) - int(0.035 * fs))
        end = min(search_end, int(search_start + offset) + int(0.045 * fs))
        peak = int(search_start + peak_local)
        if end > start and end <= r - int(0.10 * fs):
            windows.append({"start": start, "end": end, "peak": peak})
        if len(windows) >= 15:
            break
    return windows


def _preview_from_signal(
    signal: np.ndarray,
    fs: float,
    mat_name: str,
    source: str,
    channel: str,
    p_windows: list[dict] | None = None,
) -> dict | None:
    y = np.asarray(signal, dtype=float).ravel()
    if y.size < 220:
        return None

    finite = np.isfinite(y)
    if finite.sum() < 220:
        return None
    if not finite.all():
        x = np.arange(y.size)
        y = np.interp(x, x[finite], y[finite])

    y = y - np.nanmedian(y)
    scale = np.nanpercentile(np.abs(y), 95)
    if np.isfinite(scale) and scale > 1e-12:
        y = y / scale

    t_ms = np.arange(y.size) / fs * 1000.0
    return {
        "name": mat_name,
        "source": source,
        "channel": channel,
        "fs": float(fs),
        "t_ms": t_ms,
        "y": y,
        "samples": int(y.size),
        "duration_s": float(y.size / fs),
        "p_windows": p_windows or [],
    }


def _open_session_mat_source() -> tuple[str | None, str | None]:
    mat_path = st.session_state.get("signal_mat_path")
    mat_bytes = st.session_state.get("signal_mat_bytes")
    if mat_path:
        return str(mat_path), None
    if mat_bytes:
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            tmp.write(mat_bytes)
            return tmp.name, tmp.name
    return None, None


def available_carto_ecg_leads() -> list[str]:
    try:
        import h5py
    except Exception:
        return []

    source, temp_name = _open_session_mat_source()
    if source is None:
        return []
    try:
        if not h5py.is_hdf5(source):
            return []
        with h5py.File(source, "r") as handle:
            if "userdata/electric/ecg" not in handle:
                return []
            n_leads = int(handle["userdata/electric/ecg"].shape[0])
            names = _read_carto_ecg_names(handle)
        valid = [name for name in names[:n_leads] if name]
        if not valid:
            return []

        by_key = {_lead_key(name): name for name in valid}
        return [by_key[key] for key in ECG_PREVIEW_LEAD_ORDER if key in by_key]
    except Exception:
        return []
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def _build_carto_ecg_preview(
    mat_path: str | Path | None = None,
    mat_bytes: bytes | None = None,
    selected_lead: str | None = None,
) -> dict | None:
    try:
        import h5py
    except Exception:
        return None

    temp_name = None
    try:
        if mat_bytes is not None:
            with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
                tmp.write(mat_bytes)
                temp_name = tmp.name
            source = temp_name
            mat_name = st.session_state.get("signal_mat_name", "uploaded .mat")
        elif mat_path is not None:
            source = str(mat_path)
            mat_name = Path(mat_path).name
        else:
            return None

        if not h5py.is_hdf5(source):
            return None
        with h5py.File(source, "r") as handle:
            if "userdata/electric/ecg" not in handle:
                return None
            ecg = np.asarray(handle["userdata/electric/ecg"])
            if ecg.ndim not in {2, 3} or ecg.shape[0] < 1:
                return None
            names = _read_carto_ecg_names(handle)
            lead_idx, lead_label = _select_carto_lead(names, ecg.shape[0], selected_lead)
            ref_idx, _ = _select_carto_lead(names, ecg.shape[0], "II")

            if ecg.ndim == 3:
                signal = np.asarray(ecg[lead_idx, :, 0], dtype=float)
                ref_signal = np.asarray(ecg[ref_idx, :, 0], dtype=float)
                channel = f"{lead_label} / map point 1 (matches extractor)"
            else:
                signal = np.asarray(ecg[lead_idx, :], dtype=float)
                ref_signal = np.asarray(ecg[ref_idx, :], dtype=float)
                channel = lead_label

        p_windows: list[dict] = []
        try:
            import ml_v16_extract as extractor

            r_peaks = extractor.detect_r_peaks(ref_signal, fs=1000)
            p_windows = _estimate_pwave_windows(signal, r_peaks, 1000.0)
        except Exception:
            p_windows = []

        return _preview_from_signal(signal, 1000.0, mat_name, "userdata/electric/ecg", channel, p_windows)
    except Exception:
        return None
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass


def _read_mat_mapping(mat_path: str | Path | None = None, mat_bytes: bytes | None = None) -> dict:
    try:
        from scipy.io import loadmat

        if mat_bytes is not None:
            return loadmat(io.BytesIO(mat_bytes), squeeze_me=True, struct_as_record=False)
        if mat_path is not None:
            return loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        pass
    except Exception:
        pass

    try:
        import h5py
    except Exception:
        return {}

    data = {}
    temp_name = None
    try:
        if mat_bytes is not None:
            with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
                tmp.write(mat_bytes)
                temp_name = tmp.name
            source = temp_name
        elif mat_path is not None:
            source = str(mat_path)
        else:
            return {}

        with h5py.File(source, "r") as handle:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    try:
                        data[name] = np.asarray(obj)
                    except Exception:
                        pass

            handle.visititems(visit)
    except Exception:
        return {}
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
    return data


def _candidate_signal_vectors(mapping: dict) -> list[dict]:
    candidates = []
    preferred_tokens = ("pwave", "p_wave", "p-wave", "lead_ii", "leadii", "lead2", "ii", "ecg", "signal")
    for name, value in mapping.items():
        if str(name).startswith("__"):
            continue
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if arr.dtype.kind not in "biufc" or arr.size < 220:
            continue
        arr = np.squeeze(arr)
        vectors = []
        if arr.ndim == 1:
            vectors.append((arr.astype(float), "single"))
        elif arr.ndim == 2:
            if min(arr.shape) <= 32 and max(arr.shape) >= 220:
                if arr.shape[0] <= arr.shape[1]:
                    lead_idx = 1 if arr.shape[0] > 1 else 0
                    vectors.append((arr[lead_idx, :].astype(float), f"channel {lead_idx + 1}"))
                    vectors.append((arr[0, :].astype(float), "channel 1"))
                else:
                    lead_idx = 1 if arr.shape[1] > 1 else 0
                    vectors.append((arr[:, lead_idx].astype(float), f"channel {lead_idx + 1}"))
                    vectors.append((arr[:, 0].astype(float), "channel 1"))
            else:
                vectors.append((arr.astype(float).ravel(), "flattened"))
        elif arr.ndim == 3 and arr.shape[1] >= 220:
            lead_idx = 1 if arr.shape[0] > 1 else 0
            point_idx = 0
            vectors.append((arr[lead_idx, :, point_idx].astype(float), f"channel {lead_idx + 1}"))

        token_score = sum(token in str(name).lower() for token in preferred_tokens)
        for vector, channel_label in vectors:
            vector = vector[np.isfinite(vector)]
            if vector.size < 220 or np.nanstd(vector) <= 1e-12:
                continue
            candidates.append(
                {
                    "name": str(name),
                    "channel": channel_label,
                    "signal": vector,
                    "score": token_score * 1000 + min(vector.size, 200000) / 1000,
                }
            )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _infer_sampling_rate(mapping: dict) -> float:
    for key, value in mapping.items():
        name = str(key).lower()
        if any(token in name for token in ["fs", "sampling", "sample_rate", "samplerate"]):
            try:
                arr = np.asarray(value).astype(float).ravel()
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    fs = float(arr[0])
                    if 50 <= fs <= 5000:
                        return fs
            except Exception:
                continue
    return 1000.0


def build_ecg_preview(selected_lead: str | None = None) -> dict | None:
    mat_path = st.session_state.get("signal_mat_path")
    mat_bytes = st.session_state.get("signal_mat_bytes")
    mat_name = st.session_state.get("signal_mat_name", "")
    if not mat_path and not mat_bytes:
        return None

    preview = _build_carto_ecg_preview(mat_path=mat_path, mat_bytes=mat_bytes, selected_lead=selected_lead)
    if preview is not None:
        return preview

    mapping = _read_mat_mapping(mat_path=mat_path, mat_bytes=mat_bytes)
    if not mapping:
        return None
    candidates = _candidate_signal_vectors(mapping)
    if not candidates:
        return None

    candidate = candidates[0]
    return _preview_from_signal(
        np.asarray(candidate["signal"], dtype=float),
        _infer_sampling_rate(mapping),
        mat_name or (Path(mat_path).name if mat_path else "uploaded .mat"),
        candidate["name"],
        candidate["channel"],
    )


def ecg_overlay_style(prediction: dict) -> dict:
    probability = float(prediction.get("selected_probability", np.nan))
    threshold = float(prediction.get("threshold", FINAL_THRESHOLD))
    if not np.isfinite(probability):
        return {"label": "unavailable", "color": "#22c55e", "soft": "rgba(34,197,94,.18)"}

    margin = probability - threshold
    if abs(margin) <= 0.05:
        return {"label": "borderline / near threshold", "color": "#f59e0b", "soft": "rgba(245,158,11,.23)"}
    if margin > 0:
        return {"label": "supports higher recurrence-risk output", "color": "#ef4444", "soft": "rgba(239,68,68,.24)"}
    return {"label": "supports lower recurrence-risk output", "color": "#22c55e", "soft": "rgba(34,197,94,.20)"}


def preview_lead_key(label: str) -> str:
    return _lead_key(str(label).split("/")[0])


def selected_pwave_features_for_lead(prediction: dict, lead_key: str) -> tuple[list[str], list[str]]:
    lead_features: list[str] = []
    cross_lead_features: list[str] = []
    for result in prediction.get("base_results", {}).values():
        for feature in result.get("selected_features", []):
            if feature.startswith(f"pw_{lead_key}_") and feature not in lead_features:
                lead_features.append(feature)
            if feature.startswith("xl_") and feature not in cross_lead_features:
                cross_lead_features.append(feature)
    return lead_features, cross_lead_features


def ecg_preview_figure(preview: dict, overlay: dict | None = None) -> go.Figure:
    t = np.asarray(preview["t_ms"], dtype=float) / 1000.0
    y = np.asarray(preview["y"], dtype=float)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=t,
                y=y,
                mode="lines",
                line=dict(color="#2563eb", width=2.4),
                name="ECG signal",
                hovertemplate="time=%{x:.3f} s<br>amplitude=%{y:.3f}<extra></extra>",
            )
        ]
    )
    overlay = overlay or {"color": "#22c55e", "soft": "rgba(34,197,94,.18)"}
    fs = float(preview["fs"])
    for window in preview.get("p_windows", []):
        fig.add_vrect(
            x0=float(window["start"]) / fs,
            x1=float(window["end"]) / fs,
            fillcolor=overlay["color"],
            opacity=0.18,
            layer="below",
            line_width=0,
        )
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        xaxis_title="Time from start of available strip (s)",
        yaxis_title="Normalized amplitude",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#94a3b8")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    return fig


def render_live_ecg_monitor(preview: dict, overlay: dict | None = None) -> None:
    y = np.asarray(preview["y"], dtype=float)
    if len(y) < 5:
        return

    max_points = 5000
    if len(y) > max_points:
        idx = np.linspace(0, len(y) - 1, max_points).astype(int)
        y = y[idx]
    y = y - np.nanmedian(y)
    spread = np.nanpercentile(np.abs(y), 98)
    if not np.isfinite(spread) or spread <= 1e-9:
        spread = 1.0
    y = np.clip(y / spread, -1.25, 1.25)

    width = 1000.0
    height = 230.0
    xs = np.linspace(0, width, len(y))
    ys = height * 0.52 - y * 62.0
    path = "M " + " L ".join(f"{x:.1f} {yy:.1f}" for x, yy in zip(xs, ys))
    safe_name = html.escape(str(preview["name"]))
    safe_source = html.escape(str(preview["source"]))
    safe_channel = html.escape(str(preview["channel"]))
    fs = float(preview["fs"])
    duration_s = float(preview.get("duration_s", len(preview["y"]) / fs))
    samples = int(preview.get("samples", len(preview["y"])))
    overlay = overlay or {"label": "P-wave windows", "color": "#22c55e", "soft": "rgba(34,197,94,.18)"}
    overlay_color = html.escape(overlay["color"])
    overlay_label = html.escape(overlay["label"])
    sample_count = max(1, samples - 1)
    zone_rects = []
    for window in preview.get("p_windows", []):
        x0 = max(0.0, min(width, float(window["start"]) / sample_count * width))
        x1 = max(0.0, min(width, float(window["end"]) / sample_count * width))
        if x1 <= x0:
            continue
        zone_rects.append(
            f'<rect class="pwave-zone" x="{x0:.1f}" y="8" width="{max(3.0, x1 - x0):.1f}" height="214" rx="7" ry="7"/>'
        )
    zone_svg = "\n".join(zone_rects)

    components.html(
        f"""
        <div class="ecg-monitor">
          <div class="monitor-top">
            <div>
              <div class="monitor-title">FULL ECG STRIP</div>
              <div class="monitor-subtitle">{safe_name} &middot; {safe_source} &middot; {safe_channel}</div>
            </div>
            <div class="monitor-readout">
              <span class="live-dot"></span>
              <span>{fs:.0f} Hz</span>
            </div>
          </div>
          <div class="monitor-screen">
            <svg viewBox="0 0 1000 230" preserveAspectRatio="none" role="img" aria-label="Animated ECG signal strip">
              <defs>
                <linearGradient id="ecgGlow" x1="0" x2="1">
                  <stop offset="0%" stop-color="#34d399" stop-opacity=".30"/>
                  <stop offset="45%" stop-color="#5eead4" stop-opacity="1"/>
                  <stop offset="100%" stop-color="#22c55e" stop-opacity=".85"/>
                </linearGradient>
                <filter id="softGlow" x="-20%" y="-80%" width="140%" height="260%">
                  <feGaussianBlur stdDeviation="3.5" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
                <filter id="pwaveGlow" x="-80%" y="-20%" width="260%" height="140%">
                  <feGaussianBlur stdDeviation="6" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
              </defs>
              {zone_svg}
              <path class="ecg-live-path" d="{path}" pathLength="1000" fill="none" stroke="url(#ecgGlow)" stroke-width="3.6" stroke-linecap="round" stroke-linejoin="round" filter="url(#softGlow)"/>
              <line class="ecg-cursor" x1="0" y1="8" x2="0" y2="222"/>
            </svg>
          </div>
          <div class="monitor-footer">
            <span>Estimated P-wave windows: {len(preview.get("p_windows", []))} · {overlay_label}</span>
            <span>{duration_s:.2f} s · {samples} samples</span>
            <span>Research visualization only</span>
          </div>
        </div>
        <style>
          .ecg-monitor {{
            width: 100%;
            box-sizing: border-box;
            border: 1px solid rgba(52,211,153,.30);
            border-radius: 18px;
            background: linear-gradient(180deg, #08111f 0%, #050b13 100%);
            padding: 16px;
            box-shadow: 0 18px 45px rgba(2,6,23,.22), inset 0 0 0 1px rgba(255,255,255,.04);
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }}
          .monitor-top, .monitor-footer {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 14px;
            color: #d1fae5;
          }}
          .monitor-title {{
            color: #6ee7b7;
            font-size: 12px;
            font-weight: 900;
            letter-spacing: .13em;
          }}
          .monitor-subtitle {{
            margin-top: 4px;
            color: #94a3b8;
            font-size: 12px;
            font-weight: 650;
            max-width: 760px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }}
          .monitor-readout {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border: 1px solid rgba(52,211,153,.28);
            border-radius: 999px;
            padding: 7px 11px;
            background: rgba(15,23,42,.82);
            color: #a7f3d0;
            font-size: 12px;
            font-weight: 850;
            white-space: nowrap;
          }}
          .live-dot {{
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #22c55e;
            box-shadow: 0 0 14px #22c55e;
            animation: pulseDot 1s ease-in-out infinite;
          }}
          .monitor-screen {{
            position: relative;
            height: 260px;
            margin: 14px 0 10px;
            border-radius: 14px;
            overflow: hidden;
            background:
              linear-gradient(rgba(16,185,129,.12) 1px, transparent 1px),
              linear-gradient(90deg, rgba(16,185,129,.12) 1px, transparent 1px),
              linear-gradient(rgba(16,185,129,.05) 1px, transparent 1px),
              linear-gradient(90deg, rgba(16,185,129,.05) 1px, transparent 1px),
              radial-gradient(circle at 25% 30%, rgba(16,185,129,.08), transparent 32%),
              #020617;
            background-size: 100% 52px, 52px 100%, 100% 13px, 13px 100%, 100% 100%, 100% 100%;
            box-shadow: inset 0 0 30px rgba(0,0,0,.55);
          }}
          .monitor-screen svg {{
            width: 100%;
            height: 100%;
            display: block;
          }}
          .ecg-live-path {{
            stroke-dasharray: 1000;
            stroke-dashoffset: 1000;
            animation: drawEcg 7.8s linear infinite;
          }}
          .pwave-zone {{
            fill: {overlay_color};
            opacity: .18;
            filter: url(#pwaveGlow);
            animation: pulsePwave 1.7s ease-in-out infinite;
          }}
          .ecg-cursor {{
            stroke: rgba(248,250,252,.82);
            stroke-width: 2.2;
            filter: drop-shadow(0 0 7px #ecfeff);
            animation: sweepCursor 7.8s linear infinite;
          }}
          .monitor-footer {{
            color: #64748b;
            font-size: 11px;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: .08em;
          }}
          @keyframes drawEcg {{
            0% {{ stroke-dashoffset: 1000; opacity: .25; }}
            4% {{ opacity: .95; }}
            92% {{ stroke-dashoffset: 0; opacity: 1; }}
            100% {{ stroke-dashoffset: 0; opacity: .35; }}
          }}
          @keyframes sweepCursor {{
            0% {{ transform: translateX(0); opacity: 0; }}
            4% {{ opacity: 1; }}
            92% {{ transform: translateX(1000px); opacity: 1; }}
            100% {{ transform: translateX(1000px); opacity: 0; }}
          }}
          @keyframes pulsePwave {{
            0%, 100% {{ opacity: .12; }}
            50% {{ opacity: .34; }}
          }}
          @keyframes pulseDot {{
            0%, 100% {{ opacity: .45; transform: scale(.86); }}
            50% {{ opacity: 1; transform: scale(1.18); }}
          }}
        </style>
        """,
        height=360,
    )


def render_ecg_signal_preview(prediction: dict) -> None:
    st.markdown("#### Real ECG preview")
    st.markdown(
        """
        <div class="ecg-panel">
          This preview reads the loaded <strong>.mat</strong>, renders the complete available ECG strip for the selected lead,
          and highlights a widened estimated P-wave/pre-QRS region before each detected QRS complex.
          The ECG trace uses map point 1 because the V16 extractor computes the ECG/P-wave features from that same Carto point.
        </div>
        """,
        unsafe_allow_html=True,
    )

    lead_options = available_carto_ecg_leads()
    selected_lead = None
    if lead_options:
        state_key = "ecg_preview_selected_lead"
        if st.session_state.get(state_key) not in lead_options:
            st.session_state[state_key] = lead_options[0]
        label_col, select_col = st.columns([0.68, 0.32], vertical_alignment="center")
        with label_col:
            st.caption("Choose the surface ECG lead to preview. Model inference still uses the full validated feature pipeline.")
        with select_col:
            selected_lead = st.selectbox(
                "ECG lead",
                options=lead_options,
                key=state_key,
                label_visibility="collapsed",
            )

    preview = build_ecg_preview(selected_lead)
    if preview is None:
        st.info("No raw ECG signal preview could be recovered from this input. Prediction still works from the extracted feature table.")
        return

    overlay = ecg_overlay_style(prediction)
    render_live_ecg_monitor(preview, overlay)

    lead_key = preview_lead_key(preview["channel"])
    lead_features, cross_lead_features = selected_pwave_features_for_lead(prediction, lead_key)
    if lead_features:
        st.caption(
            f"Selected model features linked to this lead ({lead_key}): "
            + ", ".join(lead_features[:6])
            + ("..." if len(lead_features) > 6 else "")
        )
    elif cross_lead_features:
        st.caption(
            f"No lead-specific selected feature for {lead_key}; cross-lead selected features may still include this lead: "
            + ", ".join(cross_lead_features[:4])
            + ("..." if len(cross_lead_features) > 4 else "")
        )
    else:
        st.caption(f"No selected final-model feature is specific to lead {lead_key}; the ECG strip is shown for visual inspection.")

    st.caption(
        "Overlay color is based on the patient-level model score, not on causal attribution of an individual beat. "
        "The highlighted windows are a widened visual estimate of the P-wave/pre-QRS region: "
        f"{overlay['label']}."
    )

    show_static = st.toggle("Show interactive full ECG plot", value=False)
    if show_static:
        st.plotly_chart(ecg_preview_figure(preview, overlay), use_container_width=True)
    st.markdown(
        f"""
        <div class="ecg-meta-grid">
          <div class="ecg-meta-card"><div class="ecg-meta-label">MAT file</div><div class="ecg-meta-value">{html.escape(preview['name'])}</div></div>
          <div class="ecg-meta-card"><div class="ecg-meta-label">Signal variable</div><div class="ecg-meta-value">{html.escape(preview['source'])}</div></div>
          <div class="ecg-meta-card"><div class="ecg-meta-label">Lead/channel</div><div class="ecg-meta-value">{html.escape(preview['channel'])}</div></div>
          <div class="ecg-meta-card"><div class="ecg-meta-label">Sampling rate</div><div class="ecg-meta-value">{preview['fs']:.0f} Hz</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def first_feature_table(folder: str | Path) -> Path | None:
    folder = Path(folder)
    candidates = []
    for pattern in ("*.csv", "*.xlsx", "*.xls"):
        candidates.extend([p for p in folder.rglob(pattern) if p.is_file()])
    if not candidates:
        return None
    preferred = [
        p
        for p in candidates
        if any(token in p.name.lower() for token in ["feature", "features", "v16", "patient"])
    ]
    return (preferred or candidates)[0]


def read_feature_table(path_or_file) -> pd.DataFrame:
    name = getattr(path_or_file, "name", str(path_or_file))
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path_or_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path_or_file)
    raise ValueError(f"Unsupported feature-table file type: {suffix}")


def extract_features_from_mat(mat_path: str | Path, patient_id: str) -> pd.DataFrame:
    try:
        import ml_v16_extract as extractor
    except Exception as exc:
        raise RuntimeError(
            "Could not import ml_v16_extract.py. Raw .mat extraction requires the V16 "
            f"extraction dependencies. Details: {exc}"
        ) from exc

    features = extractor.extract_patient_features_v5(str(mat_path), str(patient_id))
    return pd.DataFrame([features])


def prepare_uploaded_input(uploaded_file, patient_id: str) -> tuple[pd.DataFrame | None, str]:
    if uploaded_file is None:
        return None, ""

    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        dst = tmpdir / Path(uploaded_file.name).name
        dst.write_bytes(uploaded_file.getbuffer())

        if suffix == ".zip":
            extract_dir = tmpdir / "unzipped_patient"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dst, "r") as zf:
                zf.extractall(extract_dir)
            mat, skipped = best_supported_mat_file(extract_dir)
            if mat is not None:
                remember_signal_mat_bytes(mat)
                return extract_features_from_mat(mat, patient_id), f"ZIP -> largest .mat: {mat.name}"
            table = first_feature_table(extract_dir)
            if table is not None:
                clear_signal_preview_state()
                return read_feature_table(table), f"ZIP -> feature table: {table.name}"
            raise ValueError(unsupported_mat_message(skipped))

        if suffix == ".mat":
            ok, reason = is_supported_carto_mat(dst)
            if not ok:
                raise ValueError(
                    f"Unsupported .mat file '{uploaded_file.name}': {reason}. "
                    "Please upload the complete Carto HDF5 map file (*_1-Map.mat or *_AE.mat), "
                    "a ZIP containing that file, or a CSV/XLSX table with already extracted V16 features."
                )
            remember_signal_mat_bytes(dst, uploaded_file.name)
            return extract_features_from_mat(dst, patient_id), f"Uploaded .mat: {uploaded_file.name}"

        if suffix in {".csv", ".xlsx", ".xls"}:
            clear_signal_preview_state()
            return read_feature_table(dst), f"Uploaded feature table: {uploaded_file.name}"

    raise ValueError(f"Unsupported upload type: {suffix}")


def prepare_local_folder(folder_path: str, patient_id: str) -> tuple[pd.DataFrame | None, str]:
    if not folder_path.strip():
        return None, ""
    folder = Path(folder_path.strip().strip('"'))
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    mat, skipped = best_supported_mat_file(folder)
    if mat is not None:
        remember_signal_mat_path(mat)
        return extract_features_from_mat(mat, patient_id), f"Local folder -> largest .mat: {mat.name}"

    table = first_feature_table(folder)
    if table is not None:
        clear_signal_preview_state()
        return read_feature_table(table), f"Local folder -> feature table: {table.name}"

    raise ValueError(unsupported_mat_message(skipped))


def choose_patient_row(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= 1:
        return df.iloc[[0]].copy()

    st.warning("The uploaded table has more than one row. Select which row to use.")
    if "patient_id" in df.columns:
        labels = [f"{i}: patient_id={pid}" for i, pid in zip(df.index, df["patient_id"].astype(str))]
    else:
        labels = [str(i) for i in df.index]
    selected = st.selectbox("Patient row", labels)
    idx = int(selected.split(":")[0])
    return df.loc[[idx]].copy()


def build_signal_matrix_for_inference(feature_row: pd.DataFrame, candidate_cols: list[str]) -> pd.DataFrame:
    import ml_v16_train as v16

    feature_row = feature_row.copy()
    pwave_cols = [c for c in feature_row.columns if c.startswith("pw_") or c == "PTFV1"]
    tech_cols = [c for c in v16.TECHNICAL_COUNTERS if c in feature_row.columns]

    if pwave_cols or tech_cols:
        X_signal = v16._safe_numeric_frame(feature_row[pwave_cols + tech_cols].copy())
        X_signal = v16.add_ecg_domain_features(X_signal)
        X_signal = X_signal.drop(columns=v16.TECHNICAL_COUNTERS, errors="ignore")
        X_signal = X_signal[[c for c in X_signal.columns if not c.startswith("lit_")]]
    else:
        X_signal = pd.DataFrame(index=feature_row.index)

    # If a table already contains final candidate columns, keep them too.
    for col in candidate_cols:
        if col in feature_row.columns:
            X_signal[col] = pd.to_numeric(feature_row[col], errors="coerce")

    for col in candidate_cols:
        if col not in X_signal.columns:
            X_signal[col] = np.nan

    return X_signal[candidate_cols]


def build_clinical_matrix(model_artifact: dict, type_af: float, redo: float) -> pd.DataFrame:
    raw = {"type_af": type_af, "redo": redo}
    cols = model_artifact["clinical_columns"]

    if model_artifact["clinical_encoding"] == "legacy_value_plus_missing_indicator":
        out = {}
        for col in cols:
            value = raw.get(col, np.nan)
            out[col] = 0.0 if pd.isna(value) else float(value)
            out[f"{col}_missing"] = 1.0 if pd.isna(value) else 0.0
        X = pd.DataFrame([out])
    else:
        out = {}
        maps = model_artifact["clinical_maps"]
        for col in cols:
            value = raw.get(col, np.nan)
            if pd.isna(value):
                out[col] = 0.0
            else:
                out[col] = float(maps[col].get(float(value), 0.0))
        X = pd.DataFrame([out])

    for col in model_artifact["clinical_feature_columns"]:
        if col not in X.columns:
            X[col] = 0.0
    return X[model_artifact["clinical_feature_columns"]]


def predict_one(artifact: dict, model_slug: str, feature_row: pd.DataFrame, type_af: float, redo: float) -> dict:
    import ml_v16_train as v16

    model_artifact = artifact["models"][model_slug]
    if "ensemble_members" in model_artifact:
        member_results = [
            predict_one(artifact, member, feature_row, type_af, redo)
            for member in model_artifact["ensemble_members"]
            if member in artifact["models"]
        ]
        probs = np.array([r["probability"] for r in member_results], dtype=float)
        if len(probs) == 0:
            prob = np.nan
        elif "power_k" in model_artifact.get("params", {}):
            k_power = float(model_artifact["params"]["power_k"])
            prob = float((np.mean(probs**k_power)) ** (1.0 / k_power))
        else:
            prob = float(np.mean(probs))
        return {
            "model_slug": model_slug,
            "label": model_artifact["label"],
            "probability": prob,
            "params": model_artifact["params"],
            "clinical_encoding": model_artifact["clinical_encoding"],
            "selected_features": [],
            "selected_feature_values": {},
            "n_candidate_missing": max((r["n_candidate_missing"] for r in member_results), default=0),
            "n_candidate_features": max((r["n_candidate_features"] for r in member_results), default=0),
            "member_predictions": member_results,
        }

    X_signal = build_signal_matrix_for_inference(feature_row, artifact["candidate_signal_columns"])
    X_signal_pre = model_artifact["preprocessor"].transform(X_signal)

    selected = model_artifact["selected_features"]
    X_clin = build_clinical_matrix(model_artifact, type_af, redo)
    X_design = v16.make_design_matrix(X_signal_pre[selected], X_clin)
    X_scaled = model_artifact["scaler"].transform(X_design)
    prob = float(model_artifact["model"].predict_proba(X_scaled)[0, 1])

    selected_values = X_signal_pre[selected].iloc[0].to_dict()
    return {
        "model_slug": model_slug,
        "label": model_artifact["label"],
        "probability": prob,
        "params": model_artifact["params"],
        "clinical_encoding": model_artifact["clinical_encoding"],
        "selected_features": selected,
        "selected_feature_values": selected_values,
        "n_candidate_missing": int(X_signal.isna().sum(axis=1).iloc[0]),
        "n_candidate_features": int(X_signal.shape[1]),
        "member_predictions": [],
    }


def predict_final_ensemble(artifact: dict, feature_row: pd.DataFrame, type_af: float, redo: float) -> dict:
    et_result = predict_one(artifact, BASE_MODEL_ET, feature_row, type_af, redo)
    bnb_result = predict_one(artifact, BASE_MODEL_BNB, feature_row, type_af, redo)
    p_et = float(et_result["probability"])
    p_bnb = float(bnb_result["probability"])
    p_pm = power_mean(p_et, p_bnb, POWER_K)
    predicted_class, risk_band, color = classify_final_risk(p_pm)
    missing_features = int(max(et_result["n_candidate_missing"], bnb_result["n_candidate_missing"]))
    feature_count = int(max(et_result["n_candidate_features"], bnb_result["n_candidate_features"]))
    return {
        "patient_id": None,
        "label": FINAL_MODEL_LABEL,
        "p_ExtraTrees": p_et,
        "p_BernoulliNB": p_bnb,
        "p_PowerMean": p_pm,
        "k_power": POWER_K,
        "threshold": FINAL_THRESHOLD,
        "predicted_class": predicted_class,
        "risk_band": risk_band,
        "risk_color": color,
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "missing_features": missing_features,
        "n_candidate_features": feature_count,
        "base_results": {
            "ExtraTrees": et_result,
            "BernoulliNB": bnb_result,
        },
    }


def predict_selected_model(
    artifact: dict,
    model_slug: str,
    feature_row: pd.DataFrame,
    type_af: float,
    redo: float,
    threshold_policy: str = "operational",
) -> dict:
    metrics = model_metrics(model_slug, artifact, threshold_policy)
    threshold = float(metrics["threshold"])
    if model_slug == "power_mean_ensemble":
        prediction = predict_final_ensemble(artifact, feature_row, type_af, redo)
        prediction["selected_model"] = model_slug
        prediction["selected_model_label"] = MODEL_CHOICES[model_slug]
        prediction["selected_probability"] = prediction["p_PowerMean"]
        prediction["threshold"] = threshold
        prediction["threshold_policy"] = metrics["threshold_policy"]
        predicted_class, risk_band, color = classify_risk(prediction["p_PowerMean"], threshold)
        prediction["predicted_class"] = predicted_class
        prediction["risk_band"] = risk_band
        prediction["risk_color"] = color
        return prediction

    result = predict_one(artifact, model_slug, feature_row, type_af, redo)
    probability = float(result["probability"])
    predicted_class, risk_band, color = classify_risk(probability, threshold)
    return {
        "patient_id": None,
        "label": MODEL_CHOICES[model_slug],
        "selected_model": model_slug,
        "selected_model_label": MODEL_CHOICES[model_slug],
        "selected_probability": probability,
        "threshold": threshold,
        "threshold_policy": metrics["threshold_policy"],
        "predicted_class": predicted_class,
        "risk_band": risk_band,
        "risk_color": color,
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "missing_features": result["n_candidate_missing"],
        "n_candidate_features": result["n_candidate_features"],
        "base_results": {MODEL_CHOICES[model_slug]: result},
    }


def selected_feature_table(prediction: dict) -> pd.DataFrame:
    rows = []
    for model_name, result in prediction["base_results"].items():
        for feature in result["selected_features"]:
            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "value_after_preprocessing": result["selected_feature_values"].get(feature, np.nan),
                }
            )
    return pd.DataFrame(rows)


def export_payload(prediction: dict, patient_id: str) -> dict:
    payload = {
        "patient_id": patient_id,
        "selected_model": prediction["selected_model"],
        "selected_model_label": prediction["selected_model_label"],
        "p_selected": prediction["selected_probability"],
        "threshold": prediction["threshold"],
        "threshold_policy": prediction.get("threshold_policy", "Operational threshold (max-all)"),
        "predicted_class": prediction["predicted_class"],
        "risk_band": prediction["risk_band"],
        "model_version": prediction["model_version"],
        "timestamp": prediction["timestamp"],
        "missing_features": prediction["missing_features"],
    }
    if prediction["selected_model"] == "power_mean_ensemble":
        payload.update(
            {
                "p_ExtraTrees": prediction["p_ExtraTrees"],
                "p_BernoulliNB": prediction["p_BernoulliNB"],
                "p_PowerMean": prediction["p_PowerMean"],
                "k_power": prediction["k_power"],
            }
        )
    return payload


def render_light_table(df: pd.DataFrame, max_rows: int | None = None) -> None:
    shown = df.copy()
    if max_rows is not None:
        shown = shown.head(max_rows)
    html = shown.to_html(index=True, classes="light-table", border=0, escape=True)
    st.markdown(f'<div class="light-table-wrap">{html}</div>', unsafe_allow_html=True)


def format_feature_value(value: object) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.8g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def render_extracted_feature_list(feature_row: pd.DataFrame) -> None:
    if feature_row.empty:
        st.info("No extracted features available for this patient.")
        return

    row = feature_row.iloc[0]
    feature_list = pd.DataFrame(
        {
            "feature": row.index.astype(str),
            "value": [format_feature_value(value) for value in row.to_numpy()],
            "status": ["missing" if pd.isna(value) else "available" for value in row.to_numpy()],
        }
    )

    query = st.text_input("Search extracted features", value="", placeholder="Example: pw_V1, area, symmetry")
    if query.strip():
        mask = feature_list["feature"].str.contains(query.strip(), case=False, regex=False)
        shown = feature_list.loc[mask].reset_index(drop=True)
    else:
        shown = feature_list

    st.caption(f"Showing {len(shown)} of {len(feature_list)} extracted features.")
    st.dataframe(shown, hide_index=True, use_container_width=True, height=560)
    st.download_button(
        "Download extracted features as CSV",
        data=feature_list.to_csv(index=False).encode("utf-8"),
        file_name="extracted_features_list.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_json_box(payload: dict) -> None:
    pretty = json.dumps(payload, indent=2, ensure_ascii=False)
    pretty_html = html.escape(pretty).replace(" ", "&nbsp;").replace("\n", "<br>")
    st.markdown(f'<div class="json-box">{pretty_html}</div>', unsafe_allow_html=True)


def clinical_value_label(value: float) -> str:
    if pd.isna(value):
        return "missing"
    try:
        return str(int(value))
    except Exception:
        return str(value)


def score_percentile(model_slug: str, probability: float) -> float:
    rows = loocv_rows_for_model(model_slug)
    if rows.empty:
        return np.nan
    return float((rows["score"].astype(float) <= probability).mean() * 100.0)


def risk_position_text(percentile: float, probability: float, threshold: float) -> str:
    if pd.isna(percentile):
        cohort_text = "No LOOCV score distribution was available for this model."
    elif percentile >= 75:
        cohort_text = "The score sits in the upper part of the training-cohort LOOCV distribution."
    elif percentile <= 25:
        cohort_text = "The score sits in the lower part of the training-cohort LOOCV distribution."
    else:
        cohort_text = "The score sits in the middle range of the training-cohort LOOCV distribution."

    margin = probability - threshold
    if abs(margin) < 0.05:
        threshold_text = "It is close to the operating threshold, so the classification is borderline."
    elif margin >= 0:
        threshold_text = "It is above the operating threshold."
    else:
        threshold_text = "It is below the operating threshold."
    return f"{cohort_text} {threshold_text}"


def model_agreement_summary(prediction: dict) -> tuple[str, str, str]:
    if prediction["selected_model"] != "power_mean_ensemble":
        return (
            "Single model",
            "Not applicable",
            "This view uses one selected model, so there is no base-model agreement check.",
        )

    p_et = float(prediction["p_ExtraTrees"])
    p_bnb = float(prediction["p_BernoulliNB"])
    diff = abs(p_et - p_bnb)
    if diff < 0.10:
        label = "High agreement"
        body = "ExtraTrees and BernoulliNB give closely aligned probabilities."
    elif diff < 0.20:
        label = "Moderate agreement"
        body = "The base models are not identical, but they remain reasonably aligned."
    else:
        label = "Model disagreement"
        body = "The base models disagree meaningfully; interpret the final fused score with extra caution."
    return label, f"{diff:.3f}", body


def selected_feature_domain(feature: str) -> str:
    if feature.startswith("pw_"):
        return "P-wave lead feature"
    if feature.startswith("xl_"):
        return "Cross-lead P-wave feature"
    if feature in {"type_af", "redo"} or feature.endswith("_missing"):
        return "Clinical feature"
    return "Other signal feature"


def selected_feature_summary_table(prediction: dict) -> pd.DataFrame:
    selected_df = selected_feature_table(prediction)
    if selected_df.empty:
        return pd.DataFrame()
    summary = (
        selected_df.assign(domain=selected_df["feature"].map(selected_feature_domain))
        .groupby(["model", "domain"], as_index=False)
        .agg(n_features=("feature", "count"))
        .sort_values(["model", "n_features"], ascending=[True, False])
    )
    return summary


def render_score_distribution(prediction: dict) -> None:
    rows = loocv_rows_for_model(prediction["selected_model"])
    if rows.empty:
        st.caption("LOOCV score distribution is unavailable for this model.")
        return

    scores = rows["score"].astype(float)
    probability = float(prediction["selected_probability"])
    threshold = float(prediction["threshold"])
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=scores,
            nbinsx=18,
            marker_color="#93c5fd",
            marker_line_color="#ffffff",
            marker_line_width=1,
            opacity=0.9,
            name="LOOCV scores",
        )
    )
    fig.add_shape(
        type="line",
        x0=threshold,
        x1=threshold,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#6b7280", dash="dash", width=2),
    )
    fig.add_shape(
        type="line",
        x0=probability,
        x1=probability,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=prediction["risk_color"], width=3),
    )
    fig.add_annotation(
        x=threshold,
        y=1.03,
        yref="paper",
        text="threshold",
        showarrow=False,
        font=dict(color="#6b7280", size=12),
    )
    fig.add_annotation(
        x=probability,
        y=1.13,
        yref="paper",
        text="patient",
        showarrow=False,
        font=dict(color=prediction["risk_color"], size=12),
    )
    fig.update_layout(
        height=310,
        margin=dict(l=10, r=10, t=35, b=25),
        xaxis_title="Model score",
        yaxis_title="LOOCV patients",
        showlegend=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        bargap=0.04,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def clinical_impact_table(
    artifact: dict,
    model_slug: str,
    feature_row: pd.DataFrame,
    type_af: float,
    redo: float,
    base_probability: float,
    threshold_policy: str,
) -> pd.DataFrame:
    rows = []
    variables = [
        ("type_af", type_af, [0.0, 1.0, np.nan]),
        ("redo", redo, [0.0, 1.0, np.nan]),
    ]
    for variable, current, candidates in variables:
        for candidate in candidates:
            if (pd.isna(current) and pd.isna(candidate)) or (not pd.isna(current) and current == candidate):
                continue
            test_type_af = candidate if variable == "type_af" else type_af
            test_redo = candidate if variable == "redo" else redo
            try:
                prediction = predict_selected_model(
                    artifact,
                    model_slug,
                    feature_row,
                    test_type_af,
                    test_redo,
                    threshold_policy,
                )
                probability = prediction["selected_probability"]
                rows.append(
                    {
                        "Variable": variable,
                        "Scenario": "Missing/unknown" if pd.isna(candidate) else str(int(candidate)),
                        "Score": probability,
                        "Delta": probability - base_probability,
                    }
                )
            except Exception:
                continue
    return pd.DataFrame(rows).sort_values("Delta", key=lambda s: s.abs(), ascending=False)


def render_impact_analysis(
    artifact: dict,
    prediction: dict,
    feature_row: pd.DataFrame,
    type_af: float,
    redo: float,
) -> None:
    st.markdown(
        f"""
        <div class="fusion-note">
          <strong>What this shows:</strong> this is a local sensitivity check, not a causal explanation. The app keeps the ECG-derived features fixed and recalculates the selected model after changing only <strong>type_af</strong> or <strong>redo</strong>. Positive deltas increase the model score; negative deltas reduce it. Current score: <strong>{probability_pct(prediction['selected_probability'])}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    percentile = score_percentile(prediction["selected_model"], prediction["selected_probability"])
    if pd.isna(percentile):
        st.caption("Training-cohort percentile is unavailable for this model because no LOOCV score distribution was found.")
    else:
        st.markdown(
            f"""
            <div class="fusion-note">
              This patient is at percentile <strong>{percentile:.0f}</strong> of the training LOOCV score distribution for the selected model.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Clinical variable sensitivity")
    impacts = clinical_impact_table(
        artifact,
        prediction["selected_model"],
        feature_row,
        type_af,
        redo,
        prediction["selected_probability"],
        prediction.get("threshold_policy", "operational"),
    )
    if impacts.empty:
        st.caption("No counterfactual clinical-variable impacts could be computed.")
        return

    st.caption("Bars are centered at zero change versus the current patient settings. Green means the alternative setting lowers the selected model score; red means it raises it.")
    max_abs = max(0.01, float(impacts["Delta"].abs().max()))
    for _, row in impacts.iterrows():
        delta = float(row["Delta"])
        width = min(50, abs(delta) / max_abs * 50)
        if delta >= 0:
            left = 50
            color = "linear-gradient(90deg, #fecdd3, #fb7185)"
            label_color = "#e11d48"
            signed = f"+{100 * delta:.1f}%"
        else:
            left = 50 - width
            color = "linear-gradient(90deg, #34d399, #a7f3d0)"
            label_color = "#059669"
            signed = f"{100 * delta:.1f}%"
        st.markdown(
            f"""
            <div class="impact-row">
              <div><strong>{row['Variable']}</strong><br><span style="color:#6b7280;font-size:.78rem;">alternative: {row['Scenario']}</span></div>
              <div><span style="color:#6b7280;font-size:.72rem;font-weight:800;text-transform:uppercase;">score</span><br>{probability_pct(row['Score'])}</div>
              <div style="display:flex;align-items:center;gap:.6rem;">
                <div class="impact-bar" style="flex:1;">
                  <div class="impact-fill" style="left:{left:.1f}%;width:{width:.1f}%;background:{color};"></div>
                </div>
                <strong style="color:{label_color};min-width:54px;">{signed}</strong>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_risk_explanation(
    artifact: dict,
    prediction: dict,
    feature_row: pd.DataFrame,
    type_af: float,
    redo: float,
) -> None:
    probability = float(prediction["selected_probability"])
    threshold = float(prediction["threshold"])
    percentile = score_percentile(prediction["selected_model"], probability)
    agreement_label, agreement_value, agreement_body = model_agreement_summary(prediction)
    margin = probability - threshold
    percentile_value = "n/a" if pd.isna(percentile) else f"{percentile:.0f}th"

    st.markdown(
        f"""
        <div class="explanation-grid">
          <div class="explanation-card">
            <div class="explanation-label">Cohort position</div>
            <div class="explanation-value">{percentile_value}</div>
            <div class="explanation-body">{html.escape(risk_position_text(percentile, probability, threshold))}</div>
          </div>
          <div class="explanation-card">
            <div class="explanation-label">Threshold margin</div>
            <div class="explanation-value">{margin:+.3f}</div>
            <div class="explanation-body">Positive values are above the operating point; negative values are below it.</div>
          </div>
          <div class="explanation-card">
            <div class="explanation-label">Model agreement</div>
            <div class="explanation-value">{html.escape(agreement_label)}</div>
            <div class="explanation-body">Difference: {html.escape(agreement_value)}. {html.escape(agreement_body)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Score in the V16 LOOCV distribution")
    render_score_distribution(prediction)

    if prediction["selected_model"] == "power_mean_ensemble":
        st.markdown("#### Base-model probabilities")
        render_light_table(
            pd.DataFrame(
                [
                    {"model": "ExtraTrees tuned", "probability": probability_pct(prediction["p_ExtraTrees"])},
                    {"model": "BernoulliNB tuned", "probability": probability_pct(prediction["p_BernoulliNB"])},
                    {"model": f"Power Mean k={POWER_K:g}", "probability": probability_pct(prediction["p_PowerMean"])},
                ]
            )
        )

    st.markdown("#### Selected feature groups")
    feature_summary = selected_feature_summary_table(prediction)
    if feature_summary.empty:
        st.caption("The selected model does not expose a fold-final selected-feature list.")
    else:
        render_light_table(feature_summary)

    st.markdown("#### Clinical-variable sensitivity")
    st.caption(
        "Local check only: ECG-derived features remain fixed and the app recalculates the score after changing only type_af or redo."
    )
    impacts = clinical_impact_table(
        artifact,
        prediction["selected_model"],
        feature_row,
        type_af,
        redo,
        probability,
        prediction.get("threshold_policy", "operational"),
    )
    if impacts.empty:
        st.caption("No counterfactual clinical-variable impacts could be computed.")
    else:
        impacts_shown = impacts.copy()
        impacts_shown["Score"] = impacts_shown["Score"].map(probability_pct)
        impacts_shown["Delta"] = impacts_shown["Delta"].map(lambda x: f"{100 * float(x):+.1f}%")
        render_light_table(impacts_shown)

    st.markdown(
        """
        <div class="fusion-note">
          This explanation is descriptive and local. It should be read as model transparency for a research PoC, not as causal attribution of recurrence to an individual ECG feature.
        </div>
        """,
        unsafe_allow_html=True,
    )


def dataframe_html(df: pd.DataFrame, max_rows: int | None = None, index: bool = False) -> str:
    if df is None or df.empty:
        return "<p>Not available.</p>"
    shown = df.copy()
    if max_rows is not None:
        shown = shown.head(max_rows)
    return shown.to_html(index=index, border=0, escape=True)


def report_metric_rows(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "AUC", "value": metric_value(metrics.get("auc", np.nan))},
            {"metric": "Average precision", "value": metric_value(metrics.get("ap", np.nan))},
            {"metric": "Brier score", "value": metric_value(metrics.get("brier", np.nan), ".4f")},
            {"metric": "Sensitivity", "value": pct(metrics.get("sensitivity", np.nan))},
            {"metric": "Specificity", "value": pct(metrics.get("specificity", np.nan))},
            {"metric": "F1", "value": pct(metrics.get("f1", np.nan))},
        ]
    )


def build_clinical_report_html(
    artifact: dict,
    prediction: dict,
    patient_id: str,
    source_note: str,
    type_af: float,
    redo: float,
    feature_row: pd.DataFrame,
    selected_df: pd.DataFrame,
    validation_metrics: dict,
) -> str:
    probability = float(prediction["selected_probability"])
    threshold = float(prediction["threshold"])
    percentile = score_percentile(prediction["selected_model"], probability)
    agreement_label, agreement_value, agreement_body = model_agreement_summary(prediction)
    feature_summary = selected_feature_summary_table(prediction)
    impacts = clinical_impact_table(
        artifact,
        prediction["selected_model"],
        feature_row,
        type_af,
        redo,
        probability,
        prediction.get("threshold_policy", "operational"),
    )
    if not impacts.empty:
        impacts = impacts.copy()
        impacts["Score"] = impacts["Score"].map(probability_pct)
        impacts["Delta"] = impacts["Delta"].map(lambda x: f"{100 * float(x):+.1f}%")

    confusion_df = pd.DataFrame(
        [
            [validation_metrics.get("tn", np.nan), validation_metrics.get("fp", np.nan)],
            [validation_metrics.get("fn", np.nan), validation_metrics.get("tp", np.nan)],
        ],
        index=["Actual no recurrence", "Actual recurrence"],
        columns=["Predicted no recurrence", "Predicted recurrence"],
    )
    base_rows = []
    if prediction["selected_model"] == "power_mean_ensemble":
        base_rows = [
            {"model": "ExtraTrees tuned", "probability": probability_pct(prediction["p_ExtraTrees"])},
            {"model": "BernoulliNB tuned", "probability": probability_pct(prediction["p_BernoulliNB"])},
            {"model": f"Power Mean k={POWER_K:g}", "probability": probability_pct(prediction["p_PowerMean"])},
        ]

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    percentile_text = "n/a" if pd.isna(percentile) else f"{percentile:.0f}th percentile"
    source_text = html.escape(str(source_note or "not specified"))
    notes = artifact.get("notes", [])
    notes_html = "".join(f"<li>{html.escape(str(note))}</li>" for note in notes)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>V16 AF recurrence risk report - {html.escape(str(patient_id))}</title>
<style>
body {{ font-family: Arial, sans-serif; color: #111827; margin: 32px; line-height: 1.45; }}
h1 {{ margin-bottom: 0; }}
h2 {{ margin-top: 28px; border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; }}
.muted {{ color: #6b7280; }}
.pill {{ display:inline-block; border-radius:999px; padding:6px 10px; background:#eff6ff; color:#1d4ed8; font-weight:700; }}
.risk {{ font-size: 34px; font-weight: 800; color: {html.escape(prediction["risk_color"])}; }}
.grid {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:12px; margin:16px 0; }}
.card {{ border:1px solid #e5e7eb; border-radius:10px; padding:14px; background:#ffffff; }}
.label {{ color:#6b7280; font-size:12px; text-transform:uppercase; font-weight:800; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0 18px; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #f9fafb; color: #374151; }}
.notice {{ background:#fffbeb; border:1px solid #fcd34d; border-radius:10px; padding:12px; }}
</style>
</head>
<body>
<div class="pill">Research PoC only</div>
<h1>AF recurrence risk report</h1>
<p class="muted">Generated {html.escape(created_at)}. This report is not validated for standalone clinical decision-making.</p>

<h2>Patient and input</h2>
<table>
<tr><th>Patient ID</th><td>{html.escape(str(patient_id))}</td></tr>
<tr><th>Source</th><td>{source_text}</td></tr>
<tr><th>type_af</th><td>{html.escape(clinical_value_label(type_af))}</td></tr>
<tr><th>redo</th><td>{html.escape(clinical_value_label(redo))}</td></tr>
<tr><th>Populated extracted columns</th><td>{int(feature_row.notna().sum(axis=1).iloc[0])}</td></tr>
</table>

<h2>Risk result</h2>
<div class="risk">{html.escape(probability_pct(probability))}</div>
<p><strong>{html.escape(prediction["risk_band"])}</strong></p>
<div class="grid">
  <div class="card"><div class="label">Model</div>{html.escape(prediction["selected_model_label"])}</div>
  <div class="card"><div class="label">Threshold</div>{threshold:.3f} ({html.escape(str(prediction.get("threshold_policy", "")))})</div>
  <div class="card"><div class="label">Cohort position</div>{html.escape(percentile_text)}</div>
</div>
<p>{html.escape(risk_position_text(percentile, probability, threshold))}</p>

<h2>Model agreement and explanation</h2>
<p><strong>{html.escape(agreement_label)}</strong>. Difference: {html.escape(agreement_value)}. {html.escape(agreement_body)}</p>
{dataframe_html(pd.DataFrame(base_rows), index=False) if base_rows else ""}
<p class="notice">Explanation is descriptive and local. It is not causal attribution of recurrence to an individual ECG feature.</p>

<h2>Clinical-variable sensitivity</h2>
{dataframe_html(impacts, index=False)}

<h2>Selected feature groups</h2>
{dataframe_html(feature_summary, index=False)}

<h2>Selected model features</h2>
{dataframe_html(selected_df, max_rows=40, index=False)}

<h2>LOOCV validation summary</h2>
{dataframe_html(report_metric_rows(validation_metrics), index=False)}
<h3>Confusion matrix at selected threshold</h3>
{dataframe_html(confusion_df, index=True)}

<h2>Model notes</h2>
<ul>{notes_html}</ul>
</body>
</html>"""


def build_clinical_summary_csv(
    prediction: dict,
    patient_id: str,
    source_note: str,
    type_af: float,
    redo: float,
    validation_metrics: dict,
) -> bytes:
    rows = [
        {"section": "patient", "field": "patient_id", "value": patient_id},
        {"section": "patient", "field": "source", "value": source_note},
        {"section": "clinical", "field": "type_af", "value": clinical_value_label(type_af)},
        {"section": "clinical", "field": "redo", "value": clinical_value_label(redo)},
        {"section": "prediction", "field": "model", "value": prediction["selected_model_label"]},
        {"section": "prediction", "field": "probability", "value": f"{float(prediction['selected_probability']):.6f}"},
        {"section": "prediction", "field": "threshold", "value": f"{float(prediction['threshold']):.6f}"},
        {"section": "prediction", "field": "threshold_policy", "value": prediction.get("threshold_policy", "")},
        {"section": "prediction", "field": "risk_band", "value": prediction["risk_band"]},
        {"section": "validation", "field": "auc", "value": metric_value(validation_metrics.get("auc", np.nan))},
        {"section": "validation", "field": "average_precision", "value": metric_value(validation_metrics.get("ap", np.nan))},
        {"section": "validation", "field": "brier", "value": metric_value(validation_metrics.get("brier", np.nan), ".4f")},
        {"section": "validation", "field": "sensitivity", "value": pct(validation_metrics.get("sensitivity", np.nan))},
        {"section": "validation", "field": "specificity", "value": pct(validation_metrics.get("specificity", np.nan))},
        {"section": "validation", "field": "f1", "value": pct(validation_metrics.get("f1", np.nan))},
    ]
    if prediction["selected_model"] == "power_mean_ensemble":
        rows.extend(
            [
                {"section": "base_model", "field": "p_ExtraTrees", "value": f"{float(prediction['p_ExtraTrees']):.6f}"},
                {"section": "base_model", "field": "p_BernoulliNB", "value": f"{float(prediction['p_BernoulliNB']):.6f}"},
                {"section": "base_model", "field": "k_power", "value": f"{POWER_K:g}"},
            ]
        )
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def render_clinical_orientation(prediction: dict) -> None:
    probability = prediction["selected_probability"]
    threshold = prediction["threshold"]
    margin = abs(probability - threshold)
    is_borderline = margin < 0.05
    disagreement = False
    if prediction["selected_model"] == "power_mean_ensemble":
        disagreement = abs(prediction["p_ExtraTrees"] - prediction["p_BernoulliNB"]) >= 0.15

    if is_borderline or disagreement:
        parts = []
        if is_borderline:
            parts.append("the score is close to the operating threshold")
        if disagreement:
            parts.append("the two base models are not tightly aligned")
        st.markdown(
            f"""
            <div class="guidance-card warning">
              <div class="guidance-title">⚠ Alerta de zona cinzenta</div>
              This case deserves cautious interpretation because {' and '.join(parts)}. Review signal quality, missingness and clinical context before drawing conclusions.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if prediction["predicted_class"]:
        action_title = "Research-oriented escalation"
        action_summary = "Higher estimated recurrence risk. In a research workflow, this can justify closer review of rhythm follow-up intensity and modifiable clinical factors."
        action_items = [
            "Review ECG/feature quality and missingness before interpreting the score.",
            "Consider closer rhythm surveillance planning inside the research protocol.",
            "Compare with clinical phenotype and established guideline-based risk criteria.",
        ]
    else:
        action_title = "Research-oriented follow-up"
        action_summary = "Lower estimated recurrence risk. Standard follow-up may be reasonable in the research workflow, with escalation only if the clinical context suggests otherwise."
        action_items = [
            "Keep standard rhythm follow-up unless clinical context indicates higher concern.",
            "Check whether the case is borderline or affected by model disagreement.",
            "Use the score as a research signal, not as a standalone decision rule.",
        ]

    item_html = "".join(f"<li>{item}</li>" for item in action_items)

    st.markdown(
        f"""
        <div class="guidance-card action">
          <div class="guidance-title">↗ Vias verdes & plano de ação</div>
          <strong>{action_title}</strong><br>
          {action_summary}
          <ul class="guidance-list">{item_html}</ul>
        </div>
        <div class="guidance-card">
          <div class="guidance-title">▣ Integração com guidelines</div>
          <em>This PoC does not replace guideline-based AF management.</em> Anticoagulation, rhythm-control strategy and follow-up intensity should remain based on validated clinical criteria and clinician judgement.
        </div>
        <div class="guidance-card">
          <div class="guidance-title">ⓘ Operating point</div>
          Selected score: <strong>{probability_pct(probability)}</strong> · Threshold: <strong>{threshold:.3f}</strong> · Margin from threshold: <strong>{margin:.3f}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_portable_app_zip() -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add Windows script
        zf.writestr("INICIAR_APP.bat", "@echo off\necho Iniciando AF Risk Predictor...\npython --version >nul 2>&1\nif errorlevel 1 (\n    echo Python nao encontrado! Instale a partir de python.org\n    pause\n    exit /b\n)\nif not exist \"venv\" (\n    echo Criando ambiente virtual...\n    python -m venv venv\n)\necho Instalando dependencias (na 1a vez demora mais)...\ncall venv\\Scripts\\activate\npip install -r requirements.txt >nul\nstreamlit run app_final.py\npause\n")
        # Add Mac script
        mac_script = "#!/bin/bash\ncd \"$(dirname \"$0\")\"\necho \"Iniciando AF Risk Predictor...\"\nif ! command -v python3 &> /dev/null; then echo \"Python3 nao encontrado.\"; exit 1; fi\nif [ ! -d \"venv\" ]; then python3 -m venv venv; fi\nsource venv/bin/activate\npip install -r requirements.txt >/dev/null\nstreamlit run app_final.py\n"
        zf.writestr("INICIAR_APP.command", mac_script)
        # Add Mac execution permissions metadata
        info = zipfile.ZipInfo("INICIAR_APP.command")
        info.external_attr = 0o755 << 16
        zf.writestr(info, mac_script)
        
        # Add source files
        try:
            with open("app_final.py", "r", encoding="utf-8") as f:
                zf.writestr("app_final.py", f.read())
            with open("requirements.txt", "r", encoding="utf-8") as f:
                zf.writestr("requirements.txt", f.read())
            if os.path.exists("ml_v16_train.py"):
                with open("ml_v16_train.py", "r", encoding="utf-8") as f:
                    zf.writestr("ml_v16_train.py", f.read())
            if os.path.exists("ml_v16_extract.py"):
                with open("ml_v16_extract.py", "r", encoding="utf-8") as f:
                    zf.writestr("ml_v16_extract.py", f.read())
            if os.path.exists("fake_patient_123.mat"):
                with open("fake_patient_123.mat", "rb") as f:
                    zf.writestr("fake_patient_123.mat", f.read())
            for root, _, files in os.walk("app_artifacts"):
                for file in files:
                    file_path = os.path.join(root, file)
                    zf.write(file_path, arcname=file_path)
        except Exception:
            pass
    return zip_buffer.getvalue()

def main() -> None:
    inject_css()
    artifact = load_artifact()

    st.markdown(
        f"""
        <div class="app-header">
          <div class="brand-mark">{brand_heart_html()}</div>
          <div>
            <div class="app-title">AF Recurrence Risk Predictor</div>
            <div class="app-subtitle">Research PoC · Final Ensemble Model</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_col, work_col = st.columns([1.12, 2.88], gap="large")

    with model_col:
        selected_model, threshold_policy = render_model_panel(artifact)
        dark_mode = st.toggle("Night mode", value=False)
        if dark_mode:
            inject_dark_mode_css()
        
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("### 💻 Versão Desktop")
        st.markdown("<span style='font-size: 0.85rem; color: #6b7280;'>Podes descarregar a app para correr no teu PC e usar pastas locais. Extrai o ZIP e clica em **INICIAR_APP**.</span>", unsafe_allow_html=True)
        st.download_button(
            label="💾 Download Portable App",
            data=generate_portable_app_zip(),
            file_name="AF_Risk_Predictor_Desktop.zip",
            mime="application/zip",
            use_container_width=True
        )

    with work_col:
        st.markdown(
            '<div class="notice"><strong>Research demonstration only.</strong> Not validated for standalone clinical decision-making.</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown("### Patient data")
            c1, c2, c3 = st.columns(3)
            with c1:
                patient_id = st.text_input("Patient ID", value="new_patient")
            with c2:
                type_af = clinical_value("type_af", "Use the same 0/1 coding as the V16 database; choose missing if unavailable.")
            with c3:
                redo = clinical_value("redo", "Use the same 0/1 coding as the V16 database; choose missing if unavailable.")

        with st.container(border=True):
            st.markdown("### Signal / feature input")
            mode = st.radio(
                "Input mode",
                ["Local folder path", "Upload ZIP / MAT / feature table"],
                horizontal=True,
            )

            feature_df = None
            source_note = ""

            try:
                if mode == "Local folder path":
                    folder_path = st.text_input("Patient folder path", placeholder="Paste patient folder path...")
                    if st.button("Load local folder", type="primary", use_container_width=True):
                        loader_slot = st.empty()
                        with loader_slot.container():
                            render_heart_loader("Loading local folder...")
                        try:
                            feature_df, source_note = prepare_local_folder(folder_path, patient_id)
                            st.session_state["feature_df"] = feature_df
                            st.session_state["source_note"] = source_note
                        finally:
                            loader_slot.empty()
                else:
                    uploaded = st.file_uploader(
                        "Upload .zip, .mat, .csv, .xlsx, or .xls",
                        type=["zip", "mat", "csv", "xlsx", "xls"],
                    )
                    if uploaded is not None and st.button("Load uploaded file", type="primary", use_container_width=True):
                        loader_slot = st.empty()
                        with loader_slot.container():
                            render_heart_loader("Loading uploaded file...")
                        try:
                            feature_df, source_note = prepare_uploaded_input(uploaded, patient_id)
                            st.session_state["feature_df"] = feature_df
                            st.session_state["source_note"] = source_note
                        finally:
                            loader_slot.empty()

                feature_df = st.session_state.get("feature_df")
                source_note = st.session_state.get("source_note", "")
            except Exception as exc:
                st.error(str(exc))

        if feature_df is None:
            st.stop()

        feature_row = choose_patient_row(feature_df)
        available_count = int(feature_row.notna().sum(axis=1).iloc[0])
        st.success(f"Loaded patient data. {source_note} · {available_count} populated columns")

        try:
            prediction = predict_selected_model(
                artifact,
                selected_model,
                feature_row,
                type_af,
                redo,
                threshold_policy,
            )
            prediction["patient_id"] = patient_id
            payload = export_payload(prediction, patient_id)
            selected_df = selected_feature_table(prediction)

            summary_tab, ecg_tab, technical_tab, explanation_tab, extracted_tab, validation_tab, orientation_tab, export_tab = st.tabs(
                ["Summary", "ECG signal", "Technical details", "Risk explanation", "Extracted features", "Validation", "Clinical orientation", "Export"]
            )

            with summary_tab:
                st.markdown(
                    f"""
                    <div class="risk-card">
                      <div class="risk-number" style="color:{prediction['risk_color']};">{probability_pct(prediction['selected_probability'])}</div>
                      <div class="risk-label">{prediction['risk_band']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if prediction["selected_model"] == "power_mean_ensemble":
                    render_probability_cards(
                        prediction["p_ExtraTrees"],
                        prediction["p_BernoulliNB"],
                        prediction["p_PowerMean"],
                        prediction["threshold"],
                        prediction["risk_band"],
                    )
                    st.markdown(
                        f"""
                        <div class="fusion-note">
                          The final risk score combines ExtraTrees and BernoulliNB probabilities using a Power Mean with k={POWER_K}.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    render_single_probability_cards(
                        prediction["selected_model_label"],
                        prediction["selected_probability"],
                        prediction["threshold"],
                        prediction["risk_band"],
                    )
                if prediction["selected_probability"] <= 0.005 or prediction["selected_probability"] >= 0.995:
                    st.markdown(
                        """
                        <div class="fusion-note">
                          Very extreme score: read this as the model's probability output for this feature vector, not as a standalone calibrated clinical risk. This can happen especially with simpler baseline models such as GNB.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                render_risk_scale(prediction["selected_probability"], prediction["threshold"])

            with ecg_tab:
                render_ecg_signal_preview(prediction)

            with technical_tab:
                st.markdown("#### Features used by base models")
                render_light_table(selected_df, max_rows=30)
                missing_summary = pd.DataFrame(
                    [
                        {
                            "model": model_name,
                            "missing_features": result["n_candidate_missing"],
                            "candidate_features": result["n_candidate_features"],
                            "clinical_encoding": result["clinical_encoding"],
                        }
                        for model_name, result in prediction["base_results"].items()
                    ]
                )
                st.markdown("#### Missing features")
                render_light_table(missing_summary)
                st.markdown("#### Prediction JSON")
                render_json_box(payload)

            with explanation_tab:
                render_risk_explanation(artifact, prediction, feature_row, type_af, redo)

            with extracted_tab:
                render_extracted_feature_list(feature_row)

            with validation_tab:
                validation_metrics = model_metrics(
                    prediction["selected_model"],
                    artifact,
                    threshold_policy,
                )
                st.markdown(f"#### {prediction['selected_model_label']} LOOCV")
                if prediction["selected_model"] == "power_mean_ensemble":
                    st.caption(
                        "Current V16 final model: ExtraTrees tuned + BernoulliNB tuned combined with Power Mean k=3. "
                        f"The displayed threshold is {prediction['threshold']:.3f} ({prediction['threshold_policy']})."
                    )
                v1, v2, v3, v4, v5, v6 = st.columns(6)
                v1.metric("AUC", metric_value(validation_metrics["auc"]))
                v2.metric("AP", metric_value(validation_metrics["ap"], ".3f"))
                v3.metric("Brier", metric_value(validation_metrics["brier"], ".4f"))
                v4.metric("Sensitivity", pct(validation_metrics["sensitivity"]))
                v5.metric("Specificity", pct(validation_metrics["specificity"]))
                v6.metric("F1", pct(validation_metrics["f1"]))
                st.markdown("#### Confusion matrix")
                if validation_metrics["available"]:
                    render_light_table(
                        pd.DataFrame(
                            [
                                [validation_metrics["tn"], validation_metrics["fp"]],
                                [validation_metrics["fn"], validation_metrics["tp"]],
                            ],
                            index=["Actual no recurrence", "Actual recurrence"],
                            columns=["Predicted no recurrence", "Predicted recurrence"],
                        )
                    )
                else:
                    st.info("No real LOOCV confusion matrix found for this model.")

                rows = loocv_rows_for_model(prediction["selected_model"])
                if not rows.empty:
                    st.markdown("#### Threshold tradeoff")
                    threshold_demo = st.slider("Educational threshold", 0.05, 0.95, float(prediction["threshold"]), 0.01)
                    y_true = rows["y_true"].astype(int)
                    y_pred = (rows["score"].astype(float) >= threshold_demo).astype(int)
                    tn = int(((y_true == 0) & (y_pred == 0)).sum())
                    fp = int(((y_true == 0) & (y_pred == 1)).sum())
                    fn = int(((y_true == 1) & (y_pred == 0)).sum())
                    tp = int(((y_true == 1) & (y_pred == 1)).sum())
                    sens = tp / (tp + fn) if (tp + fn) else np.nan
                    spec = tn / (tn + fp) if (tn + fp) else np.nan
                    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
                    st.caption("Educational only: changing this slider does not refit the model.")
                    render_light_table(
                        pd.DataFrame(
                            [{"threshold": threshold_demo, "sensitivity": pct(sens), "specificity": pct(spec), "F1": pct(f1), "TN": tn, "FP": fp, "FN": fn, "TP": tp}]
                        )
                    )

            with orientation_tab:
                render_clinical_orientation(prediction)

            with export_tab:
                export_metrics = model_metrics(
                    prediction["selected_model"],
                    artifact,
                    threshold_policy,
                )
                report_html = build_clinical_report_html(
                    artifact,
                    prediction,
                    patient_id,
                    source_note,
                    type_af,
                    redo,
                    feature_row,
                    selected_df,
                    export_metrics,
                )
                st.download_button(
                    "Download clinical report HTML",
                    data=report_html.encode("utf-8"),
                    file_name=f"{patient_id}_v16_clinical_report.html",
                    mime="text/html",
                    use_container_width=True,
                )
                st.download_button(
                    "Download clinical summary CSV",
                    data=build_clinical_summary_csv(
                        prediction,
                        patient_id,
                        source_note,
                        type_af,
                        redo,
                        export_metrics,
                    ),
                    file_name=f"{patient_id}_v16_clinical_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                st.download_button(
                    "Download final prediction JSON",
                    data=json.dumps(payload, indent=2),
                    file_name=f"{patient_id}_final_ensemble_prediction.json",
                    mime="application/json",
                    use_container_width=True,
                )
                with st.expander("Report preview", expanded=False):
                    components.html(report_html, height=720, scrolling=True)
                st.markdown("#### Raw prediction JSON")
                render_json_box(payload)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
