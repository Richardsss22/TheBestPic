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
import json
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
CORNER_ANIMATION_PATH = SCRIPT_DIR / "app_artifacts" / "corner_loop_transparent.webp"
CORNER_VIDEO_PATH = SCRIPT_DIR / "app_artifacts" / "corner_loop_seamless.mp4"
FALLBACK_CORNER_VIDEO_PATH = Path("/Users/ricardo/Downloads/O meu filme-enhanced.mp4")
HEART_LOADING_PATH = SCRIPT_DIR / "app_artifacts" / "Heart Loading.json"
FALLBACK_HEART_LOADING_PATH = Path("/Users/ricardo/Downloads/Heart Loading.json")

MODEL_VERSION = "V16 Final Power Mean Ensemble"
FINAL_MODEL_LABEL = "Power Mean Ensemble k=2.5"
BASE_MODEL_ET = "extratrees"
BASE_MODEL_BNB = "bernoulli_tuned"
POWER_K = 2.5
FINAL_THRESHOLD = 0.396
FINAL_METRICS = {
    "LOOCV AUC": 0.764,
    "Brier": 0.1887,
    "Sensitivity": 0.673,
    "Specificity": 0.747,
    "F1": 0.625,
}
FINAL_CONFUSION = {
    "threshold": FINAL_THRESHOLD,
    "n": 151,
    "tn": 74,
    "fp": 25,
    "fn": 17,
    "tp": 35,
}
EXTRATREES_METRICS = {
    "threshold": 0.44,
    "n": 151,
    "tn": 65,
    "fp": 34,
    "fn": 14,
    "tp": 38,
    "auc": 0.7377,
    "ap": 0.508,
    "brier": 0.2007,
    "sensitivity": 0.7307,
    "specificity": 0.6565,
    "f1": 0.6129,
}
MODEL_CHOICES = {
    "power_mean_ensemble": "Power Mean Ensemble k=2.5",
    "extratrees": "ExtraTrees",
    "bernoulli_tuned": "BernoulliNB tuned",
    "gnb_fixed": "GNB fixed",
}
MODEL_SHORT_LABELS = {
    "power_mean_ensemble": "PM Ensemble",
    "extratrees": "ExtraTrees",
    "bernoulli_tuned": "BernoulliNB",
    "gnb_fixed": "GNB",
}
MODEL_HELP = {
    "power_mean_ensemble": "Final model",
    "extratrees": "Base model",
    "bernoulli_tuned": "Base model",
    "gnb_fixed": "Baseline",
}
METRIC_CSV_NAMES = {
    "gnb_fixed": "GNB fixed",
    "bernoulli_tuned": "BernoulliNB tuned",
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
        .stTextInput input:focus {
          border-color: var(--blue) !important;
          box-shadow: 0 0 0 2px rgba(37,99,235,.16) !important;
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
          background: #ffffff !important;
          color: var(--ink) !important;
          border: 1px solid #d1d5db !important;
          box-shadow: none !important;
        }
        [data-testid="stFileUploader"] button:hover {
          background: #f9fafb !important;
          color: var(--ink) !important;
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
        .stTextInput input, [data-baseweb="select"] > div, [data-testid="stFileUploaderDropzone"] {
          background: #111827 !important; color: #f9fafb !important; border-color: #4b5563 !important; caret-color: #f9fafb !important;
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
        [data-testid="stMetric"] * { color: #f9fafb !important; }
        [data-testid="stMetricLabel"] * { color: #cbd5e1 !important; }
        [data-baseweb="tab"] { background: #1f2937 !important; color: #e5e7eb !important; border-color: #374151 !important; }
        [data-baseweb="tab"] p, [data-baseweb="tab"] span { color: #e5e7eb !important; }
        [data-baseweb="tab"][aria-selected="true"] { background: #2563eb !important; }
        [data-baseweb="tab"][aria-selected="true"] p,
        [data-baseweb="tab"][aria-selected="true"] span { color: #ffffff !important; }
        .guidance-card.warning { background: #422006 !important; border-color: #a16207 !important; }
        .guidance-card.action { background: #172554 !important; border-color: #1d4ed8 !important; }
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


def brier_for_loocv_model(model_slug: str) -> float:
    predictions = load_loocv_predictions()
    if predictions.empty:
        return np.nan
    rows = predictions[
        (predictions["model_slug"] == model_slug)
        & (predictions["clinical_scope"] == "preop")
        & (predictions["k"] == 18)
    ]
    if rows.empty:
        return np.nan
    return float(np.mean((rows["score"].astype(float) - rows["y_true"].astype(float)) ** 2))


def model_metrics(model_slug: str, artifact: dict) -> dict:
    if model_slug == "power_mean_ensemble":
        return {
            "available": True,
            "threshold": FINAL_THRESHOLD,
            "n": FINAL_CONFUSION["n"],
            "tn": FINAL_CONFUSION["tn"],
            "fp": FINAL_CONFUSION["fp"],
            "fn": FINAL_CONFUSION["fn"],
            "tp": FINAL_CONFUSION["tp"],
            "auc": FINAL_METRICS["LOOCV AUC"],
            "ap": np.nan,
            "brier": FINAL_METRICS["Brier"],
            "sensitivity": FINAL_METRICS["Sensitivity"],
            "specificity": FINAL_METRICS["Specificity"],
            "f1": FINAL_METRICS["F1"],
        }
    if model_slug == "extratrees":
        out = dict(EXTRATREES_METRICS)
        out["available"] = True
        return out

    metrics_df = load_confusion_metrics()
    model_name = METRIC_CSV_NAMES.get(model_slug)
    if model_name and not metrics_df.empty:
        rows = metrics_df[metrics_df["model"] == model_name]
        if not rows.empty:
            row = rows.iloc[0]
            return {
                "available": True,
                "threshold": float(row["threshold"]),
                "n": int(row["n"]),
                "tn": int(row["tn"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "tp": int(row["tp"]),
                "auc": float(row["auc_loocv"]),
                "ap": np.nan,
                "brier": brier_for_loocv_model(model_slug),
                "sensitivity": float(row["sensitivity_recall_tpr"]),
                "specificity": float(row["specificity_tnr"]),
                "f1": float(row["f1"]),
            }

    return {
        "available": False,
        "threshold": 0.5,
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
def load_corner_media_data_uri(media_path: str, mime_type: str) -> str:
    path = Path(media_path)
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def render_corner_video() -> None:
    animation_uri = load_corner_media_data_uri(str(CORNER_ANIMATION_PATH), "image/webp")
    if animation_uri:
        st.markdown(
            f'<img class="corner-animation" src="{animation_uri}" alt="" aria-hidden="true">',
            unsafe_allow_html=True,
        )
        return

    video_path = CORNER_VIDEO_PATH if CORNER_VIDEO_PATH.exists() else FALLBACK_CORNER_VIDEO_PATH
    video_uri = load_corner_media_data_uri(str(video_path), "video/mp4")
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
    animation_uri = load_corner_media_data_uri(str(CORNER_ANIMATION_PATH), "image/webp")
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


def classify_final_risk(probability: float) -> tuple[int, str, str]:
    if probability >= FINAL_THRESHOLD:
        return 1, "Higher estimated recurrence risk", "#dc2626"
    return 0, "Lower estimated recurrence risk", "#0f766e"


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


def render_model_panel(artifact: dict) -> str:
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
    metrics = model_metrics(selected_model, artifact)
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
    st.caption(f"Threshold: {threshold:.3f}")

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
    if selected_model == "extratrees":
        st.caption("ET performance summary and confusion matrix loaded from the provided final metrics.")
    elif not metrics["available"]:
        st.caption("LOOCV confusion metrics unavailable for this deployment-only view.")

    with st.expander("Base models", expanded=False):
        st.write("ExtraTrees")
        st.write("BernoulliNB")
        st.caption("The final score combines both probabilities using a Power Mean with k=2.5.")

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
    return selected_model


def render_sidebar_model_metrics(model_slug: str) -> None:
    metric_names = {
        "bernoulli_tuned": "BernoulliNB tuned",
        "gnb_fixed": "GNB fixed",
        "extratrees": "ExtraTrees",
        "ensemble": "Soft Ensemble",
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
        dst = tmpdir / uploaded_file.name
        dst.write_bytes(uploaded_file.getbuffer())

        if suffix == ".zip":
            extract_dir = tmpdir / "unzipped_patient"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dst, "r") as zf:
                zf.extractall(extract_dir)
            mat = largest_mat_file(extract_dir)
            if mat is not None:
                return extract_features_from_mat(mat, patient_id), f"ZIP -> largest .mat: {mat.name}"
            table = first_feature_table(extract_dir)
            if table is not None:
                return read_feature_table(table), f"ZIP -> feature table: {table.name}"
            raise ValueError("The ZIP did not contain a .mat, .csv, .xlsx, or .xls file.")

        if suffix == ".mat":
            return extract_features_from_mat(dst, patient_id), f"Uploaded .mat: {uploaded_file.name}"

        if suffix in {".csv", ".xlsx", ".xls"}:
            return read_feature_table(dst), f"Uploaded feature table: {uploaded_file.name}"

    raise ValueError(f"Unsupported upload type: {suffix}")


def prepare_local_folder(folder_path: str, patient_id: str) -> tuple[pd.DataFrame | None, str]:
    if not folder_path.strip():
        return None, ""
    folder = Path(folder_path.strip().strip('"'))
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    mat = largest_mat_file(folder)
    if mat is not None:
        return extract_features_from_mat(mat, patient_id), f"Local folder -> largest .mat: {mat.name}"

    table = first_feature_table(folder)
    if table is not None:
        return read_feature_table(table), f"Local folder -> feature table: {table.name}"

    raise ValueError("No .mat, .csv, .xlsx, or .xls file was found in that folder.")


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
        prob = float(np.mean([r["probability"] for r in member_results])) if member_results else np.nan
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
) -> dict:
    metrics = model_metrics(model_slug, artifact)
    threshold = float(metrics["threshold"])
    if model_slug == "power_mean_ensemble":
        prediction = predict_final_ensemble(artifact, feature_row, type_af, redo)
        prediction["selected_model"] = model_slug
        prediction["selected_model_label"] = MODEL_CHOICES[model_slug]
        prediction["selected_probability"] = prediction["p_PowerMean"]
        prediction["threshold"] = threshold
        return prediction

    result = predict_one(artifact, model_slug, feature_row, type_af, redo)
    probability = float(result["probability"])
    predicted_class = int(probability >= threshold)
    if predicted_class:
        risk_band = "Higher estimated recurrence risk"
        color = "#dc2626"
    else:
        risk_band = "Lower estimated recurrence risk"
        color = "#0f766e"
    return {
        "patient_id": None,
        "label": MODEL_CHOICES[model_slug],
        "selected_model": model_slug,
        "selected_model_label": MODEL_CHOICES[model_slug],
        "selected_probability": probability,
        "threshold": threshold,
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


def render_json_box(payload: dict) -> None:
    pretty = json.dumps(payload, indent=2, ensure_ascii=False)
    pretty_html = html.escape(pretty).replace(" ", "&nbsp;").replace("\n", "<br>")
    st.markdown(f'<div class="json-box">{pretty_html}</div>', unsafe_allow_html=True)


def score_percentile(model_slug: str, probability: float) -> float:
    predictions = load_loocv_predictions()
    if predictions.empty:
        return np.nan
    rows = predictions[
        (predictions["model_slug"] == model_slug)
        & (predictions["clinical_scope"] == "preop")
        & (predictions["k"] == 18)
    ]
    if rows.empty:
        return np.nan
    return float((rows["score"].astype(float) <= probability).mean() * 100.0)


def clinical_impact_table(
    artifact: dict,
    model_slug: str,
    feature_row: pd.DataFrame,
    type_af: float,
    redo: float,
    base_probability: float,
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
                prediction = predict_selected_model(artifact, model_slug, feature_row, test_type_af, test_redo)
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
        selected_model = render_model_panel(artifact)
        dark_mode = st.toggle("Night mode", value=False)
        if dark_mode:
            inject_dark_mode_css()

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
            prediction = predict_selected_model(artifact, selected_model, feature_row, type_af, redo)
            prediction["patient_id"] = patient_id
            payload = export_payload(prediction, patient_id)
            selected_df = selected_feature_table(prediction)

            summary_tab, technical_tab, impact_tab, extracted_tab, validation_tab, orientation_tab, export_tab = st.tabs(
                ["Summary", "Technical details", "Impact", "Extracted features", "Validation", "Clinical orientation", "Export"]
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

            with impact_tab:
                render_impact_analysis(artifact, prediction, feature_row, type_af, redo)

            with extracted_tab:
                render_light_table(feature_row.head(1), max_rows=1)

            with validation_tab:
                validation_metrics = model_metrics(prediction["selected_model"], artifact)
                st.markdown(f"#### {prediction['selected_model_label']} LOOCV")
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

                score_rows = load_loocv_predictions()
                if not score_rows.empty and prediction["selected_model"] in {"gnb_fixed", "bernoulli_tuned"}:
                    st.markdown("#### Threshold tradeoff")
                    threshold_demo = st.slider("Educational threshold", 0.05, 0.95, float(prediction["threshold"]), 0.01)
                    rows = score_rows[
                        (score_rows["model_slug"] == prediction["selected_model"])
                        & (score_rows["clinical_scope"] == "preop")
                        & (score_rows["k"] == 18)
                    ]
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
                st.download_button(
                    "Download final prediction JSON",
                    data=json.dumps(payload, indent=2),
                    file_name=f"{patient_id}_final_ensemble_prediction.json",
                    mime="application/json",
                    use_container_width=True,
                )
                render_json_box(payload)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


if __name__ == "__main__":
    main()
