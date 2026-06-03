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
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go


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
ECG_PREVIEW_FS = 1000.0

MODEL_VERSION = "V16 Final Power Mean Ensemble"
FINAL_MODEL_LABEL = "Power Mean Ensemble k=2.5"
BASE_MODEL_ET = "extratrees"
BASE_MODEL_BNB = "bernoulli_tuned"
POWER_K = 2.5
FINAL_THRESHOLD = 0.396
FINAL_METRICS = {
    "LOOCV AUC": 0.764,
    "Average Precision": 0.584,
    "Brier": 0.1887,
    "Sensitivity": 0.673,
    "Specificity": 0.758,
    "F1": 0.631,
}
FINAL_CONFUSION = {
    "threshold": FINAL_THRESHOLD,
    "n": 151,
    "tn": 75,
    "fp": 24,
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
        .threshold-panel,
        .ecg-panel,
        .pdf-panel {
          border: 1px solid var(--line);
          border-radius: 12px;
          background: #ffffff;
          padding: 1rem;
          box-shadow: 0 1px 3px rgba(15,23,42,.05);
          margin: 1rem 0;
        }
        .threshold-badge {
          display: inline-flex;
          align-items: center;
          gap: .4rem;
          border-radius: 999px;
          padding: .34rem .62rem;
          background: #eff6ff;
          color: #1d4ed8;
          border: 1px solid #bfdbfe;
          font-size: .75rem;
          font-weight: 850;
          margin-right: .35rem;
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
          font-size: 1.05rem;
          font-weight: 850;
          margin-top: .12rem;
        }
        .clinical-verdict {
          border-left: 4px solid var(--blue);
          background: #f8fafc;
          padding: .85rem .95rem;
          border-radius: 10px;
          color: var(--ink);
          font-weight: 650;
          margin-top: .8rem;
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
        .threshold-panel,
        .ecg-panel,
        .pdf-panel,
        .ecg-meta-card,
        .clinical-verdict,
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
            "ap": FINAL_METRICS["Average Precision"],
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


def classify_probability_at_threshold(probability: float, threshold: float) -> tuple[int, str, str]:
    if probability >= threshold:
        return 1, "Higher estimated recurrence risk", "#dc2626"
    return 0, "Lower estimated recurrence risk", "#0f766e"


def with_operating_threshold(prediction: dict, threshold: float) -> dict:
    updated = prediction.copy()
    predicted_class, risk_band, color = classify_probability_at_threshold(
        float(updated["selected_probability"]),
        float(threshold),
    )
    updated["threshold"] = float(threshold)
    updated["predicted_class"] = predicted_class
    updated["risk_band"] = risk_band
    updated["risk_color"] = color
    return updated


def render_threshold_controls(prediction: dict) -> float:
    base_threshold = float(prediction["threshold"])
    safe_threshold = max(0.05, base_threshold - 0.08)
    specific_threshold = min(0.95, base_threshold + 0.08)
    st.markdown(
        """
        <div class="threshold-panel">
          <span class="threshold-badge">Clinical threshold mode</span>
          <strong>Move the operating cut-off without changing the model score.</strong><br>
          <span style="color:#6b7280;font-size:.86rem;">
          Lower thresholds prioritize sensitivity and reduce false negatives; higher thresholds prioritize specificity and reduce false positives.
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    preset = st.radio(
        "Operating preset",
        ["Balanced", "Sensitive - fewer false negatives", "Specific - fewer false positives", "Custom"],
        index=0,
        horizontal=True,
        key=f"threshold_preset_{prediction['selected_model']}",
    )
    preset_value = {
        "Balanced": base_threshold,
        "Sensitive - fewer false negatives": safe_threshold,
        "Specific - fewer false positives": specific_threshold,
        "Custom": st.session_state.get(f"clinical_threshold_{prediction['selected_model']}", base_threshold),
    }[preset]
    threshold = st.slider(
        "Clinical decision threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(preset_value),
        step=0.005,
        key=f"clinical_threshold_{prediction['selected_model']}_{preset}",
        help="Educational operating threshold. It updates the class label only; the model probability is unchanged.",
    )
    return float(threshold)


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
                        data[name] = np.array(obj)
                    except Exception:
                        pass

            handle.visititems(visit)
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
    return data


def _candidate_signal_vectors(mapping: dict) -> list[dict]:
    candidates = []
    # Heavily weight ECG tokens
    preferred_tokens = {"pwave": 10000, "ecg": 10000, "surface": 3000, "lead_ii": 3000, "leadii": 3000, "lead2": 3000, "ii": 1000, "v1": 1000, "signal": 100}
    # Strongly penalize CARTO EGM/map specific data
    avoid_tokens = {"bipolar": -10000, "unipolar": -10000, "bip": -10000, "uni": -10000, "act": -10000, "egm": -10000, "voltages": -10000, "lat": -10000, "map": -10000}
    
    for name, value in mapping.items():
        name_lower = str(name).lower()
        if name_lower.startswith("__"):
            continue
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if arr.dtype.kind not in "biufc" or arr.size < 220:
            continue
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            vectors = [(arr.astype(float), "single")]
        elif arr.ndim == 2:
            vectors = []
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
                flat = arr.astype(float).ravel()
                vectors = [(flat, "flattened")]
        elif arr.ndim == 3:
            vectors = []
            try:
                shape = arr.shape
                time_axis = int(np.argmax(shape))
                other_axes = [i for i in range(3) if i != time_axis]
                
                # The smaller remaining axis is leads, the larger is points
                if shape[other_axes[0]] < shape[other_axes[1]]:
                    lead_axis = other_axes[0]
                    point_axis = other_axes[1]
                else:
                    lead_axis = other_axes[1]
                    point_axis = other_axes[0]
                
                if shape[time_axis] >= 220:
                    lead_idx = 1 if shape[lead_axis] > 1 else 0
                    
                    # Find the best map point by checking kurtosis.
                    # 50Hz sine wave noise has a kurtosis of ~1.5.
                    # A real ECG has a high kurtosis (>5) because of the sparse, sharp QRS spikes.
                    best_pt = 0
                    best_kurt = -1
                    pts_to_check = np.unique(np.linspace(0, shape[point_axis] - 1, min(20, shape[point_axis])).astype(int))
                    
                    for pt in pts_to_check:
                        slc_test = [slice(None)] * 3
                        slc_test[lead_axis] = lead_idx
                        slc_test[point_axis] = pt
                        sig = arr[tuple(slc_test)].astype(float)
                        sig = sig[np.isfinite(sig)]
                        if sig.size > 220:
                            std = np.std(sig)
                            if std > 1e-6:
                                kurt = np.mean(((sig - np.mean(sig)) / std)**4)
                                if kurt > best_kurt:
                                    best_kurt = kurt
                                    best_pt = pt
                    
                    if best_kurt == -1:
                        best_pt = 0

                    slc2 = [slice(None)] * 3
                    slc2[lead_axis] = lead_idx
                    slc2[point_axis] = best_pt
                    vectors.append((arr[tuple(slc2)].astype(float), f"channel {lead_idx + 1} (pt {best_pt})"))
                    
                    slc1 = [slice(None)] * 3
                    slc1[lead_axis] = 0
                    slc1[point_axis] = best_pt
                    vectors.append((arr[tuple(slc1)].astype(float), f"channel 1 (pt {best_pt})"))
            except Exception:
                pass
        else:
            continue

        token_score = 0
        for token, weight in preferred_tokens.items():
            if token in name_lower: token_score += weight
        for token, weight in avoid_tokens.items():
            if token in name_lower: token_score += weight
            
        for vector, channel_label in vectors:
            vector = vector[np.isfinite(vector)]
            if vector.size < 220 or np.nanstd(vector) <= 1e-12:
                continue
            candidates.append(
                {
                    "name": str(name),
                    "channel": channel_label,
                    "signal": vector,
                    # Base score on tokens, use size as a tiny tie-breaker
                    "score": token_score + min(vector.size, 200000) / 100000.0,
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
                    # Clinical ECG is almost never below 250 Hz. 
                    # CARTO map geometry update rate is often 80 Hz, which breaks the scaling.
                    if 250 <= fs <= 5000:
                        return fs
            except Exception:
                continue
    # Standard CARTO ECG sampling rate is 1000 Hz
    return 1000.0


def build_ecg_preview() -> dict | None:
    mat_path = st.session_state.get("signal_mat_path")
    mat_bytes = st.session_state.get("signal_mat_bytes")
    mat_name = st.session_state.get("signal_mat_name", "")
    if not mat_path and not mat_bytes:
        return None

    mapping = _read_mat_mapping(mat_path=mat_path, mat_bytes=mat_bytes)
    if not mapping:
        return None
    candidates = _candidate_signal_vectors(mapping)
    if not candidates:
        return None

    source_fs = _infer_sampling_rate(mapping)
    fs = ECG_PREVIEW_FS
    candidate = candidates[0]
    signal = candidate["signal"].astype(float)
    signal = signal - np.nanmedian(signal)
    scale = np.nanpercentile(np.abs(signal), 95)
    if scale > 0:
        signal = signal / scale

    peak = int(np.nanargmax(np.abs(signal)))
    half_window = int(max(180, min(signal.size // 2, 0.45 * source_fs)))
    start = max(0, peak - int(0.32 * source_fs))
    end = min(signal.size, start + 2 * half_window)
    if end - start < 220:
        start = max(0, min(signal.size - 220, peak - 110))
        end = min(signal.size, start + 220)
    segment = signal[start:end]
    t = (np.arange(start, end) - peak) / source_fs * 1000.0

    if segment.size >= 2 and np.isfinite(source_fs) and abs(source_fs - fs) > 1e-6:
        source_seconds = np.arange(segment.size, dtype=float) / source_fs
        duration_s = source_seconds[-1]
        target_len = max(2, int(round(duration_s * fs)) + 1)
        target_seconds = np.linspace(0.0, duration_s, target_len)
        segment = np.interp(target_seconds, source_seconds, segment)
        t = ((start - peak) / source_fs + target_seconds) * 1000.0

    return {
        "name": mat_name or (Path(mat_path).name if mat_path else "uploaded .mat"),
        "source": candidate["name"],
        "channel": candidate["channel"],
        "fs": fs,
        "source_fs": source_fs,
        "t_ms": t,
        "y": segment,
    }


def ecg_preview_figure(preview: dict) -> go.Figure:
    t = np.asarray(preview["t_ms"], dtype=float)
    y = np.asarray(preview["y"], dtype=float)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=t,
                y=y,
                mode="lines",
                line=dict(color="#2563eb", width=2.4),
                name="ECG signal",
                hovertemplate="time=%{x:.1f} ms<br>amplitude=%{y:.3f}<extra></extra>",
            ),
        ]
    )
    if len(t):
        fig.add_vrect(
            x0=max(float(t[0]), -220),
            x1=min(float(t[-1]), 60),
            fillcolor="#34d399",
            opacity=0.16,
            layer="below",
            line_width=0,
            annotation_text="estimated P-wave analysis window",
            annotation_position="top left",
        )
    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        xaxis_title="Time around detected beat (ms)",
        yaxis_title="Normalized amplitude",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#94a3b8")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", zeroline=False)
    return fig


def render_live_ecg_monitor(preview: dict) -> None:
    t = np.asarray(preview["t_ms"], dtype=float)
    y = np.asarray(preview["y"], dtype=float)
    if len(t) < 5:
        return

    max_points = 1200
    if len(t) > max_points:
        idx = np.linspace(0, len(t) - 1, max_points).astype(int)
        y = y[idx]
    y = y.astype(float)
    y = y - np.nanmedian(y)
    
    # Use 98th percentile to capture the QRS peak properly without extreme clipping
    spread = np.nanpercentile(np.abs(y), 98)
    if not np.isfinite(spread) or spread <= 1e-9:
        spread = 1.0
        
    # Scale and clip the signal
    y = np.clip(y / spread, -1.25, 1.25)

    # Optional: Apply a very light smoothing so the trace looks clean and professional
    try:
        y = np.convolve(y, np.ones(5)/5.0, mode='same')
    except Exception:
        pass

    width = 1000.0
    height = 230.0 
    xs = np.linspace(0, width, len(y))
    ys = height * 0.52 - y * 70.0  # Slightly increased amplitude to keep structures visible but not clipped
    path = "M " + " L ".join(f"{x:.1f} {yy:.1f}" for x, yy in zip(xs, ys))
    safe_name = html.escape(str(preview["name"]))
    safe_source = html.escape(str(preview["source"]))
    safe_channel = html.escape(str(preview["channel"]))
    fs = float(preview["fs"])

    components.html(
        f"""
        <div class="ecg-monitor">
          <div class="monitor-top">
            <div>
              <div class="monitor-title">LIVE ECG P-WAVE STRIP</div>
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
                  <stop offset="0%" stop-color="#10b981" stop-opacity=".40"/>
                  <stop offset="50%" stop-color="#34d399" stop-opacity="1"/>
                  <stop offset="100%" stop-color="#6ee7b7" stop-opacity="1"/>
                </linearGradient>
                <filter id="softGlow" x="-20%" y="-80%" width="140%" height="260%">
                  <feGaussianBlur stdDeviation="3.0" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
                <linearGradient id="sweepGradient" x1="0" x2="1">
                  <stop offset="0%" stop-color="white" stop-opacity="0.0"/>
                  <stop offset="25%" stop-color="white" stop-opacity="0.8"/>
                  <stop offset="99%" stop-color="white" stop-opacity="1.0"/>
                  <stop offset="100%" stop-color="black" stop-opacity="1.0"/>
                </linearGradient>
                <mask id="sweepMask">
                  <rect class="sweep-anim" x="-1000" y="-50" width="1000" height="330" fill="url(#sweepGradient)" />
                </mask>
              </defs>
              <path d="{path}" fill="none" stroke="#064e3b" stroke-width="1.5" opacity=".35"/>
              <g mask="url(#sweepMask)">
                <path d="{path}" fill="none" stroke="url(#ecgGlow)" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" filter="url(#softGlow)"/>
              </g>
              <g class="sweep-anim">
                <line class="ecg-cursor" x1="0" y1="0" x2="0" y2="230"/>
              </g>
            </svg>
          </div>
          <div class="monitor-footer">
            <span>Estimated P-wave morphology preview</span>
            <span>Research visualization only</span>
          </div>
        </div>
        <style>
          .ecg-monitor {{
            width: 100%;
            box-sizing: border-box;
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: 16px;
            background: linear-gradient(180deg, #020617 0%, #0f172a 100%);
            padding: 18px;
            box-shadow: 0 20px 40px rgba(0,0,0,.4), inset 0 0 0 1px rgba(255,255,255,.05);
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
            color: #10b981;
            font-size: 13px;
            font-weight: 800;
            letter-spacing: .15em;
          }}
          .monitor-subtitle {{
            margin-top: 5px;
            color: #94a3b8;
            font-size: 12px;
            font-weight: 600;
            max-width: 760px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
          }}
          .monitor-readout {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: 999px;
            padding: 6px 12px;
            background: rgba(2, 6, 23, 0.8);
            color: #34d399;
            font-size: 13px;
            font-weight: 700;
            white-space: nowrap;
          }}
          .live-dot {{
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: #10b981;
            box-shadow: 0 0 12px #10b981;
            animation: pulseDot 1.2s ease-in-out infinite;
          }}
          .monitor-screen {{
            position: relative;
            height: 280px;
            margin: 16px 0 12px;
            border-radius: 12px;
            overflow: hidden;
            background:
              linear-gradient(rgba(16,185,129,.15) 1px, transparent 1px),
              linear-gradient(90deg, rgba(16,185,129,.15) 1px, transparent 1px),
              linear-gradient(rgba(16,185,129,.05) 1px, transparent 1px),
              linear-gradient(90deg, rgba(16,185,129,.05) 1px, transparent 1px),
              radial-gradient(circle at center, rgba(16,185,129,.08), transparent 70%),
              #020617;
            background-size: 100% 56px, 56px 100%, 100% 11.2px, 11.2px 100%, 100% 100%;
            box-shadow: inset 0 0 40px rgba(0,0,0,.8);
          }}
          .monitor-screen svg {{
            width: 100%;
            height: 100%;
            display: block;
          }}
          .ecg-cursor {{
            stroke: rgba(236, 253, 245, 0.9);
            stroke-width: 2.5;
            filter: drop-shadow(0 0 8px #a7f3d0);
          }}
          .sweep-anim {{
            animation: sweepTransform 4.5s linear infinite;
          }}
          .monitor-footer {{
            color: #64748b;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .1em;
          }}
          @keyframes sweepTransform {{
            0% {{ transform: translateX(0); opacity: 0; }}
            3% {{ opacity: 1; }}
            96% {{ transform: translateX(1000px); opacity: 1; }}
            100% {{ transform: translateX(1000px); opacity: 0; }}
          }}
          @keyframes pulseDot {{
            0%, 100% {{ opacity: .5; transform: scale(.9); }}
            50% {{ opacity: 1; transform: scale(1.1); }}
          }}
        </style>
        """,
        height=360,
    )


def render_ecg_signal_preview() -> None:
    st.markdown("#### Real ECG / P-wave preview")
    st.markdown(
        """
        <div class="ecg-panel">
          This preview reads the loaded <strong>.mat</strong> and renders the ECG segment like a live monitor strip.
          Use the static view below only when you want to inspect the full waveform calmly.
        </div>
        """,
        unsafe_allow_html=True,
    )
    preview = build_ecg_preview()
    if preview is None:
        st.info("No raw ECG signal preview could be recovered from this input. Prediction still works from the extracted feature table.")
        return
    render_live_ecg_monitor(preview)
    show_static = st.toggle("Show static diagnostic ECG plot", value=False)
    if show_static:
        st.plotly_chart(ecg_preview_figure(preview), use_container_width=True)
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


def build_pdf_report(prediction: dict, patient_id: str, selected_df: pd.DataFrame) -> bytes:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), dpi=144)
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        def txt(x, y, text, size=10, color="#111827", weight="normal", ha="left", va="top"):
            ax.text(
                x,
                y,
                text,
                fontsize=size,
                color=color,
                fontweight=weight,
                ha=ha,
                va=va,
                transform=ax.transAxes,
                family="DejaVu Sans",
            )

        ax.add_patch(plt.Rectangle((0, 0.91), 1, 0.09, color="#f8fafc", transform=ax.transAxes, zorder=0))
        txt(0.065, 0.965, "AF Recurrence Risk Predictor", size=22, weight="bold")
        txt(0.065, 0.932, "Research PoC - Final Ensemble Model", size=10, color="#64748b", weight="bold")
        txt(
            0.065,
            0.885,
            "Research demonstration only. Not validated for standalone clinical decision-making.",
            size=9,
            color="#1d4ed8",
            weight="bold",
        )

        probability = float(prediction["selected_probability"])
        threshold = float(prediction["threshold"])

        ax.add_patch(
            plt.Rectangle((0.065, 0.68), 0.87, 0.17, facecolor="#ffffff", edgecolor="#e5e7eb", linewidth=1.2, transform=ax.transAxes)
        )
        txt(0.085, 0.825, f"Patient ID: {patient_id}", size=12, weight="bold")
        txt(0.085, 0.798, f"Timestamp: {prediction['timestamp']}", size=9, color="#64748b")
        txt(0.085, 0.765, f"{probability * 100:.1f}%", size=42, color=prediction["risk_color"], weight="bold")
        txt(0.255, 0.75, prediction["risk_band"], size=13, weight="bold")
        txt(0.255, 0.723, f"Operating threshold: {threshold:.3f}", size=10, color="#64748b", weight="bold")

        bar_x, bar_y, bar_w, bar_h = 0.085, 0.655, 0.83, 0.026
        ax.add_patch(plt.Rectangle((bar_x, bar_y), bar_w * threshold, bar_h, color="#6ee7b7", transform=ax.transAxes))
        ax.add_patch(plt.Rectangle((bar_x + bar_w * threshold, bar_y), bar_w * (1 - threshold), bar_h, color="#fb7185", transform=ax.transAxes))
        ax.plot([bar_x + bar_w * probability, bar_x + bar_w * probability], [bar_y - 0.008, bar_y + bar_h + 0.008], color="#111827", lw=2.4, transform=ax.transAxes)
        ax.plot([bar_x + bar_w * threshold, bar_x + bar_w * threshold], [bar_y - 0.008, bar_y + bar_h + 0.008], color="#ffffff", lw=2.0, transform=ax.transAxes)
        txt(bar_x, bar_y - 0.018, "0.00", size=8, color="#64748b")
        txt(bar_x + bar_w - 0.035, bar_y - 0.018, "1.00", size=8, color="#64748b")
        txt(bar_x + bar_w * threshold - 0.045, bar_y - 0.018, f"Threshold {threshold:.3f}", size=8, color="#64748b")

        y = 0.595
        txt(0.065, y, "Model output", size=14, weight="bold")
        lines = [
            ("Selected model", prediction["selected_model_label"]),
            ("Selected score", f"{probability:.4f}"),
            ("Threshold", f"{threshold:.3f}"),
            ("Predicted class", str(prediction["predicted_class"])),
            ("Missing features", str(prediction["missing_features"])),
        ]
        if prediction["selected_model"] == "power_mean_ensemble":
            lines.extend(
                [
                    ("p_ExtraTrees", f"{prediction['p_ExtraTrees']:.4f}"),
                    ("p_BernoulliNB", f"{prediction['p_BernoulliNB']:.4f}"),
                    ("p_PowerMean", f"{prediction['p_PowerMean']:.4f}"),
                    ("k_power", f"{prediction['k_power']:.2f}"),
                ]
            )
        for label, value in lines:
            y -= 0.028
            txt(0.085, y, label, size=9, color="#64748b")
            txt(0.33, y, value, size=9, weight="bold")

        y -= 0.055
        txt(0.065, y, "Features used by selected model", size=14, weight="bold")
        y -= 0.03
        ax.add_patch(plt.Rectangle((0.065, y - 0.012), 0.87, 0.025, facecolor="#f1f5f9", edgecolor="#e5e7eb", transform=ax.transAxes))
        txt(0.085, y + 0.006, "Model", size=8, color="#475569", weight="bold")
        txt(0.24, y + 0.006, "Feature", size=8, color="#475569", weight="bold")
        txt(0.78, y + 0.006, "Value", size=8, color="#475569", weight="bold")
        top_features = selected_df.head(10)
        for _, row in top_features.iterrows():
            y -= 0.026
            value = row.get("value_after_preprocessing", np.nan)
            txt(0.085, y, str(row.get("model", ""))[:18], size=8, color="#64748b")
            txt(0.24, y, str(row.get("feature", ""))[:48], size=8)
            txt(0.78, y, f"{float(value):.4g}" if pd.notna(value) else "n/a", size=8, weight="bold")

        txt(
            0.065,
            0.075,
            "Interpretation note: this report is generated by a research proof-of-concept and must not be used as a standalone clinical decision rule.",
            size=8,
            color="#6b7280",
        )
        pdf.savefig(fig)
        plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


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
    if selected_model == "power_mean_ensemble":
        st.caption(f"Operational threshold: {threshold:.3f}")
        st.caption("LOOCV classification metrics: Forma A, fold-specific inner-CV thresholds.")
    else:
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
        st.session_state.pop("signal_mat_path", None)
        st.session_state.pop("signal_mat_bytes", None)
        st.session_state.pop("signal_mat_name", None)

        if suffix == ".zip":
            extract_dir = tmpdir / "unzipped_patient"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dst, "r") as zf:
                zf.extractall(extract_dir)
            mat = largest_mat_file(extract_dir)
            if mat is not None:
                st.session_state["signal_mat_bytes"] = mat.read_bytes()
                st.session_state["signal_mat_name"] = mat.name
                return extract_features_from_mat(mat, patient_id), f"ZIP -> largest .mat: {mat.name}"
            table = first_feature_table(extract_dir)
            if table is not None:
                return read_feature_table(table), f"ZIP -> feature table: {table.name}"
            raise ValueError("The ZIP did not contain a .mat, .csv, .xlsx, or .xls file.")

        if suffix == ".mat":
            st.session_state["signal_mat_bytes"] = dst.read_bytes()
            st.session_state["signal_mat_name"] = uploaded_file.name
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
        st.session_state["signal_mat_path"] = str(mat)
        st.session_state["signal_mat_name"] = mat.name
        st.session_state.pop("signal_mat_bytes", None)
        return extract_features_from_mat(mat, patient_id), f"Local folder -> largest .mat: {mat.name}"

    table = first_feature_table(folder)
    if table is not None:
        st.session_state.pop("signal_mat_path", None)
        st.session_state.pop("signal_mat_bytes", None)
        st.session_state.pop("signal_mat_name", None)
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
            selected_df = selected_feature_table(prediction)

            summary_tab, ecg_tab, technical_tab, impact_tab, extracted_tab, validation_tab, orientation_tab, export_tab = st.tabs(
                ["Summary", "ECG signal", "Technical details", "Impact", "Extracted features", "Validation", "Clinical orientation", "Export"]
            )

            with summary_tab:
                clinical_threshold = render_threshold_controls(prediction)
                prediction = with_operating_threshold(prediction, clinical_threshold)
                payload = export_payload(prediction, patient_id)
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
                st.markdown(
                    f"""
                    <div class="clinical-verdict">
                      Current operating threshold: <strong>{prediction['threshold']:.3f}</strong>. 
                      The score is unchanged; only the operational class and risk category update as the threshold moves.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with ecg_tab:
                render_ecg_signal_preview()

            payload = export_payload(prediction, patient_id)

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
                if prediction["selected_model"] == "power_mean_ensemble":
                    st.caption(
                        "Forma A: threshold selected inside each outer fold's inner CV. "
                        f"The deployed operational threshold remains {prediction['threshold']:.3f}."
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
                st.markdown(
                    """
                    <div class="pdf-panel">
                      <strong>Clinical-style PDF report</strong><br>
                      <span style="color:#6b7280;font-size:.86rem;">
                      Generates a polished one-page research report with model score, operating threshold, risk band and selected features.
                      </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                pdf_bytes = build_pdf_report(prediction, patient_id, selected_df)
                st.download_button(
                    "Generate medical report PDF",
                    data=pdf_bytes,
                    file_name=f"{patient_id}_af_recurrence_research_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
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
