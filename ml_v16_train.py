#!/usr/bin/env python3
"""
ml_v16_train.py - ML v16: Addressing Professor's feedback.

Evolution from V15:
  - Compare the locked V15-style pre-operative model with and without
    blanking_period, as requested for thesis/discussion context.

Design goals (inherited from V14/V15):
  1. Strict fold-local preprocessing.
  2. Deterministic, patient-level ECG engineering.
  3. Conservative pre-specified sparse linear model (GNB k=18, vs=0.1).
  4. LOOCV as primary, Repeated Stratified 10-fold CV as sensitivity analysis.

Cache: features_v16_cache.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import LeaveOneOut, ParameterGrid, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import resample


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")
ANOVA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "anova_features")
GNB_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "gnb_fixed")
BERNOULLI_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "bernoulli_tuned")
CACHE_FILE = os.path.join(SCRIPT_DIR, "features_v16_cache.pkl")

META_COLS = ["patient_id", "recurrence", "type_af", "redo", "blanking_period", "source"]
PREOP_CLINICAL = ["type_af", "redo"]
BLANKING_CLINICAL = ["type_af", "redo", "blanking_period"]
SAFE_CLINICAL = BLANKING_CLINICAL
TECHNICAL_COUNTERS = ["n_samples", "n_rpeaks"]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    selector: str
    k: int | str
    penalty: str
    C: float
    l1_ratio: float | None = None
    scaler: str = "standard"
    class_weight: str | None = "balanced"
    estimator: str = "logistic"


# Primary V15 model: GaussianNB on 18 ANOVA-selected P-wave features + clinical bypass.
# k=18 and var_smoothing=0.1 were validated via incremental experimentation in
# v15_experiment_lab.py and v15_experiment_lab_round2.py.
# The ElasticNet model is retained as a sensitivity analysis.
V15_CONFIGS = [
    ModelConfig("anova18_gnb", "anova", 18, "gnb", 1.0, None, estimator="gnb"),
]

V15_LOGISTIC_CONFIGS = [
    ModelConfig("anova18_enet_c010_l1_07", "anova", 18, "elasticnet", 0.10, 0.70),
]

# V15 GNB var_smoothing (experimentally validated)
V15_VAR_SMOOTHING = 0.1

# BernoulliNB tuned model discovered in the V16.3 ANOVA k-sweep.
# In final training/evaluation these parameters are selected fold-locally by
# inner CV; this grid is not tuned on the held-out outer patient.
INNER_FOLDS = 3
RANDOM_STATE = 42
BERNOULLI_PARAM_GRID = list(
    ParameterGrid({"alpha": [0.1, 0.5, 1.0, 2.0, 5.0], "binarize": [0.25, 0.5, 0.75]})
)

BERNOULLI_TUNED_CONFIGS = [
    ModelConfig(
        "anova18_bernoulli_tuned",
        "anova",
        18,
        "bernoulli",
        1.0,
        None,
        scaler="minmax",
        class_weight=None,
        estimator="bernoulli_tuned",
    ),
]


def _safe_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def split_feature_sets(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    feat_cols = [c for c in df.columns if c not in META_COLS]
    ecg_prefixes = (
        "pw_",
        "fw_",
        "ecg_",
        "hr_",
        "rr_",
        "rmssd",
        "sdnn",
        "pnn",
        "tri_",
        "PTFV1",
        "n_rpeaks",
        "n_samples",
    )
    ecg_cols = [c for c in feat_cols if any(c.startswith(p) for p in ecg_prefixes)]
    egm_cols = [c for c in feat_cols if c not in ecg_cols and c != "n_points"]
    clinical_cols = [c for c in SAFE_CLINICAL if c in df.columns]
    return ecg_cols, egm_cols, clinical_cols


def add_ecg_domain_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic per-patient ECG/P-wave features."""
    X = X.copy()
    leads = ["II", "V1", "I", "aVF", "aVL"]

    # Clinically interpretable heart-rate estimate from recording counters.
    if "n_rpeaks" in X.columns and "n_samples" in X.columns:
        duration_s = X["n_samples"] / 1000.0
        X["ecg_hr_est_bpm"] = 60.0 * X["n_rpeaks"] / duration_s.replace(0, np.nan)

    ratio_pairs = [
        ("pw_aVL_rms_mean", "pw_V1_rms_mean", "xl_aVL_V1_rms_ratio"),
        ("pw_aVL_area_total_mean", "pw_V1_area_total_mean", "xl_aVL_V1_area_ratio"),
        ("pw_aVL_cwt_energy_mean", "pw_V1_cwt_energy_mean", "xl_aVL_V1_cwt_ratio"),
        ("pw_aVF_area_neg_mean", "pw_V1_area_neg_mean", "xl_aVF_V1_neg_area_ratio"),
        ("pw_II_slope_onset_mean", "pw_II_slope_offset_mean", "xl_II_slope_asymmetry"),
        ("pw_I_rms_mean", "pw_aVL_rms_mean", "xl_I_aVL_rms_ratio"),
        ("pw_V1_area_neg_mean", "pw_V1_area_total_mean", "xl_V1_terminal_neg_fraction"),
        ("pw_V1_ptf_mean", "pw_V1_duration_ms_mean", "xl_V1_ptf_per_ms"),
    ]
    for num, den, name in ratio_pairs:
        if num in X.columns and den in X.columns:
            X[name] = X[num] / (X[den].abs() + 1e-10)

    summary_metrics = [
        "duration_ms_mean",
        "duration_ms_std",
        "amp_abs_mean",
        "area_total_mean",
        "area_neg_mean",
        "ptf_mean",
        "rms_mean",
        "p_qrs_ratio_mean",
        "spectral_entropy_mean",
        "cwt_energy_mean",
        "nvg_degree_entropy_mean",
        "symmetry_mean",
    ]
    for metric in summary_metrics:
        cols = [f"pw_{lead}_{metric}" for lead in leads if f"pw_{lead}_{metric}" in X.columns]
        if len(cols) >= 3:
            vals = X[cols]
            stem = metric.replace("_mean", "")
            X[f"xl_{stem}_lead_mean"] = vals.mean(axis=1)
            X[f"xl_{stem}_lead_std"] = vals.std(axis=1)
            X[f"xl_{stem}_lead_range"] = vals.max(axis=1) - vals.min(axis=1)
            X[f"xl_{stem}_lead_cv"] = vals.std(axis=1) / (vals.mean(axis=1).abs() + 1e-10)

    dur_cols = [f"pw_{lead}_duration_ms_mean" for lead in leads if f"pw_{lead}_duration_ms_mean" in X.columns]
    if len(dur_cols) >= 3:
        X["xl_pw_duration_dispersion"] = X[dur_cols].max(axis=1) - X[dur_cols].min(axis=1)

    amp_cols = [f"pw_{lead}_amp_abs_mean" for lead in leads if f"pw_{lead}_amp_abs_mean" in X.columns]
    if len(amp_cols) >= 3:
        X["xl_pw_amp_dispersion"] = X[amp_cols].max(axis=1) - X[amp_cols].min(axis=1)

    # Literature-style P-wave indices. These are deterministic transforms of
    # already extracted ECG features and use no cohort-level fitted quantities.
    dur_mean_cols = [f"pw_{lead}_duration_ms_mean" for lead in leads if f"pw_{lead}_duration_ms_mean" in X.columns]
    if len(dur_mean_cols) >= 2:
        durations = X[dur_mean_cols]
        X["lit_pwave_pmax_ms"] = durations.max(axis=1)
        X["lit_pwave_pmin_ms"] = durations.min(axis=1)
        X["lit_pwave_mean_duration_ms"] = durations.mean(axis=1)
        X["lit_pwave_dispersion_ms"] = durations.max(axis=1) - durations.min(axis=1)
        X["lit_pwave_pmax_ge_115"] = (X["lit_pwave_pmax_ms"] >= 115).astype(float)
        X["lit_pwave_pmax_ge_120"] = (X["lit_pwave_pmax_ms"] >= 120).astype(float)
        X["lit_pwave_dispersion_ge_40"] = (X["lit_pwave_dispersion_ms"] >= 40).astype(float)

    pr_cols = [f"pw_{lead}_pr_interval_ms_mean" for lead in leads if f"pw_{lead}_pr_interval_ms_mean" in X.columns]
    if len(pr_cols) >= 2:
        pr = X[pr_cols]
        X["lit_pr_mean_ms"] = pr.mean(axis=1)
        X["lit_pr_max_ms"] = pr.max(axis=1)
        X["lit_pr_ge_200"] = (X["lit_pr_max_ms"] >= 200).astype(float)

    # Approximate P-wave peak time from duration and symmetry:
    # symmetry = onset_duration / offset_duration.
    peak_time_cols = []
    for lead in leads:
        d_col = f"pw_{lead}_duration_ms_mean"
        s_col = f"pw_{lead}_symmetry_mean"
        if d_col in X.columns and s_col in X.columns:
            out_col = f"lit_{lead}_pwave_peak_time_ms"
            sym = X[s_col].clip(lower=0)
            X[out_col] = X[d_col] * sym / (1.0 + sym)
            peak_time_cols.append(out_col)
    if len(peak_time_cols) >= 2:
        peak_times = X[peak_time_cols]
        X["lit_pwave_peak_time_max_ms"] = peak_times.max(axis=1)
        X["lit_pwave_peak_time_mean_ms"] = peak_times.mean(axis=1)

    if "PTFV1" in X.columns:
        X["lit_ptfv1_abs"] = X["PTFV1"].abs()
        # 0.04 mV*s is equivalent to 40 mV*ms if the source amplitude is in mV.
        X["lit_ptfv1_ge_40_mVms"] = (X["lit_ptfv1_abs"] >= 40).astype(float)
    if "pw_V1_ptf_mean" in X.columns:
        X["lit_v1_ptf_abs"] = X["pw_V1_ptf_mean"].abs()
        X["lit_v1_ptf_ge_40_mVms"] = (X["lit_v1_ptf_abs"] >= 40).astype(float)

    # Frontal P-wave axis surrogate from signed lead I and aVF P amplitudes.
    if "pw_I_amp_mean" in X.columns and "pw_aVF_amp_mean" in X.columns:
        axis = np.degrees(np.arctan2(X["pw_aVF_amp_mean"], X["pw_I_amp_mean"]))
        X["lit_pwave_axis_deg"] = axis
        X["lit_pwave_axis_abnormal"] = ((axis < 0) | (axis > 75)).astype(float)

    return X


def encode_clinical(df: pd.DataFrame, clinical_cols: Iterable[str]) -> pd.DataFrame:
    """Encode categorical clinical variables as missing=0, observed levels=1..n."""
    out = pd.DataFrame(index=df.index)
    for col in clinical_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        levels = sorted(s.dropna().unique().tolist())
        mapping = {level: i + 1 for i, level in enumerate(levels)}
        out[col] = s.map(mapping).fillna(0).astype(float)
    return out


def encode_clinical_legacy_missing_indicators(df: pd.DataFrame, clinical_cols: Iterable[str]) -> pd.DataFrame:
    """V15-style clinical encoding: value filled with 0 plus explicit missingness flags."""
    out = pd.DataFrame(index=df.index)
    for col in clinical_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        out[col] = s.fillna(0).astype(float)
        out[f"{col}_missing"] = s.isna().astype(float)
    return out


class FoldPreprocessor:
    """Fold-local signal cleanup with train-only fitted state."""

    def __init__(
        self,
        missing_threshold: float = 0.50,
        corr_threshold: float = 0.95,
        include_missing_indicators: bool = True,
    ):
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.include_missing_indicators = include_missing_indicators

    def fit(self, X: pd.DataFrame) -> "FoldPreprocessor":
        Xn = _safe_numeric_frame(X)
        missing_rate = Xn.isna().mean()
        keep = missing_rate[missing_rate < self.missing_threshold].index.tolist()
        keep = [c for c in keep if not Xn[c].isna().all()]
        if not keep:
            raise ValueError("No signal features survived missingness filtering.")

        self.initial_cols_ = keep
        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = pd.DataFrame(self.imputer_.fit_transform(Xn[keep]), columns=keep, index=X.index)

        std = X_imp.std(axis=0)
        var_cols = std[std > 1e-10].index.tolist()
        if not var_cols:
            raise ValueError("No signal features survived variance filtering.")

        if len(var_cols) > 1:
            corr = X_imp[var_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = {c for c in upper.columns if (upper[c] > self.corr_threshold).any()}
            self.signal_cols_ = [c for c in var_cols if c not in to_drop]
            self.corr_dropped_ = sorted(to_drop)
        else:
            self.signal_cols_ = var_cols
            self.corr_dropped_ = []

        # Missingness of retained signal features may encode failed P-wave detection.
        if self.include_missing_indicators:
            self.indicator_cols_ = [
                c for c in self.signal_cols_ if 0 < Xn[c].isna().mean() < self.missing_threshold
            ]
        else:
            self.indicator_cols_ = []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xn = _safe_numeric_frame(X)
        X_imp = pd.DataFrame(
            self.imputer_.transform(Xn[self.initial_cols_]),
            columns=self.initial_cols_,
            index=X.index,
        )
        out = X_imp[self.signal_cols_].copy()
        for col in self.indicator_cols_:
            out[f"miss__{col}"] = Xn[col].isna().astype(float)
        return out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def rank_auc_scores(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    scores = []
    for col in X.columns:
        v = X[col].to_numpy(dtype=float)
        if np.nanstd(v) <= 1e-12:
            scores.append(0.0)
            continue
        try:
            auc = roc_auc_score(y, v)
            scores.append(abs(auc - 0.5))
        except Exception:
            scores.append(0.0)
    return np.asarray(scores, dtype=float)


def feature_scores(X: pd.DataFrame, y: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return np.ones(X.shape[1], dtype=float)
    if method == "anova":
        scores, _ = f_classif(X, y)
    elif method == "rank_auc":
        scores = rank_auc_scores(X, y)
    elif method == "mi":
        n_neighbors = max(3, min(5, int(np.bincount(y).min()) - 1))
        scores = mutual_info_classif(X, y, n_neighbors=n_neighbors, random_state=42)
    else:
        raise ValueError(f"Unknown selector: {method}")
    scores = np.asarray(scores, dtype=float)
    scores[~np.isfinite(scores)] = 0.0
    return scores


def select_features(X: pd.DataFrame, y: np.ndarray, cfg: ModelConfig) -> list[str]:
    if cfg.k == "all" or cfg.selector == "none":
        return list(X.columns)
    k = min(int(cfg.k), X.shape[1])
    scores = feature_scores(X, y, cfg.selector)
    order = np.argsort(scores)[::-1]
    return list(X.columns[order[:k]])


def build_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler(quantile_range=(10, 90))
    if name == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown scaler: {name}")


def model_score(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def make_design_matrix(X_sig_selected: pd.DataFrame, X_clin_part: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [X_sig_selected.reset_index(drop=True), X_clin_part.reset_index(drop=True)],
        axis=1,
    )


def scale_train_test(Xtr: pd.DataFrame, Xte: pd.DataFrame, scaler_name: str) -> tuple[np.ndarray, np.ndarray]:
    scaler = build_scaler(scaler_name)
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    return Xtr_s, Xte_s


def inner_cv_select_bernoulli_params(
    X_sig_tr_raw: pd.DataFrame,
    X_clin_tr: pd.DataFrame,
    y_tr: np.ndarray,
    cfg: ModelConfig,
) -> tuple[dict[str, float], float]:
    """Select BernoulliNB alpha/binarize using only the outer-training patients."""
    param_scores = [[] for _ in BERNOULLI_PARAM_GRID]
    skf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for itr, iva in skf.split(X_sig_tr_raw, y_tr):
        pre = FoldPreprocessor(missing_threshold=0.50, corr_threshold=0.95)
        try:
            X_inner_tr = pre.fit_transform(X_sig_tr_raw.iloc[itr])
            X_inner_va = pre.transform(X_sig_tr_raw.iloc[iva])
        except Exception:
            for scores in param_scores:
                scores.append(0.5)
            continue

        selected = select_features(X_inner_tr, y_tr[itr], cfg)
        if not selected:
            for scores in param_scores:
                scores.append(0.5)
            continue

        Xtr = make_design_matrix(X_inner_tr[selected], X_clin_tr.iloc[itr])
        Xva = make_design_matrix(X_inner_va[selected], X_clin_tr.iloc[iva])
        Xtr_s, Xva_s = scale_train_test(Xtr, Xva, cfg.scaler)

        for param_idx, params in enumerate(BERNOULLI_PARAM_GRID):
            try:
                model = BernoulliNB(alpha=params["alpha"], binarize=params["binarize"])
                model.fit(Xtr_s, y_tr[itr])
                scores = model_score(model, Xva_s)
                param_scores[param_idx].append(roc_auc_score(y_tr[iva], scores))
            except Exception:
                param_scores[param_idx].append(0.5)

    means = [float(np.mean(scores)) if scores else 0.5 for scores in param_scores]
    best_idx = int(np.argmax(means))
    return dict(BERNOULLI_PARAM_GRID[best_idx]), means[best_idx]


def fit_bernoulli_tuned_outer(
    X_signal: pd.DataFrame,
    X_clinical: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: ModelConfig,
) -> np.ndarray:
    y_tr = y[train_idx]
    X_sig_tr_raw = X_signal.iloc[train_idx].reset_index(drop=True)
    X_clin_tr = X_clinical.iloc[train_idx].reset_index(drop=True)

    selected_params, _ = inner_cv_select_bernoulli_params(X_sig_tr_raw, X_clin_tr, y_tr, cfg)

    pre = FoldPreprocessor(missing_threshold=0.50, corr_threshold=0.95)
    X_sig_tr = pre.fit_transform(X_signal.iloc[train_idx])
    X_sig_te = pre.transform(X_signal.iloc[test_idx])

    selected = select_features(X_sig_tr, y_tr, cfg)
    if not selected:
        return np.full(len(test_idx), 0.5)

    Xtr = make_design_matrix(X_sig_tr[selected], X_clinical.iloc[train_idx])
    Xte = make_design_matrix(X_sig_te[selected], X_clinical.iloc[test_idx])
    Xtr_s, Xte_s = scale_train_test(Xtr, Xte, cfg.scaler)

    model = BernoulliNB(alpha=selected_params["alpha"], binarize=selected_params["binarize"])
    model.fit(Xtr_s, y_tr)
    return model_score(model, Xte_s)


def fit_one_config(
    X_sig_tr: pd.DataFrame,
    X_sig_te: pd.DataFrame,
    X_clin_tr: pd.DataFrame,
    X_clin_te: pd.DataFrame,
    y_tr: np.ndarray,
    cfg: ModelConfig,
) -> np.ndarray:
    selected = select_features(X_sig_tr, y_tr, cfg)
    Xtr = pd.concat([X_sig_tr[selected].reset_index(drop=True), X_clin_tr.reset_index(drop=True)], axis=1)
    Xte = pd.concat([X_sig_te[selected].reset_index(drop=True), X_clin_te.reset_index(drop=True)], axis=1)

    scaler = build_scaler(cfg.scaler)
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    if cfg.estimator == "gnb" or cfg.penalty == "gnb":
        model = GaussianNB(var_smoothing=V15_VAR_SMOOTHING)
    else:
        kwargs = {
            "C": cfg.C,
            "class_weight": cfg.class_weight,
            "max_iter": 5000,
            "random_state": 42,
        }
        if cfg.penalty == "elasticnet":
            kwargs.update({"penalty": "elasticnet", "solver": "saga", "l1_ratio": cfg.l1_ratio})
        elif cfg.penalty == "l2":
            kwargs.update({"penalty": "l2", "solver": "lbfgs"})
        else:
            raise ValueError(f"Unknown penalty: {cfg.penalty}")
        model = LogisticRegression(**kwargs)

    model.fit(Xtr_s, y_tr)
    return model.predict_proba(Xte_s)[:, 1]


def fold_predict(
    X_signal: pd.DataFrame,
    X_clinical: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    configs: list[ModelConfig],
) -> np.ndarray:
    y_tr = y[train_idx]

    preds = []
    standard_configs = [cfg for cfg in configs if cfg.estimator != "bernoulli_tuned"]
    tuned_configs = [cfg for cfg in configs if cfg.estimator == "bernoulli_tuned"]

    if standard_configs:
        pre = FoldPreprocessor(missing_threshold=0.50, corr_threshold=0.95)
        X_sig_tr = pre.fit_transform(X_signal.iloc[train_idx])
        X_sig_te = pre.transform(X_signal.iloc[test_idx])

        X_clin_tr = X_clinical.iloc[train_idx]
        X_clin_te = X_clinical.iloc[test_idx]

        for cfg in standard_configs:
            preds.append(fit_one_config(X_sig_tr, X_sig_te, X_clin_tr, X_clin_te, y_tr, cfg))

    for cfg in tuned_configs:
        preds.append(fit_bernoulli_tuned_outer(X_signal, X_clinical, y, train_idx, test_idx, cfg))

    return np.mean(np.vstack(preds), axis=0)


def threshold_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {
        "Accuracy": accuracy_score(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "PPV": precision_score(y, pred, zero_division=0),
        "NPV": npv,
        "Sensitivity": recall_score(y, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "FPR": fp / (fp + tn) if (fp + tn) else np.nan,
        "FNR": fn / (fn + tp) if (fn + tp) else np.nan,
        "Threshold": threshold,
    }


def threshold_sweep(y: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    rows = {}
    for threshold in np.round(np.arange(0.10, 0.91, 0.05), 2):
        rows[f"thr_{threshold:.2f}"] = threshold_metrics(y, prob, float(threshold))
    return pd.DataFrame(rows).T


def bootstrap_auc_ci(y: np.ndarray, prob: np.ndarray, n_boot: int) -> tuple[float, float]:
    if n_boot <= 0:
        return np.nan, np.nan
    aucs = []
    for b in range(n_boot):
        try:
            idx = resample(range(len(y)), stratify=y, random_state=b)
            aucs.append(roc_auc_score(y[idx], prob[idx]))
        except Exception:
            continue
    if len(aucs) < 10:
        return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def repeated_cv_predict(
    X_signal: pd.DataFrame,
    X_clinical: pd.DataFrame,
    y: np.ndarray,
    configs: list[ModelConfig],
    n_splits: int,
    n_repeats: int,
) -> tuple[np.ndarray, list[float]]:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    y_prob_sum = np.zeros(len(y), dtype=float)
    y_count = np.zeros(len(y), dtype=float)
    repeat_probs = [np.full(len(y), np.nan, dtype=float) for _ in range(n_repeats)]

    for fold_no, (train_idx, test_idx) in enumerate(cv.split(X_signal, y)):
        repeat_idx = fold_no // n_splits
        fold_prob = fold_predict(X_signal, X_clinical, y, train_idx, test_idx, configs)
        y_prob_sum[test_idx] += fold_prob
        y_count[test_idx] += 1
        repeat_probs[repeat_idx][test_idx] = fold_prob

    y_prob_avg = y_prob_sum / np.maximum(y_count, 1)
    repeat_aucs = [roc_auc_score(y, p) for p in repeat_probs if not np.isnan(p).any()]
    return y_prob_avg, repeat_aucs


def loocv_predict(
    X_signal: pd.DataFrame,
    X_clinical: pd.DataFrame,
    y: np.ndarray,
    configs: list[ModelConfig],
    progress_label: str = "LOOCV",
    progress_every: int = 25,
) -> np.ndarray:
    loo = LeaveOneOut()
    y_prob = np.zeros(len(y), dtype=float)
    t0 = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X_signal), start=1):
        y_prob[test_idx] = fold_predict(X_signal, X_clinical, y, train_idx, test_idx, configs)
        if progress_every and (fold_idx % progress_every == 0 or fold_idx == len(y)):
            elapsed = time.time() - t0
            eta = elapsed / fold_idx * (len(y) - fold_idx)
            print(f"    {progress_label} fold {fold_idx:3d}/{len(y)} | elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min")
    return y_prob


def plot_roc(
    y: np.ndarray,
    curves: dict[str, np.ndarray],
    title: str,
    filename: str,
    output_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for label, prob in curves.items():
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        ax.plot(fpr, tpr, linewidth=2.5, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_diagnostic_plots(
    y: np.ndarray,
    prob: np.ndarray,
    model_label: str,
    output_prefix: str,
    output_dir: str,
    threshold: float = 0.5,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(y, prob)
    auc = roc_auc_score(y, prob)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.fill_between(fpr, tpr, alpha=0.14, color="#2a9d8f")
    ax.plot(fpr, tpr, lw=2.5, color="#2a9d8f", label=f"{model_label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC - {model_label}", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_roc.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=28, fontweight="bold", color=color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Rec", "Rec"])
    ax.set_yticklabels(["No Rec", "Rec"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_label}", fontweight="bold")
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y, prob)
    ap = average_precision_score(y, prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, lw=2.5, color="#2a9d8f", label=f"AP={ap:.3f}")
    ax.axhline(y.mean(), color="gray", ls="--", alpha=0.45, label=f"Prevalence={y.mean():.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall - {model_label}", fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_precision_recall.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    frac, mean_prob = calibration_curve(y, prob, n_bins=8, strategy="quantile")
    ax.plot(mean_prob, frac, "o-", lw=2.2, color="#2a9d8f", label=model_label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, label="Perfect calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed recurrence fraction")
    ax.set_title(f"Calibration - {model_label}", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_calibration.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    bins = np.linspace(0, 1, 25)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(prob[y == 0], bins=bins, alpha=0.62, color="#457b9d", label="No recurrence", edgecolor="white")
    ax.hist(prob[y == 1], bins=bins, alpha=0.62, color="#e76f51", label="Recurrence", edgecolor="white")
    ax.axvline(threshold, color="black", ls="--", alpha=0.55, label=f"Threshold={threshold:.2f}")
    ax.set_xlabel("Predicted recurrence probability")
    ax.set_ylabel("Count")
    ax.set_title(f"Prediction Distribution - {model_label}", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.22)
    fig.savefig(os.path.join(output_dir, f"{output_prefix}_prediction_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_feature_set(
    df: pd.DataFrame,
    feature_cols: list[str],
    y: np.ndarray,
    name: str,
    clinical_cols: list[str],
    n_boot: int,
    n_splits: int,
    n_repeats: int,
    drop_technical_counters: bool,
    run_loocv: bool = True,
    run_repeated: bool = True,
    exclude_prefixes: tuple[str, ...] = (),
    configs: list[ModelConfig] | None = None,
    output_dir: str | None = None,
    roc_filename: str | None = None,
    clinical_encoder: Callable[[pd.DataFrame, Iterable[str]], pd.DataFrame] = encode_clinical,
    clinical_encoding_name: str = "professor_missing0_levels1n",
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    configs = configs or V15_CONFIGS
    X_signal = _safe_numeric_frame(df[feature_cols].copy())
    X_signal = add_ecg_domain_features(X_signal)
    if drop_technical_counters:
        X_signal = X_signal.drop(columns=TECHNICAL_COUNTERS, errors="ignore")
    for prefix in exclude_prefixes:
        X_signal = X_signal[[c for c in X_signal.columns if not c.startswith(prefix)]]
    X_clinical = clinical_encoder(df, clinical_cols)

    print(f"\n{'=' * 72}")
    print(f"  {name}")
    print(f"{'=' * 72}")
    print(f"  Patients: {len(y)} ({int((y == 0).sum())} no recurrence, {int((y == 1).sum())} recurrence)")
    print(f"  Signal features before fold-local filtering: {X_signal.shape[1]}")
    print(f"  Clinical encoding: {clinical_encoding_name}")
    print(f"  Clinical bypass columns: {list(X_clinical.columns)}")
    print(f"  Model members: {len(configs)} ({', '.join(c.name for c in configs)})")

    if run_repeated:
        print(f"\n  [1/2] Repeated Stratified {n_splits}-Fold CV x {n_repeats} repeats")
        prob_rep, repeat_aucs = repeated_cv_predict(
            X_signal, X_clinical, y, configs, n_splits=n_splits, n_repeats=n_repeats
        )
        auc_rep = roc_auc_score(y, prob_rep)
        ci_lo, ci_hi = bootstrap_auc_ci(y, prob_rep, n_boot)
        metrics = threshold_metrics(y, prob_rep, threshold=0.5)
        print(
            f"  -> averaged-OOF AUC={auc_rep:.3f} [{ci_lo:.3f}-{ci_hi:.3f}] | "
            f"repeat mean={np.mean(repeat_aucs):.3f} +/- {np.std(repeat_aucs):.3f}"
        )
    else:
        print(f"\n  [1/2] Repeated Stratified {n_splits}-Fold CV x {n_repeats} repeats skipped")
        prob_rep = np.full(len(y), np.nan)
        repeat_aucs = []
        auc_rep = np.nan
        ci_lo, ci_hi = np.nan, np.nan
        metrics = {}

    if run_loocv:
        print("\n  [2/2] Leave-One-Out CV")
        prob_loo = loocv_predict(X_signal, X_clinical, y, configs, progress_label=name[:42])
        auc_loo = roc_auc_score(y, prob_loo)
        ci_lo_loo, ci_hi_loo = bootstrap_auc_ci(y, prob_loo, n_boot)
        print(f"  -> LOOCV AUC={auc_loo:.3f} [{ci_lo_loo:.3f}-{ci_hi_loo:.3f}]")
        if not run_repeated:
            auc_rep = auc_loo
            ci_lo, ci_hi = ci_lo_loo, ci_hi_loo
            metrics = threshold_metrics(y, prob_loo, threshold=0.5)
    else:
        print("\n  [2/2] Leave-One-Out CV skipped")
        prob_loo = np.full(len(y), np.nan)
        auc_loo = np.nan
        ci_lo_loo, ci_hi_loo = np.nan, np.nan

    out = {
        "AUC-ROC": auc_rep,
        "AUC_repeat_mean": float(np.mean(repeat_aucs)) if repeat_aucs else np.nan,
        "AUC_repeat_sd": float(np.std(repeat_aucs)) if repeat_aucs else np.nan,
        "AUC_LOOCV": auc_loo,
        "CI_lower": ci_lo,
        "CI_upper": ci_hi,
        "CI_lower_LOOCV": ci_lo_loo,
        "CI_upper_LOOCV": ci_hi_loo,
        "n_signal_features_input": int(X_signal.shape[1]),
        "n_model_members": int(len(configs)),
        "clinical_encoding": clinical_encoding_name,
        "validation_mode": "LOOCV only" if (run_loocv and not run_repeated) else "Repeated CV + LOOCV",
    }
    out.update(metrics)
    if output_dir and roc_filename:
        curves = {}
        if not np.isnan(prob_rep).all():
            curves["Repeated K-fold"] = prob_rep
        if not np.isnan(prob_loo).all():
            curves["LOOCV"] = prob_loo
        if curves:
            plot_roc(y, curves, name, roc_filename, output_dir)
    return out, prob_rep, prob_loo


def run_model_group(
    df: pd.DataFrame,
    y: np.ndarray,
    pwave_cols: list[str],
    ecg_cols: list[str],
    preop_clinical_cols: list[str],
    blanking_clinical_cols: list[str],
    args: argparse.Namespace,
    model_slug: str,
    model_label: str,
    configs: list[ModelConfig],
    output_dir: str,
    clinical_encoder: Callable[[pd.DataFrame, Iterable[str]], pd.DataFrame],
    clinical_encoding_name: str,
    primary_only: bool = False,
    run_repeated: bool = True,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    sensitivity_roc_dir = os.path.join(output_dir, "sensitivity_roc")
    os.makedirs(sensitivity_roc_dir, exist_ok=True)

    result_rows = {}
    prediction_payload = {
        "patient_id": df["patient_id"].astype(str).tolist(),
        "y": y.tolist(),
        "model_label": model_label,
        "model_slug": model_slug,
        "clinical_encoding": clinical_encoding_name,
    }

    feature_sets = [
        (
            f"{model_slug}_PWave_NoLit_Clinical_Preop",
            pwave_cols + [c for c in TECHNICAL_COUNTERS if c in df.columns],
            True,
            f"Primary pre-op: {model_label} on P-wave ANOVA k=18 + type_af/redo, lit excluded",
            ("lit_",),
            preop_clinical_cols,
        ),
        (
            f"{model_slug}_PWave_NoLit_Clinical_WithBlanking",
            pwave_cols + [c for c in TECHNICAL_COUNTERS if c in df.columns],
            True,
            f"Post-procedural sensitivity: {model_label} + blanking_period",
            ("lit_",),
            blanking_clinical_cols,
        ),
        (
            f"{model_slug}_PWave_LitSensitivity_Preop",
            pwave_cols + [c for c in TECHNICAL_COUNTERS if c in df.columns],
            True,
            f"Sensitivity: {model_label} with literature transforms, pre-op clinical only",
            (),
            preop_clinical_cols,
        ),
        (
            f"{model_slug}_ECG_NoLit_Clinical_Preop",
            ecg_cols,
            True,
            f"Sensitivity: {model_label} ECG + pre-op clinical, lit excluded",
            ("lit_",),
            preop_clinical_cols,
        ),
    ]
    if primary_only:
        feature_sets = feature_sets[:1]

    for key, cols, drop_counters, title, exclude_prefixes, clinical_cols in feature_sets:
        res, prob_rep, prob_loo = evaluate_feature_set(
            df,
            cols,
            y,
            title,
            clinical_cols,
            args.boot,
            args.splits,
            args.repeats,
            drop_technical_counters=drop_counters,
            run_loocv=not args.skip_loocv,
            run_repeated=run_repeated,
            exclude_prefixes=exclude_prefixes,
            configs=configs,
            output_dir=sensitivity_roc_dir,
            roc_filename=f"roc_{key.lower()}.png",
            clinical_encoder=clinical_encoder,
            clinical_encoding_name=clinical_encoding_name,
        )
        result_rows[key] = res
        prediction_payload[f"{key}_repeated"] = prob_rep.tolist()
        prediction_payload[f"{key}_loocv"] = prob_loo.tolist()

    results_df = pd.DataFrame(result_rows).T
    results_path = os.path.join(output_dir, f"results_v16_{model_slug}.csv")
    results_df.to_csv(results_path)

    blanking_rows = [
        f"{model_slug}_PWave_NoLit_Clinical_Preop",
        f"{model_slug}_PWave_NoLit_Clinical_WithBlanking",
    ]
    if all(row in results_df.index for row in blanking_rows):
        blanking_df = results_df.loc[blanking_rows].copy()
        blanking_df["Delta_vs_preop_AUC"] = (
            blanking_df["AUC-ROC"] - float(results_df.loc[blanking_rows[0], "AUC-ROC"])
        )
        blanking_df.to_csv(os.path.join(output_dir, f"v16_{model_slug}_blanking_comparison.csv"))

    primary_rep_key = f"{model_slug}_PWave_NoLit_Clinical_Preop_repeated"
    primary_loo_key = f"{model_slug}_PWave_NoLit_Clinical_Preop_loocv"
    primary_rep = np.asarray(prediction_payload[primary_rep_key], dtype=float)
    primary_loo = np.asarray(prediction_payload[primary_loo_key], dtype=float)
    if not np.isnan(primary_rep).all():
        threshold_sweep(y, primary_rep).to_csv(os.path.join(output_dir, f"v16_{model_slug}_threshold_sweep_repeated.csv"))

    if not np.isnan(primary_loo).all():
        threshold_sweep(y, primary_loo).to_csv(os.path.join(output_dir, f"v16_{model_slug}_threshold_sweep_loocv.csv"))
        diagnostic_prob = primary_loo
        diagnostic_suffix = "LOOCV"
    elif not np.isnan(primary_rep).all():
        diagnostic_prob = primary_rep
        diagnostic_suffix = "Repeated CV"
    else:
        diagnostic_prob = np.full(len(y), np.nan)
        diagnostic_suffix = "No CV"

    if not np.isnan(diagnostic_prob).all():
        save_diagnostic_plots(
            y,
            diagnostic_prob,
            f"{model_label} ({diagnostic_suffix})",
            f"v16_{model_slug}",
            output_dir,
        )

    pred_path = os.path.join(output_dir, f"predictions_v16_{model_slug}.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(prediction_payload, f, indent=2)

    print(f"\nSaved {model_label} results: {results_path}")
    print(f"Saved {model_label} predictions: {pred_path}")
    return results_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=500)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--skip-loocv",
        action="store_true",
        help="Run only repeated K-fold CV. Useful for fast experiments.",
    )
    parser.add_argument(
        "--loocv-only",
        action="store_true",
        help="Skip repeated K-fold CV and use LOOCV as the primary evaluation.",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Run only the primary pre-operative P-wave model for each model family.",
    )
    args = parser.parse_args()
    if args.skip_loocv and args.loocv_only:
        parser.error("--skip-loocv and --loocv-only cannot be used together.")

    for folder in (OUTPUT_DIR, ANOVA_OUTPUT_DIR, GNB_OUTPUT_DIR, BERNOULLI_OUTPUT_DIR):
        os.makedirs(folder, exist_ok=True)

    print("AF Recurrence - ML v16 Training")
    print("Leakage-strict fold-local preprocessing; explicit pre-op vs blanking comparison")
    print("=" * 72)

    if not os.path.exists(CACHE_FILE):
        print(f"Cache not found: {CACHE_FILE}")
        sys.exit(1)

    with open(CACHE_FILE, "rb") as f:
        df = pickle.load(f)

    df = df.drop_duplicates(subset=["patient_id"]).dropna(subset=["recurrence"]).copy()
    y = df["recurrence"].values.astype(int)

    ecg_cols, egm_cols, all_clinical_cols = split_feature_sets(df)
    preop_clinical_cols = [c for c in PREOP_CLINICAL if c in df.columns]
    blanking_clinical_cols = [c for c in BLANKING_CLINICAL if c in df.columns]
    pwave_cols = [c for c in ecg_cols if c.startswith("pw_") or c == "PTFV1"]

    print(f"\nPatients: {len(df)} ({int((y == 0).sum())} no recurrence, {int((y == 1).sum())} recurrence)")
    print(f"ECG columns: {len(ecg_cols)} | P-wave columns: {len(pwave_cols)} | EGM/substrate columns: {len(egm_cols)}")
    print("Primary model excludes blanking_period. Rows marked WithBlanking are post-procedural sensitivity only.")
    print("Always excluded from feature matrix: source, patient_id, recurrence")
    print("Chosen clinical encodings:")
    print("  GNB fixed: professor_missing0_levels1n (best GNB LOOCV in V16 comparison)")
    print("  Bernoulli tuned: legacy_value_plus_missing_indicator (best Bernoulli LOOCV in V16 comparison)")

    gnb_results = run_model_group(
        df,
        y,
        pwave_cols,
        ecg_cols,
        preop_clinical_cols,
        blanking_clinical_cols,
        args,
        "gnb_fixed",
        "GaussianNB fixed: ANOVA k=18, var_smoothing=0.1, clinical missing=0",
        V15_CONFIGS,
        GNB_OUTPUT_DIR,
        encode_clinical,
        "professor_missing0_levels1n",
        primary_only=args.primary_only,
        run_repeated=not args.loocv_only,
    )

    bernoulli_results = run_model_group(
        df,
        y,
        pwave_cols,
        ecg_cols,
        preop_clinical_cols,
        blanking_clinical_cols,
        args,
        "bernoulli_tuned",
        "BernoulliNB tuned: ANOVA k=18, fold-local alpha/binarize, clinical *_missing",
        BERNOULLI_TUNED_CONFIGS,
        BERNOULLI_OUTPUT_DIR,
        encode_clinical_legacy_missing_indicators,
        "legacy_value_plus_missing_indicator",
        primary_only=args.primary_only,
        run_repeated=not args.loocv_only,
    )

    print(f"\n{'=' * 72}")
    print("ML v16 - RESULTS")
    print(f"{'=' * 72}")
    for group_name, df_group in (("GNB fixed", gnb_results), ("Bernoulli tuned", bernoulli_results)):
        print(f"\n{group_name}")
        for model_name, row in df_group.iterrows():
            loocv = row["AUC_LOOCV"]
            loocv_text = "skipped" if pd.isna(loocv) else f"{loocv:.3f}"
            if pd.isna(row.get("AUC_repeat_mean", np.nan)):
                print(
                    f"  {model_name:<48s} LOOCV AUC={loocv_text} "
                    f"[{row['CI_lower_LOOCV']:.3f}-{row['CI_upper_LOOCV']:.3f}]"
                )
            else:
                print(
                    f"  {model_name:<48s} Repeated AUC={row['AUC-ROC']:.3f} "
                    f"[{row['CI_lower']:.3f}-{row['CI_upper']:.3f}] | LOOCV={loocv_text}"
                )
    print(f"\nGNB fixed folder: {GNB_OUTPUT_DIR}")
    print(f"Bernoulli tuned folder: {BERNOULLI_OUTPUT_DIR}")
    print(f"ANOVA feature folder: {ANOVA_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
