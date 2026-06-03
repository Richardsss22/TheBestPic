#!/usr/bin/env python3
"""
ml_v16_extract.py — ML v16: P-Wave Focused AF Recurrence Prediction

Key innovations vs v3/v4:
  1. No blanking period filtering (all patients included)
  2. No alignment needed (data already aligned)
  3. 30+ novel P-wave features (PTFV1, dispersion, morphology, variability)
  4. Dual feature sets: ECG-only vs ECG+EGM for comparison
  5. Multi-lead P-wave analysis across II, V1, aVL, aVF, I

Usage:
    python3 ml_v16_extract.py
"""

import os, sys, pickle, time, warnings, traceback
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import h5py
from scipy import signal as sig
from scipy import stats
from scipy.stats import entropy as scipy_entropy
import pywt
from ts2vg import NaturalVG

# Data paths — Dados_sofia1 = original share, Dados_sofia2/new = new patients
DATA_ROOT = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Ambiente de Trabalho', 'dados_t_pic')
SHARE_PATH = os.path.join(DATA_ROOT, 'Dados_sofia1')
NEW_PATH = os.path.join(DATA_ROOT, 'Dados_sofia2', 'new')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(SCRIPT_DIR, 'features_v16_cache.pkl')


# ═══════════════════════════════════════════════════════════
#  SIGNAL PROCESSING UTILITIES
# ═══════════════════════════════════════════════════════════

def safe_bandpass(signal_1d, flo, fhi, fs=1000, order=3):
    """Apply bandpass filter safely."""
    try:
        if len(signal_1d) < 50:
            return np.zeros_like(signal_1d)
        sos = sig.butter(order, [flo, fhi], btype='band', fs=fs, output='sos')
        return sig.sosfiltfilt(sos, signal_1d)
    except:
        return np.zeros_like(signal_1d)

def safe_lowpass(signal_1d, fhi, fs=1000, order=3):
    """Apply lowpass filter safely."""
    try:
        if len(signal_1d) < 50:
            return np.zeros_like(signal_1d)
        sos = sig.butter(order, fhi, btype='low', fs=fs, output='sos')
        return sig.sosfiltfilt(sos, signal_1d)
    except:
        return np.zeros_like(signal_1d)

def safe_highpass(signal_1d, flo, fs=1000, order=3):
    """Apply highpass filter safely."""
    try:
        if len(signal_1d) < 50:
            return np.zeros_like(signal_1d)
        sos = sig.butter(order, flo, btype='high', fs=fs, output='sos')
        return sig.sosfiltfilt(sos, signal_1d)
    except:
        return np.zeros_like(signal_1d)


# ═══════════════════════════════════════════════════════════
#  R-PEAK AND P-WAVE DETECTION
# ═══════════════════════════════════════════════════════════

def detect_r_peaks(ecg_lead, fs=1000):
    """Detect R-peaks using bandpass + squared derivative approach."""
    try:
        # Bandpass 5-15 Hz (QRS complex energy)
        filt = safe_bandpass(ecg_lead, 5, 15, fs, order=4)
        sq = filt ** 2
        thr = np.percentile(sq, 80)
        peaks, props = sig.find_peaks(sq, distance=int(0.4 * fs), height=thr)
        
        # Refine: find actual R-peak in raw signal near each detected peak
        refined = []
        for p in peaks:
            window = int(0.05 * fs)  # 50ms window
            lo = max(0, p - window)
            hi = min(len(ecg_lead), p + window)
            segment = ecg_lead[lo:hi]
            # R-peak is the max absolute deflection
            local_peak = lo + np.argmax(np.abs(segment))
            refined.append(local_peak)
        return np.array(refined)
    except:
        return np.array([])


def detect_pwave_segment(ecg_beat_segment, fs=1000):
    """
    Given an ECG segment centered on R-peak, detect the P-wave region.
    
    Convention: The segment is [R-300ms : R+200ms].
    P-wave typically occurs 120-200ms before R-peak, lasting 80-120ms.
    
    Returns: (p_start, p_end, p_peak) indices relative to segment start, or None.
    """
    try:
        # P-wave is in the region 250ms to 80ms before R-peak
        # In our segment (R-300ms to R+200ms), R is at index 300 (at 1kHz)
        r_idx = int(0.3 * fs)  # R-peak position in segment
        
        # P-wave search window: 250ms to 80ms before R
        p_search_start = r_idx - int(0.28 * fs)  # 280ms before R
        p_search_end = r_idx - int(0.08 * fs)    # 80ms before R
        
        if p_search_start < 0:
            p_search_start = 0
        if p_search_end <= p_search_start + 10:
            return None
        
        # Low-pass filter to isolate P-wave (< 15 Hz)
        p_region = ecg_beat_segment[p_search_start:p_search_end].copy()
        if len(p_region) < 20:
            return None
        
        p_filt = safe_lowpass(p_region, 15, fs, order=3)
        
        if np.std(p_filt) < 1e-6:
            return None
        
        # Find the dominant deflection (P-wave peak)
        # P-wave can be positive (Lead II) or biphasic (V1)
        pos_peak = np.argmax(p_filt)
        neg_peak = np.argmin(p_filt)
        
        # Use the peak with larger absolute amplitude
        if abs(p_filt[pos_peak]) >= abs(p_filt[neg_peak]):
            p_peak_local = pos_peak
        else:
            p_peak_local = neg_peak
        
        p_peak = p_search_start + p_peak_local
        
        # Estimate P-wave onset and offset using derivative zero-crossings
        # Onset: look backwards from peak for the start
        deriv = np.diff(p_filt)
        
        # P-wave onset: where signal starts rising/falling before peak
        onset_local = 0
        for i in range(p_peak_local - 1, 0, -1):
            if i < len(deriv):
                # Look for near-zero derivative or sign change
                if abs(p_filt[i]) < abs(p_filt[p_peak_local]) * 0.15:
                    onset_local = i
                    break
        
        # P-wave offset: where signal returns to baseline after peak
        offset_local = len(p_filt) - 1
        for i in range(p_peak_local + 1, len(p_filt)):
            if abs(p_filt[i]) < abs(p_filt[p_peak_local]) * 0.15:
                offset_local = i
                break
        
        p_start = p_search_start + onset_local
        p_end = p_search_start + offset_local
        
        # Sanity checks
        p_duration_ms = (p_end - p_start) / fs * 1000
        if p_duration_ms < 30 or p_duration_ms > 200:
            return None
        
        return (p_start, p_end, p_peak)
    except:
        return None


# ═══════════════════════════════════════════════════════════
#  P-WAVE FEATURE EXTRACTION (NOVEL — CORE V5)
# ═══════════════════════════════════════════════════════════

def extract_pwave_features_single_beat(ecg_segment, fs=1000, lead_name='II'):
    """Extract P-wave features from a single beat ECG segment."""
    feats = {}
    prefix = f'pw_{lead_name}_'
    
    pwave = detect_pwave_segment(ecg_segment, fs)
    if pwave is None:
        # Return NaN for all features
        for k in ['amp', 'amp_abs', 'duration_ms', 'area_pos', 'area_neg', 'area_total',
                   'ptf', 'morphology_peaks', 'rms', 'slope_onset', 'slope_offset',
                   'symmetry', 'pr_interval_ms', 'p_qrs_ratio', 'spectral_dom',
                   'spectral_entropy', 'peak_sharpness', 'cwt_energy', 'cwt_max_power',
                   'nvg_degree_entropy', 'nvg_mean_degree']:
            feats[prefix + k] = np.nan
        return feats
    
    p_start, p_end, p_peak = pwave
    r_idx = int(0.3 * fs)
    p_segment = ecg_segment[p_start:p_end + 1]
    
    if len(p_segment) < 5:
        for k in ['amp', 'amp_abs', 'duration_ms', 'area_pos', 'area_neg', 'area_total',
                   'ptf', 'morphology_peaks', 'rms', 'slope_onset', 'slope_offset',
                   'symmetry', 'pr_interval_ms', 'p_qrs_ratio', 'spectral_dom',
                   'spectral_entropy', 'peak_sharpness', 'cwt_energy', 'cwt_max_power',
                   'nvg_degree_entropy', 'nvg_mean_degree']:
            feats[prefix + k] = np.nan
        return feats
    
    # Low-pass the segment for clean analysis
    p_clean = safe_lowpass(p_segment, 15, fs, order=3) if len(p_segment) > 20 else p_segment
    
    # 1. P-wave Amplitude
    feats[prefix + 'amp'] = ecg_segment[p_peak]
    feats[prefix + 'amp_abs'] = abs(ecg_segment[p_peak])
    
    # 2. P-wave Duration (ms)
    feats[prefix + 'duration_ms'] = (p_end - p_start) / fs * 1000
    
    # 3. P-wave Area (positive and negative components)
    baseline = (p_clean[0] + p_clean[-1]) / 2
    p_centered = p_clean - baseline
    feats[prefix + 'area_pos'] = np.sum(p_centered[p_centered > 0]) / fs
    feats[prefix + 'area_neg'] = np.sum(p_centered[p_centered < 0]) / fs
    feats[prefix + 'area_total'] = np.sum(np.abs(p_centered)) / fs
    
    # 4. P-wave Terminal Force (PTF) — amplitude × duration of negative component
    # Critical in V1 (PTFV1), but computed for all leads
    neg_component = p_centered[p_centered < 0]
    if len(neg_component) > 0:
        neg_amp = np.min(neg_component)  # Most negative value
        neg_dur = len(neg_component) / fs * 1000  # ms
        feats[prefix + 'ptf'] = abs(neg_amp) * neg_dur  # mV * ms
    else:
        feats[prefix + 'ptf'] = 0.0
    
    # 5. P-wave Morphology — number of distinct peaks (notching)
    try:
        peaks_pos, _ = sig.find_peaks(p_clean, distance=int(0.02 * fs))
        peaks_neg, _ = sig.find_peaks(-p_clean, distance=int(0.02 * fs))
        feats[prefix + 'morphology_peaks'] = len(peaks_pos) + len(peaks_neg)
    except:
        feats[prefix + 'morphology_peaks'] = np.nan
    
    # 6. P-wave RMS
    feats[prefix + 'rms'] = np.sqrt(np.mean(p_clean**2))
    
    # 7. P-wave Onset and Offset Slopes
    peak_local = p_peak - p_start
    onset_segment = p_clean[:max(peak_local, 1)]
    offset_segment = p_clean[peak_local:]
    
    if len(onset_segment) > 2:
        feats[prefix + 'slope_onset'] = (onset_segment[-1] - onset_segment[0]) / (len(onset_segment) / fs)
    else:
        feats[prefix + 'slope_onset'] = np.nan
    
    if len(offset_segment) > 2:
        feats[prefix + 'slope_offset'] = (offset_segment[-1] - offset_segment[0]) / (len(offset_segment) / fs)
    else:
        feats[prefix + 'slope_offset'] = np.nan
    
    # 8. P-wave Symmetry (ratio of onset to offset duration)
    onset_dur = peak_local
    offset_dur = len(p_clean) - peak_local
    feats[prefix + 'symmetry'] = onset_dur / (offset_dur + 1e-6)
    
    # 9. PR Interval (P-peak to R-peak)
    feats[prefix + 'pr_interval_ms'] = (r_idx - p_peak) / fs * 1000
    
    # 10. P/QRS Amplitude Ratio
    qrs_region = ecg_segment[r_idx - int(0.04 * fs):r_idx + int(0.04 * fs)]
    qrs_amp = np.ptp(qrs_region) if len(qrs_region) > 0 else 1e-6
    feats[prefix + 'p_qrs_ratio'] = feats[prefix + 'amp_abs'] / (qrs_amp + 1e-6)
    
    # 11. P-wave Spectral Features
    try:
        if len(p_clean) > 30:
            freqs, psd = sig.welch(p_clean, fs, nperseg=min(64, len(p_clean)))
            feats[prefix + 'spectral_dom'] = freqs[np.argmax(psd)]
            psd_n = psd / (np.sum(psd) + 1e-12)
            feats[prefix + 'spectral_entropy'] = scipy_entropy(psd_n + 1e-12)
        else:
            feats[prefix + 'spectral_dom'] = np.nan
            feats[prefix + 'spectral_entropy'] = np.nan
    except:
        feats[prefix + 'spectral_dom'] = np.nan
        feats[prefix + 'spectral_entropy'] = np.nan
    
    # 12. P-wave Peak Sharpness (curvature at peak)
    if p_peak > 0 and p_peak < len(ecg_segment) - 1:
        feats[prefix + 'peak_sharpness'] = abs(
            ecg_segment[p_peak - 1] - 2 * ecg_segment[p_peak] + ecg_segment[p_peak + 1]
        )
    else:
        feats[prefix + 'peak_sharpness'] = np.nan
    
    # 13. CWT Features (Complex Morlet)
    try:
        if len(p_segment) > 10:
            wavelet = 'cmor1.5-1.0'
            frequencies = np.arange(5, 31, 1)
            scales = pywt.frequency2scale(wavelet, frequencies / fs)
            coefs, freqs_cwt = pywt.cwt(p_segment, scales, wavelet, sampling_period=1/fs)
            power_scalogram = np.abs(coefs)**2
            feats[prefix + 'cwt_energy'] = np.sum(power_scalogram)
            feats[prefix + 'cwt_max_power'] = np.max(power_scalogram)
        else:
            feats[prefix + 'cwt_energy'] = np.nan
            feats[prefix + 'cwt_max_power'] = np.nan
    except:
        feats[prefix + 'cwt_energy'] = np.nan
        feats[prefix + 'cwt_max_power'] = np.nan
        
    # 14. NVG Features (Natural Visibility Graph)
    try:
        if len(p_segment) > 10:
            vg = NaturalVG(directed=None)
            vg.build(p_segment)
            degrees = vg.degrees
            degree_counts = np.bincount(degrees)
            probs = degree_counts[degree_counts > 0] / len(degrees)
            feats[prefix + 'nvg_degree_entropy'] = -np.sum(probs * np.log2(probs))
            feats[prefix + 'nvg_mean_degree'] = np.mean(degrees)
        else:
            feats[prefix + 'nvg_degree_entropy'] = np.nan
            feats[prefix + 'nvg_mean_degree'] = np.nan
    except:
        feats[prefix + 'nvg_degree_entropy'] = np.nan
        feats[prefix + 'nvg_mean_degree'] = np.nan
    
    return feats


def extract_pwave_multilead(ecg_data, lead_indices, lead_names, r_peaks, fs=1000):
    """
    Extract P-wave features across multiple leads, including inter-lead features.
    
    Returns dict of aggregated P-wave features.
    """
    features = {}
    
    window_pre = int(0.3 * fs)   # 300ms before R
    window_post = int(0.2 * fs)  # 200ms after R
    
    # Per-lead P-wave features (aggregated across beats)
    lead_durations = {}  # For P-wave dispersion calculation
    lead_ptfs = {}       # For PTFV1 analysis
    
    for lead_name, lead_idx in zip(lead_names, lead_indices):
        if lead_idx is None:
            continue
        
        ecg_lead = ecg_data[lead_idx, :, 0]  # ECG is same across all points
        
        beat_feats_list = []
        beat_durations = []
        
        valid_beats = [r for r in r_peaks if r >= window_pre and r + window_post < len(ecg_lead)]
        
        for r in valid_beats[:15]:  # Max 15 beats per lead
            segment = ecg_lead[r - window_pre: r + window_post]
            bf = extract_pwave_features_single_beat(segment, fs, lead_name)
            beat_feats_list.append(bf)
            
            dur_key = f'pw_{lead_name}_duration_ms'
            if dur_key in bf and not np.isnan(bf[dur_key]):
                beat_durations.append(bf[dur_key])
        
        if not beat_feats_list:
            continue
        
        # Aggregate beat features: mean and std (beat-to-beat variability)
        df_beats = pd.DataFrame(beat_feats_list)
        for col in df_beats.columns:
            vals = df_beats[col].dropna()
            if len(vals) > 0:
                features[f'{col}_mean'] = vals.mean()
                features[f'{col}_std'] = vals.std() if len(vals) > 1 else 0.0
                features[f'{col}_cv'] = vals.std() / (abs(vals.mean()) + 1e-6) if len(vals) > 1 else 0.0
        
        # Store for inter-lead analysis
        if beat_durations:
            lead_durations[lead_name] = np.mean(beat_durations)
            ptf_key = f'pw_{lead_name}_ptf_mean'
            if ptf_key in features:
                lead_ptfs[lead_name] = features[ptf_key]
    
    # ─── INTER-LEAD P-WAVE FEATURES ───
    
    # P-wave Dispersion: max duration - min duration across leads
    if len(lead_durations) >= 2:
        durations = list(lead_durations.values())
        features['pw_dispersion_ms'] = max(durations) - min(durations)
        features['pw_duration_range'] = np.ptp(durations)
        features['pw_duration_cross_std'] = np.std(durations)
    else:
        features['pw_dispersion_ms'] = np.nan
        features['pw_duration_range'] = np.nan
        features['pw_duration_cross_std'] = np.nan
    
    # PTFV1 (the most critical P-wave marker for AF)
    if 'V1' in lead_ptfs:
        features['PTFV1'] = lead_ptfs['V1']
    else:
        features['PTFV1'] = np.nan
    
    # P-wave amplitude dispersion across leads
    amp_keys = [f'pw_{ln}_amp_abs_mean' for ln in lead_names if f'pw_{ln}_amp_abs_mean' in features]
    if len(amp_keys) >= 2:
        amps = [features[k] for k in amp_keys]
        features['pw_amp_dispersion'] = max(amps) - min(amps)
        features['pw_amp_cross_std'] = np.std(amps)
    else:
        features['pw_amp_dispersion'] = np.nan
        features['pw_amp_cross_std'] = np.nan
    
    # P-wave detection rate (how many beats have detectable P-waves)
    for lead_name in lead_names:
        dur_key = f'pw_{lead_name}_duration_ms_mean'
        if dur_key in features and not np.isnan(features.get(dur_key, np.nan)):
            pass  # P-wave detected
    
    return features


# ═══════════════════════════════════════════════════════════
#  ECG GLOBAL FEATURES (HRV, QRS, COMPLEXITY)
# ═══════════════════════════════════════════════════════════

def extract_hrv_features(r_peaks, fs=1000):
    """Extract HRV features from R-peak positions."""
    feats = {}
    try:
        if len(r_peaks) < 3:
            raise ValueError("Too few R-peaks")
        
        rr = np.diff(r_peaks) / fs  # RR intervals in seconds
        
        feats['hr_mean'] = 60.0 / np.mean(rr)
        feats['hr_std'] = 60.0 / np.std(rr) if np.std(rr) > 0 else 0
        feats['rr_mean_ms'] = np.mean(rr) * 1000
        feats['rr_std_ms'] = np.std(rr) * 1000
        feats['rr_cv'] = np.std(rr) / (np.mean(rr) + 1e-6)
        
        # RMSSD
        feats['rmssd'] = np.sqrt(np.mean(np.diff(rr)**2)) * 1000 if len(rr) > 1 else np.nan
        
        # SDNN
        feats['sdnn'] = np.std(rr) * 1000
        
        # pNN50
        nn_diff = np.abs(np.diff(rr)) * 1000  # in ms
        feats['pnn50'] = np.sum(nn_diff > 50) / max(len(nn_diff), 1) * 100
        feats['pnn20'] = np.sum(nn_diff > 20) / max(len(nn_diff), 1) * 100
        
        # Triangular Index
        hist, _ = np.histogram(rr, bins=max(5, len(rr) // 3))
        feats['tri_index'] = len(rr) / (np.max(hist) + 1)
        
        # RR irregularity measures
        feats['rr_range_ms'] = np.ptp(rr) * 1000
        feats['rr_iqr_ms'] = (np.percentile(rr, 75) - np.percentile(rr, 25)) * 1000
        
        # Successive differences entropy
        if len(nn_diff) > 5:
            hist_diff, _ = np.histogram(nn_diff, bins=max(5, len(nn_diff) // 3), density=True)
            hist_diff = hist_diff[hist_diff > 0]
            feats['rr_diff_entropy'] = -np.sum(hist_diff * np.log2(hist_diff + 1e-12))
        else:
            feats['rr_diff_entropy'] = np.nan
        
    except:
        for k in ['hr_mean', 'hr_std', 'rr_mean_ms', 'rr_std_ms', 'rr_cv', 'rmssd', 'sdnn',
                   'pnn50', 'pnn20', 'tri_index', 'rr_range_ms', 'rr_iqr_ms', 'rr_diff_entropy']:
            feats[k] = np.nan
    return feats


def extract_fwave_features(ecg_signal, fs=1000, lead_name='V1'):
    """Extract fibrillatory wave (f-wave) features from ECG."""
    feats = {}
    prefix = f'fw_{lead_name}_'
    try:
        # f-wave band: 4-9 Hz (atrial fibrillatory activity)
        fw = safe_bandpass(ecg_signal, 4, 9, fs)
        
        # Amplitude
        feats[prefix + 'rms'] = np.sqrt(np.mean(fw**2))
        feats[prefix + 'amp_max'] = np.max(np.abs(fw))
        feats[prefix + 'amp_p2p'] = np.ptp(fw)
        
        # Spectral analysis of f-wave
        freqs, psd = sig.welch(fw, fs, nperseg=min(512, len(fw)))
        total = np.sum(psd) + 1e-12
        
        feats[prefix + 'dom_freq'] = freqs[np.argmax(psd)]
        feats[prefix + 'regularity'] = np.max(psd) / total  # Organization index
        
        psd_n = psd / total
        feats[prefix + 'spectral_entropy'] = scipy_entropy(psd_n + 1e-12)
        
        # Bandwidth (half-power)
        half_power = np.max(psd) * 0.5
        above = freqs[psd > half_power]
        feats[prefix + 'bandwidth'] = np.ptp(above) if len(above) > 1 else 0
        
        # f-wave regularity via autocorrelation
        fw_norm = (fw - np.mean(fw)) / (np.std(fw) + 1e-12)
        autocorr = np.correlate(fw_norm, fw_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0] + 1e-12
        
        # Find first secondary peak in autocorrelation
        peaks, _ = sig.find_peaks(autocorr[int(0.05*fs):], distance=int(0.05*fs))
        if len(peaks) > 0:
            feats[prefix + 'autocorr_peak'] = autocorr[int(0.05*fs) + peaks[0]]
        else:
            feats[prefix + 'autocorr_peak'] = np.nan
            
    except:
        for k in ['rms', 'amp_max', 'amp_p2p', 'dom_freq', 'regularity',
                   'spectral_entropy', 'bandwidth', 'autocorr_peak']:
            feats[prefix + k] = np.nan
    return feats


def extract_ecg_morphology_features(ecg_signal, fs=1000, lead_name='II'):
    """Extract global ECG morphology and complexity features."""
    feats = {}
    prefix = f'ecg_{lead_name}_'
    try:
        # Signal complexity
        # Shannon entropy
        hist, _ = np.histogram(ecg_signal, bins=50, density=True)
        hist = hist[hist > 0]
        feats[prefix + 'shannon'] = -np.sum(hist * np.log2(hist + 1e-12))
        
        # Permutation entropy
        m = 3
        N = min(len(ecg_signal), 500)
        s = ecg_signal[:N]
        perms = []
        for i in range(N - m):
            perm = tuple(np.argsort(s[i:i+m]))
            perms.append(perm)
        unique, counts = np.unique(perms, axis=0, return_counts=True)
        probs = counts / len(perms)
        feats[prefix + 'perm_entropy'] = -np.sum(probs * np.log2(probs + 1e-12))
        
        # Hjorth parameters
        dy = np.diff(ecg_signal)
        ddy = np.diff(dy)
        var_y = np.var(ecg_signal) + 1e-12
        var_dy = np.var(dy) + 1e-12
        var_ddy = np.var(ddy) + 1e-12
        feats[prefix + 'hjorth_activity'] = var_y
        feats[prefix + 'hjorth_mobility'] = np.sqrt(var_dy / var_y)
        feats[prefix + 'hjorth_complexity'] = np.sqrt(var_ddy / var_dy) / (np.sqrt(var_dy / var_y) + 1e-12)
        
        # Line length (total signal variation)
        feats[prefix + 'line_length'] = np.sum(np.abs(dy))
        
        # Teager-Kaiser energy
        feats[prefix + 'teager_energy'] = np.mean(ecg_signal[1:-1]**2 - ecg_signal[:-2] * ecg_signal[2:])
        
        # Zero crossings
        m_val = np.mean(ecg_signal)
        feats[prefix + 'zero_crossings'] = np.sum(np.diff(np.sign(ecg_signal - m_val)) != 0)
        
        # RMS and P2P
        feats[prefix + 'rms'] = np.sqrt(np.mean(ecg_signal**2))
        feats[prefix + 'p2p'] = np.ptp(ecg_signal)
        
        # Kurtosis and Skewness
        std = np.std(ecg_signal) + 1e-12
        feats[prefix + 'kurtosis'] = stats.kurtosis(ecg_signal)
        feats[prefix + 'skewness'] = stats.skew(ecg_signal)
        
        # Spectral features
        freqs, psd = sig.welch(ecg_signal, fs, nperseg=min(512, len(ecg_signal)))
        total = np.sum(psd) + 1e-12
        feats[prefix + 'dom_freq'] = freqs[np.argmax(psd)]
        feats[prefix + 'spectral_entropy'] = scipy_entropy(psd / total + 1e-12)
        
        # Band power ratios
        af_mask = (freqs >= 4) & (freqs <= 9)
        feats[prefix + 'af_power_ratio'] = np.sum(psd[af_mask]) / total
        
        low_mask = (freqs >= 0.5) & (freqs <= 4)
        feats[prefix + 'low_power_ratio'] = np.sum(psd[low_mask]) / total
        
        high_mask = (freqs >= 10) & (freqs <= 50)
        feats[prefix + 'high_power_ratio'] = np.sum(psd[high_mask]) / total
        
        # P-wave band energy (1-10 Hz)
        pw_band = safe_bandpass(ecg_signal, 1, 10, fs)
        feats[prefix + 'pwave_band_energy'] = np.sum(pw_band**2) / len(pw_band)
        
    except:
        keys = ['shannon', 'perm_entropy', 'hjorth_activity', 'hjorth_mobility',
                'hjorth_complexity', 'line_length', 'teager_energy', 'zero_crossings',
                'rms', 'p2p', 'kurtosis', 'skewness', 'dom_freq', 'spectral_entropy',
                'af_power_ratio', 'low_power_ratio', 'high_power_ratio', 'pwave_band_energy']
        for k in keys:
            feats[prefix + k] = np.nan
    return feats


# ═══════════════════════════════════════════════════════════
#  EGM & SUBSTRATE FEATURES (for comparison model)
# ═══════════════════════════════════════════════════════════

def extract_voltage_features(v_bip, v_uni):
    """Extract voltage distribution features."""
    feats = {}
    try:
        feats['v_bip_mean'] = np.mean(v_bip)
        feats['v_bip_std'] = np.std(v_bip)
        feats['v_bip_median'] = np.median(v_bip)
        feats['v_bip_iqr'] = np.percentile(v_bip, 75) - np.percentile(v_bip, 25)
        feats['v_bip_skew'] = stats.skew(v_bip)
        feats['v_bip_kurt'] = stats.kurtosis(v_bip)
        feats['v_uni_mean'] = np.mean(v_uni)
        feats['v_uni_std'] = np.std(v_uni)
        
        # Percentage in voltage zones
        feats['scar_pct'] = np.mean(v_bip < 0.5) * 100
        feats['dense_scar_pct'] = np.mean(v_bip < 0.2) * 100
        feats['borderzone_pct'] = np.mean((v_bip >= 0.5) & (v_bip < 1.5)) * 100
        feats['healthy_pct'] = np.mean(v_bip >= 1.5) * 100
        feats['low_voltage_pct'] = np.mean(v_bip < 1.0) * 100
        feats['total_abnormal_pct'] = feats['scar_pct'] + feats['borderzone_pct']
        
        # Voltage histogram features (fine-grained distribution)
        bins = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf]
        hist, _ = np.histogram(v_bip, bins=bins)
        hist_pct = hist / (len(v_bip) + 1e-12) * 100
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            label = f'v_hist_{lo:.1f}_{hi:.1f}' if hi != np.inf else f'v_hist_{lo:.1f}_plus'
            feats[label.replace('.', 'p')] = hist_pct[i]
        
        # Gini coefficient (voltage inequality)
        feats['v_gini'] = np.sum(np.abs(np.subtract.outer(v_bip, v_bip))) / (2 * len(v_bip)**2 * np.mean(v_bip) + 1e-12)
        
        # Voltage entropy
        feats['v_entropy'] = scipy_entropy(hist + 1)
        
        # Bipolar/unipolar ratio
        feats['v_bip_uni_ratio'] = np.mean(v_bip) / (np.mean(v_uni) + 1e-6)
        
    except:
        pass
    return feats


def extract_lat_features(lat):
    """Extract LAT (Local Activation Time) features."""
    feats = {}
    try:
        valid = lat[lat > -5000]
        if len(valid) < 5:
            raise ValueError("Too few valid LAT values")
        
        feats['lat_range_ms'] = np.ptp(valid)
        feats['lat_std_ms'] = np.std(valid)
        feats['lat_iqr_ms'] = np.percentile(valid, 75) - np.percentile(valid, 25)
        feats['lat_cv'] = np.std(valid) / (abs(np.mean(valid)) + 1e-6)
        feats['lat_skew'] = stats.skew(valid)
        feats['lat_kurt'] = stats.kurtosis(valid)
        
        # LAT entropy (activation dispersion)
        hist, _ = np.histogram(valid, bins=20)
        feats['lat_entropy'] = scipy_entropy(hist + 1)
        
        # Percentage of "late activation" points
        late_threshold = np.percentile(valid, 90)
        feats['lat_late_pct'] = np.mean(valid > late_threshold) * 100
        
    except:
        for k in ['lat_range_ms', 'lat_std_ms', 'lat_iqr_ms', 'lat_cv', 'lat_skew',
                   'lat_kurt', 'lat_entropy', 'lat_late_pct']:
            feats[k] = np.nan
    return feats


def extract_egm_signal_features(egm, n_sample=100, fs=1000):
    """Extract EGM signal features from sampled points.
    
    Includes features from Lucas Felix [1] and Maria Nunes [8]:
    - Morphology: p2p, rms, line_length, n_deflections, Hjorth, Shannon
    - ZCR (Zero-Crossing Rate) — Lucas Felix [1]
    - ACL (Atrial Cycle Length) statistics — Felix/Nunes [1,7]
    - Spectral: DF, AF power, regularity, spectral entropy
    - OI (Organization Index) with ±0.5 Hz band — Maria Nunes [8]
    """
    feats = {}
    try:
        n_samples_sig, n_points = egm.shape
        sample_idx = np.linspace(0, n_points - 1, min(n_sample, n_points), dtype=int)
        
        all_morph = []
        all_spectral = []
        
        for idx in sample_idx:
            s = egm[:, idx]
            morph = {}
            
            # Morphology
            morph['p2p'] = np.ptp(s)
            morph['rms'] = np.sqrt(np.mean(s**2))
            
            # Amplitude mean and std (Felix [1]: ux, sigma_x)
            morph['amp_mean'] = np.mean(s)
            morph['amp_std'] = np.std(s)
            
            dy = np.diff(s)
            morph['line_length'] = np.sum(np.abs(dy))
            
            # Fractionation (number of deflections)
            noise_floor = np.percentile(np.abs(s), 25)
            thr = max(noise_floor * 3, 0.05)
            peaks, _ = sig.find_peaks(np.abs(s), height=thr, distance=20)
            morph['n_deflections'] = len(peaks)
            
            # ── ACL: Atrial Cycle Length statistics (Felix/Nunes [1,7]) ──
            # ACL = inter-peak intervals; CFAE defined as ACL <= 120ms
            if len(peaks) >= 3:
                acl_samples = np.diff(peaks)  # inter-peak intervals in samples
                acl_ms = acl_samples / fs * 1000  # convert to ms
                morph['acl_mean'] = np.mean(acl_ms)
                morph['acl_std'] = np.std(acl_ms)
                morph['acl_cv'] = np.std(acl_ms) / (np.mean(acl_ms) + 1e-6)
                morph['acl_min'] = np.min(acl_ms)
                morph['acl_max'] = np.max(acl_ms)
                # CFAE indicator: fraction of cycles with ACL <= 120ms
                morph['cfae_pct'] = np.mean(acl_ms <= 120) * 100
            else:
                morph['acl_mean'] = np.nan
                morph['acl_std'] = np.nan
                morph['acl_cv'] = np.nan
                morph['acl_min'] = np.nan
                morph['acl_max'] = np.nan
                morph['cfae_pct'] = np.nan
            
            # ── ZCR: Zero-Crossing Rate (Lucas Felix [1]) ──
            # ZCR = (1/N) * sum(|sign(x[n]) - sign(x[n-1])|) / 2
            sign_changes = np.diff(np.sign(s))
            morph['zcr'] = np.sum(sign_changes != 0) / (len(s) - 1)
            
            # Hjorth
            var_y = np.var(s) + 1e-12
            var_dy = np.var(dy) + 1e-12
            ddy = np.diff(dy)
            var_ddy = np.var(ddy) + 1e-12
            morph['hjorth_mob'] = np.sqrt(var_dy / var_y)
            morph['hjorth_comp'] = np.sqrt(var_ddy / var_dy) / (np.sqrt(var_dy / var_y) + 1e-12)
            
            # Shannon entropy
            hist, _ = np.histogram(s, bins=30, density=True)
            hist = hist[hist > 0]
            morph['shannon'] = -np.sum(hist * np.log2(hist + 1e-12))
            
            all_morph.append(morph)
            
            # Spectral
            spec = {}
            freqs, psd = sig.welch(s, fs, nperseg=min(512, len(s)))
            total = np.sum(psd) + 1e-12
            af_mask = (freqs >= 4) & (freqs <= 9)
            
            # Dominant Frequency (Felix [1])
            df_idx = np.argmax(psd)
            spec['dom_freq'] = freqs[df_idx]
            spec['af_power'] = np.sum(psd[af_mask]) / total
            
            # Total Power (Felix [1]: Ptotal)
            spec['total_power'] = np.sum(psd) * (freqs[1] - freqs[0]) if len(freqs) > 1 else 0
            
            # ── OI: Organization Index with ±0.5 Hz band (Maria Nunes [8]) ──
            # OI = power in [DF-0.5, DF+0.5] / total power in [3-20 Hz]
            df_freq = freqs[df_idx]
            oi_band = (freqs >= df_freq - 0.5) & (freqs <= df_freq + 0.5)
            analysis_band = (freqs >= 3) & (freqs <= 20)
            oi_power = np.sum(psd[oi_band])
            analysis_total = np.sum(psd[analysis_band]) + 1e-12
            spec['oi_band'] = oi_power / analysis_total
            
            # ── RI: Regularity Index (Nunes [8]) ──
            # RI = power in [DF-0.375, DF+0.375] / total power
            ri_band = (freqs >= df_freq - 0.375) & (freqs <= df_freq + 0.375)
            spec['ri'] = np.sum(psd[ri_band]) / total
            
            # Legacy regularity (max_psd / total)
            spec['regularity'] = np.max(psd) / total
            psd_n = psd / total
            spec['spec_entropy'] = scipy_entropy(psd_n + 1e-12)
            all_spectral.append(spec)
        
        # Aggregate
        df_morph = pd.DataFrame(all_morph)
        for col in df_morph.columns:
            feats[f'egm_{col}_mean'] = df_morph[col].mean()
            feats[f'egm_{col}_std'] = df_morph[col].std()
        
        df_spec = pd.DataFrame(all_spectral)
        for col in df_spec.columns:
            feats[f'egm_spec_{col}_mean'] = df_spec[col].mean()
            feats[f'egm_spec_{col}_std'] = df_spec[col].std()
        
        # Spatial heterogeneity (variation across points)
        rms_per_point = np.sqrt(np.mean(egm[:, sample_idx]**2, axis=0))
        feats['egm_spatial_rms_cv'] = np.std(rms_per_point) / (np.mean(rms_per_point) + 1e-12)
        
    except:
        pass
    return feats


# ═══════════════════════════════════════════════════════════
#  CROSS-MODAL FEATURES (EGM vs ECG)
# ═══════════════════════════════════════════════════════════

def extract_cross_modal_features(egm, ecg_lead, sample_idx, fs=1000):
    """Extract cross-modal (EGM vs ECG) features."""
    feats = {}
    try:
        all_xcorr = []
        all_coh = []
        
        ecg_signal = ecg_lead[:, 0]  # ECG same across all points
        
        for idx in sample_idx[:30]:
            egm_sig = egm[:, idx]
            
            # Normalize
            e1 = (egm_sig - np.mean(egm_sig)) / (np.std(egm_sig) + 1e-12)
            e2 = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-12)
            
            # Cross-correlation
            corr = np.correlate(e1, e2, mode='full') / len(e1)
            mid = len(corr) // 2
            sr = min(200, mid)
            local = corr[mid-sr:mid+sr]
            
            xcorr_max = np.max(np.abs(local))
            xcorr_lag = (np.argmax(np.abs(local)) - sr) / fs * 1000
            all_xcorr.append({'max': xcorr_max, 'lag': xcorr_lag})
            
            # Coherence in AF band
            try:
                freqs_c, coh = sig.coherence(egm_sig, ecg_signal, fs=fs, nperseg=256)
                af = (freqs_c >= 4) & (freqs_c <= 9)
                all_coh.append({
                    'coh_af': np.mean(coh[af]) if af.any() else np.nan,
                    'coh_mean': np.mean(coh)
                })
            except:
                all_coh.append({'coh_af': np.nan, 'coh_mean': np.nan})
        
        # Aggregate
        df_xc = pd.DataFrame(all_xcorr)
        feats['cross_xcorr_max_mean'] = df_xc['max'].mean()
        feats['cross_xcorr_max_std'] = df_xc['max'].std()
        feats['cross_xcorr_lag_mean'] = df_xc['lag'].mean()
        feats['cross_xcorr_lag_std'] = df_xc['lag'].std()
        
        df_co = pd.DataFrame(all_coh)
        feats['cross_coh_af_mean'] = df_co['coh_af'].mean()
        feats['cross_coh_af_std'] = df_co['coh_af'].std()
        feats['cross_coh_mean_mean'] = df_co['coh_mean'].mean()
        
    except:
        for k in ['cross_xcorr_max_mean', 'cross_xcorr_max_std', 'cross_xcorr_lag_mean',
                   'cross_xcorr_lag_std', 'cross_coh_af_mean', 'cross_coh_af_std', 'cross_coh_mean_mean']:
            feats[k] = np.nan
    return feats


# ═══════════════════════════════════════════════════════════
#  MAIN PATIENT FEATURE EXTRACTION (V5)
# ═══════════════════════════════════════════════════════════

def extract_patient_features_v5(mat_file, patient_id):
    """Extract all v5 features for a single patient."""
    features = {'patient_id': patient_id}
    
    try:
        f = h5py.File(mat_file, 'r')
        elec = f['userdata']['electric']
        
        egm = np.array(elec['egm'])
        ecg_data = np.array(elec['ecg'])
        v_bip = np.array(elec['voltages']['bipolar']).flatten()
        v_uni = np.array(elec['voltages']['unipolar']).flatten()
        lat = np.array(elec['annotations']['mapAnnot']).flatten()
        
        n_samples, n_points = egm.shape
        features['n_points'] = n_points
        features['n_samples'] = n_samples
        
        # ─── FIND ECG LEADS ───
        ecg_names = []
        for i in range(elec['ecgNames'].shape[0]):
            ref = elec['ecgNames'][i, 0]
            chars = np.array(f[ref]).flatten()
            ecg_names.append(''.join(chr(c) for c in chars))
        
        def find_lead(name):
            for i, n in enumerate(ecg_names):
                if n.split('(')[0].strip() == name:
                    return i
            return None
        
        lead_map = {
            'II': find_lead('II'),
            'V1': find_lead('V1'),
            'I': find_lead('I'),
            'aVF': find_lead('aVF'),
            'aVL': find_lead('aVL'),
            'V2': find_lead('V2'),
        }
        
        # ═══════════════════════════════════════════
        #  ECG-ONLY FEATURES (Tagged with 'ecg_')
        # ═══════════════════════════════════════════
        
        # 1. R-peak detection (use Lead II as reference)
        ref_lead = lead_map.get('II') or lead_map.get('I') or 0
        ecg_ref = ecg_data[ref_lead, :, 0]
        r_peaks = detect_r_peaks(ecg_ref)
        features['n_rpeaks'] = len(r_peaks)
        
        # 2. HRV Features (REMOVED - Unstable on short windows)
        # hrv = extract_hrv_features(r_peaks)
        # features.update(hrv)
        
        # 3. P-WAVE FEATURES (Main v5 Innovation)
        # Multi-lead P-wave analysis
        pw_lead_names = []
        pw_lead_indices = []
        for ln in ['II', 'V1', 'I', 'aVF', 'aVL']:
            idx = lead_map.get(ln)
            if idx is not None:
                pw_lead_names.append(ln)
                pw_lead_indices.append(idx)
        
        pw_feats = extract_pwave_multilead(ecg_data, pw_lead_indices, pw_lead_names, r_peaks)
        features.update(pw_feats)
        
        # 4. f-wave Features (V1 and II)
        for ln in ['V1', 'II']:
            idx = lead_map.get(ln)
            if idx is not None:
                fw = extract_fwave_features(ecg_data[idx, :, 0], lead_name=ln)
                features.update(fw)
        
        # 5. ECG Morphology + Complexity (II, V1)
        for ln in ['II', 'V1']:
            idx = lead_map.get(ln)
            if idx is not None:
                morph = extract_ecg_morphology_features(ecg_data[idx, :, 0], lead_name=ln)
                features.update(morph)
        
        # ═══════════════════════════════════════════
        #  EGM + SUBSTRATE FEATURES (Tagged with 'egm_', 'v_', 'lat_')
        # ═══════════════════════════════════════════
        
        # 6. Voltage Features
        volt_feats = extract_voltage_features(v_bip, v_uni)
        features.update(volt_feats)
        
        # 7. LAT Features
        lat_feats = extract_lat_features(lat)
        features.update(lat_feats)
        
        # 8. EGM Signal Features
        egm_feats = extract_egm_signal_features(egm, n_sample=100)
        features.update(egm_feats)
        
        # 9. Cross-Modal Features
        if ref_lead is not None:
            sample_idx = np.linspace(0, n_points - 1, min(100, n_points), dtype=int)
            cross_feats = extract_cross_modal_features(egm, ecg_data[ref_lead], sample_idx)
            features.update(cross_feats)
        
        f.close()
        
    except Exception as e:
        print(f'    ❌ Error patient {patient_id}: {e}')
        traceback.print_exc()
    
    return features


# ═══════════════════════════════════════════════════════════
#  BATCH EXTRACTION
# ═══════════════════════════════════════════════════════════

def main():
    # Use NEW database2.xlsx (superset of share)
    db_path = os.path.join(NEW_PATH, 'database2.xlsx')
    if not os.path.exists(db_path):
        print(f'❌ Database not found at {db_path}')
        sys.exit(1)
    
    db = pd.read_excel(db_path)
    
    print('🫀 AF Recurrence — ML v6 Feature Extraction')
    print('  P-Wave Focus | Dual Source (share + new) | ECG-Only Capability')
    print('=' * 70)
    print(f'  Database: {len(db)} entries ({db["Recurrence"].sum():.0f} recurrence, '
          f'{(db["Recurrence"]==0).sum()} no recurrence)')
    print('=' * 70)
    
    # Build patient → folder mapping (prefer share/, fallback to new/)
    pids = db['Number'].unique()
    pid_folders = {}
    for pid in pids:
        ps = str(int(pid))
        share_folder = os.path.join(SHARE_PATH, ps)
        new_folder = os.path.join(NEW_PATH, ps)
        if os.path.isdir(share_folder):
            pid_folders[pid] = ('share', share_folder)
        elif os.path.isdir(new_folder):
            pid_folders[pid] = ('new', new_folder)
    
    print(f'  Patients with signals: {len(pid_folders)} '
          f'(share: {sum(1 for s,_ in pid_folders.values() if s=="share")}, '
          f'new: {sum(1 for s,_ in pid_folders.values() if s=="new")})')
    print('=' * 70)
    
    all_feats = []
    errors = []
    
    for i, (pid, (source, folder)) in enumerate(pid_folders.items()):
        ps = str(int(pid))
        
        # Find largest .mat file
        mats = []
        for r, d, fs in os.walk(folder):
            for fname in fs:
                if fname.endswith('.mat'):
                    fp = os.path.join(r, fname)
                    mats.append((fp, os.path.getsize(fp)))
        if not mats:
            continue
        
        mat = sorted(mats, key=lambda x: x[1], reverse=True)[0][0]
        sz = os.path.getsize(mat) / 1e6
        
        print(f'  [{i+1}/{len(pid_folders)}] {ps} [{source}] ({sz:.0f}MB)...', end=' ', flush=True)
        t0 = time.time()
        
        try:
            feats = extract_patient_features_v5(mat, ps)
            
            # Add clinical metadata
            row = db[db['Number'] == pid]
            if len(row):
                feats['recurrence'] = int(row['Recurrence'].values[0])
                feats['type_af'] = row['Type_AF'].values[0] if pd.notnull(row['Type_AF'].values[0]) else np.nan
                feats['redo'] = row['Redo'].values[0] if pd.notnull(row['Redo'].values[0]) else np.nan
                feats['blanking_period'] = row['Blanking_Period'].values[0] if pd.notnull(row['Blanking_Period'].values[0]) else np.nan
                feats['source'] = source  # Track origin for debugging
            
            all_feats.append(feats)
            
            ecg_only = len([k for k in feats if any(k.startswith(p) for p in 
                          ['pw_', 'fw_', 'ecg_', 'PTFV1', 'n_rpeaks'])])
            egm_sub = len([k for k in feats if any(k.startswith(p) for p in 
                         ['v_', 'scar', 'dense', 'border', 'healthy', 'low_volt', 'total_abn',
                          'lat_', 'egm_', 'cross_'])])
            
            elapsed = time.time() - t0
            print(f'✅ {ecg_only} ECG + {egm_sub} EGM/Sub features, {elapsed:.1f}s')
            
        except Exception as e:
            errors.append(ps)
            print(f'❌ {e}')
    
    # Save
    df = pd.DataFrame(all_feats)
    with open(CACHE_FILE, 'wb') as fout:
        pickle.dump(df, fout)
    
    # Summary
    meta_cols = ['patient_id', 'recurrence', 'type_af', 'redo', 'blanking_period', 'source']
    feat_cols = [c for c in df.columns if c not in meta_cols]
    
    ecg_cols = [c for c in feat_cols if any(c.startswith(p) for p in 
                ['pw_', 'fw_', 'ecg_', 'PTFV1', 'n_rpeaks', 'n_samples'])]
    egm_cols = [c for c in feat_cols if c not in ecg_cols and c not in ['n_points']]
    
    print(f'\n{"=" * 70}')
    print(f'  ✅ ML v6 EXTRACTION COMPLETE')
    print(f'{"=" * 70}')
    print(f'  Patients processed: {len(all_feats)} (from share: {sum(1 for f in all_feats if f.get("source")=="share")}, '
          f'from new: {sum(1 for f in all_feats if f.get("source")=="new")})')
    print(f'  Errors: {len(errors)}')
    print(f'  Recurrence: {(df["recurrence"]==1).sum()} rec / {(df["recurrence"]==0).sum()} no rec')
    print(f'  Total features: {len(feat_cols)}')
    print(f'    ECG-only features: {len(ecg_cols)}')
    print(f'    EGM/Substrate features: {len(egm_cols)}')
    print(f'  Cache saved to: {CACHE_FILE}')
    
    if errors:
        print(f'  Failed patients: {", ".join(errors)}')
    
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()
