"""
EEG Motor-Imagery Decoding Pipeline
===================================
Dataset : PhysioNet EEG Motor Movement/Imagery Database (eegmmidb)
Task    : Decode LEFT-fist vs RIGHT-fist motor imagery (runs 4, 8, 12)
Method  : Common Spatial Patterns (CSP) + Linear Discriminant Analysis (LDA)
          with 5-fold stratified cross-validation, per subject.

Requires: mne, scikit-learn, pooch, numpy, scipy, matplotlib
Run      : python eeg_decoding_pipeline.py
"""
import os
import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, pick_types
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score

mne.set_log_level("WARNING")

# ---- Configuration ----
SUBJECTS = [1, 2, 3, 4, 5]
RUNS     = [4, 8, 12]          # imagine opening/closing left vs right fist
TMIN, TMAX = 1.0, 4.0          # motor-imagery window (s) after cue
FMIN, FMAX = 7.0, 30.0         # mu (8-12 Hz) + beta (13-30 Hz) rhythm
DATA = os.path.abspath("mne_data")

# T1 (id 2) = left fist, T2 (id 3) = right fist
EVENT_ID = {"left": 2, "right": 3}


def load_subject_raw(subject):
    """Download (if needed) and concatenate the motor-imagery runs for one subject."""
    files = eegbci.load_data(subjects=subject, runs=RUNS, path=DATA, update_path=False)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in files])
    eegbci.standardize(raw)                                   # 10-05 channel names
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
    raw.rename_channels(lambda x: x.strip("."))
    return raw


def make_epochs(subject):
    """Bandpass-filter and epoch one subject into left/right imagery trials."""
    raw = load_subject_raw(subject)
    raw.filter(FMIN, FMAX, fir_design="firwin", skip_by_annotation="edge")
    events, _ = mne.events_from_annotations(raw)
    picks = pick_types(raw.info, eeg=True, exclude="bads")
    return Epochs(raw, events, event_id=EVENT_ID, tmin=TMIN, tmax=TMAX,
                  picks=picks, baseline=None, preload=True)


def decode_subject(epochs, n_components=6, n_splits=5, seed=42):
    """CSP+LDA with stratified k-fold CV. Returns (auc_scores, acc_scores)."""
    X = epochs.get_data(copy=False)
    y = epochs.events[:, -1]
    clf = Pipeline([("csp", CSP(n_components=n_components, reg="ledoit_wolf", log=True)),
                    ("lda", LDA())])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return auc, acc


if __name__ == "__main__":
    for s in SUBJECTS:
        ep = make_epochs(s)
        auc, acc = decode_subject(ep)
        print(f"S{s:02d}: AUC={auc.mean():.3f}+/-{auc.std():.3f}  "
              f"Acc={acc.mean():.3f}+/-{acc.std():.3f}  (n={len(ep)})")
