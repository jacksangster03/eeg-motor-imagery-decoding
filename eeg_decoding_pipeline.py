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
import csv
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, pick_types
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_score

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---- Configuration ----
SUBJECTS = list(range(1, 110))  # all 109 subjects
RUNS     = [4, 8, 12]
TMIN, TMAX = 1.0, 4.0
FMIN, FMAX = 7.0, 30.0
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "mne_data")
RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EVENT_ID = {"left": 2, "right": 3}


def load_subject_raw(subject):
    files = eegbci.load_data(subjects=subject, runs=RUNS, path=DATA, update_path=False)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in files])
    eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"))
    raw.rename_channels(lambda x: x.strip("."))
    return raw


def make_epochs(subject):
    raw = load_subject_raw(subject)
    raw.filter(FMIN, FMAX, fir_design="firwin", skip_by_annotation="edge")
    events, _ = mne.events_from_annotations(raw)
    picks = pick_types(raw.info, eeg=True, exclude="bads")
    return Epochs(raw, events, event_id=EVENT_ID, tmin=TMIN, tmax=TMAX,
                  picks=picks, baseline=None, preload=True)


def decode_subject(epochs, n_components=6, n_splits=5, seed=42):
    X = epochs.get_data(copy=False)
    y = epochs.events[:, -1]
    clf = Pipeline([("csp", CSP(n_components=n_components, reg="ledoit_wolf", log=True)),
                    ("lda", LDA())])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return auc, acc


def plot_distribution(rows, out_path):
    aucs = [r["auc"] for r in rows]
    accs = [r["acc"] for r in rows]
    chance = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, values, label, color in [
        (axes[0], aucs, "AUC (ROC)", "#1565C0"),
        (axes[1], accs, "Accuracy", "#2E7D32"),
    ]:
        ax.hist(values, bins=20, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(chance, color="red", linestyle="--", linewidth=1.5, label="Chance (0.5)")
        ax.axvline(np.mean(values), color="black", linestyle="-", linewidth=1.5,
                   label=f"Mean={np.mean(values):.2f}")
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel("Number of subjects", fontsize=11)
        ax.set_title(f"{label} distribution — {len(values)} subjects", fontsize=12)
        ax.legend(fontsize=9)

    n_above = sum(a > chance for a in aucs)
    n_illiterate = sum(a < chance for a in aucs)
    fig.suptitle(
        f"CSP+LDA Motor Imagery Decoding  |  PhysioNet EEGMMIDB  |  "
        f"{n_above}/{len(aucs)} above chance  |  {n_illiterate} BCI-illiterate",
        fontsize=11, y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out_path}")


if __name__ == "__main__":
    rows = []
    csv_path = os.path.join(RESULTS_DIR, "eeg_decoding_all_subjects.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "auc_mean", "auc_std", "acc_mean", "acc_std", "n_trials"])

        for s in SUBJECTS:
            try:
                ep = make_epochs(s)
                auc, acc = decode_subject(ep)
                row = {
                    "subject": s, "auc": auc.mean(), "auc_std": auc.std(),
                    "acc": acc.mean(), "acc_std": acc.std(), "n": len(ep)
                }
                rows.append(row)
                writer.writerow([f"S{s:03d}", f"{auc.mean():.3f}", f"{auc.std():.3f}",
                                 f"{acc.mean():.3f}", f"{acc.std():.3f}", len(ep)])
                f.flush()
                print(f"S{s:03d} [{s:3d}/109]: AUC={auc.mean():.3f}  Acc={acc.mean():.3f}  n={len(ep)}")
            except Exception as e:
                print(f"S{s:03d} FAILED: {e}")
                writer.writerow([f"S{s:03d}", "NA", "NA", "NA", "NA", "NA"])
                f.flush()

    if rows:
        aucs = [r["auc"] for r in rows]
        accs = [r["acc"] for r in rows]
        print(f"\n=== Summary ({len(rows)} subjects) ===")
        print(f"AUC  mean={np.mean(aucs):.3f}  median={np.median(aucs):.3f}  std={np.std(aucs):.3f}")
        print(f"Acc  mean={np.mean(accs):.3f}  median={np.median(accs):.3f}  std={np.std(accs):.3f}")
        print(f"Above chance (AUC>0.5): {sum(a>0.5 for a in aucs)}/{len(aucs)}")
        print(f"BCI-illiterate (AUC<0.5): {sum(a<0.5 for a in aucs)}/{len(aucs)}")
        plot_distribution(rows, os.path.join(RESULTS_DIR, "eeg_decoding_distribution.png"))

    print(f"\nResults saved: {csv_path}")
