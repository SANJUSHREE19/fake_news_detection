import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve
)

def save_fig(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_confusion(y_true, y_pred, outpath):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks([0,1], ["fake(0)","real(1)"])
    plt.yticks([0,1], ["fake(0)","real(1)"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True")
    save_fig(outpath)

def plot_prf_bars(metrics_df, outpath):
    # rows we care: 'fake(0)', 'real(1)'
    rows = ["fake(0)", "real(1)"]
    metrics_df = metrics_df.loc[rows, ["precision","recall","f1-score","support"]]
    # bar chart for precision/recall/f1
    plt.figure()
    x = np.arange(len(rows))
    w = 0.25
    plt.bar(x - w, metrics_df["precision"], width=w, label="Precision")
    plt.bar(x,      metrics_df["recall"],    width=w, label="Recall")
    plt.bar(x + w,  metrics_df["f1-score"],  width=w, label="F1")
    plt.xticks(x, rows)
    plt.ylim(0, 1.0)
    plt.title("Per-Class Precision / Recall / F1")
    plt.legend()
    save_fig(outpath.replace(".png", "_prf.png"))

    # support bars
    plt.figure()
    plt.bar(rows, metrics_df["support"])
    plt.title("Per-Class Support")
    plt.ylabel("Count")
    save_fig(outpath.replace(".png", "_support.png"))

def plot_roc(y_true, y_score, outpath):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (positive = real[1])")
    plt.legend(loc="lower right")
    save_fig(outpath)

def plot_pr_curve(y_true, y_score, outpath):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = np.trapz(precision[::-1], recall[::-1])  # rough area
    plt.figure()
    plt.plot(recall, precision, label=f"AP ≈ {ap:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (positive = real[1])")
    plt.legend(loc="lower left")
    save_fig(outpath)

def main(model_dir):
    md = Path(model_dir)
    metrics_path = md/"metrics.csv"
    y_true_path = md/"y_true.csv"
    y_pred_path = md/"y_pred.csv"
    y_proba_path = md/"y_proba.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}")
    if not y_true_path.exists() or not y_pred_path.exists():
        raise FileNotFoundError("Missing y_true.csv or y_pred.csv")
    # y_proba is optional (SVM may not have probabilities)
    has_proba = y_proba_path.exists()

    metrics_df = pd.read_csv(metrics_path, index_col=0)
    y_true = pd.read_csv(y_true_path).values.ravel().astype(int)
    y_pred = pd.read_csv(y_pred_path).values.ravel().astype(int)
    y_score = pd.read_csv(y_proba_path).values.ravel() if has_proba else None

    # 1) Confusion matrix
    plot_confusion(y_true, y_pred, md/"diagram_confusion.png")

    # 2) Per-class precision/recall/F1 + support
    plot_prf_bars(metrics_df, str(md/"diagram_metrics.png"))

    # 3) ROC + 4) PR (only if scores available)
    if has_proba:
        plot_roc(y_true, y_score, md/"diagram_roc.png")
        plot_pr_curve(y_true, y_score, md/"diagram_pr.png")
    else:
        print("[WARN] y_proba.csv not found — skipping ROC/PR plots.")

    print(f"[OK] Diagrams saved → {md}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="outputs/models/tfidf_lr")
    args = ap.parse_args()
    main(args.model_dir)
