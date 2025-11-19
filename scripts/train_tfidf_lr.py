import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use("Agg")  # Needed for saving images without display
import matplotlib.pyplot as plt


def save_fig(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main(input_path, outdir, text_col):
    print(f"[INFO] Loading data → {input_path}")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    # Choose text column
    if text_col not in df.columns:
        text_col = "cleaned" if "cleaned" in df.columns else "text"

    df = df.dropna(subset=[text_col, "label"])
    X = df[text_col].astype(str).tolist()
    y = df["label"].astype(int).to_numpy()

    print(f"[INFO] Samples: {len(df):,} | Text column: {text_col}")

    # Train-test split
    print("[INFO] Splitting dataset (80/20)...")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF
    print("[INFO] Vectorizing text using TF-IDF...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        strip_accents="unicode"
    )
    Xtr = tfidf.fit_transform(Xtr)
    Xte = tfidf.transform(Xte)
    print(f"[INFO] Shapes: Train={Xtr.shape}, Test={Xte.shape}")

    # Train Logistic Regression
    print("[INFO] Training Logistic Regression model...")
    clf = LogisticRegression(solver="liblinear", C=1.0, max_iter=200)
    clf.fit(Xtr, ytr)

    # Predictions and metrics
    print("[INFO] Evaluating...")
    preds = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:, 1]

    report = classification_report(
        yte, preds, target_names=["fake(0)", "real(1)"], output_dict=True
    )
    auc = roc_auc_score(yte, proba)

    # Save metrics CSV
    metrics_df = pd.DataFrame(report).T
    metrics_df.loc["roc_auc", "precision"] = auc
    metrics_df.to_csv(out / "metrics.csv")
    print(f"[OK] Metrics saved → {out/'metrics.csv'}")

    # Save raw predictions for plotting script
    pd.Series(yte, name="y_true").to_csv(out/"y_true.csv", index=False)
    pd.Series(preds, name="y_pred").to_csv(out/"y_pred.csv", index=False)
    pd.Series(proba, name="y_proba").to_csv(out/"y_proba.csv", index=False)
    print(f"[OK] Prediction arrays saved")

    # Confusion Matrix
    print("[INFO] Saving confusion matrix...")
    cm = confusion_matrix(yte, preds)
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title("Confusion Matrix — TF-IDF + Logistic Regression")
    plt.xticks([0, 1], ["fake(0)", "real(1)"])
    plt.yticks([0, 1], ["fake(0)", "real(1)"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    save_fig(out / "confusion_matrix.png")
    print(f"[OK] Confusion matrix → {out/'confusion_matrix.png'}")

    # Top features
    print("[INFO] Extracting top terms...")
    vocab = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]
    top_real_idx = np.argsort(coefs)[-20:]
    top_fake_idx = np.argsort(coefs)[:20]

    feat_df = pd.DataFrame({
        "top_real_terms": vocab[top_real_idx][::-1],
        "real_coef": coefs[top_real_idx][::-1],
        "top_fake_terms": vocab[top_fake_idx],
        "fake_coef": coefs[top_fake_idx],
    })
    feat_df.to_csv(out / "top_terms.csv", index=False)
    print(f"[OK] Top terms → {out/'top_terms.csv'}")

    print("\n✅ DONE — TF-IDF + Logistic Regression successfully trained!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/clean_combined.csv")
    parser.add_argument("--outdir", default="outputs/models/tfidf_lr")
    parser.add_argument("--text_col", default="text")
    args = parser.parse_args()
    main(args.input, args.outdir, args.text_col)
