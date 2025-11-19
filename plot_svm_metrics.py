import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

from preprocess import clean_text, ensure_nltk, setup_text_tools
from sklearn.model_selection import train_test_split

ensure_nltk()
setup_text_tools()

import os
os.makedirs("../outputs/models/tfidf_svm", exist_ok=True)
OUT = "../outputs/models/tfidf_svm"

# Load dataset
real = pd.read_csv("../data/FakeReal/True.csv")
fake = pd.read_csv("../data/FakeReal/Fake.csv")
real["label"] = 1
fake["label"] = 0
df = pd.concat([real, fake], ignore_index=True)
df["text"] = df["text"].astype(str).apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# Load model
with open("artifacts/tfidf_vectorizer_svm.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("artifacts/svm_model.pkl", "rb") as f:
    svm = pickle.load(f)

X_test_vec = tfidf.transform(X_test)
y_pred = svm.predict(X_test_vec)

# Convert decision function into probability for ROC
scores = svm.decision_function(X_test_vec)

# Save raw arrays
pd.Series(y_test, name="y_true").to_csv(f"{OUT}/y_true.csv", index=False)
pd.Series(y_pred, name="y_pred").to_csv(f"{OUT}/y_pred.csv", index=False)
pd.Series(scores, name="y_proba").to_csv(f"{OUT}/y_proba.csv", index=False)

# Save metrics.csv
report = classification_report(y_test, y_pred, output_dict=True)
df_metrics = pd.DataFrame(report).T
df_metrics.to_csv(f"{OUT}/metrics.csv")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap="Blues")
plt.title("SVM Confusion Matrix")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig(f"{OUT}/confusion_matrix.png", dpi=200)
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"SVM (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("SVM ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.savefig(f"{OUT}/roc_curve.png", dpi=200)
plt.close()

print("✨ SVM metrics exported to:", OUT)
