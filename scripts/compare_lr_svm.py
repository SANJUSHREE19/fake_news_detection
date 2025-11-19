import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess import clean_text, ensure_nltk, setup_text_tools

# Init NLP tools
ensure_nltk()
setup_text_tools()

# Load raw fake/real dataset
real = pd.read_csv("../data/FakeReal/True.csv")
fake = pd.read_csv("../data/FakeReal/Fake.csv")
real["label"] = 1
fake["label"] = 0

df = pd.concat([real, fake], ignore_index=True)
df["text"] = df["text"].astype(str).apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# -------- Load Logistic Regression artifacts --------
with open("artifacts/tfidf_vectorizer_lr.pkl", "rb") as f:
    tfidf_lr = pickle.load(f)

with open("artifacts/lr_model.pkl", "rb") as f:
    lr = pickle.load(f)

# -------- Load SVM artifacts --------
with open("artifacts/tfidf_vectorizer_svm.pkl", "rb") as f:
    tfidf_svm = pickle.load(f)

with open("artifacts/svm_model.pkl", "rb") as f:
    svm = pickle.load(f)

# Vectorize
X_test_lr = tfidf_lr.transform(X_test)
X_test_svm = tfidf_svm.transform(X_test)

y_pred_lr = lr.predict(X_test_lr)
y_pred_svm = svm.predict(X_test_svm)

results = {
    "Model": ["Logistic Regression", "SVM"],
    "Accuracy": [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_svm),
    ],
    "Precision": [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_svm),
    ],
    "Recall": [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_svm),
    ],
    "F1 Score": [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_svm),
    ],
}

results_df = pd.DataFrame(results)
print("\n===== MODEL COMPARISON =====\n")
print(results_df)

results_df.to_csv("../outputs/lr_vs_svm_results.csv", index=False)

plt.figure(figsize=(8,5))
plt.bar(results_df["Model"], results_df["Accuracy"], width=0.5)
plt.title("Accuracy Comparison: Logistic Regression vs SVM")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.savefig("../outputs/lr_vs_svm_accuracy.png", dpi=300)
plt.show()
print("Saved comparison plot to ../outputs/lr_vs_svm_accuracy.png")
