import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from preprocess import clean_text, ensure_nltk, setup_text_tools

ensure_nltk()
setup_text_tools()

real = pd.read_csv("../data/FakeReal/True.csv")
fake = pd.read_csv("../data/FakeReal/Fake.csv")
real["label"] = 1
fake["label"] = 0
df = pd.concat([real, fake], ignore_index=True)
df["text"] = df["text"].astype(str).apply(clean_text)

_, X_test, _, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

# Load LR
with open("artifacts/tfidf_vectorizer_lr.pkl", "rb") as f: tfidf_lr = pickle.load(f)
with open("artifacts/lr_model.pkl", "rb") as f: lr = pickle.load(f)
proba_lr = lr.predict_proba(tfidf_lr.transform(X_test))[:, 1]

# Load SVM
with open("artifacts/tfidf_vectorizer_svm.pkl", "rb") as f: tfidf_svm = pickle.load(f)
with open("artifacts/svm_model.pkl", "rb") as f: svm = pickle.load(f)
scores_svm = svm.decision_function(tfidf_svm.transform(X_test))

# Plot
fpr_lr, tpr_lr, _ = roc_curve(y_test, proba_lr)
fpr_svm, tpr_svm, _ = roc_curve(y_test, scores_svm)

plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc(fpr_lr, tpr_lr):.3f})")
plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {auc(fpr_svm, tpr_svm):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
plt.legend()
plt.title("Combined ROC Curve — LR vs SVM")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("../outputs/combined_roc_lr_svm.png", dpi=300)
plt.show()
print("Saved to outputs/combined_roc_lr_svm.png")
