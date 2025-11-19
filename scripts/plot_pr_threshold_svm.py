import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
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

with open("artifacts/tfidf_vectorizer_svm.pkl","rb") as f: tfidf = pickle.load(f)
with open("artifacts/svm_model.pkl","rb") as f: svm = pickle.load(f)

scores = svm.decision_function(tfidf.transform(X_test))
thresholds = np.linspace(min(scores), max(scores), 50)

precisions = []
recalls = []
for t in thresholds:
    preds = (scores >= t).astype(int)
    precisions.append(precision_score(y_test, preds))
    recalls.append(recall_score(y_test, preds))

plt.plot(thresholds, precisions, label="Precision")
plt.plot(thresholds, recalls, label="Recall")
plt.legend()
plt.title("Precision vs Recall Across Decision Threshold — SVM")
plt.xlabel("Threshold")
plt.savefig("../outputs/svm_precision_recall_threshold.png", dpi=300)
plt.show()
print("Saved to outputs/svm_precision_recall_threshold.png")
