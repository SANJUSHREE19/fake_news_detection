import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from preprocess import clean_text, ensure_nltk, setup_text_tools

ensure_nltk()
setup_text_tools()

# -----------------------
# 1. Load Dataset
# -----------------------
real = pd.read_csv("../data/FakeReal/True.csv")
fake = pd.read_csv("../data/FakeReal/Fake.csv")

real["label"] = 1   # 1 = Real
fake["label"] = 0   # 0 = Fake

df = pd.concat([real, fake], axis=0).reset_index(drop=True)
df["text"] = df["text"].astype(str).apply(clean_text)

# -----------------------
# 2. Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# -----------------------
# 3. TF-IDF
# -----------------------
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# -----------------------
# 4. SVM Model
# -----------------------
svm = LinearSVC()
svm.fit(X_train_vec, y_train)

# -----------------------
# 5. Evaluation
# -----------------------
y_pred = svm.predict(X_test_vec)

print("\n========== SVM RESULTS ==========")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------
# 6. Save artifacts
# -----------------------
import os
os.makedirs("artifacts", exist_ok=True)

with open("artifacts/tfidf_vectorizer_svm.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("artifacts/svm_model.pkl", "wb") as f:
    pickle.dump(svm, f)

print("\nArtifacts saved: tfidf_vectorizer_svm.pkl & svm_model.pkl")
print("==================================")
