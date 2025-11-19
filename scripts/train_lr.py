import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import clean_text, ensure_nltk, setup_text_tools

ensure_nltk()
setup_text_tools()

real = pd.read_csv("../data/FakeReal/True.csv")
fake = pd.read_csv("../data/FakeReal/Fake.csv")
real["label"] = 1
fake["label"] = 0

df = pd.concat([real, fake], ignore_index=True)
df["text"] = df["text"].astype(str).apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

tfidf_lr = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train_vec = tfidf_lr.fit_transform(X_train)
X_test_vec = tfidf_lr.transform(X_test)

lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_vec, y_train)

y_pred = lr.predict(X_test_vec)
print("\n========== LR RESULTS ==========")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

os.makedirs("artifacts", exist_ok=True)
with open("artifacts/tfidf_vectorizer_lr.pkl", "wb") as f:
    pickle.dump(tfidf_lr, f)
with open("artifacts/lr_model.pkl", "wb") as f:
    pickle.dump(lr, f)

print("\nArtifacts saved: tfidf_vectorizer_lr.pkl & lr_model.pkl")
print("==================================")
