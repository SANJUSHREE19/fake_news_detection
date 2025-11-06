# scripts/export_model.py
from pathlib import Path
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA = "outputs/clean_combined.csv"
ART = Path("artifacts"); ART.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    text_col = "cleaned" if "cleaned" in df.columns else "text"
    df = df.dropna(subset=[text_col, "label"])
    X = df[text_col].astype(str).tolist()
    y = df["label"].astype(int).values

    tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.9, strip_accents="unicode")
    Xtf = tfidf.fit_transform(X)

    clf = LogisticRegression(solver="liblinear", C=1.0, max_iter=200)
    clf.fit(Xtf, y)

    joblib.dump(tfidf, ART/"tfidf.pkl")
    joblib.dump(clf,   ART/"lr.pkl")
    (ART/"labels.txt").write_text("0=fake,1=real")

    print("✅ Saved:", ART/"tfidf.pkl", ART/"lr.pkl")

if __name__ == "__main__":
    main()
