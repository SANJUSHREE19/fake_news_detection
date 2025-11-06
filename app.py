# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

@st.cache_resource
def load_artifacts():
    tfidf: TfidfVectorizer = joblib.load("artifacts/tfidf.pkl")
    model: LogisticRegression = joblib.load("artifacts/lr.pkl")
    vocab = np.array(tfidf.get_feature_names_out())
    coefs = model.coef_[0]
    return tfidf, model, vocab, coefs

tfidf, model, vocab, coefs = load_artifacts()

def predict_one(text: str):
    X = tfidf.transform([text])
    proba = model.predict_proba(X)[0,1]
    pred = int(proba >= 0.5)
    return pred, proba, X

def explain_terms(X_row):
    # show top positive/negative contributing terms for this text
    # X_row is 1xN sparse; we map non-zero indices to coefs
    idx = X_row.nonzero()[1]
    scores = X_row.data * coefs[idx]
    # top "real" terms (positive contribution)
    top_pos_idx = np.argsort(scores)[-10:]
    # top "fake" terms (negative contribution)
    top_neg_idx = np.argsort(scores)[:10]
    pos = [(vocab[idx[i]], float(scores[i])) for i in top_pos_idx[::-1]]
    neg = [(vocab[idx[i]], float(scores[i])) for i in top_neg_idx]
    return pos, neg

st.title("📰 Fake News Detector (TF-IDF + Logistic Regression)")
st.caption("Trained on LIAR + Fake/Real datasets • 0=fake, 1=real")

tab1, tab2 = st.tabs(["🔎 Single Text", "📦 Batch (CSV)"])

with tab1:
    default_text = "Reuters – The White House said on Monday that ..."
    text = st.text_area("Enter an article or statement", value=default_text, height=180)
    colA, colB = st.columns([1,1])
    with colA:
        threshold = st.slider("Decision Threshold (for real=1)", 0.1, 0.9, 0.5, 0.05)
    with colB:
        run = st.button("Classify")

    if run and text.strip():
        pred, proba, Xrow = predict_one(text)
        pred_thr = int(proba >= threshold)
        label = "Real (1)" if pred_thr == 1 else "Fake (0)"
        st.markdown(f"### Prediction: **{label}**")
        st.progress(min(max(proba,0.0),1.0), text=f"Probability(real=1): {proba:.3f}")

        with st.expander("Feature contributions (top terms)"):
            pos, neg = explain_terms(Xrow)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Top terms → Real**")
                st.dataframe(pd.DataFrame(pos, columns=["term","contribution"]))
            with c2:
                st.markdown("**Top terms → Fake**")
                st.dataframe(pd.DataFrame(neg, columns=["term","contribution"]))

with tab2:
    st.write("Upload a CSV with a column named `text` (or `cleaned`).")
    file = st.file_uploader("CSV file", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        text_col = "cleaned" if "cleaned" in df.columns else "text" if "text" in df.columns else None
        if text_col is None:
            st.error("CSV must contain 'text' or 'cleaned' column.")
        else:
            X = tfidf.transform(df[text_col].astype(str).tolist())
            proba = model.predict_proba(X)[:,1]
            pred = (proba >= 0.5).astype(int)
            out = df.copy()
            out["pred_label"] = pred
            out["prob_real"] = proba
            st.success(f"Predicted {len(out)} rows.")
            st.dataframe(out.head(20))
            st.download_button("Download predictions CSV", out.to_csv(index=False), "predictions.csv", "text/csv")

st.caption("© Fake News Detection • TF-IDF + Logistic Regression • Streamlit")
