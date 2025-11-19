import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
@st.cache_resource
def load_artifacts():
    artifacts = {}
    artifacts["tfidf_lr"] = joblib.load("artifacts/tfidf_vectorizer_lr.pkl")
    artifacts["lr"] = joblib.load("artifacts/lr_model.pkl")   
    artifacts["tfidf_svm"] = joblib.load("artifacts/tfidf_vectorizer_svm.pkl")
    artifacts["svm"] = joblib.load("artifacts/svm_model.pkl")
    return artifacts
art = load_artifacts()
models = {
    "Logistic Regression": ("tfidf_lr", "lr"),
    "SVM (LinearSVC)": ("tfidf_svm", "svm")
}
def predict_one(text: str, model_name: str, threshold: float):
    tfidf_key, model_key = models[model_name]
    tfidf = art[tfidf_key]
    model = art[model_key]
    X = tfidf.transform([text])
    if model_name == "Logistic Regression":
        proba = model.predict_proba(X)[0, 1]
    else:       
        raw = model.decision_function(X)[0]
        proba = 1 / (1 + np.exp(-raw))  
    label = int(proba >= threshold)
    return label, proba
st.title("📰 Fake News Detector (TF-IDF Models)")
st.caption("Choose between Logistic Regression and SVM models • 0 = Fake, 1 = Real")
model_choice = st.selectbox("Select Model", list(models.keys()))
tab1, tab2 = st.tabs(["🔎 Single Text", "📦 Batch (CSV)"])


with tab1:
    default_text = "Reuters – The White House said on Monday that ..."
    text = st.text_area("Enter a news article or statement:", value=default_text, height=180)

    colA, colB = st.columns([1, 1])
    with colA:
        threshold = st.slider("Decision Threshold (Real = 1)", 0.1, 0.9, 0.5, 0.05)
    with colB:
        run = st.button("Classify")

    if run and text.strip():
        label, proba = predict_one(text, model_choice, threshold)
        pred_label = "Real (1)" if label == 1 else "Fake (0)"

        st.markdown(f"## Prediction: **{pred_label}**")
        st.progress(min(max(proba, 0.0), 1.0), text=f"Probability(real=1): {proba:.3f}")

with tab2:
    st.write("Upload a CSV containing a column named `text` or `cleaned`.")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        text_col = "cleaned" if "cleaned" in df.columns else "text" if "text" in df.columns else None

        if text_col is None:
            st.error("CSV must contain 'text' or 'cleaned' column.")
        else:
            tfidf_key, model_key = models[model_choice]
            tfidf = art[tfidf_key]
            model = art[model_key]

            X = tfidf.transform(df[text_col].astype(str))
            if model_choice == "Logistic Regression":
                proba = model.predict_proba(X)[:, 1]
            else:
                raw = model.decision_function(X)
                proba = 1 / (1 + np.exp(-raw))

            preds = (proba >= 0.5).astype(int)
            out = df.copy()
            out["pred_label"] = preds
            out["prob_real"] = proba

            st.success(f"Predicted {len(out)} rows using {model_choice}.")
            st.dataframe(out.head(15))
            st.download_button("Download Predictions", out.to_csv(index=False),
                               "predictions.csv", "text/csv")
st.caption("© Fake News Detection • Logistic Regression + SVM • Streamlit UI")
