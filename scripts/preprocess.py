import argparse, os, re, yaml
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# --- NLP utils ---
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def ensure_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

stop_words = set()
lemmatizer = None

def setup_text_tools():
    global stop_words, lemmatizer
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

URL_RE = re.compile(r"http\S+|www\S+|https\S+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub("", text)
    text = NON_ALPHA_RE.sub(" ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def binarize_liar(df: pd.DataFrame, label_col="label", true_labels=None):
    true_labels = set(true_labels or ["true","mostly-true","half-true"])
    return df.assign(**{
        label_col: df[label_col].apply(lambda x: 1 if str(x).strip().lower() in true_labels else 0)
    })

def load_liar_split(path_tsv: str) -> pd.DataFrame:
    # LIAR: label at col 1, text at col 2 (0-based)
    df = pd.read_csv(path_tsv, sep="\t", header=None)
    df = df[[1, 2]]
    df.columns = ["label", "text"]
    return df

def load_fake_real(true_csv: str, fake_csv: str, text_col="text") -> pd.DataFrame:
    true_df = pd.read_csv(true_csv)
    fake_df = pd.read_csv(fake_csv)
    # normalize columns
    if text_col not in true_df.columns:
        raise ValueError(f"'text' column '{text_col}' not found in True.csv")
    if text_col not in fake_df.columns:
        raise ValueError(f"'text' column '{text_col}' not found in Fake.csv")
    true_df = true_df.rename(columns={text_col: "text"}).assign(label=1)
    fake_df = fake_df.rename(columns={text_col: "text"}).assign(label=0)
    return pd.concat([true_df, fake_df], ignore_index=True)

def clean_df(df: pd.DataFrame, text_col="text") -> pd.DataFrame:
    df = df.dropna(subset=[text_col]).copy()
    tqdm.pandas(desc="Cleaning text")
    df["cleaned"] = df[text_col].progress_apply(clean_text)
    df = df.dropna(subset=["cleaned"])
    df = df.drop_duplicates(subset=["cleaned"])
    return df

def main(cfg_path: str, combine=False):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    ensure_nltk()
    setup_text_tools()

    # --- LIAR ---
    liar_train = load_liar_split(cfg["liar"]["train"])
    liar_test  = load_liar_split(cfg["liar"]["test"])
    liar_valid = load_liar_split(cfg["liar"]["valid"])

    liar_train = binarize_liar(liar_train, true_labels=cfg.get("binary_true_labels"))
    liar_test  = binarize_liar(liar_test,  true_labels=cfg.get("binary_true_labels"))
    liar_valid = binarize_liar(liar_valid, true_labels=cfg.get("binary_true_labels"))

    liar_train = clean_df(liar_train, "text")
    liar_test  = clean_df(liar_test,  "text")
    liar_valid = clean_df(liar_valid, "text")

    liar_train.to_csv(out_dir / "clean_liar_train.csv", index=False)
    liar_test.to_csv(out_dir / "clean_liar_test.csv", index=False)
    liar_valid.to_csv(out_dir / "clean_liar_valid.csv", index=False)

    print("LIAR saved:",
          (out_dir / "clean_liar_train.csv"),
          (out_dir / "clean_liar_test.csv"),
          (out_dir / "clean_liar_valid.csv"), sep="\n")

    # --- Fake/Real ---
    fr_text_col = cfg.get("text_column_overrides", {}).get("fake_real_text", "text")
    fake_real = load_fake_real(cfg["fake_real"]["true"], cfg["fake_real"]["fake"], text_col=fr_text_col)
    fake_real = clean_df(fake_real, "text")
    fake_real.to_csv(out_dir / "clean_fake_real.csv", index=False)

    print("Fake/Real saved:", (out_dir / "clean_fake_real.csv"))

    # --- Optional combined corpus for classical models
    if combine:
        combo = pd.concat([
            liar_train[["cleaned","label"]].rename(columns={"cleaned":"text"}),
            liar_valid[["cleaned","label"]].rename(columns={"cleaned":"text"}),
            liar_test[["cleaned","label"]].rename(columns={"cleaned":"text"}),
            fake_real[["cleaned","label"]].rename(columns={"cleaned":"text"}),
        ], ignore_index=True)
        combo = combo.drop_duplicates(subset=["text"])
        combo.to_csv(out_dir / "clean_combined.csv", index=False)
        print("Combined saved:", (out_dir / "clean_combined.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess LIAR and Fake/Real datasets.")
    parser.add_argument("--config", "-c", default="config/config.yaml", help="Path to YAML config.")
    parser.add_argument("--combine", action="store_true", help="Also output a merged corpus.")
    args = parser.parse_args()
    main(args.config, combine=args.combine)
