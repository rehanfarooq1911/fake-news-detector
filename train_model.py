"""
Fake News Detection - Training Script
Run:  python train_model.py
Output: model/fake_news_model.joblib
"""
import re
import string
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATA_DIR = Path("data")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

# Text cleaning (app.py bhi isi function ko import karta hai)
from text_utils import clean_text  # noqa: E402


def load_data() -> pd.DataFrame:
    fake = pd.read_csv(DATA_DIR / "Fake.csv")
    true = pd.read_csv(DATA_DIR / "True.csv")
    fake["label"] = 0   # 0 = Fake
    true["label"] = 1   # 1 = Real

    df = pd.concat([fake, true], ignore_index=True)
    df = df.dropna(subset=["text"])
    # title + text ko jod kar ek feature banate hain
    df["content"] = (df["title"].fillna("") + " " + df["text"]).apply(clean_text)
    # bohat chhoti / khali texts hata do
    df = df[df["content"].str.split().str.len() > 10]
    # duplicate khabrein hata do (train/test mein overlap se bachne ke liye)
    df = df.drop_duplicates(subset="content").reset_index(drop=True)
    return df[["content", "label"]]


def main():
    df = load_data()
    print(f"Total samples after cleaning: {len(df)}")
    print(df["label"].value_counts().rename({0: "Fake", 1: "Real"}), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=5)),
    ])

    print("Training...")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"\nTest Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, target_names=["Fake", "Real"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    # manual_testing.csv par extra check
    manual_path = DATA_DIR / "manual_testing.csv"
    if manual_path.exists():
        m = pd.read_csv(manual_path)
        m["content"] = (m["title"].fillna("") + " " + m["text"].fillna("")).apply(clean_text)
        mp = model.predict(m["content"])
        print(f"\nmanual_testing.csv accuracy: {accuracy_score(m['class'], mp):.4f} ({len(m)} samples)")

    joblib.dump(model, MODEL_DIR / "fake_news_model.joblib", compress=3)
    size = (MODEL_DIR / "fake_news_model.joblib").stat().st_size / 1e6
    print(f"\nModel saved -> model/fake_news_model.joblib ({size:.1f} MB)")


if __name__ == "__main__":
    main()
