"""
Word statistics from the training data (no re-training needed).
Run:  python build_word_stats.py
Needs: data/Fake.csv, data/True.csv, model/fake_news_model.joblib
Output: model/word_stats.joblib  (app.py reads it to show training-data % per word)
"""
import joblib
import numpy as np

from train_model import load_data


def main():
    df = load_data()
    model = joblib.load("model/fake_news_model.joblib")
    tfidf = model.named_steps["tfidf"]

    X = tfidf.transform(df["content"])
    present = (X > 0).astype(np.int8)          # word is in the article or not
    y = df["label"].values

    real_n, fake_n = int((y == 1).sum()), int((y == 0).sum())
    real_docs = np.asarray(present[y == 1].sum(axis=0)).ravel()
    fake_docs = np.asarray(present[y == 0].sum(axis=0)).ravel()

    stats = {
        "names": list(tfidf.get_feature_names_out()),
        "real_pct": real_docs / real_n * 100,
        "fake_pct": fake_docs / fake_n * 100,
    }
    joblib.dump(stats, "model/word_stats.joblib", compress=3)
    print(f"Articles used: wire={real_n}, sensational={fake_n}")
    print("Saved model/word_stats.joblib")


if __name__ == "__main__":
    main()
