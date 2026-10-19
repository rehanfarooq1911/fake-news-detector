# NewsLens – Fake News Detector

NewsLens is a Streamlit web app that looks at how a news article is **written** and estimates whether it reads like a
calm, wire-style report or like a sensational one. It uses TF-IDF features and a Logistic Regression classifier trained on
the Kaggle "Fake and Real News Dataset".

> ⚠️ **NewsLens is not a fact-checker.** It analyses writing style only, and it can be wrong. Please read the
> [Limitations](#limitations) section before relying on any result.

## Features

- Paste an article and get a wire-style vs. sensational-style score
- Word-level view showing how often words appear in real and fake articles in the training data
- Built-in example articles to try
- About page with project details, team and known limitations
- Light and dark theme

## Tech stack

- Python
- scikit-learn (TF-IDF + Logistic Regression)
- pandas, numpy, joblib
- Streamlit

## Project structure

```
├── app.py                 # Streamlit app
├── text_utils.py          # Text cleaning (shared by training and the app)
├── train_model.py         # Trains the model -> model/fake_news_model.joblib
├── build_word_stats.py    # Builds per-word statistics -> model/word_stats.joblib
├── requirements.txt       # Pinned dependencies
├── model/                 # Saved model files (needed to run the app)
└── data/                  # Dataset CSVs (not included, see below)
```

## Run locally

1. Clone the repository and open the project folder.
2. (Recommended) create and activate a virtual environment:

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac / Linux
   source venv/bin/activate
   ```

3. Install dependencies and start the app:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

The trained model files are already in the `model/` folder, so no training is needed to run the app.

## Retrain the model (optional)

The dataset is not included in this repository.

1. Download **Fake and Real News Dataset** from Kaggle.
2. Create a `data/` folder and place `Fake.csv` and `True.csv` inside it.
3. Train the model:

   ```bash
   python train_model.py
   ```

4. Rebuild the word statistics used by the app:

   ```bash
   python build_word_stats.py
   ```

Use the same scikit-learn version when training and running the app, because a saved model can fail or give warnings
when loaded with a different version.

## How it works

1. **Cleaning:** text is lower-cased, and links, HTML tags, brackets, punctuation and numbers are removed. The
   "CITY (Reuters) -" prefix and the word "Reuters" are also removed to reduce data leakage.
2. **Features:** TF-IDF with unigrams and bigrams (up to 50,000 features, English stop words removed).
3. **Classifier:** Logistic Regression.
4. **Evaluation:** an 80/20 stratified train/test split after removing duplicate and very short articles.

## Limitations

- **Style, not truth.** The model learns how articles are written, not whether the facts are correct. A false story written
  in a calm, formal style can be marked as reliable, and a true story written dramatically can be marked as sensational.
- **One kind of "reliable" source.** In the training data, almost all wire-style articles (about 99.8%) come from a single
  news agency, Reuters. Other trusted outlets that write differently can be labelled sensational. In our own testing, a
  genuine mainstream-news article was labelled sensational-style.
- **Same facts, different result.** Two articles describing the same event can get different results depending only on
  writing style. The Analyze page has two examples that show this.
- **Narrow data.** The model is trained on English-language US political news from roughly 2015 to 2018. Other topics,
  languages (including Urdu) and newer events may not work well.
- **High test accuracy is misleading.** Accuracy on the dataset's own test split is about 98%, but this does not mean
  the model works on any article from the web, because the test split comes from the same two sources as the training
  data.
- **No source or context checking.** The app does not check the publisher, author, date, images, quotes or links, and
  it does not compare the article against other reports.
- **Possible bias.** Vocabulary tied to the topics and sources in the dataset can influence results, so some words may
  push a score up or down for reasons unrelated to credibility.
- **Not for serious decisions.** Do not use this tool to make decisions about people, publishers, news moderation or
  anything with real-world consequences. Always verify important claims with trusted fact-checking sources.

**Future work:** train on news from many publishers and regions, add Urdu support, and connect to fact-checking
sources.

## Dataset credit

"Fake and Real News Dataset" (`Fake.csv` and `True.csv`) from Kaggle. About 39,000 articles are used after removing
duplicates. Credit to the original dataset creators. The cleaning pipeline, saved model and this app were built by our
team for a hackathon; our first baseline was inspired by a public Kaggle notebook.

## Team

- **Muhammad Rehan Farooq** – Team Leader, AI & Data Science
- **Habiba Siddique** – Machine Learning & Web Development
