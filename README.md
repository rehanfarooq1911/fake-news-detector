# NewsLens – Fake News Detector

NewsLens is a Streamlit web app that looks at how a news article is **written** and estimates whether it reads like a
calm, wire-style report or like a sensational one. It uses TF-IDF features and a Logistic Regression classifier trained on
the Kaggle "Fake and Real News Dataset".

> ⚠️ **NewsLens is not a fact-checker.** It analyses writing style only, and it can be wrong. Please read the
> [Limitations](#limitations) section before relying on any result.

## Table of contents

1. [Problem statement and solution](#problem-statement-and-solution)
2. [Technology stack](#technology-stack)
3. [Setup and installation](#setup-and-installation)
4. [AI methodology and implementation](#ai-methodology-and-implementation)
5. [AI model names and API usage](#ai-model-names-and-api-usage)
6. [Product screenshots](#product-screenshots)
7. [Team member contributions](#team-member-contributions)
8. [Limitations](#limitations)
9. [Dataset credit](#dataset-credit)

## Problem statement and solution

**Problem.** Misleading and sensational news spreads quickly online, and most readers cannot tell at a glance whether an
article is written in a calm reporting style or a manipulative one. Checking every story by hand takes too much time, and
many tools give a bare verdict with no explanation, so users either trust them blindly or ignore them.

**Solution.** NewsLens lets a user paste a news article and instantly see whether it reads like a wire-style report or a
sensational one. Instead of only a label, the app shows a confidence score and how often key words appear in real and fake
articles in the training data. It also states clearly what it cannot do: it judges writing style, not facts.

### Features

- Paste an article and get a wire-style vs. sensational-style score
- Word-level view showing how often words appear in real and fake articles in the training data
- Built-in example articles to try
- About page with project details, team and known limitations
- Light and dark theme

## Technology stack

| Area | Tools |
|---|---|
| Language | Python |
| Machine learning | scikit-learn (TF-IDF + Logistic Regression) |
| Data handling | pandas, numpy |
| Model saving | joblib |
| Web interface | Streamlit (custom HTML/CSS styling) |
| Hosting | GitHub (code) and Streamlit Community Cloud (app) |

## Setup and installation

### Project structure

```
├── app.py                 # Streamlit app
├── text_utils.py          # Text cleaning (shared by training and the app)
├── train_model.py         # Trains the model -> model/fake_news_model.joblib
├── build_word_stats.py    # Builds per-word statistics -> model/word_stats.joblib
├── requirements.txt       # Pinned dependencies
├── screenshots/           # Images used in this README
├── model/                 # Saved model files (needed to run the app)
└── data/                  # Dataset CSVs (not included, see below)
```

### Run locally

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

### Retrain the model (optional)

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

## AI methodology and implementation

The pipeline has five steps, and the same cleaning function is used in training and in the app so both see identical text.

1. **Data preparation.** `Fake.csv` (label 0) and `True.csv` (label 1) are combined. The article title and text are joined
   into one field. Empty rows, articles with 10 words or fewer, and duplicate articles are removed. About 39,000 articles
   remain.
2. **Text cleaning (`text_utils.py`).** Text is lower-cased, and links, HTML tags, `[brackets]`, punctuation and numbers are
   removed. The `CITY (Reuters) -` prefix and the word "Reuters" are also removed. Without this step, the model could learn
   to detect the source name instead of the writing style (data leakage).
3. **Feature extraction.** `TfidfVectorizer` with English stop words removed, unigrams and bigrams, at most 50,000
   features, `min_df=3`, `max_df=0.9` and sublinear term frequency.
4. **Classification.** `LogisticRegression` with `C=5` and `max_iter=1000`, trained on an 80/20 stratified split
   (`random_state=42`). Accuracy, a classification report and a confusion matrix are printed after training.
5. **Explainability (`build_word_stats.py`).** For every word in the vocabulary, the script calculates the percentage of real
   and fake training articles that contain it. The app uses these numbers to show users which words appear more often in
   each group.

Accuracy on the dataset's own test split is about 98%. See [Limitations](#limitations) for why this number should not be
read as real-world accuracy.

## AI model names and API usage

| Item | Details |
|---|---|
| Vectorizer | `sklearn.feature_extraction.text.TfidfVectorizer` |
| Classifier | `sklearn.linear_model.LogisticRegression` |
| Pipeline | `sklearn.pipeline.Pipeline` (TF-IDF, then Logistic Regression) |
| Saved files | `model/fake_news_model.joblib`, `model/word_stats.joblib` |
| External APIs | None |
| Large language models | None |

NewsLens does not call any external AI service, API or large language model, and no API key is needed. The model is
trained by us and runs locally inside the app, so articles pasted into the app are not sent to any third party.

## Product screenshots

| Analyze page | Result view |
|---|---|
| ![Analyze page](screenshots/analyze.png) | ![Result view](screenshots/result.png) |

| About and limitations | Dark theme |
|---|---|
| ![About page](screenshots/about.png) | ![Dark theme](screenshots/dark.png) |

## Team member contributions

| Member | Role | Contributions |
|---|---|---|
| **Muhammad Rehan Farooq** | Team Leader, AI & Data Science | Project leadership and coordination, dataset analysis and cleaning pipeline, model evaluation and testing, limitations research, deployment |
| **Habiba Siddique** | Machine Learning & Web Development | Streamlit interface and styling |

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
