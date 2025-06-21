# 🗞️ News Classification with TF-IDF & SBERT

This project demonstrates a simple yet effective approach to classifying news headlines using two different text representation techniques: **TF-IDF** and **SBERT (Sentence-BERT)**.

---

## 🎯 Objective

Classify news articles from the [AG News dataset](https://huggingface.co/datasets/ag_news) into 4 categories:

- World
- Sports
- Business
- Sci/Tech

---

## 💡 Models Used

- `TF-IDF + Logistic Regression`
- `SBERT + Logistic Regression`

Each model is evaluated on classification accuracy and visualized using UMAP projections.

---

## 🗂️ Project Structure

```
news-classification-ml/
├── main.py                  # Main script to train and evaluate
├── models/                  # Saved TF-IDF and SBERT models
├── plots/                   # UMAP visualization images
├── requirements.txt         # Required dependencies
└── README.md                # This file
```

---

## ▶️ How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the classification pipeline with default sample size:

```bash
python main.py
```

Or specify custom sample sizes:

```bash
python main.py --train_sample 3000 --test_sample 500
```

---

## 📊 Results Example

```
TF-IDF Accuracy: 0.83
SBERT Accuracy: 0.865
```

Plots will be saved to the `plots/` folder:

- `plots/tfidf_umap.png`
- `plots/sbert_umap.png`

These show how the embeddings cluster news topics visually.

---

## 🛠️ Dependencies

```
transformers
sentence-transformers
scikit-learn
matplotlib
seaborn
umap-learn
datasets
joblib
```

(Full list in `requirements.txt`)

---

## 🧠 Insights

- TF-IDF performs well with simple logistic regression.
- SBERT improves accuracy by capturing semantic relationships.
- UMAP visualizations reveal strong topic separation.

\
MIT License

