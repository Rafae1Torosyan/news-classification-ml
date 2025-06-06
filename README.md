
# 📰 News Title Classifier

A machine learning project for classifying news article titles into 4 categories: **World**, **Sports**, **Business**, **Sci/Tech**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18YJHe3GDiCql_pvt6m_L6TkVB4LYdP39?usp=sharing)

## 🚀 Models & Accuracy

| Model                     | Accuracy |
|---------------------------|----------|
| SBERT + LogisticRegression | **82.10%**   |
| TF-IDF + LogisticRegression| **79.30%**   |

## 🔧 Tools & Libraries

- AG News dataset
- `sentence-transformers` (SBERT)
- `TfidfVectorizer`
- `LogisticRegression` from scikit-learn
- `UMAP` for visualization
- `seaborn`, `matplotlib` for plotting

## 📁 Project Structure

- `notebook.ipynb` — main Google Colab notebook
- `models/` — trained models saved with `joblib`
- `requirements.txt` — Python dependencies

## 💻 How to Run

Open `notebook.ipynb` in **Google Colab**.  
All data will download automatically — no setup needed.

---

👤 *Author: [Your Name]*  
🎯 *Goal: Showcase ML/NLP skills for entry-level ML developer position*
