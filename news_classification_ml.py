import os
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import umap

# 🚀 Reproducibility
random.seed(42)
np.random.seed(42)

print("🔄 Loading AG News dataset...")
dataset = load_dataset("ag_news")
df = pd.DataFrame(dataset["train"]).sample(n=2000, random_state=42)  # 2k для скорости
df_test = pd.DataFrame(dataset["test"]).sample(n=400, random_state=42)

X_train, y_train = df["text"], df["label"]
X_test, y_test = df_test["text"], df_test["label"]

# 📚 Training TF-IDF model
print("\n📚 Training TF-IDF model...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

clf_tfidf = LogisticRegression(max_iter=1000)
clf_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = clf_tfidf.predict(X_test_tfidf)

print("\n📊 TF-IDF Accuracy:", round(accuracy_score(y_test, y_pred_tfidf), 4))
print(classification_report(y_test, y_pred_tfidf))

# 🤖 Training SBERT model
print("\n🤖 Encoding SBERT embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
X_train_sbert = model.encode(X_train.tolist(), batch_size=32, show_progress_bar=True)
X_test_sbert = model.encode(X_test.tolist(), batch_size=32, show_progress_bar=True)

print("\n🎯 Training SBERT model...")
clf_sbert = LogisticRegression(max_iter=1000)
clf_sbert.fit(X_train_sbert, y_train)
y_pred_sbert = clf_sbert.predict(X_test_sbert)

print("\n📊 SBERT Accuracy:", round(accuracy_score(y_test, y_pred_sbert), 4))
print(classification_report(y_test, y_pred_sbert))

# 💾 Save models
os.makedirs("models", exist_ok=True)
joblib.dump(clf_tfidf, "models/tfidf_classifier.joblib")
joblib.dump(tfidf, "models/tfidf_vectorizer.joblib")
joblib.dump(clf_sbert, "models/sbert_classifier.joblib")

# 📉 UMAP plots
print("\n📷 Generating UMAP plots...")
os.makedirs("plots", exist_ok=True)

# TF-IDF UMAP
umap_tfidf = umap.UMAP(random_state=42).fit_transform(X_test_tfidf.toarray())
plt.figure(figsize=(8, 6))
sns.scatterplot(x=umap_tfidf[:, 0], y=umap_tfidf[:, 1], hue=y_test, palette="tab10", s=40)
plt.title("TF-IDF + UMAP")
plt.savefig("plots/tfidf_umap.png")
plt.close()

# SBERT UMAP
umap_sbert = umap.UMAP(random_state=42).fit_transform(X_test_sbert)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=umap_sbert[:, 0], y=umap_sbert[:, 1], hue=y_test, palette="tab10", s=40)
plt.title("SBERT + UMAP")
plt.savefig("plots/sbert_umap.png")
plt.close()

print("\n✅ Models and plots saved!")
