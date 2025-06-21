import os
import random
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def load_data(sample_train=2000, sample_test=400):
    print("🔄 Loading AG News dataset...")
    dataset = load_dataset("ag_news")
    df_train = pd.DataFrame(dataset["train"]).sample(n=sample_train, random_state=42)
    df_test = pd.DataFrame(dataset["test"]).sample(n=sample_test, random_state=42)
    return df_train, df_test


def train_tfidf(X_train, y_train, X_test, y_test):
    print("\n📚 Training TF-IDF model...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 TF-IDF Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    return clf, tfidf, X_test_tfidf, y_test


def train_sbert(X_train, y_train, X_test, y_test):
    print("\n🤖 Encoding SBERT embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X_train_emb = model.encode(X_train.tolist(), batch_size=32, show_progress_bar=True)
    X_test_emb = model.encode(X_test.tolist(), batch_size=32, show_progress_bar=True)

    print("\n🎯 Training SBERT model...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_emb, y_train)
    y_pred = clf.predict(X_test_emb)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 SBERT Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    return clf, X_test_emb, y_test


def save_models(clf_tfidf, tfidf_vectorizer, clf_sbert):
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf_tfidf, "models/tfidf_classifier.joblib")
    joblib.dump(tfidf_vectorizer, "models/tfidf_vectorizer.joblib")
    joblib.dump(clf_sbert, "models/sbert_classifier.joblib")
    print("\n💾 Models saved to 'models/' directory.")


def plot_umap(X_test_tfidf, X_test_emb_sbert, y_test):
    os.makedirs("plots", exist_ok=True)
    print("\n📷 Generating UMAP plots...")

    # TF-IDF + UMAP
    umap_tfidf = umap.UMAP(random_state=42).fit_transform(X_test_tfidf.toarray())
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=umap_tfidf[:, 0], y=umap_tfidf[:, 1], hue=y_test, palette="tab10", s=40)
    plt.title("TF-IDF + UMAP")
    plt.savefig("plots/tfidf_umap.png")
    plt.close()

    # SBERT + UMAP
    umap_sbert = umap.UMAP(random_state=42).fit_transform(X_test_emb_sbert)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=umap_sbert[:, 0], y=umap_sbert[:, 1], hue=y_test, palette="tab10", s=40)
    plt.title("SBERT + UMAP")
    plt.savefig("plots/sbert_umap.png")
    plt.close()

    print("📷 UMAP plots saved to 'plots/' directory.")


def main(args):
    set_seed()
    df_train, df_test = load_data(args.train_sample, args.test_sample)

    X_train, y_train = df_train["text"], df_train["label"]
    X_test, y_test = df_test["text"], df_test["label"]

    clf_tfidf, tfidf_vectorizer, X_test_tfidf, y_test = train_tfidf(X_train, y_train, X_test, y_test)
    clf_sbert, X_test_emb_sbert, y_test = train_sbert(X_train, y_train, X_test, y_test)

    save_models(clf_tfidf, tfidf_vectorizer, clf_sbert)
    plot_umap(X_test_tfidf, X_test_emb_sbert, y_test)

    print("\n✅ All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AG News Classification with TF-IDF and SBERT")
    parser.add_argument("--train_sample", type=int, default=2000, help="Number of training samples")
    parser.add_argument("--test_sample", type=int, default=400, help="Number of test samples")
    args = parser.parse_args()

    main(args)
