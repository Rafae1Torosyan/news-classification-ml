# News Classification ML

Welcome to the News Classification project! 🚀

This project tackles the challenge of classifying news headlines into categories using two powerful approaches: classic TF-IDF with Logistic Regression, and modern SBERT embeddings with a classifier. We trained these models on the popular AG News dataset.

To get a better feel of how the models understand the data, we also generate UMAP visualizations that show how well the embeddings capture the news topics.

Here’s a quick overview of how the models performed:

| Model   | Accuracy (%) |
|---------|--------------|
| TF-IDF  | 83           |
| SBERT   | 86.5         |

Both models achieve strong results, delivering reliable precision, recall, and F1-scores across all categories.

## Getting Started

To try this project yourself, just install the required packages and run the main script:

```bash
pip install -r requirements.txt
python news_classification_ml.py
```
After running, you’ll find the trained models saved in the models/ directory and the UMAP visualization plots in the plots/ folder.

Feel free to explore, tweak, and build on top of this foundation. Happy coding! 🎉
