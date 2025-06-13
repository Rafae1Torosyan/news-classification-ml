# News Classification ML

A project for classifying news headlines using TF-IDF and SBERT models. This project trains classifiers on the AG News dataset using TF-IDF with Logistic Regression and SBERT embeddings with a classifier. UMAP visualizations are generated to assess the quality of embeddings. Models achieve strong precision, recall, and F1-score across all classes.

| Model   | Accuracy (%) |
|---------|--------------|
| TF-IDF  | 83           |
| SBERT   | 86.5         |

Install dependencies and run the script:

```bash
pip install -r requirements.txt
python news_classification_ml.py
```
Saved models are in the models/ folder and UMAP visualizations are in the plots/ folder.
