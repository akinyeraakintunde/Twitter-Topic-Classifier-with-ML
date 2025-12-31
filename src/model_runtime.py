import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


DEFAULT_LABELS = ["Politics", "Sports", "Business", "Tech", "Entertainment", "Health"]


def _pick_text_col(df: pd.DataFrame) -> str:
    # common names used in tweet datasets
    for c in ["text", "tweet", "content", "clean_text", "Tweet", "Text"]:
        if c in df.columns:
            return c
    # fallback: first object column
    for c in df.columns:
        if df[c].dtype == "object":
            return c
    raise ValueError("No text column found. Add a column named 'text' to your dataset.")


def _pick_label_col(df: pd.DataFrame) -> str:
    for c in ["label", "topic", "category", "class", "y"]:
        if c in df.columns:
            return c
    raise ValueError("No label column found. Add a column named 'label' (topic/category).")


def train_from_csv(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)

    text_col = _pick_text_col(df)
    label_col = _pick_label_col(df)

    X = df[text_col].astype(str).fillna("")
    y = df[label_col].astype(str).fillna("")

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=None)),
        ]
    )
    pipe.fit(X, y)

    labels = sorted(list(set(y))) if len(set(y)) > 0 else DEFAULT_LABELS
    return {"pipeline": pipe, "labels": labels, "text_col": text_col, "label_col": label_col}


def load_or_train(model_path: str, dataset_path: str) -> dict:
    """
    Returns:
      bundle = {"pipeline": <sklearn Pipeline>, "labels": [...], ...}
    """
    # 1) Try to load saved model
    if model_path and os.path.exists(model_path):
        try:
            bundle = joblib.load(model_path)
            # allow either raw pipeline or dict bundle
            if isinstance(bundle, Pipeline):
                return {"pipeline": bundle, "labels": DEFAULT_LABELS}
            if isinstance(bundle, dict) and "pipeline" in bundle:
                return bundle
            # if it loads but unexpected format
            return {"pipeline": bundle, "labels": DEFAULT_LABELS}
        except Exception:
            # fall back to training
            pass

    # 2) Train from dataset
    if not dataset_path or not os.path.exists(dataset_path):
        raise FileNotFoundError(
            "Model failed to load and dataset file was not found. "
            "Add a training CSV under /data (or update DATASET_PATH)."
        )

    bundle = train_from_csv(dataset_path)

    # 3) Save for faster future loads (optional)
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        try:
            joblib.dump(bundle, model_path)
        except Exception:
            # saving is optional; demo should still work
            pass

    return bundle