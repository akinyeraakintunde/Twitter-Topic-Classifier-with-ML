"""
Prediction utilities for the Twitter Topic Classifier.

This module assumes you already have a trained scikit-learn Pipeline
(e.g., from model_definitions.build_svm_pipeline()), which contains
both the TF-IDF vectoriser and the classifier.

You can either:
- pass a trained Pipeline into predict_topic(), OR
- load a saved model from disk (joblib) and call predict_topic() on it.
"""

from typing import Union, List
from sklearn.pipeline import Pipeline

from .preprocessing import clean_text

def predict_topic(
    text: Union[str, List[str]],
    model: Pipeline
) -> Union[str, List[str]]:
    """
    Predict the topic(s) of one or more input texts.

    :param text: A single text string or a list of text strings.
    :param model: A trained scikit-learn Pipeline with TF-IDF + classifier.
    :return: Predicted label(s) as string or list of strings.
    """
    if isinstance(text, str):
        cleaned = [clean_text(text)]
        preds = model.predict(cleaned)
        return preds[0]

    # Assume it's an iterable/list of texts
    cleaned_corpus = [clean_text(t) for t in text]
    preds = model.predict(cleaned_corpus)
    return list(preds)