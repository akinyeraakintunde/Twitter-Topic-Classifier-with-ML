"""
Model definitions for the Twitter Topic Classifier.

Defines a simple pipeline builder using scikit-learn.
"""

from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

def build_logistic_regression_pipeline() -> Pipeline:
    """
    TF-IDF + Logistic Regression pipeline.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])

def build_svm_pipeline() -> Pipeline:
    """
    TF-IDF + Linear SVM pipeline.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LinearSVC())
    ])

def build_naive_bayes_pipeline() -> Pipeline:
    """
    TF-IDF + Multinomial Naive Bayes pipeline.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", MultinomialNB())
    ])

def get_all_models() -> Dict[str, Pipeline]:
    """
    Convenience function returning a dictionary of candidate models.
    """
    return {
        "logreg": build_logistic_regression_pipeline(),
        "svm": build_svm_pipeline(),
        "nb": build_naive_bayes_pipeline()
    }
