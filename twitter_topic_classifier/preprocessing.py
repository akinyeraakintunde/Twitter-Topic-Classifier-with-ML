"""
Text preprocessing utilities for the Twitter Topic Classifier.
"""

import re
from typing import Iterable, List

URL_PATTERN = re.compile(r"http\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-zA-Z0-9\s]+")

def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - lowercasing
    - remove URLs, mentions, hashtags
    - strip non-alphanumeric characters
    """
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    text = HASHTAG_PATTERN.sub(" ", text)
    text = NON_ALPHANUMERIC_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_corpus(corpus: Iterable[str]) -> List[str]:
    """
    Apply clean_text to an iterable of documents.
    """
    return [clean_text(doc) for doc in corpus]
