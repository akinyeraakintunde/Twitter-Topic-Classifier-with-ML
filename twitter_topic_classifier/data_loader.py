"""
Data loading utilities for the Twitter Topic Classifier.

These helpers assume that, after extraction, each class has
its own folder containing text files, for example:

data/
  politics/
  sport/
  entertainment/
"""

import os
from pathlib import Path
from typing import List, Tuple

def load_texts_from_folder(folder_path: str, label: str) -> Tuple[List[str], List[str]]:
    """
    Load all .txt files from a folder and assign the given label.

    :param folder_path: Path to the folder containing .txt files.
    :param label: Label to assign ("Politics", "Sports", "Entertainment").
    :return: (texts, labels) lists.
    """
    folder = Path(folder_path)
    texts = []
    labels = []

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    for file in folder.glob("*.txt"):
        try:
            content = file.read_text(encoding="utf-8", errors="ignore")
            texts.append(content)
            labels.append(label)
        except Exception as e:
            print(f"Warning: could not read {file}: {e}")

    return texts, labels
