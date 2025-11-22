import os
import glob
import pandas as pd

def load_data_from_folder(folder_path, label):
    texts = []
    labels = []
    for filepath in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
            if text:
                texts.append(text)
                labels.append(label)
    return texts, labels

def load_all_data(base_dir="data"):
    X, y = [], []

    for label in ["politics", "sport", "entertainment"]:
        folder = os.path.join(base_dir, label)
        t, l = load_data_from_folder(folder, label)
        X.extend(t)
        y.extend(l)

    return X, y

if __name__ == "__main__":
    X, y = load_all_data()
    print(f"Loaded {len(X)} samples.")
    # then continue with TF–IDF, train/test split, model training, etc.
