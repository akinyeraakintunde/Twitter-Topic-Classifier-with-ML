"""
script_model.py

Train and evaluate a multi-class social media topic classifier
(Politics, Sports, Entertainment) using the twitter_topic_classifier package.
"""

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from twitter_topic_classifier import config
from twitter_topic_classifier.data_loader import load_texts_from_folder
from twitter_topic_classifier.preprocessing import clean_corpus
from twitter_topic_classifier.model_definitions import get_all_models
from twitter_topic_classifier.predict import predict_topic


def load_dataset(data_root: str = "data") -> tuple[list[str], list[str]]:
    """
    Load and label all documents from the data folders.

    Expected structure AFTER extraction:
        data/
          politics/
          sport/
          entertainment/
    """
    root = Path(data_root)

    politics_texts, politics_labels = load_texts_from_folder(
        root / "politics", label="Politics"
    )
    sport_texts, sport_labels = load_texts_from_folder(
        root / "sport", label="Sports"
    )
    entertainment_texts, entertainment_labels = load_texts_from_folder(
        root / "entertainment", label="Entertainment"
    )

    texts = politics_texts + sport_texts + entertainment_texts
    labels = politics_labels + sport_labels + entertainment_labels
    return texts, labels


def main():
    # 1. Load raw data
    print("[INFO] Loading dataset...")
    texts, labels = load_dataset("data")
    print(f"[INFO] Loaded {len(texts)} documents.")

    # 2. Clean text
    print("[INFO] Cleaning text...")
    texts_clean = clean_corpus(texts)

    # 3. Train/test split
    print("[INFO] Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts_clean,
        labels,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=labels,
    )

    # 4. Get candidate models
    models = get_all_models()

    best_model_name = None
    best_model = None
    best_f1 = 0.0

    # 5. Train and evaluate each model
    for name, model in models.items():
        print(f"\n[INFO] Training model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"[RESULTS] Model: {name}")
        print(classification_report(y_test, y_pred))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        # simple F1 macro score to pick a best model
        from sklearn.metrics import f1_score
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"[INFO] F1-macro for {name}: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    print("\n[INFO] Best model:", best_model_name, "with F1-macro:", round(best_f1, 4))

    # 6. Example prediction using best_model + package-level predict_topic
    if best_model is not None:
        example_text = "The president announced a new policy today."
        predicted_label = predict_topic(example_text, best_model)
        print("\n[DEMO] Example text:", example_text)
        print("[DEMO] Predicted label:", predicted_label)


if __name__ == "__main__":
    main()