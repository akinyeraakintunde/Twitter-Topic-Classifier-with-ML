import os
import streamlit as st
import numpy as np

from model_runtime import load_or_train

st.set_page_config(page_title="Twitter Topic Classifier (Live Demo)", layout="centered")

st.title("🧠 Twitter Topic Classifier (Live Demo)")
st.caption("Type a tweet-like sentence → get predicted topic + confidence + explanation keywords.")


# Paths (safe for Render)
MODEL_PATH = os.getenv("MODEL_PATH", "outputs/model_bundle.joblib")
DATASET_PATH = os.getenv("DATASET_PATH", "data/training.csv")


@st.cache_resource
def get_model_bundle():
    return load_or_train(MODEL_PATH, DATASET_PATH)


bundle = get_model_bundle()
pipe = bundle["pipeline"]

text = st.text_area("Enter text", height=120, placeholder="e.g., Bitcoin is pumping again…")

def top_keywords_for_text(text_in: str, top_k: int = 8):
    """
    Simple explainability:
    show top TF-IDF tokens present in the input.
    """
    try:
        tfidf = pipe.named_steps["tfidf"]
        X = tfidf.transform([text_in])
        if X.nnz == 0:
            return []
        feature_names = np.array(tfidf.get_feature_names_out())
        weights = X.toarray().ravel()
        top_idx = weights.argsort()[-top_k:][::-1]
        return [str(feature_names[i]) for i in top_idx if weights[i] > 0]
    except Exception:
        return []


if st.button("Classify"):
    if not text.strip():
        st.warning("Please type something.")
        st.stop()

    # Predict
    pred = pipe.predict([text])[0]

    # Confidence (if available)
    conf = None
    try:
        proba = pipe.predict_proba([text])[0]
        conf = float(np.max(proba))
    except Exception:
        conf = None

    st.subheader("Result")
    st.write(f"**Predicted topic:** {pred}")
    if conf is not None:
        st.write(f"**Confidence:** {conf:.2%}")

    st.subheader("Explanation keywords")
    kws = top_keywords_for_text(text)
    if kws:
        st.write(", ".join([f"`{k}`" for k in kws]))
    else:
        st.write("No strong keywords detected for this input.")