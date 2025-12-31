import os
import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Twitter Topic Classifier Demo", page_icon="🧠", layout="centered")

st.title("🧠 Twitter Topic Classifier (Live Demo)")
st.caption("Type a tweet-like sentence → get predicted topic + confidence + explanation keywords.")

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")

@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

bundle = load_model(MODEL_PATH)

if bundle is None:
    st.error(
        "Model file not found.\n\n"
        "Expected: `models/model.joblib`\n"
        "Fix: Upload your trained model to `models/model.joblib` (see instructions below)."
    )
    st.stop()

model = bundle["model"]
vectorizer = bundle["vectorizer"]
labels = bundle["labels"]  # list like ["Politics","Sports","Entertainment"]

text = st.text_area("Enter text", height=120, placeholder="Example: The match tonight was unreal 🔥")
col1, col2 = st.columns([1, 1])

with col1:
    run = st.button("Predict", use_container_width=True)

with col2:
    show_explain = st.toggle("Show explanation keywords", value=True)

def top_keywords_for_prediction(x_vec, k=10):
    # Works for linear models with coef_ (SVM linear, LogisticRegression)
    if not hasattr(model, "coef_"):
        return []
    coefs = model.coef_
    pred_idx = int(np.argmax(model.decision_function(x_vec)))
    feature_names = vectorizer.get_feature_names_out()
    weights = coefs[pred_idx]
    top_idx = np.argsort(weights)[::-1][:k]
    return [(feature_names[i], float(weights[i])) for i in top_idx]

if run:
    if not text.strip():
        st.warning("Type something first.")
        st.stop()

    X = vectorizer.transform([text])
    # confidence: best-effort (works for models with predict_proba, else uses decision_function scale)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        pred_i = int(np.argmax(probs))
        confidence = float(probs[pred_i])
    else:
        scores = model.decision_function(X)
        if scores.ndim > 1:
            pred_i = int(np.argmax(scores))
            # softmax-ish for display only
            exp = np.exp(scores - np.max(scores))
            probs = exp / np.sum(exp)
            confidence = float(probs[0][pred_i])
        else:
            pred_i = int(np.argmax(scores))
            confidence = 0.0

    pred_label = labels[pred_i]

    st.success(f"Prediction: **{pred_label}**")
    if confidence > 0:
        st.metric("Confidence (approx.)", f"{confidence:.2%}")

    if show_explain:
        kws = top_keywords_for_prediction(X, k=12)
        if kws:
            st.subheader("Why this prediction (top weighted keywords)")
            st.write(", ".join([f"`{w}`" for w, _ in kws]))
        else:
            st.info("Explanation not available for this model type (needs a linear model with `coef_`).")

st.divider()
st.caption("Tip: Keep your model lightweight for fast cold-start deploys (joblib + TF-IDF + linear SVM is perfect).")