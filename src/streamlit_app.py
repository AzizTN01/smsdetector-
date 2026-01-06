import os
import json
import joblib
import streamlit as st
import pandas as pd

MODEL_PATH = os.path.join("models", "spam_classifier.joblib")
METRICS_PATH = os.path.join("models", "spam_classifier_metrics.json")

st.set_page_config(page_title="SMS Spam Detector", layout="wide")
st.title("SMS Spam Detector")

# Load model if available
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
else:
    st.warning("Model not found. Please run training first (src.train) to create models/spam_classifier.joblib")

# Load metrics if available
metrics = None
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")

# Left column: input
col1, col2 = st.columns([2, 1])
with col1:
    st.header("Classify a message")
    text = st.text_area("Enter SMS text to classify", height=120)
    if st.button("Predict"):
        if not model:
            st.error("Model not available. Train the model first.")
        elif not text:
            st.info("Please enter some text to classify")
        else:
            pred = model.predict([text])[0]
            prob = None
            if hasattr(model, 'predict_proba'):
                prob = float(model.predict_proba([text])[0][1])
            label = 'spam' if int(pred) == 1 else 'ham'
            if prob is not None:
                st.success(f"Prediction: {label} (spam_prob={prob:.3f})")
            else:
                st.success(f"Prediction: {label}")

    st.markdown("---")
    st.header("Batch classify (upload .txt)")
    uploaded = st.file_uploader("Upload a TXT file with one message per line", type=['txt'])
    if uploaded is not None:
        try:
            content = uploaded.read().decode('utf-8').splitlines()
            if not model:
                st.error("Model not available. Train the model first.")
            else:
                preds = model.predict(content)
                probs = None
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(content)[:, 1]
                df = pd.DataFrame({'text': content, 'pred': preds})
                df['label'] = df['pred'].map({0: 'ham', 1: 'spam'})
                if probs is not None:
                    df['spam_prob'] = probs
                st.dataframe(df)
        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")

with col2:
    st.header("Model & Metrics")
    if model:
        st.write("Model loaded from:", MODEL_PATH)
        st.write(model)
    if metrics:
        st.subheader("Classification report (summary)")
        # display accuracy and per-class f1
        acc = metrics.get('classification_report', {}).get('accuracy')
        if acc is not None:
            st.metric("Accuracy", f"{acc:.4f}")
        st.write("\n**Per-class (spam=1)**")
        try:
            class1 = metrics['classification_report']['1']
            st.write(f"Precision: {class1.get('precision'):.3f}")
            st.write(f"Recall: {class1.get('recall'):.3f}")
            st.write(f"F1-score: {class1.get('f1-score'):.3f}")
            st.write(f"Support: {class1.get('support')}")
        except Exception:
            st.write(metrics.get('classification_report'))
        st.markdown("---")
        st.subheader("Confusion matrix")
        try:
            cm = metrics.get('confusion_matrix')
            if cm:
                cm_df = pd.DataFrame(cm, index=['true_ham', 'true_spam'], columns=['pred_ham', 'pred_spam'])
                st.table(cm_df)
        except Exception:
            st.write(metrics.get('confusion_matrix'))
    else:
        st.info("No metrics available. Train model to generate metrics JSON.")

st.markdown("---")
st.caption("Built with Streamlit — run: streamlit run src/streamlit_app.py")

