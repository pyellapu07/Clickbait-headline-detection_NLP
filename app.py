import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------
# Load model (LOCAL checkpoint)
# ----------------------------
MODEL_PATH = r"F:\641 PWS2 NLP\final p\distilbert_clickbait\checkpoint-800"

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Force a normal CPU load (avoids "meta tensor" issues)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    device_map=None
).to("cpu")
model.eval()

# -----------
# UI
# -----------
st.set_page_config(page_title="Clickbait Headline Detector", page_icon="📰")
st.title("Clickbait Headline Detector")
st.write("Enter a news headline to see whether the model predicts it as clickbait.")

headline = st.text_input("Headline", value="")

if st.button("Predict"):
    if headline.strip() == "":
        st.warning("Please enter a headline.")
    else:
        inputs = tokenizer(
            headline,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs[0, pred].item()

        label = "Clickbait 🚨" if pred == 1 else "Legitimate ✅"
        st.success(f"Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2%}**")

# -----------
# Footer
# -----------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:0.9em; color:#9aa0a6;'>"
    "Built as an academic NLP project by <b>Pradeep Yellapu, Harshitha Murali, Sriniketh Shankar, and Girik</b>."
    "</div>",
    unsafe_allow_html=True
)
