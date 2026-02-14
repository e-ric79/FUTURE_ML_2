import streamlit as st
import joblib
import re
import string
import numpy as np

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Ticket Classification System",
    page_icon="🎫",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL & VECTORIZER
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load('ticket_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    categories = joblib.load('categories.pkl')
    return model, vectorizer, categories

model, vectorizer, categories = load_model()

# ---------------------------------------------------
# TEXT CLEANING FUNCTION
# ---------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.title("🎫 AI Support Ticket Classification System")
st.markdown("Automatically classify support tickets into categories using Machine Learning.")
st.markdown("---")

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------
st.subheader("📝 Enter Support Ticket")

input_method = st.radio("Choose input method:", ["Type Text", "Use Example"])

examples = {
    "Hardware Issue": "My laptop screen is flickering and I cannot connect to the monitor. Need urgent hardware support.",
    "HR Support": "I need to update my personal information and access my payroll records for tax purposes.",
    "Access Request": "Please grant me access to the project management tool and shared drive for the new project.",
    "Storage Request": "I need additional cloud storage space. My current allocation is full.",
    "Purchase Request": "Need approval to purchase new software licenses for the design team.",
    "Network Issue": "The internet connection keeps dropping and I cannot access internal systems."
}

if input_method == "Type Text":
    ticket_text = st.text_area(
        "Ticket Description:",
        height=150,
        placeholder="Example: I need urgent access to the HR portal..."
    )
else:
    selected_example = st.selectbox("Select an example:", list(examples.keys()))
    ticket_text = examples[selected_example]
    st.text_area("Ticket Description:", value=ticket_text, height=120, disabled=True)

st.markdown("---")

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if st.button("🔮 Classify Ticket", type="primary"):

    if ticket_text and len(ticket_text.strip()) > 0:

        # Clean
        cleaned = clean_text(ticket_text)

        # Vectorize
        text_vec = vectorizer.transform([cleaned])

        # Predict
        prediction = model.predict(text_vec)[0]
        probabilities = model.predict_proba(text_vec)[0]

        confidence = np.max(probabilities)

        # ===============================
        # RESULTS SECTION
        # ===============================
        st.subheader("📂 Classification Results")

        st.success(f"Primary Category: {prediction}")

        st.metric("Confidence", f"{confidence*100:.1f}%")
        st.progress(float(confidence))

        # Low confidence warning
        if confidence < 0.60:
            st.warning("Low confidence prediction — manual review recommended.")

        st.markdown("---")
        st.subheader("Top 3 Predictions")

        top3_idx = probabilities.argsort()[-3:][::-1]

        for idx in top3_idx:
            st.write(f"**{categories[idx]}** — {probabilities[idx]*100:.1f}%")
            st.progress(float(probabilities[idx]))

    else:
        st.warning("⚠️ Please enter a ticket description.")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("📊 Model Information")

st.sidebar.markdown("### Model Details")
st.sidebar.write("Algorithm: Logistic Regression")
st.sidebar.write("Accuracy: 85.38%")
st.sidebar.write("Training Samples: 47,837")
st.sidebar.write("Categories: 8")

st.sidebar.markdown("---")

st.sidebar.markdown("### Feature Engineering")
st.sidebar.write("TF-IDF (5,000 features)")
st.sidebar.write("N-grams: (1,2)")
st.sidebar.write("Stopwords: English")

st.sidebar.markdown("---")

st.sidebar.subheader("Categories")
for cat in sorted(categories):
    st.sidebar.write(f"• {cat}")

st.sidebar.markdown("---")
st.sidebar.caption("Built by Eric 🚀")
st.sidebar.caption("GitHub: https://github.com/e-ric79")
