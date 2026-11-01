import streamlit as st
import pickle
import numpy as np

# Set up page configurations
st.set_page_config(
    page_title="Fraud Guard Dashboard",
    page_icon="💳",
    layout="centered"
)

# Load the trained model safely using a cache so it doesn't reload on every click
@st.cache_resource
def load_model():
    with open('fraud_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# App Header Styling
st.title("💳 Credit Card Fraud Detection System")
st.markdown("---")

# --- CONTEXTUAL EXPLANATION FOR COLLEGE EVALUATION ---
st.markdown("""
### 📊 Project Context & Dataset Architecture
This system utilizes a machine learning engine trained on real, anonymized financial transactions. 

* **Time & Amount:** Represent the actual metadata of the transaction event.
* **The V1 - V28 Dimensions:** Due to strict **Data Privacy & Confidentiality Regulations**, the original sensitive tracking metrics (such as customer card profiles, merchant IDs, transaction locations, and spending categories) have been transformed using **PCA (Principal Component Analysis)**. These represent compressed, multi-dimensional patterns of user behavior optimized for detecting anomalies without exposing personal identifiable information (PII).
""")
st.markdown("---")

# Form arrangement using columns for layout organization
st.subheader("🛠️ Input Transaction Parameters")
col1, col2 = st.columns(2)

with col1:
    time = st.number_input("Transaction Time (Seconds elapsed since first transaction)", min_value=0.0, value=0.0, step=1.0)
with col2:
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=10.0, step=0.01)

# Grouping PCA V1-V28 inputs cleanly inside an expander so the page isn't an endless scroll
with st.expander("⚙️ Advanced Latent Features (Anonymized PCA Dimensions V1 - V28)"):
    st.info("Adjust the underlying behavioral feature dimensions below. Default baseline value is 0.0.")
    v_inputs = []
    
    # Generate input boxes dynamically for all 28 latent dimensions
    for i in range(1, 29, 2):
        c1, c2 = st.columns(2)
        with c1:
            val1 = st.number_input(f"Feature Column V{i}", value=0.0, step=0.1)
            v_inputs.append(val1)
        with c2:
            if i + 1 <= 28:
                val2 = st.number_input(f"Feature Column V{i+1}", value=0.0, step=0.1)
                v_inputs.append(val2)

# Evaluation Logic Trigger
if st.button("Run Live Fraud Analysis", type="primary", use_container_width=True):
    # Construct feature matrix exactly matching training structure: [Time, V1...V28, Amount]
    features = [time] + v_inputs + [amount]
    features_array = np.array([features])
    
    # Run prediction
    prediction = model.predict(features_array)
    
    st.markdown("### 🔔 System Diagnostic Output")
    if prediction == 1:
        st.error("⚠️ CRITICAL ALERT: High probability of Fraudulent Activity detected!")
    else:
        st.success("✅ TRANSACTION VERIFIED: Low risk profile. Operation authorized.")
