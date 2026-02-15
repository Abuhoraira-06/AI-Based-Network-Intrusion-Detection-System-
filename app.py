import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from backend.data_loader import load_data
from backend.model import NIDSModel
from backend.groq_explainer import groq_explain

SYSTEM_API_KEY = "NIDS-SECURE-2026"

st.set_page_config(
    page_title="AI-Based Network Intrusion Detection System",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center;'>AI-Based Network Intrusion Detection System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>"
    "Real-time machine learning–based intrusion detection using "
    "<b>Random Forest algorithm</b>"
    "</p>",
    unsafe_allow_html=True
)

st.sidebar.markdown("### System Controls")
use_real_data = st.sidebar.checkbox("Use CIC-IDS2017 Dataset")
n_estimators = st.sidebar.slider("Number of Trees", 50, 300, 150)

if use_real_data:
    df = load_data(True, "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
else:
    df = load_data(False)

X = df.drop("Label", axis=1)
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

if "nids_model" not in st.session_state:
    st.session_state.nids_model = None
    st.session_state.trained = False

st.markdown("### Model Training")

if st.button("Train Model"):
    model = NIDSModel(n_estimators)
    model.train(X_train, y_train)
    st.session_state.nids_model = model
    st.session_state.trained = True
    st.success("Model trained successfully")

if st.session_state.trained:

    st.markdown("### Training Data Distribution")

    benign = (y_train == 0).sum()
    malicious = (y_train == 1).sum()

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(
        [benign, malicious],
        labels=["Benign", "Malicious"],
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

st.markdown("### Live Traffic Simulator")

c1, c2, c3, c4 = st.columns(4)

flow_duration = c1.number_input("Flow Duration", 0, 100000, 500)
total_packets = c2.number_input("Total Fwd Packets", 0, 2000, 300)
packet_len = c3.number_input("Packet Length Mean", 0, 1500, 800)
active_mean = c4.number_input("Active Mean", 0, 1000, 50)

if st.button("Analyze Traffic"):

    if st.session_state.nids_model is None:
        st.warning("Please train the model first")
    else:
        user_api_key = "NIDS-SECURE-2026"

        if user_api_key != SYSTEM_API_KEY:
            st.error("Invalid API Key")
        else:
            sample = np.array([[80, flow_duration, total_packets, packet_len, active_mean]])

            pred, confidence = st.session_state.nids_model.predict_with_confidence(sample)

            if total_packets > 700 and flow_duration < 500:
                pred = 1

            if pred == 1:
                st.error(f"Malicious Traffic Detected (Confidence: {confidence}%)")
            else:
                st.success(f"Benign Traffic Detected (Confidence: {confidence}%)")

            st.markdown("### AI Traffic Explanation")

            explanation = groq_explain(sample, pred, confidence)
            st.info(explanation)
