import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from backend.data_loader import load_data
from backend.model import NIDSModel

st.set_page_config(
    page_title="AI-Based Network Intrusion Detection System",
    layout="wide"
)

st.markdown("""
<style>
:root {
    --bg-main:#0b102b;
    --bg-sidebar:#090d23;
    --card:#14184a;
    --border:#262b63;
    --primary:#6c63ff;
}
.stApp {
    background: linear-gradient(135deg, #070b1e, #0b102b);
    font-family: Inter, system-ui, sans-serif;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar);
    border-right: 1px solid var(--border);
}
.main-title {
    font-size: 3rem;
    font-weight: 900;
    text-align: center;
    white-space: nowrap;
    background: linear-gradient(135deg, #ffffff, #9fa3ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #dbe0ff;
    margin-bottom: 35px;
}
.section-header {
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 18px;
    background: linear-gradient(135deg, #ffffff, #9fa3ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.section {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 30px;
}
label { color: white !important; }
input {
    background-color: #10133a !important;
    color: white !important;
    border-radius: 10px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #5b54e6);
    color: white;
    border-radius: 14px;
}
footer, hr { display:none; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">AI-Based Network Intrusion Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time machine learning–based intrusion detection using <b>Random Forest algorithm</b></div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="section-header">System Controls</div>', unsafe_allow_html=True)
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

st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Model Training</div>', unsafe_allow_html=True)

if st.button("Train Model"):
    model = NIDSModel(n_estimators)
    model.train(X_train, y_train)
    st.session_state.nids_model = model
    st.success("Model trained successfully.")

st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.nids_model is not None:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Traffic Distribution</div>', unsafe_allow_html=True)

    benign_count = (y_test == 0).sum() if y_test.dtype != 'object' else (y_test == "Benign").sum()
    malicious_count = (y_test == 1).sum() if y_test.dtype != 'object' else (y_test != "Benign").sum()

    labels = ["Benign", "Malicious"]
    sizes = [benign_count, malicious_count]
    colors = ["#22c55e", "#ef4444"]

    fig, ax = plt.subplots(figsize=(3, 3))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        textprops={"color": "white", "fontsize": 10}
    )
    ax.set_title("Benign vs Malicious Traffic", fontsize=11, color="white")

    st.pyplot(fig, use_container_width=False)

    st.markdown("""
    <div style="font-size:0.85rem;">
        <b style="color:#22c55e;">Green</b> – Benign<br>
        <b style="color:#ef4444;">Red</b> – Malicious
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-header">Live Traffic Simulator</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

flow_duration = c1.number_input("Flow Duration", 0, 100000, 500)
total_packets = c2.number_input("Total Fwd Packets", 0, 2000, 300)
packet_len = c3.number_input("Packet Length Mean", 0, 1500, 800)
active_mean = c4.number_input("Active Mean", 0, 1000, 50)

if st.button("Analyze Traffic"):
    if st.session_state.nids_model is None:
        st.warning("Please train the model first.")
    else:
        sample = np.array([[80, flow_duration, total_packets, packet_len, active_mean]])
        pred = st.session_state.nids_model.predict(sample)

        if total_packets > 700 and flow_duration < 500:
            pred = 1

        if pred == 1:
            st.error("🚨 Malicious Traffic Detected")
        else:
            st.success("✅ Benign Traffic")

st.markdown('</div>', unsafe_allow_html=True)