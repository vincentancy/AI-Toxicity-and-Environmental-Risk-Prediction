import streamlit as st
import numpy as np
import joblib
import pandas as pd
import altair as alt

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Chemical Toxicity Prediction",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Small font
st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

h1 {
    color: #0E4C92;
    text-align:center;
}

h2,h3 {
    color:#0E4C92;
}

.stButton>button{
    background:#0E4C92;
    color:white;
    border-radius:10px;
    height:45px;
    width:100%;
    font-size:16px;
}

.stButton>button:hover{
    background:#1565C0;
}

div[data-testid="stMetric"]{
    background:white;
    border-radius:12px;
    padding:15px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.1);
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)
with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/artificial-intelligence.png",
        width=80
    )

    st.title("AI Toxicity System")

    st.markdown("---")

    st.write("### Features")

    st.success("✔ Chemical Toxicity Prediction")
    st.success("✔ Environmental Risk")
    st.success("✔ Air Risk")
    st.success("✔ Water Risk")
    st.success("✔ Soil Risk")

    st.markdown("---")

    st.info("Built using\n\nPython • RDKit • XGBoost • Streamlit")
# =========================
# LOAD MODEL
# =========================
xgb = joblib.load("toxicity_xgboost_model.pkl")
# =========================
# PREDICTION HISTORY
# =========================
if "history" not in st.session_state:
    st.session_state.history = []
# =========================
# SMILES → FP
# =========================
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(list(fp), dtype=np.int8)

# =========================
# TOXICITY
# =========================
def predict_toxicity(smiles):
    fp = smiles_to_fp(smiles)

    if fp is None:
        return None, None

    fp = fp.reshape(1, -1)
    prob = xgb.predict_proba(fp)[0][1]

    label = "TOXIC" if prob > 0.35 else "NON-TOXIC"
    return prob, label

# =========================
# ENVIRONMENT RISK
# =========================
def environmental_risk(smiles):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)

    water = min(1, max(0, logp / 5))
    air = min(1, max(0, mw / 500))
    soil = min(1, max(0, tpsa / 150))

    def level(x):
        if x < 0.3:
            return "LOW"
        elif x < 0.7:
            return "MEDIUM"
        else:
            return "HIGH"

    return air, water, soil, level((air + water + soil) / 3)

# =========================
# TITLE
# =========================
st.markdown("""
# 🌍 AI-Based Chemical Toxicity Prediction

### Environmental Risk Assessment using Machine Learning

Predict the toxicity of chemical compounds from **SMILES notation**
and estimate their environmental impact.
""")

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("🧪 Input")
    smiles = st.text_input("Enter SMILES")
    col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    analyze = st.button("🔍 Analyze")

with col_btn2:
    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

if analyze:
    prob, label = predict_toxicity(smiles)

    if prob is None:
        st.error("❌ Invalid SMILES")
    else:
        air, water, soil, env = environmental_risk(smiles)

        st.success(f"⚠️ Toxicity: {prob*100:.2f}%")

        confidence = int(prob * 100)
        st.progress(confidence)

        if label == "TOXIC":
            st.error(f"🔬 Label: {label}")
        else:
            st.success(f"🔬 Label: {label}")

        if prob < 0.3:
            tox_level = "LOW"
        elif prob < 0.7:
            tox_level = "MEDIUM"
        else:
            tox_level = "HIGH"

        st.write(f"🧪 Toxicity Level: **{tox_level}**")

        st.session_state.air = air
        st.session_state.water = water
        st.session_state.soil = soil
        st.session_state.env = env

        st.session_state.history.append({
            "SMILES": smiles,
            "Prediction": label,
            "Probability": f"{prob*100:.2f}%",
            "Toxicity Level": tox_level,
            "Environmental Risk": env
        })

# =========================
# GRAPH (SAFE)
# =========================
with col2:

    st.subheader("🌱 Environmental Risk Graph")

    if "air" in st.session_state:

        df = pd.DataFrame({
            "Category": ["Air", "Water", "Soil"],
            "Risk Score": [
                st.session_state.air,
                st.session_state.water,
                st.session_state.soil
            ]
        })

        chart = alt.Chart(df).mark_bar(size=40).encode(
            x="Category",
            y="Risk Score",
            color=alt.Color("Category", scale=alt.Scale(
                domain=["Air", "Water", "Soil"],
                range=["#6BCB77", "#4D96FF","#FF6B6B"]
            ))
        )

        st.altair_chart(chart, use_container_width=True)

        st.warning(f"🌍 Final Risk: **{st.session_state.env}**")

    else:
        st.info("Run analysis first")
# =========================
# PREDICTION HISTORY
# =========================

if len(st.session_state.history) > 0:

    st.subheader("📜 Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

st.markdown(
"""
<center>

Developed by **Ancy Evelyne**

MCA Final Year Project

AI | Machine Learning | Environmental Risk Assessment

</center>
""",
unsafe_allow_html=True)
        
