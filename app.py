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
st.set_page_config(page_title="Environmental Risk System", layout="wide")

# Small font
st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
xgb = joblib.load("toxicity_xgboost_model.pkl")

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
st.title("🌍 Chemical Toxicity and Environmental Risk Prediction System")

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("🧪 Input")
    smiles = st.text_input("Enter SMILES")
    analyze = st.button("Analyze")

    if analyze:

        prob, label = predict_toxicity(smiles)

        if prob is None:
            st.error("❌ Invalid SMILES")

        else:
            air, water, soil, env = environmental_risk(smiles)

            st.success(f"⚠️ Toxicity: {prob:.4f}")
            # ✅ COLOR LABEL FIX
        if label == "TOXIC":
            st.error(f"🔬 Label: {label}")   # RED
        else:
            st.success(f"🔬 Label: {label}") # GREEN

        if prob < 0.3:
                tox_level = "LOW"
        elif prob < 0.7:
                tox_level = "MEDIUM"
        else:
                tox_level = "HIGH"

        st.write(f"🧪 Toxicity Level: **{tox_level}**")

            # SAVE FOR GRAPH
        st.session_state.air = air
        st.session_state.water = water
        st.session_state.soil = soil
        st.session_state.env = env

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
        