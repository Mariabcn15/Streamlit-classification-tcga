import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime
import json

# Load model components
@st.cache_resource
def load_model():
    try:
        model = joblib.load("grade_prediction_model.joblib")
        return model
    except Exception as e:
        st.error(f"Model gagal dimuat: {str(e)}")
        st.stop()

# Prediction function
def predict_grade(input_data, model):
    df = pd.DataFrame([input_data])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0]
    return pred, prob

# UI setup
st.set_page_config(page_title="Genetic Mutation Grade Predictor", layout="wide")
st.title("🧬 Tumor Grade Prediction Dashboard")
st.markdown("Dashboard untuk memprediksi grade tumor berdasarkan data mutasi genetik dan demografik.")

model = load_model()

# Sidebar input
with st.sidebar.form("input_form"):
    st.header("Input Pasien")
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    age = st.number_input("Age at Diagnosis", min_value=0, max_value=100, value=40)
    race = st.selectbox("Race", [0, 1, 2, 3], format_func=lambda x: ["White", "Black", "Asian", "Native"][x])
    
    gene_cols = ['IDH1','TP53','ATRX','PTEN','EGFR','CIC','MUC16','PIK3CA','NF1','PIK3R1',
                 'FUBP1','RB1','NOTCH1','BCOR','CSMD3','SMARCA4','GRIN2A','IDH2','FAT4','PDGFRA']
    
    gene_mutations = {gene: st.selectbox(gene, [0, 1]) for gene in gene_cols}
    
    submit = st.form_submit_button("🔮 Predict")

# Process prediction
if submit:
    input_data = {
        'Gender': gender,
        'Age_at_diagnosis': age,
        'Race': race,
        **gene_mutations
    }
    pred, prob = predict_grade(input_data, model)

    st.subheader("📊 Prediction Result")
    st.success(f"Predicted Grade: **{pred}**")

    fig = px.bar(x=['II', 'III', 'IV'], y=prob, labels={'x': 'Grade', 'y': 'Probability'},
                 title="Probabilitas Prediksi", color=prob, color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)

    export_data = {
        "timestamp": datetime.now().isoformat(),
        "input": input_data,
        "prediction": {
            "grade": pred,
            "probability": list(prob)
        }
    }

    st.download_button("📥 Download Prediction Result", data=json.dumps(export_data, indent=2),
                       file_name="grade_prediction_result.json", mime="application/json")

# Optional: Load dataset for visualization
if st.checkbox("📂 Tampilkan Statistik Dataset"):
    uploaded = st.file_uploader("Upload Dataset CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)

        st.markdown("### 📈 Distribusi Umur")
        st.plotly_chart(px.histogram(df, x="Age_at_diagnosis", nbins=20), use_container_width=True)

        st.markdown("### ⚧ Distribusi Gender")
        st.plotly_chart(px.pie(df, names="Gender", title="Gender Distribution", 
                               labels={0: "Male", 1: "Female"}), use_container_width=True)

        st.markdown("### 🌎 Distribusi Ras")
        st.plotly_chart(px.histogram(df, x="Race", nbins=5), use_container_width=True)

        st.markdown("### 🧬 Heatmap Mutasi Genetik")
        mut_cols = df.columns[4:]  # assume first 4 are non-gene columns
        heatmap_data = df[mut_cols].corr()
        st.plotly_chart(px.imshow(heatmap_data, text_auto=True, color_continuous_scale="RdBu"),
                        use_container_width=True)
