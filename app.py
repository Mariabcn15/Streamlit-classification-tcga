import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px

# Set page config
st.set_page_config(page_title="UAS Tumor Grade Classifier", layout="wide")
st.title("🧬 Tumor Grade Classifier Dashboard")
st.markdown("Analisis dataset pasien dan prediksi klasifikasi *tumor grade* berdasarkan fitur biologis dan demografis.")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("TCGA.csv")  # Sesuaikan nama file jika berbeda

df = load_data()

# EDA Section
st.header("📊 Exploratory Data Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribusi Grade Tumor")
    st.bar_chart(df['Grade'].value_counts())

with col2:
    st.subheader("Statistik Umur Pasien")
    st.write(df['Age_at_diagnosis'].describe())

# Data Split and Model
X = df.drop(columns=['Grade'])
y = df['Grade']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Interaktif Prediksi
st.header("🎯 Prediksi Grade Tumor Pasien")

# Daftar fitur mutasi
categorical_mutations = [
    'IDH1', 'IDH2', 'TP53', 'ATRX', 'EGFR', 'PTEN', 
    'CIC', 'FUBP1', 'PIK3CA', 'PIK3R1', 'NF1', 'PDGFRA'
]

# Input user
input_data = {}
for col in X.columns:
    if col in categorical_mutations:
        input_data[col] = st.selectbox(f"{col} (0 = not mutated, 1 = mutated)", [0, 1])
    elif col == 'Gender':
        input_data[col] = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    elif col == 'Race':
        race_labels = {
            0: "White",
            1: "Black or African American",
            2: "Asian",
            3: "American Indian or Alaska Native"
        }
        input_data[col] = st.selectbox("Race", options=list(race_labels.keys()), format_func=lambda x: race_labels[x])
    else:
        input_data[col] = st.slider(
            f"{col}",
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].mean())
        )

# Prediksi
user_df = pd.DataFrame([input_data])
predicted_grade = model.predict(user_df)[0]
st.success(f"Prediksi Grade Tumor: {'High Grade' if predicted_grade == 1 else 'Low Grade'}")

# Feature Importance
st.subheader("📊 Feature Importance")

try:
    feature_names = X.columns
    feature_importance = model.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)

    fig_importance = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance in Decision Tree Model',
        color='Importance',
        color_continuous_scale='viridis'
    )
    fig_importance.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_importance, use_container_width=True)

except Exception as e:
    st.error(f"Error displaying feature importance: {str(e)}")
