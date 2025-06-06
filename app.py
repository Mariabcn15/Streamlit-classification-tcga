import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px

st.set_page_config(page_title="UAS Tumor Grade Classifier", layout="wide")
st.title("🧬 Tumor Grade Classifier Dashboard")
st.markdown("Analisis dataset pasien dan prediksi klasifikasi *tumor grade* berdasarkan fitur biologis dan demografis.")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("TCGA.csv")  # Sesuaikan nama file jika berbeda

df = load_data()

# EDA
st.header("📊 Exploratory Data Analysis")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribusi Grade Tumor")
    st.bar_chart(df['Grade'].value_counts())

with col2:
    st.subheader("Statistik Umur Pasien")
    st.write(df['Age_at_diagnosis'].describe())

# # Korelasi
# st.subheader("🔍 Korelasi Fitur")
# fig, ax = plt.subplots(figsize=(12, 10))
# sns.heatmap(df.corr().iloc[:10, :10], annot=True, fmt=".2f", cmap='coolwarm')
# st.pyplot(fig)

# # Model Klasifikasi
# st.header("🤖 Model Klasifikasi Tumor Grade")

X = df.drop(columns=['Grade'])
y = df['Grade']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# st.subheader("Hasil Klasifikasi Decision Tree")
# st.text(classification_report(y_test, y_pred))

# Prediksi Interaktif
st.header("🎯 Prediksi Grade Tumor Pasien")

input_data = {}
for col in X.columns:
    if df[col].nunique() <= 2:
        input_data[col] = st.selectbox(f"{col}", [0, 1])
    else:
        input_data[col] = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

user_df = pd.DataFrame([input_data])
predicted_grade = model.predict(user_df)[0]
st.success(f"Prediksi Grade Tumor: {'High Grade' if predicted_grade == 1 else 'Low Grade'}")

# Feature Importance section
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
