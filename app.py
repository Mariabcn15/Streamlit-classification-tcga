import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px

# Setup halaman
st.set_page_config(page_title="🧬 Tumor Grade Classifier", layout="wide")

# Sidebar - Branding & Info
with st.sidebar:
    st.image("glioma.jpeg", width=350)
    st.markdown("## 🧬 Tentang Aplikasi")
    st.markdown("""
Aplikasi ini dirancang untuk memprediksi **Grade Tumor Otak** pasien berdasarkan informasi **demografis** dan **mutasi genetik**.

**📁 Sumber Data:** TCGA  
**🧠 Model:** Decision Tree  
**🎯 Tujuan:** Mendukung Diagnosis Glioma
    """)

    st.markdown("---")
    st.markdown("### 👩‍💻 Pengembang")
    st.markdown("""
**Azzula Cerliana Zahro**  
[🔗 LinkedIn Profil](https://www.linkedin.com/in/azzulacerliana)
    """)
    st.markdown("""
**Maria Bernadette Chayeenee Norman**  
[🔗 LinkedIn Profil](https://www.linkedin.com/in/mariabernadette15)
    """)
    st.markdown("""
**Estri Pramudia Pangestu**  
[🔗 LinkedIn Profil](https://www.linkedin.com/in/estri-pramudia-pangestu-1457a4263)
    """)


# Load data
@st.cache_data
def load_data():
    return pd.read_csv("TCGA.csv")

df = load_data()

# Siapkan model
X = df.drop(columns=["Grade"])
y = df["Grade"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Fitur genetik
gen_features = ['IDH1', 'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 'MUC16', 'PIK3CA', 'NF1',
                'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4', 'GRIN2A',
                'IDH2', 'FAT4', 'PDGFRA']

# Tabs utama
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🔍 Prediksi Tumor", "📈 Insight Model"])

# =======================================
# TAB 1: EDA
# =======================================
with tab1:
    st.header("📊 Exploratory Data Analysis (EDA)")
    st.markdown("Data terdiri dari berbagai mutasi gen dan fitur demografis untuk memprediksi tumor low/high grade.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Grade Tumor")
        df_grade = df.copy()
        grade_map = {0: "LGG", 1: "GBM"}
        df_grade["Grade"] = df_grade["Grade"].map(grade_map)

        fig_grade = px.histogram(df_grade, x="Grade", color="Grade", barmode="group",
                                 color_discrete_sequence=["#2ecc71", "#e74c3c"])
        st.plotly_chart(fig_grade, use_container_width=True)

    with col2:
        st.subheader("Gender Pasien Tumor")

        df_gender = df.copy()
        gender_map = {0: "Male", 1: "Female"}
        df_gender["Gender"] = df_gender["Gender"].map(gender_map)

        fig_gender = px.histogram(df_gender, x="Gender", color="Gender", 
                                  color_discrete_sequence=["#3498db", "#9b59b6"])
        st.plotly_chart(fig_gender, use_container_width=True)

# =======================================
# TAB 2: Prediksi Interaktif
# =======================================
with tab2:
    if "predict_clicked" not in st.session_state:
        st.session_state.predict_clicked = False

    st.header("🔍 Prediksi Interaktif Tumor Grade")
    st.markdown("Masukkan data pasien untuk memprediksi apakah tumor termasuk **High Grade** atau **Low Grade**.")

    col_left, col_right = st.columns([1.5, 1])
    input_data = {}

    with col_left:
        st.subheader("🧍‍♀️ Data Pasien")

        gender_label = st.selectbox("🚻 Gender", ["Male", "Female"], key="gender_input")
        race_label = st.selectbox("🌍 Ras", 
            ["White", "Black or African American", "Asian", "American Indian or Alaska Native"], key="race_input")
        age_value = st.slider("📅 Umur saat Diagnosis", 
                              float(df["Age_at_diagnosis"].min()), 
                              float(df["Age_at_diagnosis"].max()), 
                              float(df["Age_at_diagnosis"].mean()), step=1.0, key="age_input")

        gender_map = {"Male": 0, "Female": 1}
        race_map = {
            "White": 0,
            "Black or African American": 1,
            "Asian": 2,
            "American Indian or Alaska Native": 3
        }

        input_data["Gender"] = gender_map[gender_label]
        input_data["Race"] = race_map[race_label]
        input_data["Age_at_diagnosis"] = age_value

        # Mutasi genetik dalam expander
        with st.expander("🧬 Klik untuk masukkan mutasi genetik"):
            st.markdown("*Pilih status mutasi setiap gen: 0 = Tidak bermutasi, 1 = Bermutasi*")
            gen_cols = st.columns(3)

            for idx, gene in enumerate(gen_features):
                with gen_cols[idx % 3]:
                    status = st.radio(
                        f"{gene}", 
                        ["-- Pilih --", "Not Mutated", "Mutated"], 
                        horizontal=True, 
                        key=f"gen_{gene}"
                    )
                    if status == "Not Mutated":
                        input_data[gene] = 0
                    elif status == "Mutated":
                        input_data[gene] = 1
                    else:
                        input_data[gene] = None  # Belum dipilih

    with col_right:
        st.subheader("📌 Ringkasan Input")
        st.write(f"**Gender:** {gender_label}")
        st.write(f"**Ras:** {race_label}")
        st.write(f"**Umur:** {int(age_value)} tahun")

        gen_filled = [input_data[g] for g in gen_features if input_data[g] is not None]
        st.write(f"**Jumlah Gen Bermutasi:** {sum(gen_filled)} / {len(gen_features)}")

        # Tombol Prediksi
        if st.button("🔮 Prediksi Tumor Grade", use_container_width=True):
            if None in input_data.values():
                st.warning("⚠️ Harap isi semua input termasuk mutasi genetik sebelum memprediksi.")
            else:
                try:
                    user_df = pd.DataFrame([input_data])
                    user_df = user_df[X.columns]
                    predicted_grade = model.predict(user_df)[0]

                    grade_text = "🟥 High Grade" if predicted_grade == 1 else "🟩 Low Grade"
                    grade_color = "#e74c3c" if predicted_grade == 1 else "#2ecc71"

                    st.markdown(
                        f"""
                        <div style='padding: 30px; background-color: {grade_color}; color: white; border-radius: 12px; 
                        text-align: center; font-size: 32px; font-weight: bold;'>
                            {grade_text}
                        </div>
                        """, unsafe_allow_html=True
                    )
                    st.session_state.predict_clicked = True
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat prediksi: {str(e)}")

        # Tombol Reset akan muncul jika prediksi sudah dilakukan
        if st.session_state.predict_clicked:
            if st.button("🔁 Reset"):
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()

# =======================================
# TAB 3: Feature Importance
# =======================================
with tab3:
    st.header("📈 Feature Importance dari Model")
    st.markdown("Berikut adalah visualisasi feature importance yang dihasilkan dari Google Colab.")

    # Tampilkan gambar hasil dari Colab
    st.image("feature_importance.png", caption="Feature Importance dari Model Decision Tree", use_container_width=True)

# # =======================================
# # Footer
# # =======================================
# st.markdown("---")
# st.markdown(
#     "<center>© 2025 - Azzula Cerliana Zahro | "
#     "<a href='https://www.linkedin.com/in/linkedin-kamu' target='_blank'>LinkedIn</a></center>",
#     unsafe_allow_html=True
# )
