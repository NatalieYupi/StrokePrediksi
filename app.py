import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Konfigurasi Halaman
st.set_page_config(
    page_title="Aplikasi Skreening Dini Risiko Stroke",
    page_icon="💛",
    layout="centered"
)

# Load Model Pipeline yang Sudah Dilatih
@st.cache_resource
def load_model():
    return joblib.load('xgboost_stroke_pipeline.joblib')

pipeline = load_model()

# Custom CSS hanya untuk penataan teks, icon, dan tombol (tanpa merubah background)
st.markdown("""
    <style>
    /* Section Labels */
    .section-label {
        font-size: 13px;
        font-weight: 700;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    
    /* Header Icon & Title */
    .header-container {
        text-align: center;
        margin-bottom: 25px;
    }
    .header-icon {
        font-size: 50px;
        line-height: 1;
    }
    .header-title {
        font-size: 26px;
        font-weight: 700;
        margin-top: 10px;
    }

    /* Header Form Inside */
    .form-title-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 15px;
    }
    .form-title-icon {
        background-color: rgba(2, 132, 199, 0.1);
        color: #0284c7;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 20px;
    }
    .form-title-text h3 {
        margin: 0;
        font-size: 18px;
        font-weight: 700;
    }
    .form-title-text p {
        margin: 0;
        font-size: 13px;
        opacity: 0.7;
    }

    /* Full Width Submit Button */
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #1a73e8 !important;
        color: white !important;
        border-radius: 8px !important;
        width: 100% !important;
        height: 50px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border: none !important;
        margin-top: 20px !important;
    }
    </style>
""", unsafe_allow_html=True)

# 1. Header Bagian Atas
st.markdown("""
    <div class="header-container">
        <div class="header-icon">💛</div>
        <div class="header-title">Aplikasi Skreening Dini Risiko Stroke</div>
    </div>
""", unsafe_allow_html=True)

# 2. Form Input Data Pasien
with st.form("stroke_form"):
    # Subheader Formulir
    st.markdown("""
        <div class="form-title-container">
            <div class="form-title-icon">📄</div>
            <div class="form-title-text">
                <h3>Formulir Data Kesehatan</h3>
                <p>Isi parameter di bawah ini berdasarkan kondisi riil Anda</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # SEKSI A: INFORMASI DEMOGRAFI
    st.markdown('<div class="section-label">A. INFORMASI DEMOGRAFI</div>', unsafe_allow_html=True)
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        age = st.number_input("Usia Anda (Tahun)", min_value=1.0, max_value=120.0, value=45.0, step=1.0)
    with col_a2:
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])

    # SEKSI B: PENGUKURAN FISIK
    st.markdown('<div class="section-label">B. PENGUKURAN FISIK</div>', unsafe_allow_html=True)
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
    with col_b2:
        avg_glucose_level = st.number_input("Rata-rata Kadar Glukosa Darah (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, step=0.1)

    # SEKSI C: RIWAYAT KLINIS & FAKTOR RISIKO KRITIS
    st.markdown('<div class="section-label">C. RIWAYAT KLINIS & FAKTOR RISIKO KRITIS</div>', unsafe_allow_html=True)
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        hypertension = st.selectbox("Tekanan Darah Tinggi (Hipertensi)", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        ever_married = st.selectbox("Pernah Menikah?", ["Yes", "No"])
    with col_c2:
        heart_disease = st.selectbox("Riwayat Disease/Penyakit Jantung", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
        Residence_type = st.selectbox("Tipe Tempat Tinggal", ["Urban", "Rural"])

    # SEKSI D: PERILAKU & GAYA HIDUP
    st.markdown('<div class="section-label">D. PERILAKU & GAYA HIDUP</div>', unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        smoking_status = st.selectbox("Status Merokok", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    with col_d2:
        work_type = st.selectbox("Tipe Pekerjaan", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])

    # Tombol Jalankan Analisis
    submit_button = st.form_submit_button(label="⚙️ Jalankan Analisis Risiko")

# 3. Proses Prediksi & Tampilan Hasil
if submit_button:
    # Buat DataFrame dari Input
    input_data = pd.DataFrame([{
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': Residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }])
    
    # Prediksi Probabilitas
    stroke_probability = pipeline.predict_proba(input_data)[0][1]
    stroke_prediction = pipeline.predict(input_data)[0]
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Hasil Analisis Risiko")
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.metric(
            label="Probabilitas Risiko Stroke",
            value=f"{stroke_probability * 100:.1f}%"
        )
        
    with col_res2:
        if stroke_probability >= 0.5:
            st.error("⚠️ **Risiko Tinggi:** Pasien terdeteksi memiliki potensi risiko stroke yang signifikan.")
            st.warning("Rekomendasi: Disarankan untuk segera melakukan pemeriksaan medis lanjutan dan konsultasi dokter spesialis saraf.")
        else:
            st.success("✅ **Risiko Rendah:** Pasien terdeteksi memiliki probabilitas stroke yang rendah.")
            st.info("Rekomendasi: Tetap jaga pola hidup sehat, atur pola makan, dan olahraga teratur.")