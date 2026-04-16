import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Load Model and Preprocessing Objects ---
@st.cache_resource
def load_assets():
    try:
        with open('huber_regressor_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('standard_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        return model, scaler, label_encoders, feature_columns
    except FileNotFoundError:
        st.error("Error: Salah satu atau lebih file yang diperlukan tidak ditemukan. "
                 "Pastikan file .pkl berada di direktori yang sama dengan app.py.")
        st.stop()

model, scaler, label_encoders, feature_columns = load_assets()

# --- 2. Streamlit UI ---
st.set_page_config(page_title="Prediksi Gaji Awal Pelatihan Vokasi", layout="centered")

st.title("💰 Prediksi Gaji Awal Lulusan Pelatihan Vokasi")
st.markdown("Aplikasi ini memprediksi estimasi gaji awal lulusan pelatihan vokasi berdasarkan karakteristik individu.")

# Input Form
with st.form("prediction_form"):
    st.header("Data Diri & Pelatihan")

    # Input Numerik
    usia = st.slider("Usia", min_value=17, max_value=60, value=25)
    durasi_jam = st.slider("Durasi Pelatihan (Jam)", min_value=30, max_value=120, value=60)
    nilai_ujian = st.slider("Nilai Ujian", min_value=50, max_value=100, value=80)

    # Input Kategorikal
    pendidikan_options = label_encoders['Pendidikan'].classes_
    jurusan_options = label_encoders['Jurusan'].classes_
    jenis_kelamin_options = ['Laki-laki', 'Wanita']
    status_bekerja_options = ['Sudah Bekerja', 'Belum Bekerja']

    # PERBAIKAN: Konversi hasil np.where ke int() untuk menghindari error tipe data pada Streamlit
    idx_pendidikan = int(np.where(pendidikan_options == 'SMK')[0][0]) if 'SMK' in pendidikan_options else 0
    pendidikan = st.selectbox("Pendidikan", options=pendidikan_options, index=idx_pendidikan)

    idx_jurusan = int(np.where(jurusan_options == 'Teknik Listrik')[0][0]) if 'Teknik Listrik' in jurusan_options else 0
    jurusan = st.selectbox("Jurusan", options=jurusan_options, index=idx_jurusan)

    jenis_kelamin = st.selectbox("Jenis Kelamin", options=jenis_kelamin_options, index=0)
    status_bekerja = st.selectbox("Status Bekerja", options=status_bekerja_options, index=0)

    submitted = st.form_submit_button("Prediksi Gaji")

    if submitted:
        # --- 3. Pra-pemrosesan Input Pengguna ---
        input_data = pd.DataFrame([[
            usia, durasi_jam, nilai_ujian, pendidikan, jurusan, jenis_kelamin, status_bekerja
        ]], columns=['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan', 'Jenis_Kelamin', 'Status_Bekerja'])

        # Label Encoding
        for col in ['Pendidikan', 'Jurusan']:
            input_data[col] = label_encoders[col].transform(input_data[col])

        # One-Hot Encoding Manual (menyesuaikan dengan feature_columns)
        one_hot_cols = [
            'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
            'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja'
        ]
        
        for col in one_hot_cols:
            input_data[col] = 0

        if jenis_kelamin == 'Laki-laki':
            input_data['Jenis_Kelamin_Laki-laki'] = 1
        else:
            input_data['Jenis_Kelamin_Wanita'] = 1
        
        if status_bekerja == 'Belum Bekerja':
            input_data['Status_Bekerja_Belum Bekerja'] = 1
        else:
            input_data['Status_Bekerja_Sudah Bekerja'] = 1

        # Drop kolom asli yang sudah di-encode
        input_data = input_data.drop(columns=['Jenis_Kelamin', 'Status_Bekerja'])
        
        # Scaling fitur numerik
        numerical_features = ['Usia', 'Durasi_Jam', 'Nilai_Ujian']
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])

        # Menyesuaikan urutan kolom dengan model (menggunakan feature_columns.pkl)
        processed_input = pd.DataFrame(0, index=[0], columns=feature_columns)
        for col in feature_columns:
            if col in input_data.columns:
                processed_input[col] = input_data[col].values[0]
        
        processed_input = processed_input.apply(pd.to_numeric)

        # --- 4. Hasil Prediksi ---
        try:
            prediction = model.predict(processed_input)[0]
            st.subheader("Hasil Prediksi Gaji Awal:")
            st.success(f"### Rp {prediction:.2f} Juta")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

st.markdown("---")
st.info("Pastikan versi library (scikit-learn) di lingkungan deploy sama dengan versi saat training model.")
