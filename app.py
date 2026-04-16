
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Load Model and Preprocessing Objects ---
# Menggunakan st.cache_resource agar aset hanya dimuat sekali
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
        st.error("Error: Salah satu atau lebih file yang diperlukan (model, scaler, encoders, feature columns) tidak ditemukan. "
                 "Pastikan 'huber_regressor_model.pkl', 'standard_scaler.pkl', 'label_encoders.pkl', "
                 "dan 'feature_columns.pkl' berada di direktori yang sama.")
        st.stop()

model, scaler, label_encoders, feature_columns = load_assets()

# --- 2. Streamlit UI ---
st.set_page_config(page_title="Prediksi Gaji Awal Pelatihan Vokasi", layout="centered")

st.title("💰 Prediksi Gaji Awal Lulusan Pelatihan Vokasi")
st.markdown("Aplikasi ini memprediksi estimasi gaji awal lulusan pelatihan vokasi "
            "berdasarkan data pelatihan dan karakteristik individu.")

# Input Form
with st.form("prediction_form"):
    st.header("Data Diri & Pelatihan")

    # Input Numerik
    usia = st.slider("Usia", min_value=17, max_value=60, value=25)
    durasi_jam = st.slider("Durasi Pelatihan (Jam)", min_value=30, max_value=120, value=60)
    nilai_ujian = st.slider("Nilai Ujian", min_value=50, max_value=100, value=80)

    # Input Kategorikal (menggunakan nilai unik dari label_encoders dan kategori one-hot)
    pendidikan_options = label_encoders['Pendidikan'].classes_
    jurusan_options = label_encoders['Jurusan'].classes_
    jenis_kelamin_options = ['Laki-laki', 'Wanita'] # Berdasarkan pemetaan manual
    status_bekerja_options = ['Sudah Bekerja', 'Belum Bekerja'] # Berdasarkan kolom one-hot

    pendidikan = st.selectbox("Pendidikan", options=pendidikan_options,
                            index=np.where(pendidikan_options == 'SMK')[0][0] if 'SMK' in pendidikan_options else 0)
    jurusan = st.selectbox("Jurusan", options=jurusan_options,
                         index=np.where(jurusan_options == 'Teknik Listrik')[0][0] if 'Teknik Listrik' in jurusan_options else 0)
    jenis_kelamin = st.selectbox("Jenis Kelamin", options=jenis_kelamin_options, index=0)
    status_bekerja = st.selectbox("Status Bekerja", options=status_bekerja_options, index=0)

    submitted = st.form_submit_button("Prediksi Gaji")

    if submitted:
        # --- 3. Pra-pemrosesan Input Pengguna ---
        # Buat DataFrame dari input
        input_data = pd.DataFrame([[
            usia, durasi_jam, nilai_ujian, pendidikan, jurusan, jenis_kelamin, status_bekerja
        ]], columns=['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan', 'Jenis_Kelamin', 'Status_Bekerja'])

        # Terapkan Label Encoding (Pendidikan, Jurusan)
        for col in ['Pendidikan', 'Jurusan']:
            try:
                input_data[col] = label_encoders[col].transform(input_data[col])
            except ValueError as e:
                st.error(f"Error saat mengkodekan '{col}': {e}. Pastikan input '{input_data[col].iloc[0]}' adalah kategori yang valid.")
                st.stop()

        # Terapkan One-Hot Encoding (Jenis_Kelamin, Status_Bekerja)
        # Secara manual membuat kolom one-hot dengan 0, lalu set 1 berdasarkan input
        one_hot_cols_expected = [
            'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
            'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja'
        ]
        
        # Inisialisasi semua kolom one-hot yang diharapkan ke 0
        for col in one_hot_cols_expected:
            input_data[col] = 0

        # Set nilai 1 untuk kategori yang dipilih
        if input_data['Jenis_Kelamin'].iloc[0] == 'Laki-laki':
            input_data['Jenis_Kelamin_Laki-laki'] = 1
        elif input_data['Jenis_Kelamin'].iloc[0] == 'Wanita':
            input_data['Jenis_Kelamin_Wanita'] = 1
        
        if input_data['Status_Bekerja'].iloc[0] == 'Belum Bekerja':
            input_data['Status_Bekerja_Belum Bekerja'] = 1
        elif input_data['Status_Bekerja'].iloc[0] == 'Sudah Bekerja':
            input_data['Status_Bekerja_Sudah Bekerja'] = 1

        # Hapus kolom kategorikal asli yang sudah di-one-hot encoding
        input_data = input_data.drop(columns=['Jenis_Kelamin', 'Status_Bekerja'])
        
        # Scaling Fitur Numerik (Usia, Durasi_Jam, Nilai_Ujian)
        numerical_features = ['Usia', 'Durasi_Jam', 'Nilai_Ujian']
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])

        # Pastikan urutan fitur cocok dengan data pelatihan
        # Buat DataFrame baru dengan urutan kolom yang benar dan isi dengan data yang sudah diproses
        processed_input = pd.DataFrame(columns=feature_columns)
        for col in feature_columns:
            if col in input_data.columns:
                processed_input[col] = input_data[col]
            else:
                # Jika ada kolom one-hot yang tidak aktif untuk input saat ini, isi dengan 0
                processed_input[col] = 0
        
        # Konversi ke tipe numerik jika ada kolom yang terdeteksi object
        processed_input = processed_input.apply(pd.to_numeric, errors='coerce')

        # --- 4. Lakukan Prediksi ---
        try:
            prediction = model.predict(processed_input)[0]
            st.subheader("Hasil Prediksi Gaji Awal:")
            st.success(f"### Rp {prediction:.2f} Juta")
            st.info("Prediksi ini adalah estimasi dan dapat bervariasi.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# --- Instruksi Menjalankan Aplikasi ---
st.markdown("--- ---")
st.subheader("Cara Menjalankan Aplikasi Ini:")
st.markdown("1. Pastikan Anda telah menyimpan file `huber_regressor_model.pkl`, `standard_scaler.pkl`, "
            "`label_encoders.pkl`, dan `feature_columns.pkl` di direktori yang sama dengan `app.py`.")
st.markdown("2. Setelah sel di atas dieksekusi, file `app.py` akan dibuat.")
st.markdown("3. Buka terminal atau Command Prompt di komputer Anda.")
st.markdown("4. Navigasi ke direktori tempat Anda menyimpan `app.py`.")
st.markdown("5. Jalankan perintah: `streamlit run app.py`")
st.markdown("6. Aplikasi Streamlit akan terbuka secara otomatis di browser web Anda.")
