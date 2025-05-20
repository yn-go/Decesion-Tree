import streamlit as st
import pandas as pd
import joblib

# Fungsi untuk memuat model dan objek pendukung
# @st.cache_data (untuk data/dataframe) atau @st.cache_resource (untuk model/koneksi)
# Gunakan @st.cache_resource untuk model agar tidak re-load setiap interaksi
@st.cache_resource
def load_model_assets():
    model = joblib.load('model_tpt_indo.joblib')
    feature_names = joblib.load('model_feature_names.joblib')
    default_values = joblib.load('default_input_values.joblib')
    # Ambil juga kelas target yang diprediksi oleh model
    # Jika Anda menyimpannya saat training, bagus. Jika tidak, kita bisa ambil dari model.
    # Ini penting agar kita tahu label kelasnya (misal: 'Rendah', 'Sedang', 'Tinggi')
    # Saat training Anda bisa: joblib.dump(model.classes_, 'model_classes.joblib')
    # Untuk sekarang, kita hardcode berdasarkan output training atau coba ambil dari model
    try:
        target_classes = model.classes_
    except AttributeError:
        # Jika model.classes_ tidak ada (misalnya pipeline), hardcode sementara atau pastikan disimpan
        st.warning("Atribut 'classes_' tidak ditemukan pada model. Harap simpan kelas target saat training.")
        st.warning("Menggunakan kelas default: ['Rendah', 'Sedang', 'Tinggi']. Sesuaikan jika perlu.")
        target_classes = ['Rendah', 'Sedang', 'Tinggi'] # Ganti jika kategori Anda berbeda
        # Jika qcut hanya menghasilkan 2 kategori, sesuaikan: target_classes = ['Rendah', 'Tinggi']

    return model, feature_names, default_values, target_classes

model, feature_names, default_values, target_classes = load_model_assets()

# --- Antarmuka Aplikasi Streamlit ---
st.title("Prediksi Tingkat Pengangguran Terbuka (TPT) Indonesia")
st.write("""
Aplikasi ini memprediksi kategori TPT Indonesia (Rendah, Sedang, atau Tinggi)
berdasarkan data TPT provinsi lain dan periode waktu.
Silakan masukkan nilai-nilai di bawah ini:
""")

# Membuat form input
# Kita akan menggunakan kolom untuk tata letak yang lebih baik jika banyak input
# Atau cukup tampilkan satu per satu jika tidak terlalu banyak

input_data = {}

st.sidebar.header("Input Data:")

# Input untuk Periode dan Bulan
# Pastikan default_values['Periode'] dan default_values['Bulan_Numerik'] ada dan bertipe benar
default_periode = int(default_values.get('Periode', 2023)) # Ambil dari default atau set manual
min_periode = 2009 # Sesuaikan dengan data Anda
max_periode = 2030 # Beri ruang untuk prediksi ke depan

input_data['Periode'] = st.sidebar.number_input(
    "Tahun Periode:",
    min_value=min_periode,
    max_value=max_periode,
    value=default_periode,
    step=1
)

default_bulan = int(default_values.get('Bulan_Numerik', 2)) # 2 untuk Februari, 8 untuk Agustus
bulan_options = {2: "Februari", 8: "Agustus"}
selected_bulan_label = st.sidebar.selectbox(
    "Bulan:",
    options=list(bulan_options.values()),
    index=list(bulan_options.keys()).index(default_bulan) if default_bulan in bulan_options else 0
)
input_data['Bulan_Numerik'] = [k for k, v in bulan_options.items() if v == selected_bulan_label][0]

st.subheader("Input TPT Provinsi Lain (%):")
# Membuat input untuk setiap fitur provinsi
# Gunakan 2 kolom untuk kerapian jika banyak provinsi
col1, col2 = st.columns(2)

feature_count = 0
for feature in feature_names:
    if feature not in ['Periode', 'Bulan_Numerik']: # Ini sudah dihandle di atas
        # Ambil nilai default, jika tidak ada, fallback ke 0.0 atau nilai lain yang masuk akal
        default_val_feature = float(default_values.get(feature, 0.0))

        # Distribusikan input ke kolom
        current_column = col1 if feature_count % 2 == 0 else col2
        
        input_data[feature] = current_column.number_input(
            f"TPT {feature} (%):",
            min_value=0.0,
            max_value=100.0, # TPT biasanya tidak lebih dari 100%
            value=default_val_feature,
            step=0.1,
            format="%.2f" # format dua angka desimal
        )
        feature_count += 1


# Tombol Prediksi
if st.button("Prediksi Kategori TPT Indonesia"):
    # Siapkan DataFrame input sesuai urutan fitur saat training
    try:
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Lakukan prediksi
        prediction_idx = model.predict(input_df)[0] # Hasilnya adalah label kategori langsung
        prediction_proba = model.predict_proba(input_df)[0]

        # Tampilkan hasil
        st.subheader("Hasil Prediksi:")
        # Jika model.predict() mengembalikan label kategori langsung (misal 'Rendah')
        st.success(f"Kategori TPT Indonesia diprediksi: **{prediction_idx}**")

        st.write("Probabilitas untuk setiap kategori:")
        # Pastikan target_classes sesuai dengan urutan probabilitas
        # Biasanya model.classes_ memberikan urutan yang benar
        for i, class_label in enumerate(target_classes):
            st.write(f"- {class_label}: {prediction_proba[i]*100:.2f}%")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
        st.error("Pastikan semua input numerik diisi dengan benar dan model_feature_names.joblib sesuai.")
        st.error(f"Input yang diterima: {input_data}")
        st.error(f"Fitur yang diharapkan model: {feature_names}")

st.markdown("---")
st.markdown("Dibuat oleh Yusuf Naufal - Proyek Machine Learning - Tingkat Pengangguran Terbuka")
