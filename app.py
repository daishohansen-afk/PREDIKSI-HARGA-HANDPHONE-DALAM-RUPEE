import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Konfigurasi halaman website
st.set_page_config(page_title="Prediksi Harga Smartphone AI", layout="centered")

# Fungsi untuk memuat data, memproses EDA dasar, dan melatih model langsung di server
@st.cache_resource
def build_and_train_model():
    # 1. Memuat dataset langsung dari repositori
    df = pd.read_csv('smartphones.csv')
    
    # 2. Pembersihan data dasar (Penanganan missing value & duplikat)
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['price'], inplace=True)
    
    features = ['brand_name', 'ram', 'storage', 'chipset', 'screen_size', 'battery_capacity']
    target = 'price'
    
    # Penanganan Outlier dengan Metode IQR untuk target price
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)].copy()
    
    X = df_clean[features].copy()
    y = df_clean[target].copy()
    
    # 3. Feature Engineering / Label Encoding Manual
    categorical_cols = ['brand_name', 'chipset']
    dropdown_options = {}
    
    for col in categorical_cols:
        X[col] = X[col].astype(str)
        unique_vals = sorted(X[col].unique())
        dropdown_options[col] = unique_vals
        X[col] = X[col].apply(lambda x: unique_vals.index(x))
        
    dropdown_options['ram'] = sorted([int(x) for x in X['ram'].unique()])
    dropdown_options['storage'] = sorted([int(x) for x in X['storage'].unique()])
    
    # 4. Splitting Data & Training Model Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 5. Hitung Metrik Evaluasi
    y_pred = model.predict(X_test)
    mae_score = mean_absolute_error(y_test, y_pred)
    r2_val = r2_score(y_test, y_pred)
    
    return model, dropdown_options, mae_score, r2_val

try:
    # Menjalankan proses pembuatan model secara otomatis di server cloud
    with st.spinner("Sedang memuat data dan mengonfigurasi kecerdasan buatan, mohon tunggu sebentar..."):
        model, options, mae_score, r2_val = build_and_train_model()
    
    # Bagian Header Aplikasi Web
    st.title("Aplikasi Prediksi Harga Smartphone")
    st.write("Masukkan spesifikasi smartphone di bawah ini untuk melihat hasil estimasi prediksi harganya.")
    
    # Menampilkan performa model secara formal di website (Syarat wajib kriteria UAS)
    st.info(f"Informasi Performa Model: Aplikasi ini menggunakan algoritma Random Forest Regressor dengan tingkat keakuratan R2 Score: {r2_val:.4f} dan rata-rata error MAE: Rp {mae_score:,.2f}.")
    st.markdown("---")

    # Pembuatan Menu Pilihan Dinamis Sesuai Kolom Dataset
    st.subheader("Atur Spesifikasi Perangkat")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_brand = st.selectbox("Pilih Merk/Brand:", options['brand_name'])
        selected_ram = st.selectbox("Kapasitas RAM (GB):", options['ram'])
        selected_storage = st.selectbox("Kapasitas Memori Internal (GB):", options['storage'])

    with col2:
        selected_chipset = st.selectbox("Pilih Tipe Chipset:", options['chipset'])
        screen_size = st.number_input("Ukuran Layar (Inchi):", min_value=4.0, max_value=8.0, value=6.5, step=0.1)
        battery_capacity = st.number_input("Kapasitas Baterai (mAh):", min_value=2000, max_value=7000, value=5000, step=100)

    st.markdown("---")

    # Tombol Eksekusi Prediksi Harga
    if st.button("Prediksi Harga Sekarang", type="primary"):
        # Lakukan encoding input teks dropdown ke index angka numerik
        brand_encoded = options['brand_name'].index(selected_brand)
        chipset_encoded = options['chipset'].index(selected_chipset)
        
        # Susun parameter input data sesuai bentuk kolom saat training model
        input_data = np.array([[brand_encoded, selected_ram, selected_storage, brand_encoded, screen_size, battery_capacity]])
        
        # Jalankan algoritma prediksi harga
        prediction = model.predict(input_data)[0]
        
        # Menampilkan Hasil Prediksi Harga ke Layar
        st.success("Estimasi Prediksi Harga Perangkat:")
        st.subheader(f"Rp {prediction:,.2f}")
        st.caption("Catatan: Nilai di atas merupakan hasil kalkulasi kecerdasan buatan (AI) berdasarkan tren data historis dataset.")

except FileNotFoundError:
    st.error("Error: File 'smartphones.csv' tidak ditemukan di repositori utama GitHub Anda. Pastikan Anda telah mengunggah dataset tersebut.")
except Exception as e:
    st.error(f"Terjadi kendala teknis: {e}")
