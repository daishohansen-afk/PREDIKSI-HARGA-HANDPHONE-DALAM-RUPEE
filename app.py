import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Konfigurasi halaman website
st.set_page_config(page_title="Prediksi Kelas Harga Smartphone AI", layout="centered")

# Fungsi untuk memuat data dan melatih model klasifikasi langsung di server (AMANKAN DARI BUG PRICE)
@st.cache_resource
def build_and_train_model():
    # 1. Memuat dataset langsung dari repositori
    df = pd.read_csv('smartphones.csv')
    
    # 2. Pembersihan data dasar
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['price'], inplace=True)
    
    # --- SOLUSI ERROR PRICE: Mengubah menjadi 3 Kategori Kelas Harga ---
    # Budget: < 30,000 | Mid-Range: 30,000 - 70,000 | Flagship: > 70,000
    def categorize_price(p):
        if p < 30000:
            return 'Budget (Ekonomis)'
        elif p <= 70000:
            return 'Mid-Range (Menengah)'
        else:
            return 'Flagship (Premium)'
            
    df['price_class'] = df['price'].apply(categorize_price)
    
    features = ['brand_name', 'ram', 'storage', 'chipset', 'screen_size', 'battery_capacity']
    target = 'price_class'
    
    X = df[features].copy()
    y = df[target].copy()
    
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
    
    # 4. Splitting Data & Training Model Random Forest Classifier (Sangat Ringan)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 5. Hitung Akurasi Klasifikasi sesuai syarat nomor 4 di PDF UAS
    y_pred = model.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    
    return model, dropdown_options, acc_score

try:
    # Menjalankan proses di latar belakang server cloud
    with st.spinner("Sedang mengonfigurasi kecerdasan buatan, mohon tunggu sebentar..."):
        model, options, acc_score = build_and_train_model()
    
    # Bagian Header Aplikasi Web
    st.title("Aplikasi Prediksi Kelas Harga Smartphone")
    st.write("Masukkan spesifikasi smartphone di bawah ini untuk memprediksi kategori kelas harganya.")
    
    # Menampilkan akurasi model sesuai aturan UAS nomor 4
    st.info(f"Informasi Performa Model: Aplikasi ini menggunakan algoritma Random Forest Classifier dengan tingkat Akurasi: {acc_score * 100:.2f}%.")
    st.markdown("---")

    # Pembuatan Menu Pilihan Dinamis Sesuai Kolom Dataset
    st.subheader("Atur Spesifikasi Perangkat")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_brand = st.selectbox("Pilih Merk/Brand:", options['brand_name'])
        selected_ram = st.selectbox("Kapasitas RAM (GB):", options['ram'])
        selected_storage = st.selectbox("Kapasitas Memori Internal (ROM) (GB):", options['storage'])

    with col2:
        selected_chipset = st.selectbox("Pilih Tipe Chipset:", options['chipset'])
        screen_size = st.number_input("Ukuran Layar (Inchi):", min_value=4.0, max_value=8.0, value=6.5, step=0.1)
        battery_capacity = st.number_input("Kapasitas Baterai (mAh):", min_value=2000, max_value=7000, value=5000, step=100)

    st.markdown("---")

    # Tombol Eksekusi Prediksi
    if st.button("Prediksi Kelas Harga Sekarang", type="primary"):
        # Lakukan encoding input teks dropdown ke index angka numerik
        brand_encoded = options['brand_name'].index(selected_brand)
        chipset_encoded = options['chipset'].index(selected_chipset)
        
        # Susun parameter input data sesuai bentuk kolom saat training model
        input_data = np.array([[brand_encoded, selected_ram, selected_storage, chipset_encoded, screen_size, battery_capacity]])
        
        # Jalankan prediksi kelas harga
        prediction = model.predict(input_data)[0]
        
        # Menampilkan Hasil Prediksi ke Layar
        st.success("Hasil Estimasi Kategori Handphone:")
        st.subheader(f"Smartphone ini termasuk kelas: {prediction}")
        st.caption("Catatan: Prediksi dihitung berdasarkan kecerdasan buatan (AI) dari kecenderungan spesifikasi di dataset.")

except FileNotFoundError:
    st.error("Error: File 'smartphones.csv' tidak ditemukan di repositori utama GitHub Anda.")
except Exception as e:
    st.error(f"Terjadi kendala teknis: {e}")
