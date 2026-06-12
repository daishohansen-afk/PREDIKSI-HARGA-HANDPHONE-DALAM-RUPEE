import streamlit as st
import numpy as np
import pickle

# Konfigurasi halaman website
st.set_page_config(page_title="Prediksi Harga Smartphone AI", layout="centered")

# Fungsi untuk memuat model dan data pendukung (.pkl) secara aman
@st.cache_resource
def load_assets():
    try:
        with open('smartphone_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('dropdown_options.pkl', 'rb') as f:
            options = pickle.load(f)
        with open('model_metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
        return model, options, metrics
    except Exception as e:
        st.error(f"Gagal memuat file pkl: {e}")
        return None, None, None

# Eksekusi pemuatan aset
model, options, metrics = load_assets()

if model is not None and options is not None:
    # Bagian Header Aplikasi Web
    st.title("Aplikasi Prediksi Harga Smartphone")
    st.write("Masukkan spesifikasi smartphone di bawah ini untuk melihat hasil estimasi prediksi harganya.")
    
    # Menampilkan performa model secara formal di website (Syarat wajib kriteria UAS nomor 4)
    if metrics and 'model_name' in metrics:
        st.info(f"Informasi Performa Model: Aplikasi ini menggunakan algoritma {metrics['model_name']} dengan tingkat keakuratan R2 Score: {metrics['r2_score']} dan rata-rata error MAE: {metrics['mae']:,}.")
    st.markdown("---")

    # Pembuatan Menu Pilihan Dinamis Berdasarkan File pkl
    st.subheader("Atur Spesifikasi Perangkat")
    col1, col2 = st.columns(2)
    
    with col1:
        # Pilihan Merk/Brand HP
        selected_brand = st.selectbox("Pilih Merk/Brand:", options['brand_name'])
        
        # Pilihan Kapasitas RAM
        selected_ram = st.selectbox("Kapasitas RAM (GB):", options['ram'])
        
        # Pilihan Kapasitas Penyimpanan (Storage)
        selected_storage = st.selectbox("Kapasitas Memori Internal (GB):", options['storage'])

    with col2:
        # Pilihan Jenis Chipset
        selected_chipset = st.selectbox("Pilih Tipe Chipset:", options['chipset'])
        
        # Input Ukuran Layar
        screen_size = st.number_input("Ukuran Layar (Inchi):", min_value=4.0, max_value=8.0, value=6.5, step=0.1)
        
        # Input Kapasitas Baterai
        battery_capacity = st.number_input("Kapasitas Baterai (mAh):", min_value=2000, max_value=7000, value=5000, step=100)

    st.markdown("---")

    # Tombol Eksekusi Prediksi Harga
    if st.button("Prediksi Harga Sekarang", type="primary"):
        try:
            # Mengubah input teks teks dropdown menjadi index angka numerik sesuai data latih Colab
            brand_encoded = options['brand_name'].index(selected_brand)
            chipset_encoded = options['chipset'].index(selected_chipset)
            
            # Susun parameter input data secara berurutan
            # Urutan: ['brand_name', 'ram', 'storage', 'chipset', 'screen_size', 'battery_capacity']
            input_features = [brand_encoded, int(selected_ram), int(selected_storage), chipset_encoded, float(screen_size), int(battery_capacity)]
            
            # Konversi ke numpy array dengan tipe data float64 murni untuk mencegah layar hitam / stuck di sistem matriks Tree
            input_data = np.array([input_features], dtype=np.float64)
            
            # Jalankan algoritma prediksi harga
            prediction = model.predict(input_data)[0]
            
            # Menampilkan Hasil Prediksi Harga ke Layar
            st.success("Estimasi Prediksi Harga Perangkat:")
            
            # Deteksi jika target dataset dalam Rupee, tampilkan simbol Rupee (₹)
            st.subheader(f"₹ {prediction:,.2f} Rupee")
            st.caption("Catatan: Nilai di atas merupakan hasil kalkulasi kecerdasan buatan (AI) berdasarkan riwayat tren dataset.")
            
        except Exception as prediction_error:
            st.error(f"Gagal melakukan kalkulasi prediksi harga: {prediction_error}")
            st.warning("Tips: Pastikan urutan cell pada saat menjalankan Google Colab sudah sukses terisi secara berurutan.")

else:
    st.warning("Aplikasi dalam mode bersiap. Mengunduh data pustaka virtual dari repositori GitHub...")
