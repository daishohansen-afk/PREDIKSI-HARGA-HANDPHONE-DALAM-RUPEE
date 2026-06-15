import streamlit as st
import pandas as pd
import pickle

# Load model dan encoder
model = pickle.load(open('smartphone_model.pkl', 'rb'))
brand_encoder = pickle.load(open('brand_encoder.pkl', 'rb'))
chipset_encoder = pickle.load(open('chipset_encoder.pkl', 'rb'))

# Load dataset untuk mengambil pilihan dropdown
df = pd.read_csv('smartphones.csv')

st.title("Prediksi Harga Smartphone")

# Dropdown
brand = st.selectbox(
    "Pilih Merk",
    sorted(df['brand_name'].unique())
)

ram = st.selectbox(
    "RAM (GB)",
    sorted(df['ram'].unique())
)

storage = st.selectbox(
    "Storage (GB)",
    sorted(df['storage'].unique())
)

chipset = st.selectbox(
    "Chipset",
    sorted(df['chipset'].unique())
)

screen_size = st.slider(
    "Ukuran Layar",
    float(df['screen_size'].min()),
    float(df['screen_size'].max()),
    float(df['screen_size'].mean())
)

battery = st.slider(
    "Kapasitas Baterai",
    int(df['battery_capacity'].min()),
    int(df['battery_capacity'].max()),
    int(df['battery_capacity'].mean())
)

# Tombol prediksi
if st.button("Prediksi Harga"):

    brand_encoded = brand_encoder.transform([brand])[0]
    chipset_encoded = chipset_encoder.transform([chipset])[0]

    input_data = pd.DataFrame([[
        brand_encoded,
        ram,
        storage,
        chipset_encoded,
        screen_size,
        battery
    ]], columns=[
        'brand_name',
        'ram',
        'storage',
        'chipset',
        'screen_size',
        'battery_capacity'
    ])

    prediction = model.predict(input_data)[0]

    st.success(
        f"Perkiraan Harga Smartphone: Rp {prediction:,.0f}"
    )
