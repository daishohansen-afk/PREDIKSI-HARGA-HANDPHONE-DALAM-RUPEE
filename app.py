import streamlit as st
import pandas as pd
import pickle

# =========================
# LOAD FILE
# =========================

with open("model_smartphone.pkl", "rb") as f:
    model = pickle.load(f)

with open("dropdown_options.pkl", "rb") as f:
    dropdown_options = pickle.load(f)

with open("model_metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

# =========================
# HEADER
# =========================

st.set_page_config(
    page_title="Prediksi Harga Smartphone",
    page_icon="📱",
    layout="centered"
)

st.title("📱 Prediksi Harga Smartphone")
st.write("Masukkan spesifikasi smartphone untuk memperkirakan harga.")

# =========================
# SIDEBAR
# =========================

st.sidebar.header("Informasi Model")

st.sidebar.write(
    f"**Model:** {metrics['model_name']}"
)

st.sidebar.write(
    f"**R² Score:** {metrics['r2_score']}"
)

st.sidebar.write(
    f"**MAE:** {metrics['mae']}"
)

# =========================
# INPUT USER
# =========================

brand = st.selectbox(
    "Merk Smartphone",
    dropdown_options["brand_name"]
)

ram = st.selectbox(
    "RAM (GB)",
    dropdown_options["ram"]
)

storage = st.selectbox(
    "Storage (GB)",
    dropdown_options["storage"]
)

chipset = st.selectbox(
    "Chipset",
    dropdown_options["chipset"]
)

screen_size = st.slider(
    "Ukuran Layar (Inch)",
    min_value=4.0,
    max_value=8.0,
    value=6.5,
    step=0.1
)

battery_capacity = st.slider(
    "Kapasitas Baterai (mAh)",
    min_value=2000,
    max_value=8000,
    value=5000,
    step=100
)

# =========================
# PREDIKSI
# =========================

if st.button("🔍 Prediksi Harga"):

    try:

        brand_encoded = dropdown_options["brand_name"].index(brand)

        chipset_encoded = dropdown_options["chipset"].index(chipset)

        input_data = pd.DataFrame(
            [[
                brand_encoded,
                ram,
                storage,
                chipset_encoded,
                screen_size,
                battery_capacity
            ]],
            columns=[
                "brand_name",
                "ram",
                "storage",
                "chipset",
                "screen_size",
                "battery_capacity"
            ]
        )

        prediction = model.predict(input_data)[0]

        st.success(
            f"💰 Perkiraan Harga: Rp {prediction:,.0f}"
        )

    except Exception as e:
        st.error(f"Terjadi error: {e}")
