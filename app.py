import streamlit as st
import pickle
import pandas as pd
import os

st.set_page_config(page_title="Prediksi Harga Smartphone")

st.title("📱 Prediksi Harga Smartphone")

# Debug file
st.subheader("Debug File")

st.write("File yang ditemukan:")
st.write(os.listdir())

# Cek model
try:

    with open("smartphone_model.pkl", "rb") as f:
        model = pickle.load(f)

    st.success("Model berhasil dimuat!")

except Exception as e:

    st.error("Model gagal dimuat")
    st.exception(e)
    st.stop()

# Cek dropdown
try:
    with open("dropdown_options.pkl", "rb") as f:
        dropdown_options = pickle.load(f)

    st.success("Dropdown berhasil dimuat!")

except Exception as e:
    st.error("Dropdown gagal dimuat")
    st.exception(e)
    st.stop()

# Input user
brand = st.selectbox(
    "Brand",
    dropdown_options["brand_name"]
)

ram = st.selectbox(
    "RAM",
    dropdown_options["ram"]
)

storage = st.selectbox(
    "Storage",
    dropdown_options["storage"]
)

chipset = st.selectbox(
    "Chipset",
    dropdown_options["chipset"]
)

screen_size = st.number_input(
    "Screen Size",
    value=6.5
)

battery_capacity = st.number_input(
    "Battery Capacity",
    value=5000
)

# Prediksi
if st.button("Prediksi Harga"):

    try:

        brand_encoded = dropdown_options["brand_name"].index(brand)
        chipset_encoded = dropdown_options["chipset"].index(chipset)

        input_data = pd.DataFrame([[
            brand_encoded,
            ram,
            storage,
            chipset_encoded,
            screen_size,
            battery_capacity
        ]], columns=[
            "brand_name",
            "ram",
            "storage",
            "chipset",
            "screen_size",
            "battery_capacity"
        ])

        prediction = model.predict(input_data)[0]

        st.success(
            f"Perkiraan Harga: ₹ {prediction:,.0f}"
        )

    except Exception as e:
        st.error("Prediksi gagal")
        st.exception(e)
