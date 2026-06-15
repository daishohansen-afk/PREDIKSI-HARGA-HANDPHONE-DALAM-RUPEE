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

# Input User

brand = st.selectbox(
    "Brand",
    dropdown_options["Brand"]
)

model_name = st.selectbox(
    "Model",
    dropdown_options["Model"]
)

color = st.selectbox(
    "Color",
    dropdown_options["Color"]
)

free_shipping = st.selectbox(
    "Free Shipping",
    dropdown_options["Free"]
)

# Prediksi
if st.button("Prediksi Harga"):

    try:

        brand_encoded = dropdown_options["Brand"].index(brand)

        model_encoded = dropdown_options["Model"].index(model_name)

        color_encoded = dropdown_options["Color"].index(color)

        free_encoded = dropdown_options["Free"].index(free_shipping)

        input_data = pd.DataFrame([[
            brand_encoded,
            model_encoded,
            color_encoded,
            free_encoded
        ]], columns=[
            "Brand",
            "Model",
            "Color",
            "Free"
        ])

        prediction = model.predict(input_data)[0]

        st.success(
            f"Perkiraan Harga Smartphone: ₹ {prediction:,.2f}"
        )

    except Exception as e:

        st.error("Prediksi gagal")
        st.exception(e)
