import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(
page_title="Prediksi Harga Smartphone",
page_icon="📱",
layout="centered"
)

st.title("📱 Prediksi Harga Smartphone")

# =====================

# DEBUG INFO

# =====================

st.subheader("Debug Info")

try:
st.write("Daftar file:")
st.write(os.listdir())

```
if os.path.exists("model_smartphone.pkl"):
    st.write(
        "Ukuran model_smartphone.pkl:",
        os.path.getsize("model_smartphone.pkl"),
        "bytes"
    )
else:
    st.error("model_smartphone.pkl tidak ditemukan")
```

except Exception as e:
st.error(e)

# =====================

# LOAD FILE

# =====================

try:

```
with open("model_smartphone.pkl", "rb") as f:
    model = pickle.load(f)

with open("dropdown_options.pkl", "rb") as f:
    dropdown_options = pickle.load(f)

with open("model_metrics.pkl", "rb") as f:
    metrics = pickle.load(f)
```

except Exception as e:

```
st.error("Gagal membaca file pickle")
st.exception(e)
st.stop()
```

# =====================

# SIDEBAR

# =====================

st.sidebar.header("Informasi Model")

try:
st.sidebar.write(
f"Model : {metrics['model_name']}"
)

```
st.sidebar.write(
    f"R² Score : {metrics['r2_score']}"
)

st.sidebar.write(
    f"MAE : {metrics['mae']}"
)
```

except:
pass

# =====================

# INPUT

# =====================

brand = st.selectbox(
"Merk Smartphone",
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

screen_size = st.slider(
"Ukuran Layar (Inch)",
4.0,
8.0,
6.5,
0.1
)

battery_capacity = st.slider(
"Kapasitas Baterai (mAh)",
2000,
8000,
5000,
100
)

# =====================

# PREDIKSI

# =====================

if st.button("Prediksi Harga"):

```
try:

    brand_encoded = dropdown_options[
        "brand_name"
    ].index(brand)

    chipset_encoded = dropdown_options[
        "chipset"
    ].index(chipset)

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

    prediction = model.predict(
        input_data
    )[0]

    st.success(
        f"💰 Perkiraan Harga: Rp {prediction:,.0f}"
    )

except Exception as e:

    st.error(
        "Prediksi gagal dijalankan"
    )

    st.exception(e)
```
