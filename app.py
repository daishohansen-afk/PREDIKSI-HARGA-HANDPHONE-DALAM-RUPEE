import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Price Predictor", page_icon="📱", layout="centered")

st.title("📱 Smartphone Price Prediction App")
st.markdown("Enter specifications below to generate an immediate valuation matrix built from our model pipeline.")

model_path = 'phone_price_model.pkl'
meta_path = 'categories_metadata.pkl'

if not os.path.exists(model_path) or not os.path.exists(meta_path):
    st.error("⚠️ CRITICAL ERROR: Model files are missing from the repository directory!")
    st.stop()

@st.cache_resource
def load_assets():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    return model, meta

try:
    model, metadata = load_assets()
    
    # Performance metrics display panel
    st.sidebar.header("📊 Model Accuracy Stats")
    st.sidebar.markdown(f"""
    - **Selected Model:** Decision Tree (4-Param)
    - **R² Score:** {metadata['metrics']['r2']}
    - **Mean Absolute Error:** ₹{metadata['metrics']['mae']:,}
    - **Root MSE:** ₹{metadata['metrics']['rmse']:,}
    """)

    st.subheader("Specify Target Smartphone Metrics")
    
    # Extract lists safely with fallbacks to avoid any KeyError from older cache versions
    brands_list = metadata.get('brands', [])
    ram_list = metadata.get('ram_options', [2, 4, 6, 8, 12, 16])
    storage_list = metadata.get('storage_options', [32, 64, 128, 256, 512])
    
    # Dynamic key safety fallback for colors / model options
    if 'colors' in metadata:
        color_list = metadata['colors']
    elif 'models' in metadata:
        color_list = metadata['models']
    else:
        color_list = ["Black", "Blue", "White", "Gray", "Silver", "Gold"]

    # Render 4 parameter UI layout
    col1, col2 = st.columns(2)
    with col1:
        chosen_brand = st.selectbox("Brand Name", options=brands_list)
        chosen_ram = st.selectbox("RAM (GB)", options=ram_list)
    with col2:
        chosen_color = st.selectbox("Device Property / Color", options=color_list)
        chosen_storage = st.selectbox("Internal Storage (GB)", options=storage_list)

    st.markdown("---")
    
    if st.button("🔮 Generate Price Evaluation", type="primary"):
        # Create input dataframe aligning to the 4 parameters
        query_df = pd.DataFrame([{
            'Brand': chosen_brand,
            'RAM': float(chosen_ram),
            'Storage': float(chosen_storage),
            'Color': chosen_color
        }])
        
        # Real-time evaluation mapping
        prediction = model.predict(query_df)[0]
            
        st.success("### Final Valuation Estimate")
        st.metric(label="Calculated Base Value", value=f"₹ {prediction:,.2f}")
        st.caption(f"Configured Layout: {chosen_brand} | {chosen_color} | {chosen_ram}GB RAM / {chosen_storage}GB Storage")

except Exception as e:
    st.error(f"An unexpected parsing exception occurred: {e}")
    st.info("💡 Quick Fix: If you just pushed updates, open the App Settings menu in the bottom-right corner of Streamlit Cloud, click 'Clear Cache', and 'Reboot App'.")
