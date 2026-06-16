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
    st.error("⚠️ CRITICAL ERROR: Model components are missing from the root repository folder!")
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
    
    # Sidebar Metrics Dashboard
    st.sidebar.header("📊 Model Accuracy Stats")
    st.sidebar.markdown(f"""
    - **Selected Model:** Decision Tree (4-Param)
    - **R² Score:** {metadata['metrics']['r2']}
    - **Mean Absolute Error:** ₹{metadata['metrics']['mae']:,}
    - **Root MSE:** ₹{metadata['metrics']['rmse']:,}
    """)

    st.subheader("Specify Target Smartphone Metrics")
    
    # User Input Selectboxes
    col1, col2 = st.columns(2)
    with col1:
        chosen_brand = st.selectbox("Brand Name", options=metadata['brands'])
        chosen_ram = st.selectbox("RAM (GB)", options=metadata['ram_options'])
    with col2:
        chosen_color = st.selectbox("Device Color", options=metadata['colors'])
        chosen_storage = st.selectbox("Internal Storage (GB)", options=metadata['storage_options'])

    st.markdown("---")
    
    if st.button("🔮 Generate Price Evaluation", type="primary"):
        # Create a raw input dataframe matching your Colab X features exactly
        query_df = pd.DataFrame([{
            'Brand': chosen_brand,
            'RAM': float(chosen_ram),
            'Storage': float(chosen_storage),
            'Color': chosen_color
        }])
        
        # The pipeline handles raw string encoding internally, eliminating feature name errors!
        prediction = model.predict(query_df)[0]
            
        st.success("### Final Valuation Estimate")
        st.metric(label="Calculated Base Value", value=f"₹ {prediction:,.2f}")
        st.caption(f"Configured Layout: {chosen_brand} | {chosen_color} | {chosen_ram}GB RAM / {chosen_storage}GB Storage")

except Exception as e:
    st.error(f"Prediction process halted: {e}")
