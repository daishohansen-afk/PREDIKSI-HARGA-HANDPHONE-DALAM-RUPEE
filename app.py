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
    st.error("⚠️ CRITICAL ERROR: Model elements are missing from the root directory folder!")
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
    
    # Extract structural map components safely
    brand_map = metadata['brand_map']
    color_map = metadata['color_map']
    
    # Render user interactive parameters requested
    st.sidebar.header("📊 Model Accuracy Stats")
    st.sidebar.markdown(f"""
    - **Selected Model:** Decision Tree (4-Param)
    - **R² Score:** {metadata['metrics']['r2']}
    - **Mean Absolute Error:** ₹{metadata['metrics']['mae']:,}
    - **Root MSE:** ₹{metadata['metrics']['rmse']:,}
    """)

    st.subheader("Specify Target Smartphone Metrics")
    
    # 4 Input Parameters layout selection arrays
    col1, col2 = st.columns(2)
    with col1:
        chosen_brand = st.selectbox("Brand Name", options=list(brand_map.keys()))
        chosen_ram = st.selectbox("RAM (GB)", options=metadata['ram_options'])
    with col2:
        chosen_color = st.selectbox("Device Color", options=list(color_map.keys()))
        chosen_storage = st.selectbox("Internal Storage (GB)", options=metadata['storage_options'])

    st.markdown("---")
    
    if st.button("🔮 Generate Price Evaluation", type="primary"):
        # Map the selected string values to their corresponding numeric values
        mapped_brand = brand_map[chosen_brand]
        mapped_color = color_map[chosen_color]
        
        # Build the exact array shape matching our training layout: ['Brand', 'RAM', 'Storage', 'Color']
        query_data = pd.DataFrame([{
            'Brand': int(mapped_brand),
            'RAM': float(chosen_ram),
            'Storage': float(chosen_storage),
            'Color': int(mapped_color)
        }])
        
        # Calculate real-time mathematical inference
        prediction = model.predict(query_data)[0]
            
        st.success("### Final Valuation Estimate")
        st.metric(label="Calculated Base Value", value=f"₹ {prediction:,.2f}")
        st.caption(f"Configured Layout: {chosen_brand} | {chosen_color} | {chosen_ram}GB RAM / {chosen_storage}GB Storage")

except Exception as e:
    st.error(f"Prediction process halted: {e}")
