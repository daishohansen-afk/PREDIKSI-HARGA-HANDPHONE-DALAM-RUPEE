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
    st.error("⚠️ CRITICAL ERROR: Model files are completely missing from the server repository!")
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
    
    st.sidebar.header("📊 Model Accuracy Stats")
    st.sidebar.markdown(f"""
    - **Selected Model:** Decision Tree
    - **R² Score:** {metadata['metrics']['r2']}
    - **Mean Absolute Error:** ₹{metadata['metrics']['mae']:,}
    - **Root MSE:** ₹{metadata['metrics']['rmse']:,}
    """)

    st.subheader("Specify Target Smartphone Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        chosen_brand = st.selectbox("Brand Name", options=metadata['brands'])
        chosen_ram = st.selectbox("RAM (GB)", options=metadata['ram_options'])
    with col2:
        chosen_model = st.selectbox("Model Series", options=metadata['models'])
        chosen_storage = st.selectbox("Internal Storage (GB)", options=metadata['storage_options'])

    st.markdown("---")
    
    if st.button("🔮 Generate Price Evaluation", type="primary"):
        query_df = pd.DataFrame([{
            'Brand': chosen_brand,
            'Model': chosen_model,
            'RAM': float(chosen_ram),
            'Storage': float(chosen_storage)
        }])
        
        # Let's run prediction. If it's a structural pipeline or model, it executes cleanly
        try:
            prediction = model.predict(query_df)[0]
        except AttributeError:
            # Fallback if pickle attributes don't match exactly due to backend updates
            st.warning("🔄 Re-syncing local pipeline definitions...")
            prediction = model.steps[-1][1].predict(model.steps[0][1].transform(query_df))[0]
            
        st.success("### Final Valuation Estimate")
        st.metric(label="Calculated Base Value", value=f"₹ {prediction:,.2f}")
        st.caption(f"Configured Layout: {chosen_brand} {chosen_model} ({chosen_ram}GB / {chosen_storage}GB)")

except Exception as e:
    st.error(f"An internal mismatch occurred while unpacking artifacts. Please re-run your Colab cells to sync your models! Error details: {e}")
