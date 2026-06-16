import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Price Predictor", page_icon="📱", layout="centered")

st.title("📱 Smartphone Price Prediction App")
st.markdown("Enter specifications below to generate an immediate valuation matrix built from our model pipeline.")

@st.cache_resource
def load_assets():
    with open('phone_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('categories_metadata.pkl', 'rb') as f:
        meta = pickle.load(f)
    return model, meta

try:
    model, metadata = load_assets()
    
    # Sidebar stats displaying requested validation accuracy/metrics
    st.sidebar.header("📊 Model Accuracy Stats")
    st.sidebar.markdown(f"""
    - **Selected Model:** Decision Tree
    - **R² Score:** {metadata['metrics']['r2']}
    - **Mean Absolute Error:** ₹{metadata['metrics']['mae']:,}
    - **Root MSE:** ₹{metadata['metrics']['rmse']:,}
    """)

    st.subheader("Specify Target Smartphone Metrics")
    
    # 4 Input Parameters layout
    col1, col2 = st.columns(2)
    with col1:
        chosen_brand = st.selectbox("Brand Name", options=metadata['brands'])
        chosen_ram = st.selectbox("RAM (GB)", options=metadata['ram_options'])
    with col2:
        chosen_model = st.selectbox("Model Series", options=metadata['models'])
        chosen_storage = st.selectbox("Internal Storage (GB)", options=metadata['storage_options'])

    st.markdown("---")
    
    if st.button("🔮 Generate Price Evaluation", type="primary"):
        # Formatting data matching preprocessing pipeline expectations
        query_df = pd.DataFrame([{
            'Brand': chosen_brand,
            'Model': chosen_model,
            'RAM': float(chosen_ram),
            'Storage': float(chosen_storage)
        }])
        
        prediction = model.predict(query_df)[0]
        
        st.success("### Final Valuation Estimate")
        st.metric(label="Calculated Base Value", value=f"₹ {prediction:,.2f}")
        st.caption(f"Configured Layout: {chosen_brand} {chosen_model} ({chosen_ram}GB / {chosen_storage}GB)")

except Exception as e:
    st.error(f"Waiting for files to be uploaded or executed properly. Details: {e}")
