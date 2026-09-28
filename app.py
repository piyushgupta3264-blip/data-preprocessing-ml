import streamlit as st
import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="House Price Predictor & Preprocessing Pipeline",
    page_icon="🏡",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 14px rgba(0,0,0,0.1);
    }
    .price-display {
        font-size: 2.5rem;
        font-weight: 800;
        color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join("models", "house_price_pipeline.joblib")
METRICS_PATH = os.path.join("models", "model_metrics.json")
DATA_PATH = os.path.join("data", "housing_data.csv")

@st.cache_resource
def load_trained_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_metrics_and_data():
    metrics = None
    data = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    if os.path.exists(DATA_PATH):
        data = pd.read_csv(DATA_PATH)
    return metrics, data

pipeline = load_trained_model()
metrics, data = load_metrics_and_data()

st.markdown('<div class="main-title">🏡 House Price Estimation & ML Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">End-to-End Data Preprocessing, Imputation, One-Hot Encoding, and Regression Modeling</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Price Estimator", "📊 Model Comparison & Metrics", "🔍 Data & Preprocessing Pipeline"])

# --- TAB 1: ESTIMATOR ---
with tab1:
    col_inputs, col_result = st.columns([3, 2], gap="large")
    
    with col_inputs:
        st.subheader("🛠️ Property Details")
        
        sqft = st.slider("Square Footage (sqft)", min_value=500, max_value=6000, value=2200, step=50)
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            bedrooms = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
            year_built = st.slider("Year Built", min_value=1960, max_value=2024, value=2016, step=1)
        with col_c2:
            bathrooms = st.selectbox("Bathrooms", options=[1, 2, 3, 4, 5], index=1)
            has_garage = st.radio("Garage Included", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
            
        location = st.selectbox("Location / Neighborhood", options=['Downtown', 'Suburbs', 'Uptown', 'Rural', 'Waterfront'], index=0)

    with col_result:
        st.subheader("💡 Valuation Result")
        if pipeline is not None:
            input_data = pd.DataFrame([{
                'square_footage': sqft,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'year_built': year_built,
                'has_garage': has_garage,
                'location': location
            }])
            
            predicted_price = pipeline.predict(input_data)[0]
            
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 1rem; opacity: 0.9;">Estimated Market Value</div>
                <div class="price-display">${predicted_price:,.2f}</div>
                <div style="font-size: 0.85rem; margin-top: 8px; opacity: 0.85;">
                    Estimated Rate: <b>${predicted_price/sqft:,.2f} / sqft</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.info(f"""
            **Features Summary:**
            - **Living Space:** {sqft:,} sqft ({bedrooms} beds, {bathrooms} baths)
            - **Location:** {location}
            - **Age:** {2024 - year_built} years old
            - **Garage:** {"Yes" if has_garage == 1 else "No"}
            """)
        else:
            st.warning("⚠️ Model pipeline not trained yet. Run `python run_project.py` or the training script.")

# --- TAB 2: MODEL COMPARISON ---
with tab2:
    st.subheader("📈 Regression Model Benchmark")
    if metrics:
        st.write(f"**Best Performing Model:** `{metrics.get('best_model')}`")
        
        all_results = metrics.get('all_results', {})
        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        results_df = results_df.rename(columns={'MAE': 'MAE ($)', 'RMSE': 'RMSE ($)', 'R2_Score': 'R² Score'})
        
        st.dataframe(results_df.style.highlight_max(subset=['R² Score'], color='#D1FAE5').highlight_min(subset=['MAE ($)', 'RMSE ($)'], color='#D1FAE5'), use_container_width=True)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        best = metrics.get('best_metrics', {})
        col_m1.metric("R² Score (Accuracy)", f"{best.get('R2_Score', 0):.4f}")
        col_m2.metric("Mean Absolute Error (MAE)", f"${best.get('MAE', 0):,.2f}")
        col_m3.metric("Root Mean Squared Error (RMSE)", f"${best.get('RMSE', 0):,.2f}")
    else:
        st.info("Metrics will appear here once training completes.")

# --- TAB 3: DATA PREPROCESSING ---
with tab3:
    st.subheader("🧹 Preprocessing & Data Pipeline Details")
    st.markdown("""
    1. **Missing Value Imputation**:
       - *Numerical (`square_footage`, `year_built`)*: Imputed using **Median strategy**.
       - *Categorical (`location`)*: Imputed using **Most Frequent (Mode) strategy**.
    2. **Categorical Encoding**:
       - *`location`*: Encoded via `OneHotEncoder(handle_unknown='ignore')`.
    3. **Feature Scaling**:
       - *Numerical Features*: Scaled with `StandardScaler` to ensure zero mean and unit variance.
    4. **Leakage Prevention**:
       - Preprocessor and Regressors are encapsulated inside `sklearn.pipeline.Pipeline`.
    """)
    
    if data is not None:
        st.write("#### 📋 Raw Data Sample")
        st.dataframe(data.head(10), use_container_width=True)
