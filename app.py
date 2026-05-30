import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page configuration
st.set_page_config(
    page_title="Stacking Regression App",
    page_icon="🏠",
    layout="wide"
)

# Soft Ice Blue Theme CSS (Dropdown safe)
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f0f5fa;
}
[data-testid="stSidebar"] {
    background-color: #e2ecf5;
    border-right: 1px solid #cbd5e1;
}
h1, h2, h3, h4, h5, h6, .stMarkdown p, label, .stMetricValue, [data-testid="stHeader"] {
    color: #0f172a !important;
    font-family: 'Segoe UI', system-ui, sans-serif;
}
.hero-banner {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
    padding: 35px;
    border-radius: 16px;
    border: 1px solid #cbd5e1;
    border-bottom: 4px solid #0284c7;
    box-shadow: 0px 10px 30px rgba(15, 23, 42, 0.05);
    margin-bottom: 30px;
    text-align: center;
}
.hero-banner h1 {
    color: #0369a1 !important;
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 5px;
}
.hero-banner p {
    color: #475569 !important;
    font-size: 15px;
}
div[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
.input-card {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
}
.stDataFrame {
    background-color: white !important;
    border-radius: 8px;
}
.stButton > button {
    background: linear-gradient(135deg, #0284c7, #0369a1);
    color: #ffffff !important;
    font-weight: 700;
    border-radius: 8px;
    padding: 12px 30px;
    border: none;
    cursor: pointer;
    box-shadow: 0px 4px 15px rgba(3, 105, 161, 0.2);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0369a1, #1d4ed8);
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <h1>CALIFORNIA PROPERTY VALUATION TOOL</h1>
    <p>Ensemble Stacking Regressor (Decision Tree + Random Forest + AdaBoost) • Meta-Learner: Ridge</p>
</div>
""", unsafe_allow_html=True)

# Load artifacts
try:
    model = joblib.load("models/stacking_reg_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    df = pd.read_csv("data/california_housing.csv")
except FileNotFoundError:
    st.error("Model artifacts not found. Please run model_training.py first.")
    st.stop()

with st.sidebar:
    st.markdown("### Stacking Architecture")
    st.info("**Base Learners:**\n1. Decision Tree\n2. Random Forest\n3. AdaBoost")
    st.info("**Meta-Learner:**\nRidge Regression")

tab_dashboard, tab_diagnostics, tab_calculator = st.tabs([
    "📊 Executive Summary", 
    "⚙️ Model Diagnostics & Comparison", 
    "🔮 Price Estimator"
])

# ========================================
# TAB 1: EXECUTIVE SUMMARY
# ========================================
with tab_dashboard:
    st.subheader("Key Portfolio Indicators")
    average_value = df["Price"].mean() * 100000

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Analyzed Assets", f"{df.shape[0]:,}")
    col2.metric("Predictive Dimensions", df.shape[1] - 1)
    col3.metric("Average Block Valuation", f"${average_value:,.2f}")
    col4.metric("Regression Pipeline", "Stacking Ensembles")

    st.subheader("Asset Matrix Preview")
    st.dataframe(df.head(), use_container_width=True)

# ========================================
# TAB 2: MODEL DIAGNOSTICS & COMPARISON
# ========================================
with tab_diagnostics:
    st.subheader("Performance Comparison (Stacking vs Base Learners)")

    X = df.drop("Price", axis=1)
    y = df["Price"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)

    # Predictions
    stacking_pred = model.predict(X_test_scaled)
    
    # Base Learner predictions
    dt_pred = model.named_estimators_['dt'].predict(X_test_scaled)
    rf_pred = model.named_estimators_['rf'].predict(X_test_scaled)
    ada_pred = model.named_estimators_['ada'].predict(X_test_scaled)

    # Create evaluation metrics table
    comparison_data = {
        "Model": ["Decision Tree (Base)", "Random Forest (Base)", "AdaBoost (Base)", "STACKING MODEL (Meta)"],
        "MAE": [
            mean_absolute_error(y_test, dt_pred),
            mean_absolute_error(y_test, rf_pred),
            mean_absolute_error(y_test, ada_pred),
            mean_absolute_error(y_test, stacking_pred)
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(y_test, dt_pred)),
            np.sqrt(mean_squared_error(y_test, rf_pred)),
            np.sqrt(mean_squared_error(y_test, ada_pred)),
            np.sqrt(mean_squared_error(y_test, stacking_pred))
        ],
        "R² Score": [
            r2_score(y_test, dt_pred),
            r2_score(y_test, rf_pred),
            r2_score(y_test, ada_pred),
            r2_score(y_test, stacking_pred)
        ]
    }
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    col_plot, col_info = st.columns(2)
    with col_plot:
        st.subheader("Actual vs Predicted House Values")
        fig, ax = plt.subplots(facecolor="#f0f5fa")
        ax.set_facecolor("#ffffff")
        ax.scatter(y_test, stacking_pred, alpha=0.2, color="#0284c7")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel("Actual Price ($100k)")
        ax.set_ylabel("Predicted Price ($100k)")
        ax.tick_params(colors='#0f172a')
        st.pyplot(fig)
    with col_info:
        st.subheader("Why Stacking Succeeded")
        st.write("""
        Stacking works by leveraging the unique predictive strengths of diverse algorithms:
        * **Decision Trees** identify fast, split-based partitions.
        * **Random Forests** stabilize variance and capture non-linear relationships.
        * **AdaBoost** focuses on sequential residual errors.
        
        The **Ridge Meta-Learner** synthesizes these predictions without leaking data, creating a more robust output.
        """)

# ========================================
# TAB 3: REAL-TIME ESTIMATOR
# ========================================
with tab_calculator:
    st.subheader("Estimate Real Estate Valuations")
    col_dem, col_fin, col_rel = st.columns(3)

    with col_dem:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Demographics")
        med_inc = st.slider("Median Block Income ($10k scale)", 0.5, 15.0, 4.0, step=0.1)
        population = st.slider("Block Population Count", 3, 35000, 1400)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_fin:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("### 🏢 Structural Metrics")
        house_age = st.slider("Median Property Age (Years)", 1, 52, 28)
        ave_rooms = st.slider("Average Rooms Per Household", 1.0, 15.0, 5.0, step=0.1)
        ave_bedrooms = st.slider("Average Bedrooms Per Household", 0.5, 5.0, 1.1, step=0.1)
        ave_occup = st.slider("Average Household Occupancy", 0.8, 10.0, 2.9, step=0.1)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_rel:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.markdown("### 📍 Geographic Mapping")
        latitude = st.slider("Block Latitude Coordinates", 32.5, 42.5, 35.6, step=0.1)
        longitude = st.slider("Block Longitude Coordinates", -124.3, -114.3, -119.5, step=0.1)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("CALCULATE VALUATION PROFILE", use_container_width=True):
        data_row = np.array([[med_inc, house_age, ave_rooms, ave_bedrooms, population, ave_occup, latitude, longitude]])
        scaled_row = scaler.transform(data_row)
        prediction = model.predict(scaled_row)[0]
        predicted_dollars = prediction * 100000

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #0284c7, #0369a1); padding: 30px; border-radius: 12px; text-align: center; font-size: 26px; font-weight: 700; color: white;">
            Predicted Asset Value: ${predicted_dollars:,.2f}
        </div>
        """, unsafe_allow_html=True)
        