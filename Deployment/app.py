import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import xgboost

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EcoVolt AI | Arbitrage",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #41444b; }
    h1, h2, h3 { color: #00d4ff !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. LOAD RESOURCES & DATA ---
@st.cache_resource
def load_resources():
    model = joblib.load('model_price_forecast.joblib')
    scaler = joblib.load('scaler.joblib')
    # Load data to get the column names and averages
    df_history = pd.read_csv('processed_dataset.csv')
    return model, scaler, df_history

try:
    model, scaler, df_history = load_resources()
except FileNotFoundError:
    st.error("⚠️ Critical Error: Missing files. Ensure 'model_price_forecast.joblib', 'scaler.joblib', and 'processed_dataset.csv' are in the folder.")
    st.stop()

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("⚡ Control Panel")
    days_to_forecast = st.slider("📅 Forecast Horizon (Days)", 1, 7, 3)
    st.markdown("### 🔋 Battery Specs")
    battery_capacity = st.number_input("Capacity (MWh)", value=1.0, step=0.1)
    charge_rate = st.number_input("Max Charge Rate (MW)", value=0.5, step=0.1)
    st.markdown("---")
    run_btn = st.button("🚀 Run AI Simulation", type="primary", use_container_width=True)

# --- 3. ROBUST FEATURE GENERATOR ---
def generate_future_data(days, history_df):
    # A. Create Future Time Index
    start_date = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=days*24, freq='H')
    future_df = pd.DataFrame({'timestamp': future_dates}) # specific name might not matter, but keeping it clean
    
    # B. Generate Time Features (Must match training)
    future_df['hour'] = future_df['timestamp'].dt.hour
    future_df['day_of_week'] = future_df['timestamp'].dt.dayofweek
    future_df['month'] = future_df['timestamp'].dt.month
    future_df['quarter'] = future_df['timestamp'].dt.quarter
    future_df['day_of_year'] = future_df['timestamp'].dt.dayofyear
    future_df['year'] = future_df['timestamp'].dt.year
    
    # C. Fill Missing "Generation" Columns with Historical Averages
    # This fixes the "Feature names seen at fit time" error
    
    # 1. Identify columns needed (excluding the target and time columns we just made)
    # We drop 'price_actual' because that is what we want to PREDICT
    # We drop 'timestamp' if it exists
    columns_to_exclude = ['price_actual', 'timestamp', 'hour', 'day_of_week', 'month', 'quarter', 'day_of_year', 'year']
    
    # Get all columns from the training file
    required_columns = [c for c in history_df.columns if c not in columns_to_exclude]
    
    # 2. Fill future_df with the MEAN of those columns from history
    for col in required_columns:
        if col in history_df.columns:
            # Use the average value from historical data as a "forecast"
            mean_val = history_df[col].mean()
            future_df[col] = mean_val
            
    # D. Final Column Ordering
    # The scaler expects columns in a specific order. We must match history_df exactly (minus target)
    # We create the list of input features expected by the model
    input_features = [c for c in history_df.columns if c != 'price_actual' and c != 'timestamp']
    
    # Reorder future_df to match input_features
    final_df = future_df[input_features]
    
    return final_df, future_dates

# --- MAIN APP LOGIC ---
st.title("⚡ EcoVolt: AI Energy Arbitrage")
st.markdown("Real-time Electricity Price Forecasting & Battery Optimization Engine")

if run_btn:
    with st.status("🤖 AI System Running...", expanded=True) as status:
        
        st.write("🔄 Generating future scenario (using historical averages for generation)...")
        X_input, date_index = generate_future_data(days_to_forecast, df_history)
        
        st.write("🧠 Neural Network Inference...")
        try:
            # Scale and Predict
            X_scaled = scaler.transform(X_input)
            predictions_scaled = model.predict(X_scaled)
            predicted_prices = predictions_scaled
        except ValueError as e:
            st.error(f"❌ Data Shape Error: {e}")
            st.stop()
            
        # Create Results DataFrame
        results = pd.DataFrame({'Datetime': date_index, 'Predicted_Price': predicted_prices})

        st.write("💰 Optimizing Arbitrage Strategy...")
        # Strategy Logic
        low_threshold = np.percentile(predicted_prices, 25)
        high_threshold = np.percentile(predicted_prices, 75)
        
        actions, soc, profits, cash, current_soc = [], [0.0], [0.0], 0.0, 0.0
        
        for price in predicted_prices:
            action = "HOLD"
            if price <= low_threshold and current_soc < battery_capacity:
                energy = min(charge_rate, battery_capacity - current_soc)
                current_soc += energy
                cash -= energy * price
                action = "BUY (Charge)"
            elif price >= high_threshold and current_soc > 0:
                energy = min(charge_rate, current_soc)
                current_soc -= energy
                cash += energy * price
                action = "SELL (Discharge)"
            
            actions.append(action)
            soc.append(current_soc)
            profits.append(cash)
            
        results['Action'] = actions
        status.update(label="✅ Optimization Complete!", state="complete", expanded=False)

    # --- DASHBOARD ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Projected Profit", f"€{profits[-1]:.2f}") # Assuming Euro based on data look, or change to $
    col2.metric("Avg. Price", f"{results['Predicted_Price'].mean():.2f}")
    col3.metric("Buy Signals", len(results[results['Action']=="BUY (Charge)"]))
    col4.metric("Sell Signals", len(results[results['Action']=="SELL (Discharge)"]))

    tab1, tab2, tab3 = st.tabs(["📈 Price Forecast", "🔋 Battery Strategy", "📋 Raw Data"])
    
    with tab1:
        fig = px.line(results, x='Datetime', y='Predicted_Price', title="Electricity Price Forecast")
        fig.add_hline(y=low_threshold, line_dash="dash", line_color="green", annotation_text="Buy Zone")
        fig.add_hline(y=high_threshold, line_dash="dash", line_color="red", annotation_text="Sell Zone")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        fig_soc = px.area(x=results['Datetime'], y=soc[1:], title="Battery Charge Level")
        fig_soc.update_traces(line_color='#00d4ff')
        fig_soc.update_layout(template="plotly_dark")
        st.plotly_chart(fig_soc, use_container_width=True)
        
    with tab3:
        st.dataframe(results, use_container_width=True)

else:
    st.info("👈 Select settings and click 'Run AI Simulation' to start.")
