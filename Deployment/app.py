import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI-Driven Energy Arbitrage",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (For Professional UI) ---
st.markdown("""
    <style>
    /* Main Background */
    .main {
        background-color: #0e1117;
    }
    
    /* Custom Metric Cards */
    div.css-1r6slb0.e1tzin5v2 {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric Value Text */
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        color: #00d4ff !important;
        font-weight: 700;
    }
    
    /* Metric Label Text */
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
        color: #9ca3af !important;
    }

    /* Custom Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #2563eb, #00d4ff);
        color: white;
        border: none;
        font-weight: bold;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. LOAD RESOURCES & DATA ---
@st.cache_resource
def load_resources():
    # NOTE: Ensure these files exist in your directory
    model = joblib.load('model_price_forecast.joblib')
    scaler = joblib.load('scaler.joblib')
    df_history = pd.read_csv('processed_dataset.csv')
    return model, scaler, df_history

# --- 2. SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=60)
    st.title("⚙️ Control Panel")
    
    st.markdown("### 📅 Forecast Parameters")
    days_to_forecast = st.slider("Forecast Horizon (Days)", 1, 7, 3, help="How many days into the future to predict?")
    
    st.markdown("### 🔋 Battery Specs")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        battery_capacity = st.number_input("Capacity (MWh)", value=8.0, step=0.5)
    with col_sb2:
        charge_rate = st.number_input("Max Rate (MW)", value=2.0, step=0.5)
        
    st.markdown("---")
    st.info("💡 **Tip:** Adjust battery specs to see how capacity affects potential profit.")
    
    run_btn = st.button("🚀 Run AI Simulation", type="primary", use_container_width=True)

# --- 3. ROBUST FEATURE GENERATOR ---
def generate_future_data(days, history_df):
    start_date = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=days*24, freq='H')
    future_df = pd.DataFrame({'timestamp': future_dates})
    
    # Time Features
    future_df['hour'] = future_df['timestamp'].dt.hour
    future_df['day_of_week'] = future_df['timestamp'].dt.dayofweek
    future_df['month'] = future_df['timestamp'].dt.month
    future_df['quarter'] = future_df['timestamp'].dt.quarter
    future_df['day_of_year'] = future_df['timestamp'].dt.dayofyear
    future_df['year'] = future_df['timestamp'].dt.year
    
    # Fill Missing Columns
    columns_to_exclude = ['price_actual', 'timestamp', 'hour', 'day_of_week', 'month', 'quarter', 'day_of_year', 'year']
    required_columns = [c for c in history_df.columns if c not in columns_to_exclude]
    
    for col in required_columns:
        if col in history_df.columns:
            future_df[col] = history_df[col].mean()
            
    input_features = [c for c in history_df.columns if c != 'price_actual' and c != 'timestamp']
    final_df = future_df[input_features]
    
    return final_df, future_dates

# --- MAIN APP LOGIC ---
st.title("⚡ AI-Driven Electricity Price Forecasting")
st.markdown("### *For Energy Arbitrage Optimization*")

# Introduction / How it works (Makes it easily understandable)
with st.expander("ℹ️ How this AI Model works"):
    st.markdown("""
    1.  **Forecasting:** The system uses an **XGBoost/Neural Network** model to predict electricity prices for the next 1-7 days.
    2.  **Optimization:** An algorithm analyzes the predicted prices to find the lowest points to **Buy (Charge)** and highest points to **Sell (Discharge)**.
    3.  **Simulation:** It simulates a battery storage system executing these trades to calculate potential profit.
    """)

try:
    model, scaler, df_history = load_resources()
except FileNotFoundError:
    st.error("⚠️ **System Halted:** Model files not found. Please check your directory.")
    st.stop()

if run_btn:
    # Progress Bar for professional feel
    progress_text = "Running AI Inference..."
    my_bar = st.progress(0, text=progress_text)
    
    # 1. Generate Data
    X_input, date_index = generate_future_data(days_to_forecast, df_history)
    my_bar.progress(40, text="Generating Forecasting Scenarios...")
    
    # 2. Predict
    try:
        X_scaled = scaler.transform(X_input)
        predicted_prices = model.predict(X_scaled)
        my_bar.progress(70, text="Optimizing Battery Schedule...")
    except ValueError as e:
        st.error(f"Data Shape Error: {e}")
        st.stop()
        
    # 3. Optimize Strategy
    results = pd.DataFrame({'Datetime': date_index, 'Predicted_Price': predicted_prices})
    low_threshold = np.percentile(predicted_prices, 25)
    high_threshold = np.percentile(predicted_prices, 75)
    
    actions, soc, profits, cash, current_soc = [], [0.0], [0.0], 0.0, 0.0
    
    for price in predicted_prices:
        action = "HOLD"
        if price <= low_threshold and current_soc < battery_capacity:
            energy = min(charge_rate, battery_capacity - current_soc)
            current_soc += energy
            cash -= energy * price
            action = "BUY"
        elif price >= high_threshold and current_soc > 0:
            energy = min(charge_rate, current_soc)
            current_soc -= energy
            cash += energy * price
            action = "SELL"
        
        actions.append(action)
        soc.append(current_soc)
        profits.append(cash)
        
    results['Action'] = actions
    
    my_bar.progress(100, text="Computation Complete.")
    my_bar.empty()

    # --- DASHBOARD UI ---
    
    # Top Level Metrics
    st.markdown("#### 📊 Simulation Results")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    # Calculate Profit
    final_profit = profits[-1]
    profit_color = "normal" if final_profit > 0 else "off"
    
    kpi1.metric("💰 Projected Profit", f"€{final_profit:,.2f}", delta="ROI Analysis")
    kpi2.metric("📉 Avg. Electricity Price", f"€{results['Predicted_Price'].mean():.2f}")
    kpi3.metric("🟢 Charge Cycles (Buy)", len(results[results['Action']=="BUY"]))
    kpi4.metric("🔴 Discharge Cycles (Sell)", len(results[results['Action']=="SELL"]))

    st.markdown("---")

    # Visualization Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Price Forecast & Strategy", "🔋 Battery State of Charge", "📋 Detailed Data"])
    
    with tab1:
        # Professional Chart with Zones
        fig = go.Figure()
        
        # Price Line
        fig.add_trace(go.Scatter(x=results['Datetime'], y=results['Predicted_Price'], 
                                 mode='lines', name='Price (€/MWh)', line=dict(color='#00d4ff', width=2)))
        
        # Buy Zone Shading
        fig.add_hrect(y0=0, y1=low_threshold, line_width=0, fillcolor="green", opacity=0.1, annotation_text="Buy Zone (Charge)")
        
        # Sell Zone Shading
        fig.add_hrect(y0=high_threshold, y1=results['Predicted_Price'].max(), line_width=0, fillcolor="red", opacity=0.1, annotation_text="Sell Zone (Discharge)")

        fig.update_layout(
            title="Electricity Price Forecast & Arbitrage Zones",
            xaxis_title="Timeline",
            yaxis_title="Price (€/MWh)",
            template="plotly_dark",
            hovermode="x unified",
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        fig_soc = px.area(x=results['Datetime'], y=soc[1:], title="Battery State of Charge (SoC)")
        fig_soc.update_traces(line_color='#2ecc71', fillcolor="rgba(46, 204, 113, 0.2)")
        fig_soc.update_layout(template="plotly_dark", yaxis_title="Stored Energy (MWh)", hovermode="x unified")
        st.plotly_chart(fig_soc, use_container_width=True)
        
    with tab3:
        st.dataframe(results.style.format({"Predicted_Price": "€{:.2f}"}), use_container_width=True)

else:
    # Empty State / Call to Action
    st.warning("⚠️ **Waiting for Input:** Adjust parameters in the sidebar and click 'Run AI Simulation' to start.")
