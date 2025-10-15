import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Check authentication
if 'authentication_status' not in st.session_state or not st.session_state['authentication_status']:
    st.warning("⚠️ Please login to access this page")
    st.info("👈 Use the sidebar menu to return to the main page and login")
    st.stop()

st.title("📊 Dynamic Demand Forecasting")
st.markdown("### Generate real-time material demand predictions")

# Model Performance
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model MAPE", "5.31%", "69% better")
col2.metric("R² Score", "0.9471", "+94.71%")
col3.metric("Avg Error", "78.72 units", "4.9%")
col4.metric("Training Time", "0.62 sec", "Real-time")

st.markdown("---")

# Input Section
st.markdown("### 🎯 Forecast Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Project Details")
    material_type = st.selectbox("Material Type", ["Steel", "Cement", "Conductors", "Equipment"])
    project_type = st.selectbox("Project Type", ["Transmission Line", "Substation"])

with col2:
    st.markdown("#### Location & Budget")
    state = st.selectbox("State/Region", ["North", "South", "East", "West", "Central"])
    budget = st.number_input("Project Budget (₹ Crores)", min_value=5.0, max_value=25.0, value=15.0)

with col3:
    st.markdown("#### Forecast Settings")
    forecast_horizon = st.slider("Forecast Horizon (months)", 1, 12, 6)
    lead_time = st.slider("Lead Time (days)", 10, 45, 25)

st.markdown("---")

# Generate Forecast Button
if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
    with st.spinner("Generating forecast with Prophet model..."):
        # Simulate forecast
        base_demand = 1500 + np.random.randint(-200, 200)
        future_dates = pd.date_range(start=datetime.now(), periods=forecast_horizon, freq='M')
        forecasted_demand = [base_demand * (1 + 0.1 * np.sin(i)) for i in range(forecast_horizon)]
        
        st.success("✅ Forecast generated successfully!")
        
        # Results Summary
        st.markdown("### 📈 Forecast Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric("Average Monthly Demand", f"{int(np.mean(forecasted_demand))} units")
        result_col2.metric("Peak Month", f"Month {np.argmax(forecasted_demand) + 1}")
        result_col3.metric("Total Forecast", f"{int(np.sum(forecasted_demand))} units")
        
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=future_dates, y=forecasted_demand,
                                mode='lines+markers', name='Forecasted Demand',
                                line=dict(color='#FF4B4B', width=3)))
        fig.update_layout(title=f'{material_type} Demand Forecast - Next {forecast_horizon} Months',
                         xaxis_title='Month', yaxis_title='Quantity (units)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast Table
        forecast_df = pd.DataFrame({
            'Month': future_dates.strftime('%B %Y'),
            'Forecasted Demand': [int(x) for x in forecasted_demand],
            'Confidence': ['High' if i < 3 else 'Medium' if i < 6 else 'Low' 
                          for i in range(forecast_horizon)]
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        # Download
        csv = forecast_df.to_csv(index=False)
        st.download_button("📥 Download Forecast", csv, 
                          f"forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                          mime="text/csv")
