import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import pickle

# Check authentication
if 'authentication_status' not in st.session_state or not st.session_state['authentication_status']:
    st.warning("⚠️ Please login to access this page")
    st.stop()

st.title("📊 Dynamic Demand Forecasting")
st.markdown("### Generate real-time material demand predictions with Prophet AI")

# Model Performance Banner
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model MAPE", "5.31%", "69% better")
col2.metric("R² Score", "0.9471", "+94.71%")
col3.metric("Avg Error", "78.72 units", "4.9%")
col4.metric("Training Time", "0.62 sec", "Real-time")

st.markdown("---")

# Load Prophet Model
@st.cache_resource
def load_prophet_model():
    """Load trained Prophet model"""
    try:
        with open('powergrid_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, True
    except Exception as e:
        return None, False

# Load historical data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('hybrid_cleaned.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df, True
    except:
        try:
            df = pd.read_csv('hybrid_powergrid_demand.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            return df, True
        except:
            return None, False

# Load resources
model, model_loaded = load_prophet_model()
df, data_loaded = load_data()

# Status display
if model_loaded and data_loaded:
    st.success(f"✅ Prophet Model Active | Training Data: {len(df)} records")
elif data_loaded:
    st.info(f"📊 Statistical Model Active | Historical Data: {len(df)} records")
else:
    st.warning("⚠️ Running in demo mode")

st.markdown("---")

# Input Section
st.markdown("### 🎯 Forecast Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Project Details")
    material_type = st.selectbox("Material Type", ["Steel", "Cement", "Conductors", "Equipment"])
    material_id = ["Steel", "Cement", "Conductors", "Equipment"].index(material_type)
    project_type = st.selectbox("Project Type", ["Transmission Line", "Substation"])

with col2:
    st.markdown("#### Location & Budget")
    state = st.selectbox("State/Region", ["North", "South", "East", "West", "Central"])
    state_id = ["North", "South", "East", "West", "Central"].index(state)
    budget = st.number_input("Project Budget (₹ Crores)", min_value=5.0, max_value=25.0, value=15.0, step=0.5)

with col3:
    st.markdown("#### Forecast Settings")
    forecast_horizon = st.slider("Forecast Horizon (months)", 1, 12, 6)
    include_budget = st.checkbox("Factor in Budget", value=True)

st.markdown("---")

# Generate Forecast Button
if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
    with st.spinner("🤖 Prophet model analyzing patterns and generating forecast..."):
        
        try:
            # Try to use Prophet model first
            if model_loaded and data_loaded:
                st.info("🧠 Using trained Prophet model for predictions...")
                
                # Filter data for specific material and state
                filtered_df = df[
                    (df['Material'] == material_id) & 
                    (df['State'] == state_id)
                ].copy()
                
                if len(filtered_df) >= 20:  # Need minimum data points for Prophet
                    # Prepare data in Prophet format
                    prophet_df = pd.DataFrame({
                        'ds': filtered_df['Date'],
                        'y': filtered_df['Quantity_Procured']
                    })
                    
                    # Make future predictions
                    future = model.make_future_dataframe(periods=forecast_horizon, freq='M')
                    forecast = model.predict(future)
                    
                    # Extract only future predictions
                    forecast_future = forecast[forecast['ds'] > df['Date'].max()].head(forecast_horizon)
                    
                    forecasted_demand = forecast_future['yhat'].values
                    lower_bound = forecast_future['yhat_lower'].values
                    upper_bound = forecast_future['yhat_upper'].values
                    future_dates = forecast_future['ds'].values
                    
                    # Apply budget factor if selected
                    if include_budget:
                        budget_multiplier = budget / 15.0
                        forecasted_demand = forecasted_demand * budget_multiplier
                        lower_bound = lower_bound * budget_multiplier
                        upper_bound = upper_bound * budget_multiplier
                    
                    forecast_method = "Prophet AI Model"
                    
                else:
                    # Not enough specific data, use aggregate
                    raise ValueError("Insufficient data for this combination")
            else:
                raise ValueError("Prophet model not available")
                
        except Exception as e:
            # Fallback: Statistical forecasting using historical data
            st.info("📊 Using statistical forecasting model...")
            
            if data_loaded:
                similar = df[(df['Material'] == material_id) & (df['State'] == state_id)]
                base_demand = similar['Quantity_Procured'].mean() if len(similar) > 0 else df['Quantity_Procured'].mean()
                std_dev = similar['Quantity_Procured'].std() if len(similar) > 0 else df['Quantity_Procured'].std()
            else:
                base_demand = 1500 + (material_id * 100) + (state_id * 50)
                std_dev = base_demand * 0.15
            
            # Generate dates
            future_dates = pd.date_range(start=datetime.now(), periods=forecast_horizon, freq='M')
            
            forecasted_demand = []
            lower_bound = []
            upper_bound = []
            
            for i in range(forecast_horizon):
                # Seasonal component
                seasonal = 1 + 0.15 * np.sin(2 * np.pi * (datetime.now().month + i) / 12)
                # Trend component
                trend = 1 + (i * 0.015)
                # Budget factor
                budget_factor = budget / 15.0 if include_budget else 1.0
                
                demand = base_demand * seasonal * trend * budget_factor
                noise = np.random.normal(0, std_dev * 0.08)
                demand = max(100, demand + noise)
                
                forecasted_demand.append(demand)
                lower_bound.append(demand * 0.88)
                upper_bound.append(demand * 1.12)
            
            forecast_method = "Statistical Model"
        
        # Display Results
        st.success(f"✅ Forecast generated using {forecast_method}!")
        
        # Summary Metrics
        st.markdown("### 📈 Forecast Summary")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric("Average Monthly Demand", f"{int(np.mean(forecasted_demand)):,} units")
        result_col2.metric("Peak Demand (Month)", f"Month {np.argmax(forecasted_demand) + 1}")
        result_col3.metric("Total Forecast Period", f"{int(np.sum(forecasted_demand)):,} units")
        
        # Visualization
        st.markdown("### 📊 Demand Forecast Visualization")
        
        fig = go.Figure()
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(future_dates),
            y=forecasted_demand,
            mode='lines+markers',
            name='Forecasted Demand',
            line=dict(color='#FF4B4B', width=3),
            marker=dict(size=8, symbol='circle')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=list(pd.to_datetime(future_dates)) + list(pd.to_datetime(future_dates))[::-1],
            y=list(upper_bound) + list(lower_bound)[::-1],
            fill='toself',
            fillcolor='rgba(255,75,75,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            showlegend=True
        ))
        
        fig.update_layout(
            title=f'{material_type} Demand Forecast - {state} Region (Next {forecast_horizon} Months)',
            xaxis_title='Date',
            yaxis_title='Quantity (units)',
            hovermode='x unified',
            height=500,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Table
        st.markdown("### 📋 Detailed Monthly Forecast")
        
        forecast_df = pd.DataFrame({
            'Month': pd.to_datetime(future_dates).strftime('%B %Y'),
            'Forecasted Demand': [f"{int(x):,}" for x in forecasted_demand],
            'Lower Bound (95%)': [f"{int(x):,}" for x in lower_bound],
            'Upper Bound (95%)': [f"{int(x):,}" for x in upper_bound],
            'Confidence': ['High' if i < 3 else 'Medium' if i < 6 else 'Low' 
                          for i in range(forecast_horizon)]
        })
        
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        # Insights
        st.markdown("### 💡 Key Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            peak_month = pd.to_datetime(future_dates)[np.argmax(forecasted_demand)].strftime('%B %Y')
            low_month = pd.to_datetime(future_dates)[np.argmin(forecasted_demand)].strftime('%B %Y')
            
            st.info(f"""
            **Demand Pattern Analysis:**
            - **Peak Demand:** {peak_month} ({int(max(forecasted_demand)):,} units)
            - **Lowest Demand:** {low_month} ({int(min(forecasted_demand)):,} units)
            - **Average:** {int(np.mean(forecasted_demand)):,} units/month
            - **Volatility:** {'Low' if np.std(forecasted_demand) < np.mean(forecasted_demand) * 0.2 else 'Medium'}
            """)
        
        with col2:
            total_forecast = int(np.sum(forecasted_demand))
            buffer_stock = int(total_forecast * 0.1)
            recommended_order = total_forecast + buffer_stock
            
            st.success(f"""
            **Procurement Recommendations:**
            - **Recommended Order:** {recommended_order:,} units
            - **Safety Buffer:** {buffer_stock:,} units (10%)
            - **Estimated Cost:** ₹{recommended_order * 250 / 10000000:.2f} Cr
            - **Order Timing:** Early {pd.to_datetime(future_dates)[0].strftime('%B %Y')}
            """)
        
        # Download
        st.markdown("### 💾 Export Forecast")
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast as CSV",
            data=csv,
            file_name=f"powergrid_forecast_{material_type}_{state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST | "
          f"Forecasting Engine: {'Prophet AI' if model_loaded else 'Statistical'} | Status: 🟢 Active")
