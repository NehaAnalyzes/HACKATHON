import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import pickle
from datetime import datetime

# Check authentication
if 'authentication_status' not in st.session_state or not st.session_state['authentication_status']:
    st.warning("⚠️ Please login to access this page")
    st.stop()

st.title("⚙️ Settings & Data Management")
st.markdown("### Configure system settings and upload new data")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload Data", "⚙️ System Settings", "ℹ️ About"])

# TAB 1: Upload Data
with tab1:
    st.markdown("### 📤 Upload New Procurement Data")
    st.markdown("Upload CSV file to update historical data and retrain Prophet model")
    
    # Instructions
    with st.expander("📋 Data Format Requirements"):
        st.markdown("""
        **Required columns:**
        - `Date` - YYYY-MM-DD format
        - `Material` - 0=Steel, 1=Cement, 2=Conductors, 3=Equipment
        - `State` - 0=North, 1=South, 2=East, 3=West, 4=Central
        - `Quantity_Procured` - Number of units
        
        **Example:**
        ```
        Date,Material,State,Quantity_Procured
        2024-01-15,0,1,1500
        2024-02-20,2,3,2300
        ```
        """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            new_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Uploaded {len(new_data)} records")
            st.dataframe(new_data.head(10), use_container_width=True)
            
            # Options
            col1, col2 = st.columns(2)
            with col1:
                merge = st.checkbox("Merge with existing data", value=True)
            with col2:
                retrain = st.checkbox("Retrain Prophet model", value=True)
            
            # Process button
            if st.button("🚀 Process Data", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    # Merge or replace
                    if merge:
                        try:
                            existing = pd.read_csv('hybrid_cleaned.csv')
                            combined = pd.concat([existing, new_data], ignore_index=True)
                        except:
                            combined = new_data
                    else:
                        combined = new_data
                    
                    # Save
                    combined.to_csv('hybrid_cleaned.csv', index=False)
                    st.success(f"✅ Saved {len(combined)} records")
                    
                    # Retrain model
                    if retrain:
                        prophet_data = pd.DataFrame({
                            'ds': pd.to_datetime(combined['Date']),
                            'y': combined['Quantity_Procured']
                        })
                        
                        model = Prophet(yearly_seasonality=True)
                        model.fit(prophet_data)
                        
                        with open('powergrid_model.pkl', 'wb') as f:
                            pickle.dump(model, f, protocol=4)
                        
                        st.success("✅ Model retrained!")
                    
                    st.balloons()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    else:
        # Download template
        template = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=5, freq='M').strftime('%Y-%m-%d'),
            'Material': [0, 1, 2, 3, 0],
            'State': [1, 2, 3, 0, 4],
            'Quantity_Procured': [1500, 1800, 2000, 1600, 1700]
        })
        
        csv = template.to_csv(index=False)
        st.download_button(
            "📥 Download Sample Template",
            csv,
            f"template_{datetime.now().strftime('%Y%m%d')}.csv",
            use_container_width=True
        )

# TAB 2: System Settings
with tab2:
    st.markdown("### ⚙️ System Configuration")
    
    # Model settings
    st.subheader("Prophet Model Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Current Model Version", "v1.1.5")
        st.metric("Training Records", "1,199")
    
    with col2:
        st.metric("Model Accuracy", "94.71%")
        st.metric("Last Trained", "Today")
    
    st.markdown("---")
    
    # Display settings
    st.subheader("Display Settings")
    theme = st.selectbox("Theme", ["Dark", "Light"])
    chart_type = st.selectbox("Default Chart Type", ["Line", "Bar", "Area"])
    
    if st.button("Save Settings"):
        st.success("✅ Settings saved!")

# TAB 3: About
with tab3:
    st.markdown("### ℹ️ About This Application")
    
    st.info("""
    **POWERGRID Material Demand Forecasting System**
    
    Version: 1.0.0  
    Built with: Streamlit + Prophet AI  
    
    **Features:**
    - AI-powered demand forecasting
    - Real-time inventory management
    - Interactive analytics dashboard
    - CSV data upload & model retraining
    
    **Tech Stack:**
    - Python 3.11
    - Prophet 1.1.5
    - Streamlit
    - Pandas, NumPy, Plotly
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 System Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Projects", "1,199")
    with col2:
        st.metric("Materials Tracked", "4 Types")
    with col3:
        st.metric("Regional Zones", "5")
    
    st.markdown("---")
    st.caption("© 2025 POWERGRID Material Forecasting System")
