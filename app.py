# app.py - LINEAR WORKFLOW VERSION
import streamlit as st
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import json
from datetime import datetime

# --------------------------------------------------------------------
# Page config (MUST be first)
# --------------------------------------------------------------------
st.set_page_config(
    page_title="POWERGRID Material Forecasting System",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------
MODEL_PATH = Path("powergrid_model.pkl")
HIST_CSV = Path("hybrid_cleaned.csv")
METRICS_JSON = Path("model_metrics.json")

# --------------------------------------------------------------------
# Preprocessing Functions
# --------------------------------------------------------------------
def preprocess_raw_csv(df_raw):
    """Convert RAW → CLEANED format with logging"""
    log = []
    log.append("🔧 **Starting Data Preprocessing**\n")
    
    # Check if already cleaned
    if all(col in df_raw.columns for col in ['Date', 'State', 'Material', 'Quantity_Procured']):
        if df_raw['State'].dtype in [np.int64, np.int32] and df_raw['Material'].dtype in [np.int64, np.int32]:
            log.append("✅ Data already in cleaned format")
            return df_raw, log
    
    df = df_raw.copy()
    original_rows = len(df)
    log.append(f"📊 Original: {original_rows} rows\n")
    
    # Step 1: State mapping
    if 'State' in df.columns and df['State'].dtype == 'object':
        state_map = {
            'Assam': 0, 'Gujarat': 1, 'Maharashtra': 2,
            'Tamil Nadu': 3, 'Tamil Nad': 3,
            'Uttar Pradesh': 4, 'Uttar Prad': 4
        }
        df['State'] = df['State'].map(state_map)
        log.append("✅ **Step 1:** State names → codes (0-4)")
    
    # Step 2: Material mapping
    if 'Material' in df.columns and df['Material'].dtype == 'object':
        material_map = {'Cable': 0, 'Cement': 1, 'Insulator': 2, 'Steel': 3}
        df['Material'] = df['Material'].map(material_map)
        log.append("✅ **Step 2:** Material names → codes (0-3)")
    
    # Step 3: Date standardization
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    log.append(f"✅ **Step 3:** Dates standardized ({original_rows - len(df)} invalid removed)")
    
    # Step 4: Extract Year/Month
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    log.append("✅ **Step 4:** Year/Month extracted")
    
    # Step 5: Handle missing values
    df = df.dropna(subset=['Quantity_Procured'])
    if 'Budget_Cr' in df.columns:
        df['Budget_Cr'] = df['Budget_Cr'].fillna(df['Budget_Cr'].median())
    log.append("✅ **Step 5:** Missing values handled")
    
    # Step 6: Remove duplicates
    dupes = df.duplicated(subset=['Date', 'State', 'Material']).sum()
    df = df.drop_duplicates(subset=['Date', 'State', 'Material'], keep='first')
    log.append(f"✅ **Step 6:** Removed {dupes} duplicates")
    
    # Step 7: Outlier removal (IQR)
    Q1 = df['Quantity_Procured'].quantile(0.25)
    Q3 = df['Quantity_Procured'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df['Quantity_Procured'] < lower) | (df['Quantity_Procured'] > upper)).sum()
    df = df[(df['Quantity_Procured'] >= lower) & (df['Quantity_Procured'] <= upper)]
    log.append(f"✅ **Step 7:** Removed {outliers} outliers")
    
    # Step 8: Sort chronologically
    df = df.sort_values('Date').reset_index(drop=True)
    log.append("✅ **Step 8:** Sorted chronologically")
    
    # Step 9: Calculate cost per unit
    if 'Budget_Cr' in df.columns and 'Quantity_Used' in df.columns:
        df['CostPerUnitUsed'] = df['Budget_Cr'] / df['Quantity_Used'].replace(0, np.nan)
        df['CostPerUnitUsed'] = df['CostPerUnitUsed'].fillna(0)
        log.append("✅ **Step 9:** CostPerUnitUsed calculated")
    
    log.append(f"\n📊 **Final:** {len(df)} rows (removed {original_rows - len(df)})")
    log.append("✅ **Cleaning complete!**")
    
    return df, log


def train_prophet_model(csv_path):
    """Train Prophet on cleaned CSV"""
    try:
        from prophet import Prophet
        
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['Date', 'Quantity_Procured'])
        
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['Date']),
            'y': df['Quantity_Procured']
        })
        
        model = Prophet(
            yearly_seasonality=True,
            seasonality_mode='multiplicative'
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.fit(prophet_df)
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f, protocol=4)
        
        return model, None
    except Exception as e:
        return None, str(e)


def compute_metrics(model, csv_path, validation_months=6):
    """Compute MAPE and R² from last N months"""
    try:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['Date', 'Quantity_Procured'])
        df['ds'] = pd.to_datetime(df['Date'])
        df['y'] = pd.to_numeric(df['Quantity_Procured'], errors='coerce')
        df = df.dropna(subset=['y']).sort_values('ds').reset_index(drop=True)
        
        if len(df) < validation_months:
            return None, None, None
        
        val_df = df.tail(validation_months)
        predict_dates = pd.DataFrame({'ds': val_df['ds'].values})
        preds = model.predict(predict_dates)
        
        merged = val_df[['ds','y']].merge(preds[['ds','yhat']], on='ds', how='inner')
        
        # MAPE
        merged_nz = merged[merged['y'] != 0]
        mape = (np.abs((merged_nz['y'] - merged_nz['yhat']) / merged_nz['y'])).mean() * 100.0 if len(merged_nz) > 0 else None
        
        # R²
        r2 = float(r2_score(merged['y'], merged['yhat']))
        accuracy = max(0, 100 - mape) if mape else None
        
        return mape, r2, accuracy
    except:
        return None, None, None


# --------------------------------------------------------------------
# Session State
# --------------------------------------------------------------------
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
if 'name' not in st.session_state:
    st.session_state['name'] = None
if 'data_uploaded' not in st.session_state:
    st.session_state['data_uploaded'] = False
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False
if 'forecast_generated' not in st.session_state:
    st.session_state['forecast_generated'] = False
if 'forecast_df' not in st.session_state:
    st.session_state['forecast_df'] = None

# --------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------
def check_login(username, password):
    users = {'admin': 'admin123', 'manager': 'manager123'}
    return users.get(username) == password

with st.sidebar:
    st.markdown("### 🔌 POWERGRID Forecast")
    st.markdown("Ministry of Power")
    st.markdown("---")
    
    if st.session_state['authentication_status']:
        st.success(f"✅ Logged in as: **{st.session_state['name']}**")
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Workflow Progress")
        if st.session_state['data_uploaded']:
            st.success("✅ Data Uploaded & Cleaned")
        else:
            st.info("⏳ Upload Data")
        
        if st.session_state['model_trained']:
            st.success("✅ Model Trained")
        else:
            st.info("⏳ Train Model")
        
        if st.session_state['forecast_generated']:
            st.success("✅ Forecast Generated")
        else:
            st.info("⏳ Generate Forecast")
        
    else:
        st.markdown("### 🔐 Login")
        username_input = st.text_input("Username", key="username_input")
        password_input = st.text_input("Password", type="password", key="password_input")
        
        if st.button("Login", type="primary", use_container_width=True):
            if check_login(username_input, password_input):
                st.session_state['authentication_status'] = True
                st.session_state['name'] = username_input.capitalize()
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
        
        st.markdown("---")
        st.info("**Demo:**\n\n`admin` / `admin123`")

# --------------------------------------------------------------------
# Main Application - LINEAR WORKFLOW
# --------------------------------------------------------------------
if st.session_state['authentication_status']:
    st.title("🔌 POWERGRID Material Demand Forecasting System")
    st.markdown("### Supply Chain Intelligence Platform")
    st.markdown("---")
    
    # ========== STEP 1: UPLOAD DATA ==========
    st.markdown("## 📤 Step 1: Upload & Preprocess Data")
    
    uploaded_file = st.file_uploader(
        "Choose CSV file (raw or cleaned format)",
        type=['csv'],
        help="Upload hybrid_powergrid_demand.csv or hybrid_cleaned.csv"
    )
    
    if uploaded_file:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            st.success(f"✅ Uploaded: {uploaded_file.name} ({len(df_uploaded):,} rows)")
            
            with st.expander("📊 Data Preview (first 10 rows)"):
                st.dataframe(df_uploaded.head(10), use_container_width=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🔧 Preprocess & Train Model", type="primary", use_container_width=True):
                    with st.spinner("Processing..."):
                        # Preprocess
                        cleaned_df, log = preprocess_raw_csv(df_uploaded)
                        
                        st.markdown("### ✅ Preprocessing Complete")
                        for entry in log:
                            st.markdown(entry)
                        
                        # Show cleaned preview
                        with st.expander("📊 Cleaned Data Preview"):
                            st.dataframe(cleaned_df.head(10), use_container_width=True)
                        
                        # Save
                        cleaned_df.to_csv(HIST_CSV, index=False)
                        st.success(f"💾 Saved to: `{HIST_CSV}`")
                        
                        # Train model
                        st.info("🤖 Training Prophet model...")
                        model, error = train_prophet_model(HIST_CSV)
                        
                        if error:
                            st.error(f"❌ Training failed: {error}")
                        else:
                            st.success("✅ Model trained successfully!")
                            
                            # Update state
                            st.session_state['data_uploaded'] = True
                            st.session_state['model_trained'] = True
                            
                            st.balloons()
                            st.info("✅ Ready for forecasting! Scroll down to Step 2.")
            
            with col2:
                # Download template
                template = pd.DataFrame({
                    'Date': pd.date_range('2024-01-01', periods=5, freq='M').strftime('%Y-%m-%d'),
                    'State': ['Gujarat', 'Maharashtra', 'Assam', 'Tamil Nadu', 'Uttar Pradesh'],
                    'Material': ['Steel', 'Cement', 'Insulator', 'Cable', 'Steel'],
                    'Quantity_Procured': [1500, 1800, 2000, 1600, 1700]
                })
                csv = template.to_csv(index=False)
                st.download_button(
                    "📥 Download Template",
                    csv,
                    f"template_{datetime.now():%Y%m%d}.csv",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    else:
        st.info("👆 Upload a CSV file to begin")
    
    st.markdown("---")
    
    # ========== STEP 2: GENERATE FORECAST ==========
    st.markdown("## 🔮 Step 2: Generate Forecast")
    
    if not st.session_state['model_trained']:
        st.warning("⚠️ Please upload data and train model first (Step 1)")
    else:
        try:
            model = pickle.load(open(MODEL_PATH, 'rb'))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                periods = st.number_input("Forecast months", min_value=1, max_value=36, value=6)
            
            with col2:
                show_history = st.checkbox("Show historical data", value=True)
            
            with col3:
                if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
                    with st.spinner("Generating forecast..."):
                        try:
                            future = model.make_future_dataframe(periods=int(periods), freq='M')
                            forecast = model.predict(future)
                            st.session_state['forecast_df'] = forecast
                            st.session_state['forecast_generated'] = True
                            st.success("✅ Forecast generated!")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
            
            # Display forecast
            if st.session_state.get('forecast_df') is not None:
                forecast = st.session_state['forecast_df']
                
                st.markdown("### 📈 Forecast Visualization")
                plot_df = forecast.set_index('ds')[['yhat','yhat_lower','yhat_upper']]
                
                if show_history:
                    st.line_chart(plot_df[['yhat']])
                else:
                    st.line_chart(plot_df.tail(periods)[['yhat']])
                
                # Table
                st.markdown(f"### 📊 Next {periods} Months Predictions")
                future_only = forecast.tail(periods)[['ds','yhat','yhat_lower','yhat_upper']].copy()
                future_only['Date'] = pd.to_datetime(future_only['ds']).dt.strftime('%Y-%m-%d')
                future_only['Forecast'] = future_only['yhat'].round(0).astype(int)
                future_only['Lower_Bound'] = future_only['yhat_lower'].round(0).astype(int)
                future_only['Upper_Bound'] = future_only['yhat_upper'].round(0).astype(int)
                
                display = future_only[['Date','Forecast','Lower_Bound','Upper_Bound']]
                st.dataframe(display.reset_index(drop=True), use_container_width=True)
                
                # Download
                csv_export = display.to_csv(index=False)
                st.download_button(
                    "📥 Download Forecast CSV",
                    csv_export,
                    f"forecast_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Model not found: {e}")
    
    st.markdown("---")
    
    # ========== STEP 3: MODEL METRICS ==========
    st.markdown("## 📊 Step 3: Model Performance Metrics")
    
    if not st.session_state['model_trained']:
        st.warning("⚠️ Train model first to see metrics")
    else:
        try:
            model = pickle.load(open(MODEL_PATH, 'rb'))
            
            # Check for saved metrics
            if METRICS_JSON.exists():
                with open(METRICS_JSON, 'r') as f:
                    metrics = json.load(f)
                mape = metrics.get('mape')
                r2 = metrics.get('r2')
                accuracy = metrics.get('percent_accuracy')
            else:
                # Calculate metrics
                st.info("📊 Calculating metrics from last 6 months...")
                mape, r2, accuracy = compute_metrics(model, HIST_CSV)
                
                if mape is not None:
                    # Save metrics
                    metrics = {
                        'mape': mape,
                        'r2': r2,
                        'percent_accuracy': accuracy,
                        'calculated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    with open(METRICS_JSON, 'w') as f:
                        json.dump(metrics, f, indent=2)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Model Accuracy",
                    f"{accuracy:.2f}%" if accuracy else "N/A",
                    help="Derived as (100 - MAPE)"
                )
            
            with col2:
                st.metric(
                    "MAPE",
                    f"{mape:.2f}%" if mape else "N/A",
                    help="Mean Absolute Percentage Error"
                )
            
            with col3:
                st.metric(
                    "R² Score",
                    f"{r2:.4f}" if r2 else "N/A",
                    help="Coefficient of Determination"
                )
            
            with col4:
                st.metric(
                    "Materials Tracked",
                    "4 Types",
                    help="Steel, Cement, Conductors, Equipment"
                )
            
            # Dataset info
            if HIST_CSV.exists():
                df_info = pd.read_csv(HIST_CSV)
                st.markdown("### 📈 Dataset Information")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", f"{len(df_info):,}")
                with col2:
                    st.metric("Date Range", f"{df_info['Date'].min()} to {df_info['Date'].max()}")
                with col3:
                    st.metric("Avg Quantity", f"{df_info['Quantity_Procured'].mean():.0f} units")
        
        except Exception as e:
            st.error(f"❌ Error loading metrics: {e}")

else:
    st.title("🔌 POWERGRID Material Demand Forecasting")
    st.markdown("### Supply Chain Intelligence Platform")
    st.markdown("---")
    st.info("### 🔐 Authentication Required\n\nPlease login using the sidebar.")

st.markdown("---")
st.caption("© 2025 POWERGRID Material Forecasting System | Powered by Prophet AI")
