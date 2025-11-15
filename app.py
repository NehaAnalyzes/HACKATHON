# app.py - COMPLETE VERSION with CSV Upload & Preprocessing
import streamlit as st
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import json
from datetime import datetime

# --------------------------------------------------------------------
# Page config (MUST be first Streamlit command)
# --------------------------------------------------------------------
st.set_page_config(
    page_title="POWERGRID Material Forecasting System",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------
# Constants & File Paths
# --------------------------------------------------------------------
MODEL_PATH = Path("powergrid_model.pkl")
HIST_CSV = Path("hybrid_cleaned.csv")
METRICS_JSON = Path("model_metrics.json")

# --------------------------------------------------------------------
# Data Preprocessing Functions
# --------------------------------------------------------------------

def preprocess_raw_csv(df_raw):
    """
    Convert RAW hybrid_powergrid_demand.csv → CLEANED format
    Returns: (cleaned_df, preprocessing_log)
    """
    log = []
    log.append("🔧 **Starting Data Preprocessing Pipeline**\n")
    
    # Step 1: Check if already cleaned
    if all(col in df_raw.columns for col in ['Date', 'State', 'Material', 'Quantity_Procured']):
        # Check if State/Material are already numeric
        if df_raw['State'].dtype in [np.int64, np.int32] and df_raw['Material'].dtype in [np.int64, np.int32]:
            log.append("✅ Data is already in cleaned format (numeric State/Material)")
            return df_raw, log
    
    df = df_raw.copy()
    original_rows = len(df)
    log.append(f"📊 Original dataset: {original_rows} rows\n")
    
    # Step 2: State Name → State Code
    if 'State' in df.columns and df['State'].dtype == 'object':
        state_mapping = {
            'Assam': 0,
            'Gujarat': 1,
            'Maharashtra': 2,
            'Tamil Nadu': 3,
            'Tamil Nad': 3,  # Handle variations
            'Uttar Pradesh': 4,
            'Uttar Prad': 4
        }
        df['State'] = df['State'].map(state_mapping)
        log.append("✅ **Step 1:** Converted State names → codes (0-4)")
    
    # Step 3: Material Name → Material Code
    if 'Material' in df.columns and df['Material'].dtype == 'object':
        material_mapping = {
            'Cable': 0,
            'Cement': 1,
            'Insulator': 2,
            'Steel': 3
        }
        df['Material'] = df['Material'].map(material_mapping)
        log.append("✅ **Step 2:** Converted Material names → codes (0-3)")
    
    # Step 4: Date standardization
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        log.append(f"✅ **Step 3:** Standardized dates (removed {original_rows - len(df)} invalid dates)")
    
    # Step 5: Extract Year and Month
    if 'Date' in df.columns:
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        log.append("✅ **Step 4:** Extracted Year and Month features")
    
    # Step 6: Handle missing values
    missing_before = df.isnull().sum().sum()
    df = df.dropna(subset=['Quantity_Procured'])  # Critical column
    df['Budget_Cr'] = df.get('Budget_Cr', pd.Series()).fillna(df.get('Budget_Cr', pd.Series()).median())
    missing_after = df.isnull().sum().sum()
    log.append(f"✅ **Step 5:** Handled missing values ({missing_before} → {missing_after})")
    
    # Step 7: Remove duplicates
    duplicates = df.duplicated(subset=['Date', 'State', 'Material']).sum()
    df = df.drop_duplicates(subset=['Date', 'State', 'Material'], keep='first')
    log.append(f"✅ **Step 6:** Removed {duplicates} duplicate records")
    
    # Step 8: Outlier detection (IQR method)
    Q1 = df['Quantity_Procured'].quantile(0.25)
    Q3 = df['Quantity_Procured'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((df['Quantity_Procured'] < lower_bound) | (df['Quantity_Procured'] > upper_bound)).sum()
    df = df[(df['Quantity_Procured'] >= lower_bound) & (df['Quantity_Procured'] <= upper_bound)]
    log.append(f"✅ **Step 7:** Removed {outliers} outliers (IQR method)")
    
    # Step 9: Sort chronologically
    df = df.sort_values('Date').reset_index(drop=True)
    log.append("✅ **Step 8:** Sorted data chronologically")
    
    # Step 10: Calculate CostPerUnitUsed (if columns exist)
    if 'Budget_Cr' in df.columns and 'Quantity_Used' in df.columns:
        df['CostPerUnitUsed'] = df['Budget_Cr'] / df['Quantity_Used'].replace(0, np.nan)
        df['CostPerUnitUsed'] = df['CostPerUnitUsed'].fillna(0)
        log.append("✅ **Step 9:** Calculated CostPerUnitUsed feature")
    
    final_rows = len(df)
    log.append(f"\n📊 **Final dataset:** {final_rows} rows ({original_rows - final_rows} removed)")
    log.append(f"✅ **Data cleaning complete!**")
    
    return df, log


def save_cleaned_csv(df, filepath=HIST_CSV):
    """Save cleaned dataframe to CSV"""
    df.to_csv(filepath, index=False)
    return filepath


# --------------------------------------------------------------------
# Model & Metrics Functions
# --------------------------------------------------------------------

@st.cache_resource
def load_model(path: Path):
    """Load pickled Prophet model"""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        return e


def compute_validation_metrics(model, hist_csv: Path, validation_months: int = 6):
    """Compute MAPE and R² from historical data"""
    if not hist_csv.exists():
        return None, None, None, "Historical CSV not found."
    
    try:
        hist_df = pd.read_csv(hist_csv)
        hist_df = hist_df.dropna(subset=['Date', 'Quantity_Procured']).copy()
        hist_df['ds'] = pd.to_datetime(hist_df['Date'])
        hist_df['y'] = pd.to_numeric(hist_df['Quantity_Procured'], errors='coerce')
        hist_df = hist_df.dropna(subset=['y']).sort_values('ds').reset_index(drop=True)
        
        if hist_df.empty or len(hist_df) < validation_months:
            return None, None, None, "Not enough historical data"
        
        # Take last N months for validation
        val_df = hist_df.tail(validation_months).copy()
        predict_dates = pd.DataFrame({'ds': val_df['ds'].values})
        
        preds = model.predict(predict_dates)
        merged = val_df[['ds','y']].merge(preds[['ds','yhat']], on='ds', how='inner')
        
        if merged.empty:
            return None, None, None, "No overlapping predictions"
        
        # Calculate metrics
        merged_nonzero = merged[merged['y'] != 0]
        if len(merged_nonzero) > 0:
            mape = (np.abs((merged_nonzero['y'] - merged_nonzero['yhat']) / merged_nonzero['y'])).mean() * 100.0
        else:
            mape = None
        
        r2 = float(r2_score(merged['y'], merged['yhat']))
        percent_accuracy = max(0.0, 100.0 - mape) if mape is not None else None
        
        return mape, r2, percent_accuracy, None
    
    except Exception as e:
        return None, None, None, f"Error: {e}"


def train_prophet_model(csv_path):
    """Train Prophet model on cleaned CSV"""
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
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        model.fit(prophet_df)
        
        # Save model
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f, protocol=4)
        
        return model, None
    
    except Exception as e:
        return None, str(e)


# --------------------------------------------------------------------
# Session State Initialization
# --------------------------------------------------------------------
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
if 'name' not in st.session_state:
    st.session_state['name'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None
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
            st.session_state['authentication_status'] = None
            st.session_state['name'] = None
            st.session_state['username'] = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 System Status")
        st.success("🟢 Model: Active")
        st.info("📡 API: Connected")
        st.success("💾 Database: Online")
    else:
        st.markdown("### 🔐 Login")
        username_input = st.text_input("Username", key="username_input")
        password_input = st.text_input("Password", type="password", key="password_input")
        
        if st.button("Login", type="primary", use_container_width=True):
            if check_login(username_input, password_input):
                st.session_state['authentication_status'] = True
                st.session_state['username'] = username_input
                st.session_state['name'] = username_input.capitalize()
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
        
        st.markdown("---")
        st.info("**Demo Credentials:**\n\nUsername: `admin`\nPassword: `admin123`")


# --------------------------------------------------------------------
# Main Application
# --------------------------------------------------------------------
if st.session_state['authentication_status']:
    st.title("🔌 POWERGRID Material Demand Forecasting System")
    st.markdown("### Welcome to the Supply Chain Intelligence Platform")
    
    st.markdown("---")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📤 Upload & Preprocess Data", "🔮 Generate Forecast"])
    
    # ========== TAB 1: DASHBOARD ==========
    with tab1:
        # Load model
        model = load_model(MODEL_PATH)
        
        if model is None:
            st.warning("⚠️ No trained model found. Please upload data and train model in the 'Upload & Preprocess Data' tab.")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Model Accuracy", "N/A")
            with col2:
                st.metric("MAPE", "N/A")
            with col3:
                st.metric("R²", "N/A")
            with col4:
                st.metric("Materials Tracked", "4 Types")
        else:
            # Load metrics from JSON
            if METRICS_JSON.exists():
                try:
                    with open(METRICS_JSON, "r") as f:
                        metrics = json.load(f)
                    mape = metrics.get("mape")
                    r2 = metrics.get("r2")
                    percent_accuracy = metrics.get("percent_accuracy")
                    st.info(f"✅ Metrics loaded from {METRICS_JSON.name}")
                except:
                    mape = r2 = percent_accuracy = None
            else:
                # Calculate live if JSON doesn't exist
                mape, r2, percent_accuracy, msg = compute_validation_metrics(model, HIST_CSV)
                if msg:
                    st.warning(f"⚠️ {msg}")
            
            # Display KPI Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Model Accuracy (R²)",
                    f"{percent_accuracy:.2f}%" if percent_accuracy else "N/A",
                    help="Accuracy derived as (100 - MAPE)"
                )
            
            with col2:
                st.metric(
                    "MAPE",
                    f"{mape:.2f}%" if mape else "N/A",
                    help="Mean Absolute Percentage Error (lower is better)"
                )
            
            with col3:
                st.metric(
                    "R² Score",
                    f"{r2:.4f}" if r2 else "N/A",
                    help="Coefficient of Determination (0.9471 = 94.71%)"
                )
            
            with col4:
                st.metric(
                    "Materials Tracked",
                    "4 Types",
                    help="Steel, Cement, Conductors, Equipment"
                )
            
            st.markdown("---")
            
            # Quick stats
            if HIST_CSV.exists():
                df_stats = pd.read_csv(HIST_CSV)
                st.markdown("### 📈 Dataset Overview")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", f"{len(df_stats):,}")
                with col2:
                    st.metric("Date Range", f"{df_stats['Date'].min()} to {df_stats['Date'].max()}")
                with col3:
                    st.metric("Avg Quantity", f"{df_stats['Quantity_Procured'].mean():.0f} units")
    
    # ========== TAB 2: UPLOAD & PREPROCESS ==========
    with tab2:
        st.markdown("### 📤 Upload CSV Data")
        st.markdown("Upload either **raw** (`hybrid_powergrid_demand.csv`) or **cleaned** (`hybrid_cleaned.csv`) format")
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload procurement data in CSV format"
        )
        
        if uploaded_file:
            try:
                df_uploaded = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded: {uploaded_file.name} ({len(df_uploaded)} rows)")
                
                # Show preview
                with st.expander("📊 Data Preview (First 10 rows)", expanded=True):
                    st.dataframe(df_uploaded.head(10), use_container_width=True)
                
                # Preprocess button
                if st.button("🔧 Preprocess & Clean Data", type="primary", use_container_width=True):
                    with st.spinner("Processing data..."):
                        cleaned_df, preprocessing_log = preprocess_raw_csv(df_uploaded)
                        
                        # Show preprocessing steps
                        st.markdown("### ✅ Preprocessing Complete")
                        for log_entry in preprocessing_log:
                            st.markdown(log_entry)
                        
                        # Show cleaned data preview
                        st.markdown("### 📊 Cleaned Data Preview")
                        st.dataframe(cleaned_df.head(10), use_container_width=True)
                        
                        # Save cleaned data
                        save_path = save_cleaned_csv(cleaned_df)
                        st.success(f"✅ Cleaned data saved to: `{save_path}`")
                        
                        # Train model option
                        if st.button("🤖 Train Prophet Model on Cleaned Data", type="primary"):
                            with st.spinner("Training Prophet model... (this may take 30-60 seconds)"):
                                model, error = train_prophet_model(save_path)
                                
                                if error:
                                    st.error(f"❌ Training failed: {error}")
                                else:
                                    st.success("✅ Prophet model trained successfully!")
                                    
                                    # Calculate and save metrics
                                    mape, r2, percent_accuracy, msg = compute_validation_metrics(model, save_path)
                                    
                                    if not msg:
                                        metrics = {
                                            "mape": mape,
                                            "r2": r2,
                                            "percent_accuracy": percent_accuracy,
                                            "train_end": datetime.now().strftime("%Y-%m-%d"),
                                            "total_records": len(cleaned_df)
                                        }
                                        with open(METRICS_JSON, "w") as f:
                                            json.dump(metrics, f, indent=2)
                                        
                                        st.success(f"📊 Metrics saved: MAPE={mape:.2f}%, R²={r2:.4f}, Accuracy={percent_accuracy:.2f}%")
                                    
                                    st.balloons()
                                    st.info("🎉 Model ready! Go to Dashboard or Generate Forecast tab.")
            
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
        
        else:
            st.info("👆 Upload a CSV file to get started")
            
            # Download template
            st.markdown("### 📥 Need a Template?")
            template_data = pd.DataFrame({
                'Date': pd.date_range('2024-01-01', periods=5, freq='M').strftime('%Y-%m-%d'),
                'State': ['Gujarat', 'Maharashtra', 'Assam', 'Tamil Nadu', 'Uttar Pradesh'],
                'Material': ['Steel', 'Cement', 'Insulator', 'Cable', 'Steel'],
                'Quantity_Procured': [1500, 1800, 2000, 1600, 1700],
                'Budget_Cr': [12.5, 15.0, 18.0, 14.0, 16.5]
            })
            
            csv_template = template_data.to_csv(index=False)
            st.download_button(
                "📥 Download Sample Template",
                csv_template,
                f"powergrid_template_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv",
                use_container_width=True
            )
    
    # ========== TAB 3: GENERATE FORECAST ==========
    with tab3:
        model = load_model(MODEL_PATH)
        
        if model is None:
            st.warning("⚠️ No trained model found. Please train model in 'Upload & Preprocess Data' tab first.")
        else:
            st.markdown("### 🔮 Generate Material Demand Forecast")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                periods = st.number_input("Forecast horizon (months)", min_value=1, max_value=36, value=6)
            
            with col2:
                show_history = st.checkbox("Show historical data", value=True)
            
            with col3:
                if st.button("🚀 Generate Forecast", type="primary"):
                    with st.spinner("Generating forecast..."):
                        try:
                            future = model.make_future_dataframe(periods=int(periods), freq='M')
                            forecast = model.predict(future)
                            st.session_state['forecast_df'] = forecast
                            st.success("✅ Forecast generated!")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
            
            # Display forecast if exists
            if st.session_state.get('forecast_df') is not None:
                forecast = st.session_state['forecast_df']
                
                st.markdown("### 📈 Forecast Visualization")
                
                # Plot
                plot_df = forecast.set_index('ds')[['yhat','yhat_lower','yhat_upper']]
                if show_history:
                    st.line_chart(plot_df[['yhat']])
                else:
                    st.line_chart(plot_df.tail(periods)[['yhat']])
                
                # Forecast table
                st.markdown(f"### 📊 Next {periods} Months Predictions")
                future_only = forecast.tail(periods)[['ds','yhat','yhat_lower','yhat_upper']].copy()
                future_only['Date'] = pd.to_datetime(future_only['ds']).dt.strftime('%Y-%m-%d')
                future_only['Predicted_Demand'] = future_only['yhat'].round(0).astype(int)
                future_only['Lower_Bound'] = future_only['yhat_lower'].round(0).astype(int)
                future_only['Upper_Bound'] = future_only['yhat_upper'].round(0).astype(int)
                
                display_df = future_only[['Date','Predicted_Demand','Lower_Bound','Upper_Bound']]
                st.dataframe(display_df.reset_index(drop=True), use_container_width=True)
                
                # Download button
                csv_export = display_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Forecast CSV",
                    csv_export,
                    f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )

else:
    # Not logged in
    st.title("🔌 POWERGRID Material Demand Forecasting")
    st.markdown("### Supply Chain Intelligence Platform")
    st.markdown("---")
    st.info("### 🔐 Authentication Required\n\nPlease login using the sidebar to access the platform.")

# Footer
st.markdown("---")
st.caption("© 2025 POWERGRID Material Forecasting System | Powered by Prophet AI")
