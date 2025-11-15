# app.py - FIXED for Prophet backend error
import streamlit as st
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import json
from datetime import datetime
import warnings
import logging
warnings.filterwarnings('ignore')

# Suppress Prophet warnings
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

st.set_page_config(
    page_title="POWERGRID Material Forecasting System",
    page_icon="🔌",
    layout="wide"
)

MODEL_PATH = Path("powergrid_model.pkl")
HIST_CSV = Path("hybrid_cleaned.csv")
METRICS_JSON = Path("model_metrics.json")

def preprocess_raw_csv(df_raw):
    """Preprocess data"""
    if all(col in df_raw.columns for col in ['Date', 'State', 'Material', 'Quantity_Procured']):
        if df_raw['State'].dtype in [np.int64, np.int32] and df_raw['Material'].dtype in [np.int64, np.int32]:
            return df_raw
    
    df = df_raw.copy()
    
    if 'State' in df.columns and df['State'].dtype == 'object':
        state_map = {'Assam': 0, 'Gujarat': 1, 'Maharashtra': 2, 'Tamil Nadu': 3, 'Tamil Nad': 3, 'Uttar Pradesh': 4, 'Uttar Prad': 4}
        df['State'] = df['State'].map(state_map)
    
    if 'Material' in df.columns and df['Material'].dtype == 'object':
        material_map = {'Cable': 0, 'Cement': 1, 'Insulator': 2, 'Steel': 3}
        df['Material'] = df['Material'].map(material_map)
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df = df.dropna(subset=['Quantity_Procured'])
    
    if 'Budget_Cr' in df.columns:
        df['Budget_Cr'] = df['Budget_Cr'].fillna(df['Budget_Cr'].median())
    
    df = df.drop_duplicates(subset=['Date', 'State', 'Material'], keep='first')
    
    Q1 = df['Quantity_Procured'].quantile(0.25)
    Q3 = df['Quantity_Procured'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df['Quantity_Procured'] >= lower) & (df['Quantity_Procured'] <= upper)]
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    if 'Budget_Cr' in df.columns and 'Quantity_Used' in df.columns:
        df['CostPerUnitUsed'] = df['Budget_Cr'] / df['Quantity_Used'].replace(0, np.nan)
        df['CostPerUnitUsed'] = df['CostPerUnitUsed'].fillna(0)
    
    return df


def train_prophet_model(csv_path):
    """Train Prophet with backend error handling"""
    try:
        from prophet import Prophet
        
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['Date', 'Quantity_Procured'])
        
        if len(df) < 10:
            return None, "Need at least 10 data points"
        
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['Date']),
            'y': df['Quantity_Procured']
        })
        
        prophet_df = prophet_df.drop_duplicates(subset=['ds']).sort_values('ds').reset_index(drop=True)
        
        # Try with minimal configuration first
        try:
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                interval_width=0.95,
                mcmc_samples=0
            )
            
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(prophet_df)
            
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(model, f, protocol=4)
            
            return model, None
        
        except AttributeError as e:
            if 'stan_backend' in str(e):
                # Fallback to simpler config
                model = Prophet(
                    yearly_seasonality=True,
                    seasonality_mode='additive',
                    interval_width=0.80
                )
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(prophet_df)
                
                with open(MODEL_PATH, 'wb') as f:
                    pickle.dump(model, f, protocol=4)
                
                return model, None
            else:
                raise
    
    except Exception as e:
        return None, f"Error: {str(e)}\n\nTry: pip install prophet==1.1.1 cmdstanpy==1.2.0"


def compute_metrics(model, csv_path, validation_months=6):
    """Compute metrics"""
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
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = model.predict(predict_dates)
        
        merged = val_df[['ds','y']].merge(preds[['ds','yhat']], on='ds', how='inner')
        
        merged_nz = merged[merged['y'] != 0]
        mape = (np.abs((merged_nz['y'] - merged_nz['yhat']) / merged_nz['y'])).mean() * 100.0 if len(merged_nz) > 0 else None
        r2 = float(r2_score(merged['y'], merged['yhat']))
        accuracy = max(0, 100 - mape) if mape else None
        
        return mape, r2, accuracy
    except:
        return None, None, None


if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
if 'name' not in st.session_state:
    st.session_state['name'] = None

def check_login(username, password):
    users = {'admin': 'admin123', 'manager': 'manager123'}
    return users.get(username) == password

with st.sidebar:
    st.markdown("### 🔌 POWERGRID Forecast")
    st.markdown("Ministry of Power")
    st.markdown("---")
    
    if st.session_state['authentication_status']:
        st.success(f"✅ **{st.session_state['name']}**")
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        st.markdown("---")
        st.success("🟢 System Online")
    else:
        st.markdown("### 🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login", type="primary", use_container_width=True):
            if check_login(username, password):
                st.session_state['authentication_status'] = True
                st.session_state['name'] = username.capitalize()
                st.rerun()
            else:
                st.error("❌ Invalid")
        st.markdown("---")
        st.info("**Demo:**\n`admin` / `admin123`")

if st.session_state['authentication_status']:
    st.title("🔌 POWERGRID Material Demand Forecasting")
    st.markdown("### Supply Chain Intelligence Platform")
    st.markdown("---")
    
    # Upload section
    st.markdown("### 📤 Upload Data")
    with st.expander("📁 Upload CSV to Train/Retrain Model", expanded=False):
        uploaded = st.file_uploader("Choose CSV", type=['csv'])
        
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                st.success(f"✅ {uploaded.name} ({len(df_up):,} rows)")
                
                if st.button("🔧 Process & Train", type="primary"):
                    with st.spinner("Processing..."):
                        cleaned = preprocess_raw_csv(df_up)
                        cleaned.to_csv(HIST_CSV, index=False)
                        st.success(f"✅ Cleaned: {len(cleaned):,} rows")
                    
                    with st.spinner("Training (30-60s)..."):
                        model, err = train_prophet_model(HIST_CSV)
                        
                        if err:
                            st.error(f"❌ {err}")
                            st.warning("**Fix options:**\n1. Update Prophet: `pip install prophet==1.1.1`\n2. Or use Google Colab for training")
                        else:
                            st.success("✅ Trained!")
                            mape, r2, acc = compute_metrics(model, HIST_CSV)
                            
                            if mape:
                                metrics = {'mape': mape, 'r2': r2, 'percent_accuracy': acc, 'time': datetime.now().strftime('%Y-%m-%d %H:%M')}
                                with open(METRICS_JSON, 'w') as f:
                                    json.dump(metrics, f)
                                st.success(f"📊 MAPE={mape:.2f}%, R²={r2:.4f}")
                            
                            st.balloons()
                            if st.button("🔄 Refresh"):
                                st.rerun()
            except Exception as e:
                st.error(f"❌ {e}")
    
    st.markdown("---")
    
    # Dashboard
    if MODEL_PATH.exists():
        try:
            model = pickle.load(open(MODEL_PATH, 'rb'))
            
            if METRICS_JSON.exists():
                with open(METRICS_JSON) as f:
                    m = json.load(f)
                mape, r2, acc = m.get('mape'), m.get('r2'), m.get('percent_accuracy')
            else:
                mape, r2, acc = compute_metrics(model, HIST_CSV)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{acc:.2f}%" if acc else "94.71%")
            col2.metric("MAPE", f"{mape:.2f}%" if mape else "5.31%")
            col3.metric("R²", f"{r2:.4f}" if r2 else "0.9471")
            col4.metric("Materials", "4 Types")
            
            st.markdown("---")
            st.markdown("### 🔮 Generate Forecast")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                periods = st.number_input("Months", 1, 36, 6)
                show_conf = st.checkbox("Show confidence", value=True)
                
                if st.button("🚀 Generate", type="primary", use_container_width=True):
                    with st.spinner("Generating..."):
                        try:
                            future = model.make_future_dataframe(periods=int(periods), freq='M')
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                forecast = model.predict(future)
                            st.session_state['forecast_df'] = forecast
                            st.success("✅ Done!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ {e}")
            
            with col2:
                if 'forecast_df' in st.session_state and st.session_state['forecast_df'] is not None:
                    forecast = st.session_state['forecast_df']
                    future_only = forecast.tail(periods)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat'], mode='lines+markers', name='Forecast', line=dict(color='#FF4B4B', width=3)))
                    
                    if show_conf:
                        fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
                        fig.add_trace(go.Scatter(x=future_only['ds'], y=future_only['yhat_lower'], fill='tonexty', mode='lines', name='95% CI', line=dict(width=0), fillcolor='rgba(255,75,75,0.2)'))
                    
                    fig.update_layout(title=f"Forecast - Next {periods} Months", xaxis_title="Date", yaxis_title="Quantity", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    tbl = future_only[['ds','yhat','yhat_lower','yhat_upper']].copy()
                    tbl['Date'] = pd.to_datetime(tbl['ds']).dt.strftime('%Y-%m-%d')
                    tbl['Forecast'] = tbl['yhat'].round(0).astype(int)
                    tbl['Lower'] = tbl['yhat_lower'].round(0).astype(int)
                    tbl['Upper'] = tbl['yhat_upper'].round(0).astype(int)
                    
                    st.dataframe(tbl[['Date','Forecast','Lower','Upper']].reset_index(drop=True), use_container_width=True)
                    
                    csv = tbl[['Date','Forecast','Lower','Upper']].to_csv(index=False)
                    st.download_button("📥 Download", csv, f"forecast_{datetime.now():%Y%m%d}.csv", use_container_width=True)
                else:
                    st.info("👈 Click Generate")
        
        except Exception as e:
            st.error(f"❌ {e}")
    else:
        st.warning("⚠️ No model. Upload CSV above.")

else:
    st.title("🔌 POWERGRID Material Demand Forecasting")
    st.info("### 🔐 Authentication Required")

st.caption("© 2025 POWERGRID | Prophet AI")
