# app.py - complete
import streamlit as st
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score  # pip install scikit-learn if not installed
import json

# --------------------------------------------------------------------
# IMPORTANT: set_page_config must be the first Streamlit command
# --------------------------------------------------------------------
st.set_page_config(
    page_title="POWERGRID Material Demand Forecasting System",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------
# Constants & helpers
# --------------------------------------------------------------------
MODEL_PATH = Path("powergrid_model.pkl")
HIST_CSV = Path("hybrid_cleaned.csv")

@st.cache_resource
def load_model(path: Path):
    """Load and return the pickled Prophet model (cached). Returns Exception on failure."""
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        return e

def make_forecast(model, periods_months: int = 6):
    """Return Prophet forecast DataFrame for periods_months ahead ('M' frequency expected)."""
    future = model.make_future_dataframe(periods=periods_months, freq='M')
    forecast = model.predict(future)
    return forecast

# add near the top of app.py imports

# replace previous compute_validation_metrics with this one
def compute_validation_metrics(model, hist_csv: Path, validation_months: int = 6, min_points: int = 6):
    """
    Compute MAPE and R^2 comparing model predictions to historical values.
    This version predicts exactly on the historical dates (avoids misalignment).
    Returns: (mape, r2, percent_accuracy, msg)
    """
    if not hist_csv.exists():
        return None, None, None, "Historical CSV not found."

    try:
        hist_df = pd.read_csv(hist_csv)
    except Exception as e:
        return None, None, None, f"Error reading CSV: {e}"

    # required columns check
    if 'Date' not in hist_df.columns or 'Quantity_Procured' not in hist_df.columns:
        return None, None, None, "CSV missing required columns 'Date' and/or 'Quantity_Procured'."

    # prepare historical data
    hist_df = hist_df.dropna(subset=['Date', 'Quantity_Procured']).copy()
    hist_df['ds'] = pd.to_datetime(hist_df['Date'])
    hist_df['y'] = pd.to_numeric(hist_df['Quantity_Procured'], errors='coerce')
    hist_df = hist_df.dropna(subset=['y']).sort_values('ds').reset_index(drop=True)

    if hist_df.empty:
        return None, None, None, "No valid historical rows after cleaning."

    # take the last validation_months rows (if you want calendar months, ensure data is monthly)
    # we use last N observed points as the validation window
    val_df = hist_df.tail(validation_months).copy()
    if len(val_df) < min_points:
        # if not enough points, use all available (but warn)
        if len(hist_df) >= min_points:
            val_df = hist_df.tail(min_points).copy()
        else:
            return None, None, None, f"Not enough historical points for validation (found {len(hist_df)})."

    # create a dataframe containing the exact ds dates we want predictions for
    predict_dates = pd.DataFrame({'ds': val_df['ds'].values})

    # predict at those dates
    try:
        preds = model.predict(predict_dates)
    except Exception as e:
        return None, None, None, f"Error running model.predict on historical dates: {e}"

    # preds contains yhat aligned to predict_dates; merge to val_df
    merged = val_df[['ds','y']].merge(preds[['ds','yhat']], on='ds', how='inner').sort_values('ds')

    if merged.empty:
        return None, None, None, "No overlapping predictions for validation dates (possible freq mismatch)."

    # compute MAPE excluding y==0
    merged_nonzero = merged[merged['y'] != 0]
    if len(merged_nonzero) == 0:
        mape = None
    else:
        mape = (np.abs((merged_nonzero['y'] - merged_nonzero['yhat']) / merged_nonzero['y'])).mean() * 100.0

    # compute R^2 using sklearn if available
    try:
        r2 = float(r2_score(merged['y'], merged['yhat']))
    except Exception:
        # fallback manual
        y_true = merged['y'].values
        y_pred = merged['yhat'].values
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None

    percent_accuracy = max(0.0, 100.0 - mape) if mape is not None else None

    return mape, r2, percent_accuracy, None

# --------------------------------------------------------------------
# Session state defaults (for forecast persistence)
# --------------------------------------------------------------------
st.session_state.setdefault('forecast_df', None)

# --------------------------------------------------------------------
# Sidebar: minimal login (same as before)
# --------------------------------------------------------------------
def check_login(username, password):
    users = {'admin': 'admin123', 'manager': 'manager123'}
    return users.get(username) == password

with st.sidebar:
    st.markdown("### 🔌 POWERGRID Forecast")
    st.markdown("Ministry of Power")
    st.markdown("---")
    if st.session_state.get('authentication_status'):
        st.success(f"✅ Logged in as: **{st.session_state.get('name')}**")
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
# Main app content
# --------------------------------------------------------------------
if st.session_state.get('authentication_status'):
    st.title("🔌 POWERGRID Material Demand Forecasting System")
    st.markdown("### Welcome to the Supply Chain Intelligence Platform")
    st.markdown("---")

    # load model
    model_or_error = load_model(MODEL_PATH)
    if isinstance(model_or_error, Exception):
        st.error(f"Error loading model: {model_or_error}")
        model = None
    else:
        model = model_or_error

    # if model missing, show instruction and placeholders
    if model is None:
        st.warning(
            "No trained model found at `powergrid_model.pkl`.\n\n"
            "Run the training: `python train_prophet_model.py` (after installing prophet), "
            "or place `powergrid_model.pkl` in this folder."
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Accuracy", "N/A")
        with col2:
            st.metric("MAPE", "N/A")
        with col3:
            st.metric("R²", "N/A")
        with col4:
            st.metric("Materials Tracked", "4 Types", "")
    else:
        # -----------------------------
        # Compute validation metrics IMMEDIATELY after loading the model
        # -----------------------------
        # Load precomputed metrics saved by the training script (model_metrics.json)
        METRICS_OUT = Path("model_metrics.json")

        if METRICS_OUT.exists():
            try:
                with open(METRICS_OUT, "r") as f:
                    metrics = json.load(f)
                mape = metrics.get("mape")
                r2 = metrics.get("r2")
                percent_accuracy = metrics.get("percent_accuracy")
                # optional info line:
                st.info(f"Loaded metrics from {METRICS_OUT.name} (trained up to {metrics.get('train_end')})")
            except Exception as e:
                mape = None; r2 = None; percent_accuracy = None
                st.warning(f"Failed to read {METRICS_OUT.name}: {e}")
        else:
            # fallback: JSON not found
            mape = None; r2 = None; percent_accuracy = None
            st.info("model_metrics.json not found — run train_prophet_model.py to generate it.")

        # Display metrics (from JSON or N/A)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Accuracy", f"{percent_accuracy:.2f}%" if percent_accuracy is not None else "N/A",
                    help="Derived as (100 - MAPE) when MAPE available")
        with col2:
            st.metric("MAPE", f"{mape:.2f}%" if mape is not None else "N/A",
                    help="MAPE from training holdout (model_metrics.json)")
        with col3:
            st.metric("R²", f"{r2:.3f}" if r2 is not None else "N/A",
                    help="R² from training holdout (model_metrics.json)")
        with col4:
            st.metric("Materials Tracked", "4 Types", "")

        st.markdown("---")

        # -----------------------------
        # Forecast controls (user can generate forecast)
        # -----------------------------
        cols = st.columns([1,1,1,2])
        with cols[0]:
            periods = st.number_input("Forecast horizon (months)", min_value=1, max_value=36, value=6, step=1)
        with cols[1]:
            show_history = st.checkbox("Show historical series on chart", value=True)
        with cols[2]:
            if st.button("Generate Forecast"):
                # generate forecast and persist to session_state so chart is visible after rerun
                try:
                    forecast = make_forecast(model, periods_months=int(periods))
                    st.session_state['forecast_df'] = forecast
                except Exception as e:
                    st.error(f"Error during forecasting: {e}")
                    st.session_state['forecast_df'] = None
                # optionally recompute metrics on demand (uncomment if needed)
                # mape, r2, percent_accuracy, msg = compute_validation_metrics(model, HIST_CSV, validation_months)
                # st.experimental_rerun()

        # If a forecast is already in session_state (generated earlier), display it; otherwise nothing
        if st.session_state.get('forecast_df') is not None:
            forecast = st.session_state['forecast_df']
            plot_df = forecast.set_index('ds')[['yhat','yhat_lower','yhat_upper']]
            st.subheader("Forecast Chart")
            if show_history:
                st.line_chart(plot_df[['yhat']])
            else:
                st.line_chart(plot_df.tail(periods)[['yhat']])

            future_only = forecast.tail(periods)[['ds','yhat','yhat_lower','yhat_upper']].copy()
            future_only = future_only.assign(
                Forecast=lambda d: d['yhat'].round(0),
                Lower=lambda d: d['yhat_lower'].round(0),
                Upper=lambda d: d['yhat_upper'].round(0)
            )[['ds','Forecast','Lower','Upper']].rename(columns={'ds':'Date'})
            future_only['Date'] = pd.to_datetime(future_only['Date']).dt.date
            st.subheader(f"Next {periods} months forecast")
            st.dataframe(future_only.reset_index(drop=True), use_container_width=True)

else:
    st.title("🔌 POWERGRID Material Demand Forecasting")
    st.markdown("### Supply Chain Intelligence Platform")
    st.markdown("---")
    st.info("### 🔐 Authentication Required\n\nPlease login using the sidebar to access the platform.")
