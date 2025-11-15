# app.py - POWERGRID Forecasting (with cleaning + upload + live metrics)

import streamlit as st
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

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
# Constants
# --------------------------------------------------------------------
MODEL_PATH = Path("powergrid_model.pkl")
HIST_CSV = Path("hybrid_cleaned.csv")          # default cleaned file (already used in training)

# --------------------------------------------------------------------
# Utility: load Prophet model
# --------------------------------------------------------------------
@st.cache_resource
def load_model(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        return e

def make_forecast(model, periods_months: int = 6):
    future = model.make_future_dataframe(periods=periods_months, freq='M')
    forecast = model.predict(future)
    return forecast

# --------------------------------------------------------------------
# Cleaning: from raw hybrid_powergrid_demand.csv → hybrid_cleaned-like
# --------------------------------------------------------------------
def clean_powergrid_csv(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstructs preprocessing similar to hybrid_cleaned.csv from hybrid_powergrid_demand.csv.
    Output columns:
    ['Date','State','ProjectType','TowerType','SubstationType','BudgetCr',
     'GSTRate','Material','SteelPriceIndex','CementPriceIndex','SupplierLeadTimeDays',
     'RegionCostFactor','QuantityUsed','QuantityProcured','Year','Month','CostPerUnitUsed']
    """

    df = df_raw.copy()

    # 1. Required columns
    required_cols = [
        'Date','State','ProjectType','TowerType','SubstationType',
        'BudgetCr','GSTRate','Material','SteelPriceIndex','CementPriceIndex',
        'SupplierLeadTimeDays','RegionCostFactor','QuantityUsed','QuantityProcured'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw CSV: {missing}")

    # 2. Basic cleaning: drop bad rows, parse types
    df = df.dropna(subset=['Date','QuantityProcured'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    # numeric conversions
    num_cols = [
        'BudgetCr','GSTRate','SteelPriceIndex','CementPriceIndex',
        'SupplierLeadTimeDays','RegionCostFactor','QuantityUsed','QuantityProcured'
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['QuantityUsed','QuantityProcured'])

    # 3. Label-encode categorical columns to integers (0..K-1)
    cat_cols = ['State','ProjectType','TowerType','SubstationType','Material']
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes

    # 4. Derive time features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # 5. CostPerUnitUsed (approximation consistent with your cleaned file)
    # Avoid division by zero
    df['CostPerUnitUsed'] = np.where(
        df['QuantityUsed'] != 0,
        (df['BudgetCr'] * df['RegionCostFactor']) / df['QuantityUsed'],
        0.0
    )

    # 6. Sort and select columns in the same order as hybrid_cleaned
    cols_out = [
        'Date','State','ProjectType','TowerType','SubstationType','BudgetCr',
        'GSTRate','Material','SteelPriceIndex','CementPriceIndex',
        'SupplierLeadTimeDays','RegionCostFactor','QuantityUsed',
        'QuantityProcured','Year','Month','CostPerUnitUsed'
    ]
    df = df[cols_out].sort_values('Date').reset_index(drop=True)

    return df

# --------------------------------------------------------------------
# Metrics computation
# --------------------------------------------------------------------
def compute_validation_metrics(model, hist_csv: Path, validation_months: int = 6, min_points: int = 6):
    """
    Compute MAPE and R^2 comparing model predictions to historical values.
    Predicts exactly on the historical dates.
    Returns: (mape, r2, percent_accuracy, msg)
    """
    if not hist_csv.exists():
        return None, None, None, f"Historical CSV not found: {hist_csv}"

    try:
        hist_df = pd.read_csv(hist_csv)
    except Exception as e:
        return None, None, None, f"Error reading CSV {hist_csv.name}: {e}"

    if 'Date' not in hist_df.columns or 'QuantityProcured' not in hist_df.columns:
        return None, None, None, "CSV missing required columns 'Date' and/or 'QuantityProcured'."

    hist_df = hist_df.dropna(subset=['Date','QuantityProcured']).copy()
    hist_df['ds'] = pd.to_datetime(hist_df['Date'], errors='coerce')
    hist_df['y'] = pd.to_numeric(hist_df['QuantityProcured'], errors='coerce')
    hist_df = hist_df.dropna(subset=['ds','y']).sort_values('ds').reset_index(drop=True)

    if hist_df.empty:
        return None, None, None, "No valid historical rows after cleaning."

    # last N points as validation window
    val_df = hist_df.tail(validation_months).copy()
    if len(val_df) < min_points:
        if len(hist_df) >= min_points:
            val_df = hist_df.tail(min_points).copy()
        else:
            return None, None, None, f"Not enough historical points for validation (found {len(hist_df)})."

    predict_dates = pd.DataFrame({'ds': val_df['ds'].values})

    try:
        preds = model.predict(predict_dates)
    except Exception as e:
        return None, None, None, f"Error running model.predict on historical dates: {e}"

    merged = val_df[['ds','y']].merge(preds[['ds','yhat']], on='ds', how='inner').sort_values('ds')
    if merged.empty:
        return None, None, None, "No overlapping predictions for validation dates (possible freq mismatch)."

    # MAPE excluding y == 0
    nonzero = merged[merged['y'] != 0]
    if len(nonzero) == 0:
        mape = None
    else:
        mape = (np.abs((nonzero['y'] - nonzero['yhat']) / nonzero['y'])).mean() * 100.0

    # R²
    try:
        r2 = float(r2_score(merged['y'], merged['yhat']))
    except Exception:
        y_true = merged['y'].values
        y_pred = merged['yhat'].values
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else None

    percent_accuracy = max(0.0, 100.0 - mape) if mape is not None else None

    return mape, r2, percent_accuracy, None

# --------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------
st.session_state.setdefault('forecast_df', None)

# --------------------------------------------------------------------
# Sidebar login
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
# Main content
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

    if model is None:
        st.warning(
            "No trained model found at `powergrid_model.pkl`.\n\n"
            "Run the training: `python train_prophet_model.py` (after installing prophet), "
            "or place `powergrid_model.pkl` in this folder."
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Model Accuracy", "N/A")
        with c2:
            st.metric("MAPE", "N/A")
        with c3:
            st.metric("R²", "N/A")
        with c4:
            st.metric("Materials Tracked", "4 Types", "")
    else:
        # -----------------------------
        # Data source selection + upload
        # -----------------------------
        st.subheader("📂 Data Source")

        st.markdown(
            "- By default, the app uses **`hybrid_cleaned.csv`** (same schema as used in training).\n"
            "- Optionally, upload a **raw POWERGRID CSV** (like `hybrid_powergrid_demand.csv`).\n"
            "  The app will clean it to the same structure and then compute validation metrics."
        )

        uploaded_file = st.file_uploader(
            "Upload new raw POWERGRID demand file (optional)",
            type=["csv"],
            key="raw_csv_uploader"
        )

        hist_path_for_metrics = HIST_CSV

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                cleaned_df = clean_powergrid_csv(raw_df)

                cleaned_path = Path("hybrid_cleaned_runtime.csv")
                cleaned_df.to_csv(cleaned_path, index=False)

                hist_path_for_metrics = cleaned_path

                st.success(f"✅ Uploaded and cleaned file. Using `{cleaned_path.name}` for validation.")
                st.write("Preview of cleaned data:")
                st.dataframe(cleaned_df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Error cleaning uploaded CSV: {e}")
                st.info("Falling back to default `hybrid_cleaned.csv` for metrics.")
        else:
            st.info("No file uploaded. Using default `hybrid_cleaned.csv` for validation.")

        st.markdown("---")

        # -----------------------------
        # Validation metrics
        # -----------------------------
        mape, r2, percent_accuracy, msg = compute_validation_metrics(
            model,
            hist_path_for_metrics,
            validation_months=6,
            min_points=6
        )

        if msg:
            st.warning(f"Validation info: {msg}")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "Model Accuracy",
                f"{percent_accuracy:.2f}%" if percent_accuracy is not None else "N/A",
                help="Derived as (100 - MAPE) when MAPE available."
            )
        with c2:
            st.metric(
                "MAPE",
                f"{mape:.2f}%" if mape is not None else "N/A",
                help="MAPE on last N historical points from the selected CSV."
            )
        with c3:
            st.metric(
                "R²",
                f"{r2:.3f}" if r2 is not None else "N/A",
                help="R² on last N historical points from the selected CSV."
            )
        with c4:
            st.metric("Materials Tracked", "4 Types", "")

        st.markdown("---")

        # -----------------------------
        # Forecast controls
        # -----------------------------
        cols = st.columns([1, 1, 1, 2])
        with cols[0]:
            periods = st.number_input("Forecast horizon (months)", min_value=1, max_value=36, value=6, step=1)
        with cols[1]:
            show_history = st.checkbox("Show historical series on chart", value=True)
        with cols[2]:
            if st.button("Generate Forecast"):
                try:
                    forecast = make_forecast(model, periods_months=int(periods))
                    st.session_state['forecast_df'] = forecast
                except Exception as e:
                    st.error(f"Error during forecasting: {e}")
                    st.session_state['forecast_df'] = None

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
