# app.py - ONLY model fixed, CSV must be uploaded

import streamlit as st
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# --------------------------------------------------------------------
# Page config
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

# --------------------------------------------------------------------
# Load Prophet model
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
# Cleaning for raw POWERGRID CSV (hybrid_powergrid_demand‑like)
# --------------------------------------------------------------------
def clean_powergrid_csv(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw POWERGRID CSV into hybrid_cleaned‑like structure.
    """

    df = df_raw.copy()

    required_cols = [
        'Date','State','ProjectType','TowerType','SubstationType',
        'BudgetCr','GSTRate','Material','SteelPriceIndex','CementPriceIndex',
        'SupplierLeadTimeDays','RegionCostFactor','QuantityUsed','QuantityProcured'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw CSV: {missing}")

    # dates and numeric types
    df = df.dropna(subset=['Date','QuantityProcured'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    num_cols = [
        'BudgetCr','GSTRate','SteelPriceIndex','CementPriceIndex',
        'SupplierLeadTimeDays','RegionCostFactor','QuantityUsed','QuantityProcured'
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['QuantityUsed','QuantityProcured'])

    # label‑encode categoricals
    cat_cols = ['State','ProjectType','TowerType','SubstationType','Material']
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes

    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    df['CostPerUnitUsed'] = np.where(
        df['QuantityUsed'] != 0,
        (df['BudgetCr'] * df['RegionCostFactor']) / df['QuantityUsed'],
        0.0
    )

    cols_out = [
        'Date','State','ProjectType','TowerType','SubstationType','BudgetCr',
        'GSTRate','Material','SteelPriceIndex','CementPriceIndex',
        'SupplierLeadTimeDays','RegionCostFactor','QuantityUsed',
        'QuantityProcured','Year','Month','CostPerUnitUsed'
    ]
    df = df[cols_out].sort_values('Date').reset_index(drop=True)
    return df

# --------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------
def compute_validation_metrics_from_df(model, hist_df: pd.DataFrame,
                                       validation_months: int = 6, min_points: int = 6):
    """
    Same as before but takes a DataFrame directly (already cleaned).
    Requires columns: Date, QuantityProcured
    """
    if 'Date' not in hist_df.columns or 'QuantityProcured' not in hist_df.columns:
        return None, None, None, "Uploaded CSV needs 'Date' and 'QuantityProcured' columns."

    df = hist_df.dropna(subset=['Date','QuantityProcured']).copy()
    df['ds'] = pd.to_datetime(df['Date'], errors='coerce')
    df['y'] = pd.to_numeric(df['QuantityProcured'], errors='coerce')
    df = df.dropna(subset=['ds','y']).sort_values('ds').reset_index(drop=True)

    if df.empty:
        return None, None, None, "No valid historical rows after cleaning."

    val_df = df.tail(validation_months).copy()
    if len(val_df) < min_points:
        if len(df) >= min_points:
            val_df = df.tail(min_points).copy()
        else:
            return None, None, None, f"Not enough historical points for validation (found {len(df)})."

    predict_dates = pd.DataFrame({'ds': val_df['ds'].values})

    try:
        preds = model.predict(predict_dates)
    except Exception as e:
        return None, None, None, f"Error running model.predict on historical dates: {e}"

    merged = val_df[['ds','y']].merge(preds[['ds','yhat']], on='ds', how='inner').sort_values('ds')
    if merged.empty:
        return None, None, None, "No overlapping predictions for validation dates (possible freq mismatch)."

    nonzero = merged[merged['y'] != 0]
    if len(nonzero) == 0:
        mape = None
    else:
        mape = (np.abs((nonzero['y'] - nonzero['yhat']) / nonzero['y'])).mean() * 100.0

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
# Simple login (same as before)
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

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
if st.session_state.get('authentication_status'):
    st.title("🔌 POWERGRID Material Demand Forecasting System")
    st.markdown("---")

    # Load model only
    model_or_error = load_model(MODEL_PATH)
    if isinstance(model_or_error, Exception):
        st.error(f"Error loading model: {model_or_error}")
        model = None
    else:
        model = model_or_error

    if model is None:
        st.warning(
            "No trained model found at `powergrid_model.pkl`.\n"
            "Train Prophet and save the model, then restart the app."
        )
    else:
        # ------------------ upload CSV (required) ------------------
        st.subheader("📂 Upload data (required)")
        st.markdown(
            "- You **must** upload a CSV on every run.\n"
            "- If it is **raw** like `hybrid_powergrid_demand.csv`, app will clean it.\n"
            "- If it is already **cleaned** like `hybrid_cleaned.csv`, app will use it directly."
        )

        uploaded_file = st.file_uploader(
            "Upload POWERGRID CSV",
            type=["csv"],
            key="any_csv_uploader"
        )

        if uploaded_file is None:
            st.info("Please upload a CSV to continue.")
        else:
            try:
                df_in = pd.read_csv(uploaded_file)

                # decide: raw vs already-cleaned
                raw_cols = {'State','ProjectType','TowerType','SubstationType','Material'}
                cleaned_cols = {'Year','Month','CostPerUnitUsed'}

                if raw_cols.issubset(df_in.columns) and not cleaned_cols.issubset(df_in.columns):
                    st.write("Detected raw schema → applying cleaning.")
                    hist_df = clean_powergrid_csv(df_in)
                else:
                    st.write("Detected cleaned / compatible schema → using as is.")
                    hist_df = df_in.copy()
                    hist_df['Date'] = pd.to_datetime(hist_df['Date'], errors='coerce')
                    hist_df = hist_df.dropna(subset=['Date'])

                st.write("Preview of data used for metrics:")
                st.dataframe(hist_df.head(), use_container_width=True)

                # ------------------ metrics ------------------
                st.markdown("---")
                st.subheader("📈 Validation metrics (last 6 points)")
                mape, r2, percent_accuracy, msg = compute_validation_metrics_from_df(
                    model, hist_df, validation_months=6, min_points=6
                )
                if msg:
                    st.warning(f"Validation info: {msg}")

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric(
                        "Accuracy",
                        f"{percent_accuracy:.2f}%" if percent_accuracy is not None else "N/A"
                    )
                with c2:
                    st.metric(
                        "MAPE",
                        f"{mape:.2f}%" if mape is not None else "N/A"
                    )
                with c3:
                    st.metric(
                        "R²",
                        f"{r2:.3f}" if r2 is not None else "N/A"
                    )

                # ------------------ forecast controls ------------------
                st.markdown("---")
                cols = st.columns([1,1,1,2])
                with cols[0]:
                    periods = st.number_input("Forecast horizon (months)", 1, 36, 6, 1)
                with cols[1]:
                    show_history = st.checkbox("Show full history on chart", True)
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

            except Exception as e:
                st.error(f"Error processing uploaded CSV: {e}")

else:
    st.title("🔌 POWERGRID Material Demand Forecasting")
    st.markdown("---")
    st.info("Please login from the sidebar to access the platform.")
