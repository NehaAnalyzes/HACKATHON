# app.py – Train-on-upload Prophet version
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import r2_score
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="POWERGRID Material Forecasting - Prophet AI",
    page_icon="🔌",
    layout="wide"
)

# ========== SESSION STATE ==========
if "authentication_status" not in st.session_state:
    st.session_state["authentication_status"] = None

def check_login(u, p):
    return {"admin": "admin123", "manager": "manager123"}.get(u) == p

# ========== HELPER FUNCTIONS ==========
def preprocess_csv(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean uploaded data, map categories and sort by Date."""
    df = df_raw.copy()

    if "State" in df.columns and df["State"].dtype == "object":
        state_map = {
            "Assam": 0,
            "Gujarat": 1,
            "Maharashtra": 2,
            "Tamil Nadu": 3,
            "Uttar Pradesh": 4,
        }
        df["State"] = df["State"].map(state_map)

    if "Material" in df.columns and df["Material"].dtype == "object":
        material_map = {"Cable": 0, "Cement": 1, "Insulator": 2, "Steel": 3}
        df["Material"] = df["Material"].map(material_map)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Quantity_Procured"] = pd.to_numeric(
        df["Quantity_Procured"], errors="coerce"
    )
    df = df.dropna(subset=["Date", "Quantity_Procured"])
    df = df.drop_duplicates(subset=["Date"], keep="first")
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def to_monthly_prophet_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert cleaned daily data to monthly aggregated Prophet frame."""
    df = df.copy()
    df["ds"] = pd.to_datetime(df["Date"], errors="coerce")
    df["y"] = pd.to_numeric(df["Quantity_Procured"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

    # monthly total quantity
    df_m = df.set_index("ds").resample("M")["y"].sum().reset_index()
    return df_m

def safe_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return None
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("### 🔌 POWERGRID AI")
    st.markdown("**Prophet Forecasting**")
    st.markdown("Ministry of Power")
    st.markdown("---")

    if st.session_state["authentication_status"]:
        st.success(f"✅ **{st.session_state.get('name')}**")
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        st.markdown("---")
        st.markdown("### 📊 System Status")
        st.success("🟢 Ready to train")
        st.info("📡 Upload → Train → Forecast")
    else:
        st.markdown("### 🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", type="primary", use_container_width=True):
            if check_login(username, password):
                st.session_state["authentication_status"] = True
                st.session_state["name"] = username.capitalize()
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
        st.markdown("---")
        st.info("**Demo:**  `admin` / `admin123`")

# ========== MAIN APP ==========
if st.session_state["authentication_status"]:
    st.title("🔌 POWERGRID Material Demand Forecasting System")
    st.markdown("### Prophet AI‑Powered Supply Chain Intelligence")
    st.markdown("---")

    # ===== 1. UPLOAD + TRAIN =====
    st.markdown("### 📤 Upload Dataset & Train Prophet")

    uploaded = st.file_uploader(
        "Upload CSV with `Date` and `Quantity_Procured` columns",
        type=["csv"],
        key="upload",
    )

    if uploaded:
        try:
            df_upload = pd.read_csv(uploaded)
            st.success(f"✅ **{uploaded.name}** uploaded successfully!")
            st.info(f"**Rows:** {len(df_upload):,} | **Columns:** {len(df_upload.columns)}")

            st.markdown("#### 📊 Data preview")
            st.dataframe(df_upload.head(15), use_container_width=True)

            # quick stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total rows", f"{len(df_upload):,}")
            col2.metric("Columns", len(df_upload.columns))
            if "Quantity_Procured" in df_upload.columns:
                col3.metric("Avg demand", f"{df_upload['Quantity_Procured'].mean():.0f}")
                col4.metric("Max demand", f"{df_upload['Quantity_Procured'].max():.0f}")

            if st.button(
                "🚀 Train Prophet on this dataset",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Cleaning data and training Prophet (monthly)..."):
                    cleaned = preprocess_csv(df_upload)
                    monthly = to_monthly_prophet_frame(cleaned)

                    if len(monthly) < 6:
                        st.error("❌ Not enough data after cleaning/aggregation.")
                    else:
                        # chronological split
                        TEST_MONTHS = 6
                        if len(monthly) > TEST_MONTHS + 3:
                            train_df = monthly.iloc[:-TEST_MONTHS].copy()
                            test_df = monthly.iloc[-TEST_MONTHS:].copy()
                        else:
                            train_df = monthly.copy()
                            test_df = pd.DataFrame(columns=monthly.columns)

                        # fit Prophet
                        model = Prophet(
                            yearly_seasonality=True,
                            weekly_seasonality=False,
                            daily_seasonality=False,
                            seasonality_mode="multiplicative",
                            changepoint_prior_scale=0.05,
                            interval_width=0.95,
                        )
                        model.add_seasonality(
                            name="monthly", period=30.5, fourier_order=5
                        )
                        model.fit(train_df[["ds", "y"]])

                        # evaluate
                        mape_val, r2_val = None, None
                        if len(test_df) > 0:
                            preds = model.predict(test_df[["ds"]])
                            y_true = test_df["y"].values
                            y_pred = preds["yhat"].values
                            mape_val = safe_mape(y_true, y_pred)
                            r2_val = r2_score(y_true, y_pred)

                        # save in session
                        st.session_state["model"] = model
                        st.session_state["monthly"] = monthly
                        st.session_state["mape"] = mape_val
                        st.session_state["r2"] = r2_val
                        st.success("✅ Training complete! Scroll down to generate forecast.")

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

    st.markdown("---")

    # ===== 2. FORECAST (GRAPH + TABLE) =====
    st.markdown("### 🔮 Forecast")

    if "model" not in st.session_state:
        st.info("Upload a CSV and train Prophet above to enable forecasting.")
    else:
        model = st.session_state["model"]
        monthly = st.session_state["monthly"]

        col_left, col_right = st.columns([1, 2])

        with col_left:
            horizon = st.number_input(
                "Forecast horizon (months)",
                min_value=1,
                max_value=36,
                value=6,
                help="Number of future months to forecast",
            )
            show_conf = st.checkbox(
                "Show 95% confidence interval",
                value=True,
            )

            if st.button(
                "📈 Generate forecast",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Generating forecast..."):
                    future = model.make_future_dataframe(
                        periods=int(horizon), freq="M"
                    )
                    forecast = model.predict(future)
                    st.session_state["forecast_df"] = forecast.tail(int(horizon))
                    st.success("✅ Forecast generated below.")

        with col_right:
            if "forecast_df" in st.session_state:
                future_only = st.session_state["forecast_df"]

                # ----- chart -----
                fig = go.Figure()

                # historical monthly demand
                fig.add_trace(
                    go.Scatter(
                        x=monthly["ds"],
                        y=monthly["y"],
                        mode="lines",
                        name="History",
                        line=dict(color="gray"),
                    )
                )

                # forecast central line
                fig.add_trace(
                    go.Scatter(
                        x=future_only["ds"],
                        y=future_only["yhat"],
                        mode="lines+markers",
                        name="Forecast",
                        line=dict(color="#FF4B4B", width=3),
                        marker=dict(size=8),
                    )
                )

                if show_conf:
                    fig.add_trace(
                        go.Scatter(
                            x=future_only["ds"],
                            y=future_only["yhat_upper"],
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=future_only["ds"],
                            y=future_only["yhat_lower"],
                            fill="tonexty",
                            mode="lines",
                            name="95% interval",
                            line=dict(width=0),
                            fillcolor="rgba(255, 75, 75, 0.2)",
                        )
                    )

                fig.update_layout(
                    title=f"Prophet monthly demand forecast – next {int(horizon)} months",
                    xaxis_title="Date",
                    yaxis_title="Quantity (units)",
                    hovermode="x unified",
                    template="plotly_white",
                    height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

    # ===== 3. FORECAST TABLE WITH LOW / MEDIUM / HIGH =====
    if "forecast_df" in st.session_state:
        f = st.session_state["forecast_df"].copy()
        f["Date"] = pd.to_datetime(f["ds"]).dt.strftime("%Y-%m-%d")
        f["Forecast"] = f["yhat"].round(0).astype(int)
        f["Lower_95"] = f["yhat_lower"].round(0).astype(int)
        f["Upper_95"] = f["yhat_upper"].round(0).astype(int)

        # derive Low / Medium / High numeric bands inside the 95% interval
        width = (f["Upper_95"] - f["Lower_95"]).replace(0, np.nan)
        f["Low"] = f["Lower_95"].astype(int)
        f["Medium"] = (f["Lower_95"] + width * (2 / 3.0)).round(0).astype("Int64")
        f["High"] = f["Upper_95"].astype(int)

        st.markdown("#### 📊 Forecast table (with confidence bands)")
        display_cols = ["Date", "Forecast", "Low", "Medium", "High"]
        st.dataframe(f[display_cols].reset_index(drop=True), use_container_width=True)

        csv_export = f[display_cols].to_csv(index=False)
        st.download_button(
            "📥 Download forecast as CSV",
            csv_export,
            f"prophet_forecast_{datetime.now():%Y%m%d_%H%M%S}.csv",
            "text/csv",
            use_container_width=True,
        )

    st.markdown("---")

    # ===== 4. EVALUATION METRICS =====
    st.markdown("### 📏 Model evaluation")

    mape_val = st.session_state.get("mape")
    r2_val = st.session_state.get("r2")

    col1, col2 = st.columns(2)
    if mape_val is not None:
        col1.metric("MAPE", f"{mape_val:.2f}%")
    else:
        col1.metric("MAPE", "N/A")

    if r2_val is not None:
        col2.metric("R² score", f"{r2_val:.4f}")
    else:
        col2.metric("R² score", "N/A")

else:
    st.title("🔌 POWERGRID Material Demand Forecasting")
    st.markdown("### Prophet AI‑Powered Supply Chain Intelligence")
    st.markdown("---")
    st.info("### 🔐 Authentication required\n\nPlease login using the sidebar to access the forecasting system.")

st.markdown("---")
st.caption(
    "© 2025 POWERGRID Corporation of India | Powered by Prophet (Meta) | Ministry of Power"
)
