# train_prophet_model.py (fixed: use resample to create ds column)
import pandas as pd
import numpy as np
from prophet import Prophet
import pickle
from pathlib import Path
from datetime import datetime
import json
from sklearn.metrics import r2_score
import os
import sys

# -------------- CONFIG --------------
CSV_PATH = Path("hybrid_cleaned.csv")
MODEL_OUT = Path("powergrid_model.pkl")
METRICS_OUT = Path("model_metrics.json")

# How many last months to reserve for test/validation
TEST_MONTHS = 6

# Use log(1 + y) transform? Try True if series skewed
USE_LOG_TRANSFORM = False

PROPHEt_PARAMS = {
    "yearly_seasonality": True,
    "weekly_seasonality": False,
    "daily_seasonality": False,
    "seasonality_mode": "multiplicative",
    "changepoint_prior_scale": 0.05,
    "interval_width": 0.95
}
MONTHLY_SEASONALITY = {"name": "monthly", "period": 30.5, "fourier_order": 5}
# ------------------------------------

def safe_mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return None
    return (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100.0

def main():
    print("="*60)
    print("POWERGRID Prophet Model Training (fixed: monthly aggregation via resample)")
    print("="*60)

    if not CSV_PATH.exists():
        print(f"❌ Error: {CSV_PATH} not found.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    if 'Date' not in df.columns or 'Quantity_Procured' not in df.columns:
        print("❌ CSV must contain 'Date' and 'Quantity_Procured' columns.")
        print(f"Found columns: {df.columns.tolist()}")
        sys.exit(1)

    # --- Prepare and aggregate to monthly totals ---
    df = df.dropna(subset=['Date', 'Quantity_Procured']).copy()
    # parse dates
    df['ds'] = pd.to_datetime(df['Date'], errors='coerce')
    df['y'] = pd.to_numeric(df['Quantity_Procured'], errors='coerce')
    df = df.dropna(subset=['ds','y']).sort_values('ds').reset_index(drop=True)

    # Ensure ds is timezone-naive datetime (Prophet likes that)
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

    # Aggregate by month (sum of Quantity_Procured per month) using resample
    df_monthly = df.set_index('ds').resample('M')['y'].sum().reset_index()
    if df_monthly.empty:
        print("❌ No monthly aggregated data after processing.")
        sys.exit(1)

    print(f"✅ Aggregated to {len(df_monthly)} monthly rows, range {df_monthly['ds'].min().date()} -> {df_monthly['ds'].max().date()}")

    # Train-test split (chronological)
    if len(df_monthly) <= TEST_MONTHS + 3:
        print("⚠️ Not enough monthly rows for test split — using all for training.")
        train_df = df_monthly.copy()
        test_df = pd.DataFrame(columns=df_monthly.columns)
    else:
        train_df = df_monthly.iloc[:-TEST_MONTHS].copy()
        test_df = df_monthly.iloc[-TEST_MONTHS:].copy()

    print(f"Training months: {len(train_df)}, Test months: {len(test_df)}")

    # Optional log transform
    used_transform = False
    if USE_LOG_TRANSFORM:
        used_transform = True
        train_df['y'] = np.log1p(train_df['y'])

    # Fit Prophet
    model = Prophet(**PROPHEt_PARAMS)
    model.add_seasonality(**MONTHLY_SEASONALITY)

    print("\n[Training] Fitting Prophet model...")
    t0 = datetime.now()
    model.fit(train_df[['ds','y']])
    t_elapsed = (datetime.now() - t0).total_seconds()
    print(f"✅ Model trained in {t_elapsed:.2f}s")

    # Save model
    try:
        with open(MODEL_OUT, "wb") as f:
            pickle.dump(model, f, protocol=4)
        print(f"✅ Saved model to {MODEL_OUT}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")

    # Evaluate on test set if any
    mape_val = None
    r2_val = None
    percent_accuracy = None
    test_msg = None

    if len(test_df) > 0:
        predict_dates = pd.DataFrame({'ds': test_df['ds'].values})
        print("\n[Evaluation] Predicting on test months...")
        try:
            preds = model.predict(predict_dates[['ds']])
            y_true = test_df['y'].values
            y_pred = preds['yhat'].values
            if used_transform:
                y_pred = np.expm1(y_pred)

            mape_val = safe_mape(y_true, y_pred)
            try:
                r2_val = float(r2_score(y_true, y_pred))
            except Exception:
                ss_res = np.sum((y_true - y_pred)**2)
                ss_tot = np.sum((y_true - np.mean(y_true))**2)
                r2_val = 1.0 - (ss_res/ss_tot) if ss_tot>0 else None

            if mape_val is not None:
                percent_accuracy = max(0.0, 100.0 - mape_val)

            print(f"Test MAPE: {mape_val:.2f}%") if mape_val is not None else print("MAPE: N/A")
            print(f"Test R²: {r2_val:.4f}") if r2_val is not None else print("R²: N/A")

        except Exception as e:
            test_msg = f"Error predicting on test months: {e}"
            print("❌", test_msg)
    else:
        print("⚠️ No test months available. Skipping evaluation.")

    # Save metrics JSON
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "train_end": train_df['ds'].max().isoformat() if len(train_df)>0 else None,
        "test_start": test_df['ds'].min().isoformat() if len(test_df)>0 else None,
        "test_end": test_df['ds'].max().isoformat() if len(test_df)>0 else None,
        "num_train": int(len(train_df)),
        "num_test": int(len(test_df)),
        "mape": float(mape_val) if mape_val is not None else None,
        "r2": float(r2_val) if r2_val is not None else None,
        "percent_accuracy": float(percent_accuracy) if percent_accuracy is not None else None,
        "used_transform_log1p": bool(used_transform),
        "prophet_params": PROPHEt_PARAMS,
        "monthly_seasonality": MONTHLY_SEASONALITY,
        "note": test_msg
    }
    try:
        with open(METRICS_OUT, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Saved metrics to {METRICS_OUT}")
    except Exception as e:
        print(f"❌ Error saving metrics: {e}")

    # Optional display of predictions vs actuals
    if len(test_df) > 0:
        display_df = predict_dates.copy()
        display_df['pred'] = preds['yhat'].values
        if used_transform:
            display_df['pred'] = np.expm1(display_df['pred'])
        display_df['actual'] = test_df['y'].values
        print("\nTest preds (date, actual, pred):")
        print(display_df.to_string(index=False))

    print("\nDone.")

if __name__ == "__main__":
    main()
