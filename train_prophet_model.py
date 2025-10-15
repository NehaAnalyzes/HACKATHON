"""
POWERGRID Prophet Model Training Script
Run this in VS Code with Python 3.11 to create powergrid_model.pkl
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import pickle
from datetime import datetime

print("=" * 60)
print("POWERGRID Prophet Model Training")
print("=" * 60)

# Step 1: Load Historical Data
print("\n[1/4] Loading historical data...")
try:
    df = pd.read_csv('hybrid_cleaned.csv')
    print(f"✅ Loaded {len(df)} records from hybrid_cleaned.csv")
except FileNotFoundError:
    print("❌ Error: hybrid_cleaned.csv not found!")
    print("Make sure the file is in the same directory as this script.")
    exit()

# Step 2: Prepare Data for Prophet
print("\n[2/4] Preparing data for Prophet...")

# Prophet requires columns named 'ds' (date) and 'y' (target value)
prophet_data = pd.DataFrame({
    'ds': pd.to_datetime(df['Date']),
    'y': df['Quantity_Procured']
})

# Remove any missing values
prophet_data = prophet_data.dropna()

print(f"✅ Prepared {len(prophet_data)} data points")
print(f"   Date range: {prophet_data['ds'].min()} to {prophet_data['ds'].max()}")
print(f"   Avg demand: {prophet_data['y'].mean():.2f} units")

# Step 3: Train Prophet Model
print("\n[3/4] Training Prophet model...")
print("   This may take 30-60 seconds...")

# Initialize Prophet with optimized settings
model = Prophet(
    yearly_seasonality=True,      # Capture yearly patterns
    weekly_seasonality=False,      # Not relevant for monthly data
    daily_seasonality=False,       # Not relevant for monthly data
    seasonality_mode='multiplicative',  # Better for percentage changes
    changepoint_prior_scale=0.05,  # Moderate flexibility for trend changes
    interval_width=0.95            # 95% confidence intervals
)

# Add monthly seasonality (important for material demand)
model.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=5
)

# Fit the model
start_time = datetime.now()
model.fit(prophet_data)
training_time = (datetime.now() - start_time).total_seconds()

print(f"✅ Model trained successfully in {training_time:.2f} seconds!")

# Step 4: Save Model
print("\n[4/4] Saving model...")

try:
    # Save with protocol 4 for Python 3.11 compatibility
    with open('powergrid_model.pkl', 'wb') as f:
        pickle.dump(model, f, protocol=4)
    
    print("✅ Model saved as 'powergrid_model.pkl'")
    
    # Get file size
    import os
    file_size = os.path.getsize('powergrid_model.pkl') / (1024 * 1024)  # Convert to MB
    print(f"   File size: {file_size:.2f} MB")
    
except Exception as e:
    print(f"❌ Error saving model: {str(e)}")
    exit()

# Step 5: Quick Validation Test
print("\n[5/5] Validating model...")

# Make a test prediction
future = model.make_future_dataframe(periods=6, freq='M')
forecast = model.predict(future)

# Get last 6 predictions
test_forecast = forecast.tail(6)

print("✅ Model validation successful!")
print("\nTest forecast (next 6 months):")
print("-" * 60)
for idx, row in test_forecast.iterrows():
    print(f"   {row['ds'].strftime('%Y-%m-%d')}: {row['yhat']:.0f} units "
          f"({row['yhat_lower']:.0f} - {row['yhat_upper']:.0f})")

print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\nNext steps:")
print("1. Upload 'powergrid_model.pkl' to your GitHub repository")
print("2. Make sure 'runtime.txt' contains: python-3.11")
print("3. Push changes to GitHub")
print("4. Wait for Streamlit to redeploy")
print("\nYour Prophet model is ready to use! 🚀")
