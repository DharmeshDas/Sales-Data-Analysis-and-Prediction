import pandas as pd
import os
import sys

# 1. Core Imports from the src/ directory
from src.utils import load_config
from src.data_loader import load_raw_data

# Business logic modules
from src.data_preprocessing import preprocess_for_ts, add_time_features
from src.eda_visualization import perform_eda

#FIX: Import the new prediction function name, 'prophet_predict'
from src.sales_prediction import prophet_predict 


def main():
    """
    Main entry point for batch execution (CLI/Reporting).
    Orchestrates the data analysis and prediction pipeline.
    """
    print("=====================================================")
    print("Starting Sales Data Analysis and Prediction System...")
    print("=====================================================")

    # 1. Load Configuration
    try:
        config = load_config(config_path='config.yaml')
        print("Configuration loaded successfully.")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: {e}. Please ensure config.yaml exists in the root folder.")
        sys.exit(1)
    
    # 2. Load Data
    df_raw = load_raw_data(config).copy()

    if df_raw.empty:
        print("Data loading failed. Exiting pipeline.")
        sys.exit(1)

    # --- Data Preparation ---
    date_col = config.get('DATE_COL', 'Order Date')
    sales_col = config.get('SALES_COL', 'Sales')
    
    # 3a. Prepare data for EDA (adds Month/Year columns needed by eda_visualization)
    df_eda = add_time_features(df_raw, date_col)
    
    # 3b. Prepare data specifically for Time Series (Prophet model)
    # df_ts is in the 'ds'/'y' format
    df_ts = preprocess_for_ts(
        df_raw.copy(), 
        date_col, 
        sales_col
    )
    
    # --- Execute Business Logic ---
    
    # 4. Perform Exploratory Data Analysis (EDA)
    perform_eda(df_eda)

    # 5. Predict Sales (Now using the advanced Prophet function)
    print("\nStarting Advanced Time Series Prediction...")
    
    #FIX: Call the new function, pass the time-series data (df_ts), and read parameters from config
    model, forecast, metrics = prophet_predict(
        df_ts=df_ts, 
        forecast_days=config.get('FORECAST_PERIOD_DAYS', 90), # Read from config
        test_size_months=config.get('TEST_SIZE_MONTHS', 12)    # Read from config
    )
    
    # Optional: Save forecast plot for reporting purposes
    # from src.sales_prediction import plot_forecast
    # fig = plot_forecast(model, forecast)
    # fig.write_image("reports/figures/sales_forecast.png")


    print("=====================================================")
    print("Project execution completed successfully!")
    print("=====================================================")


if __name__ == "__main__":
    main()