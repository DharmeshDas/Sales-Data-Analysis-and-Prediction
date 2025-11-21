import pandas as pd

def preprocess_for_ts(df, date_col, sales_col):
    """
    Cleans data, handles dates, and aggregates sales for time series analysis.
    This prepares the data for the Prophet model (ds/y format).
    """
    print("Preparing data for time series prediction...")
    
    # 1. Convert to datetime, drop rows with invalid dates
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=[date_col, sales_col], inplace=True)

    # 2. Aggregate Daily Sales
    # Note: Prophet works best on daily data, even if your original data is sporadic.
    daily_sales = df.groupby(date_col)[sales_col].sum().reset_index()

    # 3. Rename columns to Prophet format
    daily_sales.rename(columns={date_col: 'ds', sales_col: 'y'}, inplace=True)
    
    print(f"Data prepared. Total time points: {len(daily_sales)}")
    return daily_sales

def add_time_features(df, date_col):
    """
    Adds basic time features (Month, Year) used by EDA and the old Linear Regression model.
    This is kept for compatibility with eda_visualization.py.
    """
    if date_col in df.columns:
        df['Order Date'] = pd.to_datetime(df[date_col], errors='coerce')
        df.dropna(subset=['Order Date'], inplace=True)
        df['Month'] = df['Order Date'].dt.month
        df['Year'] = df['Order Date'].dt.year
    return df