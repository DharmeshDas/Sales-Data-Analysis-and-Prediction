import pandas as pd

def preprocess_data(df):
    print("Cleaning and preparing data...")

    # Convert order date to datetime safely
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        df['Month'] = df['Order Date'].dt.month
        df['Year'] = df['Order Date'].dt.year

    # Remove duplicates and missing values
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)

    print("Data cleaned successfully.")
    return df
