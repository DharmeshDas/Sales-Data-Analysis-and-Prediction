import pandas as pd
import streamlit as st
import chardet
import os

# Function 1: Robust CSV Reader (similar to your original main.py/app.py logic)
def read_csv_safely(file_path):
    """
    Reads a CSV file safely, handling common encoding issues (UTF-8, Latin-1) 
    and falling back to automatic detection.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except UnicodeDecodeError:
            # Automatic detection
            with open(file_path, 'rb') as f:
                # Read a chunk for detection
                result = chardet.detect(f.read(100000)) 
                detected_encoding = result['encoding']
                df = pd.read_csv(file_path, encoding=detected_encoding)
    return df

# Function 2: Streamlit Cached Data Loader
@st.cache_data(show_spinner=True, ttl=3600) # Caches the data for 1 hour (3600 seconds)
def load_raw_data(config):
    """
    Loads and caches the raw data using the path defined in the configuration.
    
    The @st.cache_data decorator ensures this function only runs once 
    per session/hour unless the input (config dict) changes.
    """
    data_path = config.get('DATA_PATH')
    
    if not data_path:
        st.error("Error: 'DATA_PATH' not found in config file.")
        return pd.DataFrame() # Return empty DataFrame on error

    df = read_csv_safely(data_path)
    
    print(f"Data loaded successfully from {data_path}. Shape: {df.shape}")
    return df