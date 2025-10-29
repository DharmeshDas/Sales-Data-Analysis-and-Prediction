import pandas as pd
import chardet
from src.data_preprocessing import preprocess_data
from src.eda_visualization import perform_eda
from src.sales_prediction import predict_sales

def read_csv_safely(file_path):
    """
    Reads a CSV file safely, handling encoding issues automatically.
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            print("UTF-8 failed, retrying with Latin-1 encoding...")
            df = pd.read_csv(file_path, encoding='latin1')
        except UnicodeDecodeError:
            print("Both UTF-8 and Latin-1 failed. Detecting encoding automatically...")
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read(10000))
                detected_encoding = result['encoding']
                print(f"Detected encoding: {detected_encoding}")
                df = pd.read_csv(file_path, encoding=detected_encoding)
    print("CSV loaded successfully!")
    return df

def main():
    print("Starting Sales Data Analysis and Prediction System...")
    df = read_csv_safely("data/superstore.csv")

    # Step 1: Clean and preprocess
    df = preprocess_data(df)

    # Step 2: Perform EDA
    perform_eda(df)

    # Step 3: Predict Sales
    predict_sales(df)

    print("Project executed successfully!")

if __name__ == "__main__":
    main()
