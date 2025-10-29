import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def predict_sales(df):
    print("Training model for sales prediction...")

    if 'Month' not in df.columns or 'Sales' not in df.columns:
        print("Missing required columns for prediction. Skipping model.")
        return

    monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()

    X = monthly_sales[['Month']]
    y = monthly_sales['Sales']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Visualization
    plt.figure(figsize=(8,5))
    plt.plot(X, y, label="Actual Sales", marker='o')
    plt.plot(X, y_pred, label="Predicted Sales", linestyle='--')
    plt.title("Actual vs Predicted Monthly Sales")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.legend()
    plt.show()

    print("Sales prediction completed successfully.")
