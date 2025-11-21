from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import plotly.graph_objects as go 

# --- 1. Model Evaluation Function ---

def evaluate_model(y_true, y_pred, model_name="Prophet"):
    """
    Calculates key regression evaluation metrics (MAE and RMSE).
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"MAE (Mean Absolute Error): {mae:,.2f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:,.2f}")
    
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2)}

# --- 2. Prophet Prediction Function ---

def prophet_predict(df_ts, forecast_days, test_size_months, seasonality_mode='additive'):
    """
    Trains and evaluates a Prophet model, then forecasts future sales.
    
    Args:
        df_ts (pd.DataFrame): Data in the Prophet 'ds' (datetime) and 'y' (sales) format.
        forecast_days (int): Number of days to forecast into the future.
        test_size_months (int): Number of historical months to reserve for model testing.
    
    Returns:
        tuple: (fitted_model, forecast_df, metrics)
    """
    print("Starting Prophet time series forecasting...")

    # 1. Temporal Splitting
    if test_size_months > 0:
        split_date = df_ts['ds'].max() - pd.DateOffset(months=test_size_months)
        train_df = df_ts[df_ts['ds'] <= split_date]
        test_df = df_ts[df_ts['ds'] > split_date].copy()
    else:
        # If no test period is specified, use all data for training
        train_df = df_ts.copy()
        test_df = pd.DataFrame() 

    # 2. Model Initialization and Training
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False, # Assuming data is aggregated weekly/monthly, adjust if daily
        seasonality_mode=seasonality_mode
    )
    
    model.fit(train_df)
    
    # 3. Evaluation (only if test data exists)
    metrics = {"MAE": None, "RMSE": None}
    if not test_df.empty:
        forecast_test = model.predict(test_df[['ds']])
        metrics = evaluate_model(test_df['y'], forecast_test['yhat'])
    
    # 4. Out-of-Sample Forecasting
    future = model.make_future_dataframe(periods=forecast_days, freq='D')
    forecast = model.predict(future)
    
    print("Forecasting complete.")
    return model, forecast, metrics

# --- 3. Plotting Function (for Streamlit app.py) ---

def plot_forecast(model, forecast):
    """Generates a Plotly chart of the Prophet forecast with confidence intervals."""
    
    fig = go.Figure()

    # Historical Sales
    fig.add_trace(go.Scatter(
        x=model.history['ds'], y=model.history['y'],
        mode='lines', name='Historical Sales', line=dict(color='darkblue')
    ))

    # Forecasted Sales (yhat)
    # Filter to only show future forecast, not historical fit
    forecast_future = forecast[forecast['ds'] > model.history['ds'].max()]
    
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'], y=forecast_future['yhat'],
        mode='lines', name='Forecasted Sales', line=dict(color='red', dash='dot')
    ))

    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'], y=forecast_future['yhat_upper'],
        fill=None, mode='lines', line=dict(color='rgba(255,0,0,0)'), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'], y=forecast_future['yhat_lower'],
        fill='tonexty', mode='lines', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,0,0,0)'), name='Uncertainty'
    ))

    fig.update_layout(title="Prophet Sales Forecast", 
                      xaxis_title="Date", 
                      yaxis_title="Sales ($)",
                      hovermode="x unified")

    return fig