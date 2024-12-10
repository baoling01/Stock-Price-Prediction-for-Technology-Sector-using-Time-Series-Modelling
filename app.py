import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings("ignore")


# Function to fit ARIMA and forecast
def fit_arima_and_forecast(train, test, p=2, d=1, q=2):
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()
    y_pred = model_fit.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test, y_pred))
    r_squared = r2_score(test, y_pred)

    # Forecast for the next 10 years into the future (approx. 252 trading days * 10 years)
    forecast_steps = 10 * 252
    forecast = model_fit.forecast(steps=forecast_steps)
    return rmse, r_squared, y_pred, forecast


# Fit SARIMA and forecast
def fit_sarima_and_forecast(train, test):
    sarima = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 252))
    sarima_fit = sarima.fit(disp=False)

    y_pred = sarima_fit.get_forecast(steps=len(test)).predicted_mean
    rmse = np.sqrt(mean_squared_error(test, y_pred))
    r_squared = r2_score(test, y_pred)

    forecast_steps = 10 * 252
    forecast = sarima_fit.get_forecast(steps=forecast_steps).predicted_mean
    return rmse, r_squared, y_pred, forecast


# Forecast with Prophet
def forecast_with_prophet(train_df, test):
    df_prophet = pd.DataFrame({"ds": train_df.index, "y": train_df})
    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=len(test), freq="B")
    forecast = model.predict(future)

    rmse = np.sqrt(mean_squared_error(test, forecast["yhat"][-len(test):]))
    r_squared = r2_score(test, forecast["yhat"][-len(test):])

    forecast_steps = 10 * 252
    future_forecast = model.make_future_dataframe(periods=forecast_steps, freq="B")
    future_forecast = model.predict(future_forecast)
    return rmse, r_squared, forecast["yhat"][-len(test):], future_forecast["yhat"]


# Forecast with LSTM
def lstm_and_forecast(train, test):
    # Prepare LSTM input data
    sequence_length = 60  # Time steps for LSTM lookback
    X_train, y_train = [], []
    
    # Create sequences for LSTM training
    for i in range(sequence_length, len(train)):
        X_train.append(train[i - sequence_length:i])
        y_train.append(train[i])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Create LSTM test sequences
    X_test = []
    for i in range(sequence_length, len(test)):
        X_test.append(test[i - sequence_length:i])
    X_test = np.array(X_test)

    # Predict using the LSTM model
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(test[sequence_length:], y_pred))
    r_squared = r2_score(test[sequence_length:], y_pred)
    return rmse, r_squared, y_pred


# Streamlit UI
st.title("Stock Forecasting Comparison")

st.write("""
Compare performance between ARIMA, SARIMA, Prophet, LSTM, and RNN for stock forecasting.
Select a stock, split data into 80/20, train models, visualize predictions, and view model performance comparison.
""")

# Allow user to select a stock symbol
stock_symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "INTC", "GOOG", "META"]
selected_stock = st.selectbox("Select a Stock Symbol", stock_symbols)

file_path = f"Dataset/Cleaned_Dataset/{selected_stock}_cleaned.csv"
try:
    # Load data
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    st.success("Data loaded successfully")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

target_var = "Adj Close"
X = data[target_var]
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42, shuffle=False)

# Train and evaluate models
# ARIMA
rmse_arima, r2_arima, y_pred_arima, forecast_arima = fit_arima_and_forecast(X_train, X_test)
# SARIMA
rmse_sarima, r2_sarima, y_pred_sarima, forecast_sarima = fit_sarima_and_forecast(X_train, X_test)
# Prophet
rmse_prophet, r2_prophet, y_pred_prophet, forecast_prophet = forecast_with_prophet(data, X_test)
# LSTM
rmse_lstm, r2_lstm, y_pred_lstm = lstm_and_forecast(X_train, X_test)

# Show metrics comparison table
metrics_df = pd.DataFrame({
    "Model": ["ARIMA", "SARIMA", "Prophet", "LSTM"],
    "RMSE": [rmse_arima, rmse_sarima, rmse_prophet, rmse_lstm],
    "R^2": [r2_arima, r2_sarima, r2_prophet, r2_lstm]
})

st.subheader("Model Performance Comparison")
st.dataframe(metrics_df)

# Plot comparison visualization
st.subheader("Test Forecast Comparison")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(X_test.index, y_pred_arima, label="ARIMA")
ax.plot(X_test.index, y_pred_sarima, label="SARIMA")
ax.plot(X_test.index, y_pred_prophet, label="Prophet")
ax.plot(X_test.index, y_pred_lstm, label="LSTM")
ax.set_title("Comparison of Model Forecasts on Test Data")
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend()
st.pyplot(fig)
