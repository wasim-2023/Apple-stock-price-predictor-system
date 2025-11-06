import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import yfinance as yf

# Page config
st.set_page_config(page_title="AAPL Stock Predictor", layout="wide")
st.title("🍏 AAPL Stock Price Prediction App")
st.markdown("This app predicts Apple's closing stock price using machine learning trained on historical data.")

# Load trained model and scaler
model = joblib.load("stock_price_predictor.pkl")
scaler = joblib.load("scaler.pkl")

# Load live AAPL stock data from Yahoo Finance
st.header("📄 AAPL Dataset (Live via yfinance)")
data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
data.reset_index(inplace=True)  # move Date from index to column
st.dataframe(data.head(100), use_container_width=True)

# Visualizations
st.header("📊 Visual Analysis")

# Close price over time
st.subheader("Closing Price Over Time")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(data['Date'], data['Close'], color='blue', label='Close')
ax1.set_title("AAPL Closing Price (2020–2024)")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price ($)")
ax1.legend()
st.pyplot(fig1)

# High vs Low
st.subheader("High vs Low Prices")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(data['Date'], data['High'], label='High', color='green')
ax2.plot(data['Date'], data['Low'], label='Low', color='red')
ax2.set_title("AAPL Daily High and Low Prices")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price ($)")
ax2.legend()
st.pyplot(fig2)

# Prediction Input
st.header("🔮 Predict Tomorrow's Closing Price")

col1, col2, col3 = st.columns(3)
with col1:
    open_price = st.number_input("Open Price", value=150.0, step=0.1)
with col2:
    high_price = st.number_input("High Price", value=155.0, step=0.1)
with col3:
    low_price = st.number_input("Low Price", value=148.0, step=0.1)

if st.button("Predict Closing Price"):
    try:
        input_scaled = scaler.transform([[open_price, high_price, low_price]])
        prediction = model.predict(input_scaled)
        st.success(f"📉 Predicted Closing Price: **${float(prediction[0]):.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.caption("Model: Linear Regression | Features: Open, High, Low | Data Source: Yahoo Finance | Built with Streamlit")
