# Forecasting Netflix Stock Prices with LSTM

## 📌 Project Overview
This project focuses on forecasting Netflix (NFLX) stock closing prices using a **Long Short-Term Memory (LSTM)** neural network.  
The model leverages historical stock market data to capture temporal dependencies and predict future price movements.

The application integrates:
- **Machine Learning (LSTM)**
- **Data Visualization**
- **Flask Web Framework**
- **Frontend UI (HTML/CSS)**

This project was developed as part of a B.Tech academic mini project and demonstrates the practical application of deep learning in financial time-series forecasting.

---

## 🎯 Problem Statement
Stock price prediction is a challenging task due to market volatility, non-linearity, and dependency on historical trends.  
Traditional models often fail to capture long-term temporal patterns.

**Objective:**  
To design and implement an LSTM-based model capable of predicting Netflix stock closing prices using historical price and volume data.

---

## 🧠 Solution Approach
- Historical stock data for Netflix is fetched using **Yahoo Finance (yfinance)**.
- Data is preprocessed and normalized using **MinMaxScaler**.
- An **LSTM neural network** is trained to learn temporal patterns.
- A **Flask-based web application** allows users to input stock parameters and receive predicted closing prices.
- Interactive candlestick charts are used for visualization.

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Libraries & Frameworks
- TensorFlow / Keras (LSTM Model)
- Scikit-learn (Preprocessing)
- Pandas, NumPy (Data Handling)
- yfinance (Stock Data)
- Plotly (Visualization)
- Flask (Backend)
- HTML & CSS (Frontend)

---

## 📂 Project Structure
