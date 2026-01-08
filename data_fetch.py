import yfinance as yf
import pandas as pd
from datetime import date, timedelta
import plotly.graph_objects as go


def fetch_data():
    end_date = date.today()
    start_date = end_date - timedelta(days=5000)

    data = yf.download("NFLX", start=start_date, end=end_date, progress=False)
    data.reset_index(inplace=True)
    return data


def plot_candlestick(data):
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=data["Date"],
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
            )
        ]
    )
    fig.update_layout(title="Netflix Stock Price Analysis")
    fig.show()


if __name__ == "__main__":
    df = fetch_data()
    plot_candlestick(df)