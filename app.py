from flask import Flask, render_template, request, jsonify
import numpy as np

from data_fetch import fetch_data
from model import prepare_data, split_and_reshape, build_lstm_model

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    open_price = float(data["Open"])
    high_price = float(data["High"])
    low_price = float(data["Low"])
    volume = float(data["Volume"])

    # Fetch historical data
    df = fetch_data()

    # Prepare data
    features, target, scaler_x, scaler_y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_and_reshape(features, target)

    # Build and train model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Prepare input for prediction
    input_features = np.array([[open_price, high_price, low_price, volume]])
    input_features = scaler_x.transform(input_features)
    input_features = input_features.reshape((1, 1, input_features.shape[1]))

    predicted_price = model.predict(input_features)
    predicted_price = scaler_y.inverse_transform(predicted_price)

    return jsonify(
        {"predicted_price": round(float(predicted_price[0][0]), 2)}
    )


if __name__ == "__main__":
    app.run(debug=True)