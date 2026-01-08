import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


def prepare_data(data):
    features = data[["Open", "High", "Low", "Volume"]].values
    target = data["Close"].values

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    features = scaler_x.fit_transform(features)
    target = scaler_y.fit_transform(target.reshape(-1, 1))

    return features, target, scaler_x, scaler_y


def split_and_reshape(features, target):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    return X_train, X_test, y_train, y_test


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")
    return model