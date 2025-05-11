import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px

# Load and preprocess data
data = pd.read_csv('stock_prices.csv')

# Ensure the dates are in a consistent format
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')

# Drop rows with invalid date entries
data = data.dropna(subset=['Date'])

data.set_index('Date', inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Ensure sufficient data points
if len(scaled_data) < 50:  # Adjusted to handle smaller datasets
    raise ValueError("Not enough data points. Please provide a larger dataset.")

# Create training and testing sets
train_data, test_data = scaled_data[:int(len(scaled_data) * 0.8)], scaled_data[int(len(scaled_data) * 0.8):]

# Check shapes of train_data and test_data
print("train_data shape:", train_data.shape)
print("test_data shape:", test_data.shape)

# Create dataset for LSTM
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 5  # Reduced look-back period for smaller datasets
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# Check shapes before reshaping
print("X_train shape before reshape:", X_train.shape)
print("X_test shape before reshape:", X_test.shape)

# Ensure there is data to reshape
if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Not enough data after processing. Adjust your look-back period or provide more data.")

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print("X_train shape after reshape:", X_train.shape)
print("X_test shape after reshape:", X_test.shape)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=10, return_sequences=True, input_shape=(X_train.shape[1], 1)))  # Adjusted units for smaller data
model.add(LSTM(units=10))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)  # Reduced epochs and batch size

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plot interactive chart using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual Stock Price'))
fig.add_trace(go.Scatter(x=data.index[-len(predictions):], y=predictions.flatten(), mode='lines', name='Predicted Stock Price'))
fig.update_layout(title='Stock Price Prediction',
                  xaxis_title='Date',
                  yaxis_title='Stock Price',
                  legend_title='Legend')
fig.show()
