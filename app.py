
import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.io as pio
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), 0])
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_html = None
    error = None

    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)

            try:
                data = pd.read_csv(filepath)
                if 'Date' not in data.columns or 'Close' not in data.columns:
                    raise ValueError("CSV must contain 'Date' and 'Close' columns.")

                data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
                data = data.dropna(subset=['Date'])
                data.set_index('Date', inplace=True)

                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

                if len(scaled_data) < 50:
                    raise ValueError("Not enough data. Upload at least 50 data points.")

                look_back = 5
                train_data = scaled_data[:int(len(scaled_data) * 0.8)]
                test_data = scaled_data[int(len(scaled_data) * 0.8) - look_back:]

                X_train, y_train = create_dataset(train_data, look_back)
                X_test, y_test = create_dataset(test_data, look_back)

                if len(X_train) == 0 or len(X_test) == 0:
                    raise ValueError("Not enough data after processing.")

                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                model = Sequential()
                model.add(LSTM(units=10, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                model.add(LSTM(units=10))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

                predictions = model.predict(X_test)
                predictions = scaler.inverse_transform(predictions)

                prediction_dates = data.index[-len(predictions):]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Actual Price'))
                fig.add_trace(go.Scatter(x=prediction_dates, y=predictions.flatten(), mode='lines', name='Predicted Price'))
                fig.update_layout(title='Stock Price Prediction',
                                  xaxis_title='Date',
                                  yaxis_title='Price')

                plot_html = pio.to_html(fig, full_html=False)

            except Exception as e:
                error = str(e)

    return render_template('index.html', plot_html=plot_html, error=error)

if __name__ == '__main__':
    app.run(debug=True)
