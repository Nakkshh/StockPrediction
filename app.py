import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Date range for historical data
start = '2000-01-01'
end = '2025-03-31'

st.title('Stock Price Predictions')

# User input
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)

# Show data summary
st.subheader('Data from 2000 to 2025')
st.write(df.describe())

# Closing Price vs Time
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close, color='black')
plt.xlabel("Date")
plt.ylabel("Closing Price")
st.pyplot(fig)

# Closing Price with 100MA
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, color='red', label='100MA')
plt.plot(df.Close, color='black', label='Close')
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig)

# Closing Price with 100MA & 200MA
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, color='red', label='100MA')
plt.plot(ma200, color='green', label='200MA')
plt.plot(df.Close, color='black', label='Close')
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig)

# Data preparation
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

# Load LSTM model
model = load_model('Stock_Price_LSTM_model.h5')

# Prepare testing data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict
y_predicted = model.predict(x_test)

# Rescale back
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plot predictions vs actual
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
