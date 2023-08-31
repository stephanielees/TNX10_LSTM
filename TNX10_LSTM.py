import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

#PREPROCESSING
# df = pd.read_csv('daily-treasury-rates-2018.csv', parse_dates=['Date'])
# #sort the dataset based on Date in ascending order
# df = df.sort_values('Date')
# print(f'There are {df.shape[0]} days in 2018')
#
# #read other files, sort the dates, then stack it with the old df
# for f in ['daily-treasury-rates-2019.csv', 'daily-treasury-rates-2020.csv',
#           'daily-treasury-rates-2021.csv', 'daily-treasury-rates-2022.csv', 'daily-treasury-rates-2023_Aug.csv']:
#     new_df = pd.read_csv(f, parse_dates=['Date']).sort_values('Date')
#     print(f"There are {new_df.shape[0]} days in {re.findall('[0-9]{4}', f)}")
#     df = pd.concat([df, new_df], ignore_index=True)
# df.to_csv('full-daily-treasury-rates-20182023.csv', index=False)
# print(df.shape)

df = pd.read_csv('full-daily-treasury-rates-20182023.csv', parse_dates=['Date'])
y10 = df['10 Yr']
# plt.plot(df['Date'], y10)
# plt.title('Yield curve rates for 10-year maturity bond\n from 2018 to Aug 2023')
# plt.ylabel('%')
# plt.show()

WINDOW_SIZE = 10
BATCH_SIZE = 32

x = tf.data.Dataset.from_tensor_slices(y10)
x = x.window(WINDOW_SIZE+1, shift=1, drop_remainder=True)
x = x.flat_map(lambda a: a.batch(WINDOW_SIZE+1))
x = x.map(lambda window: (window[:-1], window[-1]))
x = x.shuffle(1000)
x = x.batch(BATCH_SIZE).prefetch(1)

#MODEL BUILDING AND TRAINING
model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda dat: tf.expand_dims(dat, axis=-1), input_shape=[WINDOW_SIZE]),  #one feature only
    tf.keras.layers.LSTM(WINDOW_SIZE),
    tf.keras.layers.Dense(1)
    ])
model.compile(loss='mae', optimizer='SGD')
history = model.fit(x, epochs=100, verbose=2, workers=3, use_multiprocessing=True)

#plot the loss curve
plt.plot(history.history['loss'])
plt.xlabel('iterations')
plt.ylabel('Mean Absolute Error')
plt.show()

#FORECASTING
new_data = np.array(y10[-10:])
preds = []
N_AHEAD = 8
for i in range (N_AHEAD):
    input = new_data[i:(i+WINDOW_SIZE)]
    input = input[np.newaxis, :]
    pred = model.predict(input)
    new_data = np.append(new_data, pred[0][0])
    preds.append(pred[0][0])

print(preds)

#plot the predictions
plt.plot(df['Date'], y10, label='train data')
xaxis_preds = pd.period_range(start=df.Date.iloc[-1], periods=N_AHEAD, freq='D')
plt.plot(xaxis_preds, preds, color='hotpink', label='predictions')
plt.legend()
plt.show()