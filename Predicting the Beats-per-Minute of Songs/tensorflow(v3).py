import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('/Users/peiteer/Documents/train.csv')

X_tag = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
       'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
       'TrackDurationMs', 'Energy']

X = df[X_tag]
Y = df['BeatsPerMinute']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=47)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = models.Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1)  
])

model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

history = model.fit(X_train_scaled, Y_train, 
                    validation_split=0.1, 
                    epochs=20, batch_size=64, verbose=1)

Y_pred = model.predict(X_test_scaled).ravel()

print("MAE:", mean_absolute_error(Y_test, Y_pred))
print("MSE:", mean_squared_error(Y_test, Y_pred))
print("R2:", r2_score(Y_test, Y_pred))
