import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv('/Users/peiteer/Documents/train.csv')

X_tag = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
       'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
       'TrackDurationMs', 'Energy']

X_train = df[X_tag]
Y_train = df['BeatsPerMinute']

test = pd.read_csv('/Users/peiteer/Documents/test.csv')
X_test = test[X_tag]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsRegressor(n_neighbors=1500, n_jobs=-1)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)

rf = RandomForestRegressor(n_estimators=600, n_jobs=-1)
rf.fit(X_train_scaled, Y_train)
Y_pred_rf = rf.predict(X_test_scaled)


model = models.Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1)  
])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

history = model.fit(X_train_scaled, Y_train, 
                    validation_split=0.1, 
                    epochs=100, 
                    batch_size=64,
                    verbose=1,
                    callbacks=[early_stopping]
)

Y_pred_keras = model.predict(X_test_scaled).ravel()


Y_pred = Y_pred_knn * 0.3 + Y_pred_rf * 0.3 + Y_pred_keras * 0.4

ans = pd.read_csv('/Users/peiteer/Documents/sample_submission.csv')

ans['BeatsPerMinute'] = Y_pred

ans.to_csv('/Users/peiteer/Documents/ans.csv', index=False)