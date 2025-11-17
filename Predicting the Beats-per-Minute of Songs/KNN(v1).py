import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('/Users/peiteer/Documents/train.csv')

X = df[['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
       'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
       'TrackDurationMs', 'Energy']]
Y = df['BeatsPerMinute']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=47)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsRegressor(n_neighbors=100, n_jobs=-1)
knn.fit(X_train_scaled, Y_train)

Y_pred = knn.predict(X_test_scaled)

print("MSE:", mean_squared_error(Y_test, Y_pred))
print("R2:", r2_score(Y_test, Y_pred))