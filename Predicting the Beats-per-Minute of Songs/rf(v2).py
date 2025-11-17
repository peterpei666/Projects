import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

df = pd.read_csv('/Users/peiteer/Documents/train.csv')

X_tag = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
       'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
       'TrackDurationMs', 'Energy']

X = df[X_tag]
Y = df['BeatsPerMinute']

for i in X_tag:
    print(np.corrcoef(df[i], df['BeatsPerMinute']))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=47)


rf = RandomForestRegressor(n_estimators=500, max_depth=100, n_jobs=-1)
rf.fit(X_train, Y_train)

Y_pred = rf.predict(X_test)

print("MAE:", mean_absolute_error(Y_test, Y_pred))
print("MSE:", mean_squared_error(Y_test, Y_pred))
print("R2:", r2_score(Y_test, Y_pred))

joblib.dump(rf, '/Users/peiteer/Documents/rf_model')