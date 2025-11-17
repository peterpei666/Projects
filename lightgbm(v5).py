import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/peiteer/Documents/train.csv')
X_tag = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
         'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
         'TrackDurationMs', 'Energy']

X = df[X_tag].values
Y = df['BeatsPerMinute']

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

lgb_train = lgb.Dataset(X_train_scaled, label=Y_train)
lgb_val = lgb.Dataset(X_val_scaled, label=Y_val, reference=lgb_train)

params = {
    'objective': 'regression_l1',
    'metric': 'mse',
    'learning_rate': 0.001,
    'num_leaves': 527,
    'n_estimators': 2000,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 43
}

gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=100000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=['train', 'valid'],
    callbacks=[
        early_stopping(stopping_rounds=100),
        log_evaluation(period=100)
    ]
)

Y_pred_val = gbm.predict(X_val, num_iteration=gbm.best_iteration)
Y_pred_val1 = Y_pred_val - Y_pred_val + Y_pred_val.mean()


print("MSE:", mean_squared_error(Y_val, Y_pred_val))
print("MAE:", mean_absolute_error(Y_val, Y_pred_val))
print("R2:", r2_score(Y_val, Y_pred_val))

print("MSE:", mean_squared_error(Y_val, Y_pred_val1))
print("MAE:", mean_absolute_error(Y_val, Y_pred_val1))
print("R2:", r2_score(Y_val, Y_pred_val1))

print("mean:", Y_pred_val1.mean(), Y_val.mean())
print("variance:", Y_pred_val1.var(), Y_val.var())

test = pd.read_csv('/Users/peiteer/Documents/test.csv')
X_test = test[X_tag].values
X_test_scaled = scaler.transform(X_test)

Y_pred_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)

ans = pd.read_csv('/Users/peiteer/Documents/sample_submission.csv')
ans['BeatsPerMinute'] = Y_pred_test
ans.to_csv('/Users/peiteer/Documents/ans_lgbm.csv', index=False)