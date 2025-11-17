import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import optuna
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import joblib


df = pd.read_csv('/Users/peiteer/Documents/bpm/train.csv')

X_tag = ['RhythmScore', 'AudioLoudness', 'VocalContent', 'AcousticQuality',
         'InstrumentalScore', 'LivePerformanceLikelihood', 'MoodScore',
         'TrackDurationMs', 'Energy']

X = df[X_tag]
Y = df['BeatsPerMinute']

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)

lgb_train = lgb.Dataset(X_train, label=Y_train)
lgb_val = lgb.Dataset(X_val, label=Y_val, reference=lgb_train)

best_rmse = float("inf")

def objective(trial):
    global best_rmse

    params = {
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 1.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 1.0),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.01, 0.2),
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 43,
        'n_jobs': -1
    }
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=100000,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(period=1000)
        ]
    )
    Y_pred_val = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    rmse = root_mean_squared_error(Y_val, Y_pred_val)
    print("RMSE:", rmse)
    print("MAE:", mean_absolute_error(Y_val, Y_pred_val))
    print("R2:", r2_score(Y_val, Y_pred_val))
    if rmse < best_rmse:
        print("Found a better model, saving...")
        best_rmse = rmse
        joblib.dump(gbm, '/Users/peiteer/Documents/bpm/best_model.pkl')
        print("Best RMSE updated:", best_rmse)

    return rmse

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print(study.best_trial.params)

best_model = joblib.load('/Users/peiteer/Documents/bpm/best_model.pkl')

test = pd.read_csv('/Users/peiteer/Documents/bpm/test.csv')
X_test = test[X_tag]

Y_pred_test = best_model.predict(X_test, num_iteration=best_model.best_iteration)

ans = pd.read_csv('/Users/peiteer/Documents/bpm/sample_submission.csv')
ans['BeatsPerMinute'] = Y_pred_test
ans.to_csv('/Users/peiteer/Documents/bpm/ans_lgbm.csv', index=False)
