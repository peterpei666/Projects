import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 读取数据
df = pd.read_csv('/Users/peiteer/Documents/ST1131/hdb_2017_2025Feb_sample.csv')

#task 1
print(df.shape)
print(df.head())

#task 2
plt.figure(figsize=(10,6))
plt.hist(df['floor_area_sqm'], bins=45, range=(30,250), color='brown', edgecolor='black')
plt.title('The Frequencies of Floor Area')
plt.xlabel('Floor Area')
plt.ylabel('Frequency')
plt.show()

#task 3
counts = df['flat_type'].value_counts().sort_index()
categories = counts.index
frequencies = counts.values

plt.figure(figsize=(10,6))
bars = plt.bar(categories, frequencies, color='blue', edgecolor='black')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}', 
             ha='center', va='bottom', fontsize=9)
plt.title('The Frequencies of Flat Type')
plt.xlabel('Flat Type')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

#task 4
plt.figure(figsize=(10,6))
data_by_type = [df[df['flat_type'] == t]['floor_area_sqm'].dropna() for t in categories]
plt.boxplot(data_by_type, vert=True, patch_artist=True)
plt.xticks(range(1, len(categories)+1), categories)
plt.title('Spread of Floor Area by Flat Type')
plt.xlabel('Flat Type')
plt.ylabel('Floor Area')
plt.show()

#task 5
plt.figure(figsize=(10,6))
plt.hexbin(
    df['floor_area_sqm'],
    df['resale_price'],
    gridsize=150,  
    cmap='plasma',
    bins='log',
    mincnt=1
)
plt.title('Resale Price vs Floor Area')
plt.xlabel('Floor Area')
plt.ylabel('Resale Price')
plt.show()

corr = np.corrcoef(df['floor_area_sqm'], df['resale_price'])[0,1]
print(f"Correlation coefficient: {corr:.4f}")

#task 6
df_clean = df[['floor_area_sqm', 'resale_price']].dropna()

model = LinearRegression().fit(df_clean[['floor_area_sqm']], df_clean['resale_price'])
y_pred = model.predict(df_clean[['floor_area_sqm']])

plt.figure(figsize=(10,6))
plt.hexbin(
    df['floor_area_sqm'],
    df['resale_price'],
    gridsize=150,  
    cmap='plasma',
    bins='log',
    mincnt=1
)
plt.plot(df_clean['floor_area_sqm'], y_pred, color='red', linewidth=2, label='Fitted line')
plt.title('Resale Price vs Floor Area (with Fitted Line)')
plt.xlabel('Floor Area')
plt.ylabel('Resale Price')
plt.legend()
plt.show()

print(f"y = {model.coef_[0]:.2f} * x + {model.intercept_:.2f}")
mae = mean_absolute_error(df_clean['resale_price'], y_pred)
mse = mean_squared_error(df_clean['resale_price'], y_pred)
r2 = r2_score(df_clean['resale_price'], y_pred)
print('MSE:', mse)
print('MAE:', mae)
print('R²:', r2)

#extra (better model)
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

df['street_name'] = df['street_name'].astype('category')
df['flat_type'] = df['flat_type'].astype('category')
df['flat_model'] = df['flat_model'].astype('category')

X_tag = ['street_name', 'floor_area_sqm', 'flat_model', 'lease_commence_date']
X, Y = df[X_tag], df['resale_price']
categorical_features = ['street_name', 'flat_model']

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=0.9, random_state=43)

lgb_train = lgb.Dataset(X_train, label=Y_train, categorical_feature=categorical_features)
lgb_val = lgb.Dataset(X_val, label=Y_val, reference=lgb_train, categorical_feature=categorical_features)

params = {
    'objective': 'regression',
    'metric': 'mse',
    'learning_rate': 0.01,
    'num_leaves': 327,
    'max_depth': 25,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0,
    'min_gain_to_split': 0.05,
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
        early_stopping(stopping_rounds=60),
        log_evaluation(period=1000)
    ]
)

Y_pred_val = gbm.predict(X_val, num_iteration=gbm.best_iteration)

print("MSE:", mean_squared_error(Y_val, Y_pred_val))
print("MAE:", mean_absolute_error(Y_val, Y_pred_val))
print("R2:", r2_score(Y_val, Y_pred_val))
