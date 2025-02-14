import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Load the data
df = pd.read_csv('newMappedData.csv')

# Seperate the Date column into year, month, and day
df['year'] = pd.DatetimeIndex(df['Date']).year
df['month'] = pd.DatetimeIndex(df['Date']).month
df['day'] = pd.DatetimeIndex(df['Date']).day

# Since the game week is cyclical, we can represent it using sin and cos
df['game_week_sin'] = np.sin(2 * np.pi * df['Game Week'] / 38)
df['game_week_cos'] = np.cos(2 * np.pi * df['Game Week'] / 38)

# Drop the Date and Game week column
df = df.drop(['Date'], axis=1)
df = df.drop(['Game Week'], axis=1)

# Seperate the Time column into hour and minute
df['hour'] = pd.DatetimeIndex(df['Time']).hour
df['minute'] = pd.DatetimeIndex(df['Time']).minute

# Drop the Time column
df = df.drop(['Time'], axis=1)

# Identify the features and target
X = df.drop(columns=['Away Team', 'Away Score', 'Home Score'])
y = df['Home Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize the XGBoost regressor
xg_reg = xgb.XGBRegressor(
    objective='reg:squarederror',  # For regression tasks
    colsample_bytree=0.3,          # Fraction of features to consider at each split
    learning_rate=0.1,             # Step size in each iteration
    max_depth=10,                  # Maximum depth of the trees
    alpha=10,                      # L2 regularization term
    n_estimators=100,              # Number of boosting rounds (trees)
    random_state=42
)

# Train the model
xg_reg.fit(X_train, y_train)

# Make predictions
y_pred = xg_reg.predict(X_test)

# Evaluate the model
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")


import matplotlib.pyplot as plt

# Predicted vs Actual Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values (XGBoost)')
plt.legend()
plt.show()


