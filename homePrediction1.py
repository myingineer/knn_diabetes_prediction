import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV
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

'''
------------------------------------------------------------------------------------------------------------------------
This was used to find the best hyperparameters for the model
------------------------------------------------------------------------------------------------------------------------
Define the hyperparameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Best Parameters: {grid_search.best_params_}")
'''

# Define the model
regressor = RandomForestRegressor(
    random_state=42,
    max_depth=10, 
    min_samples_leaf=4, 
    min_samples_split=10, 
    n_estimators=100
)

# Train the model
regressor.fit(X_train, y_train)

# Predict the test set results
y_pred = regressor.predict(X_test)

# Evaluate the model
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

import matplotlib.pyplot as plt

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.legend()
plt.show()
