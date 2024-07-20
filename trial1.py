#Team ID=SWTID1720435231

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import pickle

# Load data
train = pd.read_csv('trains.csv')
store = pd.read_csv('stores.csv')
feature = pd.read_csv('features.csv')

# Data Preprocessing and Merging
data = train.merge(feature, on=['Store', 'Date'], how='inner').merge(store, on=['Store'], how='inner')

# Fill NaNs
data['MarkDown1'] = data['MarkDown1'].replace(np.nan, 0)
data['MarkDown2'] = data['MarkDown2'].replace(np.nan, 0)
data['MarkDown3'] = data['MarkDown3'].replace(np.nan, 0)
data['MarkDown4'] = data['MarkDown4'].replace(np.nan, 0)
data['MarkDown5'] = data['MarkDown5'].replace(np.nan, 0)

# Filter data
data = data[data['Weekly_Sales'] >= 0]

# Convert categorical variables to dummy/indicator variables
data = pd.get_dummies(data, columns=['Type'])
data['Date'] = pd.to_datetime(data['Date'])

# Extract features from date
data['month'] = data['Date'].dt.month
data['year'] = data['Date'].dt.year
data['dayofweek_name'] = data['Date'].dt.day_name()
data['is_weekend'] = data['dayofweek_name'].isin(['Sunday', 'Saturday']).astype(int)

# Drop unnecessary columns
data = data.drop(columns=['dayofweek_name', 'Date'])

# Prepare features and target variable
X = data[["Store", "Dept", "Size", "IsHoliday_x", "CPI", "Temperature", "Type_B", "Type_C", "month", "year", "is_weekend"]]
y = data["Weekly_Sales"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on test data
y_pred = rf.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")

# Example input for prediction
new_data = [[30, 5, 2000, 0, 211.0, 45.0, 1, 0, 7, 2023, 0]]  # Adjust values accordingly

# Predicting
pred1 = rf.predict(new_data)
print(f"Prediction for new data: {pred1}")

# Serialize the model into a pickle file
with open('models.pkl', 'wb') as file:
    pickle.dump(rf, file)