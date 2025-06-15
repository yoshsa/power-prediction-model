import pandas as pd
import numpy as np

# Create a sample dataset
data = {
    'hour': np.random.randint(0, 24, size=1000),
    'day_of_week': np.random.randint(0, 7, size=1000),
    'month': np.random.randint(1, 13, size=1000),
    'previous_power': np.random.uniform(1000, 3000, size=1000),
    'current': np.random.uniform(5, 30, size=1000),
    'voltage': np.random.uniform(210, 250, size=1000),
    'frequency': np.random.uniform(49, 51, size=1000),
    'pf': np.random.uniform(0.8, 1.0, size=1000),
    'va': np.random.uniform(1000, 3000, size=1000),
    'var': np.random.uniform(100, 500, size=1000),
    'power': np.random.uniform(1000, 3000, size=1000)  # Target variable
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Print first few rows to check
print(df.head())
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Feature columns
X = df[['hour', 'day_of_week', 'month', 'previous_power', 'current', 'voltage', 'frequency', 'pf', 'va', 'var']]

# Target column
y = df['power']

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model (RandomForestRegressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Save the trained model to a file
joblib.dump(model, 'power_consumption_model.pkl')
print("Model saved as 'power_consumption_model.pkl'")
