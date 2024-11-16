import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the CSV file
results_df = pd.read_csv('model2_train.csv')

# Extract actual and predicted prices
actual_prices = results_df['ActualPrice']
predicted_prices = results_df['PredictedPrice']

# Calculate MSE and MAE
mse = mean_squared_error(actual_prices, predicted_prices)
mae = mean_absolute_error(actual_prices, predicted_prices)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")