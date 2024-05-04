import pandas as pd
import json
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Correctly load the trained model using joblib
model_path = '/Users/sujitgyawali/Downloads/model.joblib'  # Path to your model file
rf = load(model_path)  # Load the model

# Load the JSON data
json_path = '/Users/sujitgyawali/Downloads/farmiculture-57e37-default-rtdb-export.json'  # Path to your JSON file
with open(json_path, 'r') as f:
    data = json.load(f)  # Load the JSON data

# Extract sensor data and convert to DataFrame
sensor_data_list = []
for sensor_id, sensor_data in data['sensorData'].items():
    sensor_data_list.append(sensor_data['fields'])

df = pd.DataFrame(sensor_data_list)  # Create a DataFrame with the sensor data

# Select the features for prediction
X = df[['humidity', 'temperature', 'soilMoisture']]

# Initialize the StandardScaler
scaler = StandardScaler()  # Initialize the scaler

# Fit and transform the data
X_scaled = scaler.fit_transform(X)

# Predict the labels for the given data
y_pred = rf.predict(X_scaled)  # Make predictions with the model

# Print the last predicted label
print("Predicted Labels:", y_pred[-1])

# Create a DataFrame with predictions and save to a CSV file
predictions_df = pd.DataFrame({'': [y_pred[-1]]})  # Use a list for the scalar value
csv_path = '/Users/sujitgyawali/farmi_culture/assets/predictions.csv'  # Path to save the CSV file
predictions_df.to_csv(csv_path, index=False)  # Save the DataFrame to CSV

# Notify that the CSV file has been saved
print(f"Predictions saved to: {csv_path}")
