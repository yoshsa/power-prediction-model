from flask import Flask, request, jsonify
import joblib
import numpy as np
import os  # For accessing environment variables

app = Flask(__name__)

# Load the trained model
model = joblib.load('power_consumption_model.pkl')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the data from the frontend (AJAX)

    # Extract features from the JSON data
    features = np.array([data['hour'], data['day_of_week'], data['month'], 
                         data['previous_power'], data['current'], 
                         data['voltage'], data['frequency'], data['pf'], 
                         data['va'], data['var']]).reshape(1, -1)

    # Predict the power consumption
    prediction = model.predict(features)

    # Return the prediction as a JSON response
    return jsonify({'predicted_power': prediction[0]})

# Run the app, binding to the port provided by Render
if __name__ == '__main__':
    # Get the port number from the environment variable, default to 5000 if not provided
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)  # Ensure it binds to the correct port
