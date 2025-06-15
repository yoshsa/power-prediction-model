from flask import Flask, request, jsonify
import joblib
import numpy as np

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

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
