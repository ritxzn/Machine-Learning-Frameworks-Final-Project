import joblib
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the saved model from the full path
model = joblib.load(r'C:\Users\ravir\Downloads\application\house_price_model.pkl')

# Define the home route
@app.route('/')
def home():
    return "Welcome to the House Price Prediction API!"

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json()  # Expecting JSON input
        
        # Check if 'features' key is present in the received data
        if 'features' not in data:
            return jsonify({'error': 'No features found in the request data'}), 400
        
        # Get the features
        features = data['features']
        
        # Ensure features are in the correct format (list of numerical values)
        if not isinstance(features, list) or not all(isinstance(x, (int, float)) for x in features):
            return jsonify({'error': 'Features should be a list of numerical values'}), 400
        
        # Convert the features into a numpy array and reshape for prediction
        features = np.array(features).reshape(1, -1)

        # Make prediction using the loaded model
        prediction = model.predict(features)

        # Return prediction as JSON response
        return jsonify({'predicted_price': prediction[0]})

    except Exception as e:
        # Catch any unexpected errors
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
