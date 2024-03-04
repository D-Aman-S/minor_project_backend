from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open('model/gradientbooster.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json
        
        # Check if data is present and contains the required keys
        if not data or not all(key in data for key in ['ja_sup', 'ja_sub', 'Re_V', 'Re_L', 'prv']):
            return jsonify({'error': 'Invalid request data'}), 400

        # Prepare data for prediction
        X_pred = pd.DataFrame(data, index=[0])

        # Perform prediction
        prediction = model.predict(X_pred)

        # Return the prediction
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def homeget():
    return jsonify({'message': 'Hello World, are you lost?'})

if __name__ == '__main__':
    app.run(host="0.0.0.0")
