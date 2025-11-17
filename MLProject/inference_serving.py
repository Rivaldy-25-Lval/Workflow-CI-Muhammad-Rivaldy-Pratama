"""
Simple Flask Inference Server for Docker
Serves Heart Disease ML model predictions
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and scaler
try:
    model = joblib.load('Heart_Disease_RandomForest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if model is not None and scaler is not None:
        return jsonify({"status": "healthy", "model": "loaded"}), 200
    return jsonify({"status": "unhealthy", "model": "not_loaded"}), 503

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        if model is None or scaler is None:
            return jsonify({"error": "Model not loaded"}), 503
        
        # Get input data
        data = request.get_json()
        
        # Expected features
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        # Parse input
        if 'data' in data:
            # Format: {"data": [[values]]}
            input_data = data['data']
        elif 'instances' in data:
            # Format: {"instances": [[values]]}
            input_data = data['instances']
        else:
            # Direct format: {"age": 63, "sex": 1, ...}
            input_data = [[data.get(f, 0) for f in feature_names]]
        
        # Convert to DataFrame
        df = pd.DataFrame(input_data, columns=feature_names)
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        # Predict
        prediction = model.predict(X_scaled)
        probability = model.predict_proba(X_scaled)
        
        # Return result
        result = {
            "prediction": int(prediction[0]),
            "probability": {
                "no_disease": float(probability[0][0]),
                "disease": float(probability[0][1])
            },
            "diagnosis": "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        "service": "Heart Disease ML Inference",
        "model": "RandomForestClassifier",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        },
        "example": {
            "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
            "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
            "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
        }
    }), 200

if __name__ == '__main__':
    print("🚀 Starting Heart Disease ML Inference Server...")
    print("📊 Model: RandomForestClassifier")
    print("🌐 Port: 5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
