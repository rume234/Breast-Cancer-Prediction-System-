from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model and scaler from the 'model' directory
# Ensure these files were created in Part A
MODEL_PATH = os.path.join('model', 'breast_cancer_model.pkl')
SCALER_PATH = os.path.join('model', 'scaler.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract features from form
            features = [float(x) for x in request.form.values()]
            final_features = [np.array(features)]
            
            # Scale features
            scaled_features = scaler.transform(final_features)
            
            # Prediction
            prediction = model.predict(scaled_features)
            output = "Benign" if prediction[0] == 1 else "Malignant"
            
            # Determine color for the result
            res_class = "benign-text" if output == "Benign" else "malignant-text"

            return render_template('index.html', 
                                 prediction_text=f'Diagnosis: {output}',
                                 res_class=res_class)
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)