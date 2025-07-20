from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from PIL import Image
import os

# importing updated models
cnn_model = tf.keras.models.load_model('soil_cnn_model.h5')
rfc_model = joblib.load('rfc_model.joblib')
ms = joblib.load('minmaxscaler.joblib')

# Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Crop dictionary
crop_dict = {
    1: 'rice', 2: 'maize', 3: 'chickpea', 4: 'kidneybeans', 5: 'pigeonpeas',
    6: 'mothbeans', 7: 'mungbean', 8: 'blackgram', 9: 'lentil', 10: 'pomegranate',
    11: 'banana', 12: 'mango', 13: 'grapes', 14: 'watermelon', 15: 'muskmelon',
    16: 'apple', 17: 'orange', 18: 'papaya', 19: 'coconut', 20: 'cotton',
    21: 'jute', 22: 'coffee'
}

# Soil dictionary
soil_dict = {'Alluvial': 1, 'Black': 2, 'Cinder': 3, 'Clay': 4, 'Laterite': 5,
             'peat': 6, 'Read': 7, 'Yellow': 8}

# Soil to compatible crops
soil_crop_mapping = {
    'Alluvial': ['Rice', 'Wheat', 'Sugarcane', 'Cotton', 'Jute', 'Pulses'],
    'Black': ['Cotton', 'Soybean', 'Sorghum', 'Millets'],
    'Cinder': ['Grapes', 'Apple'],
    'Clay': ['Rice', 'Sugarcane'],
    'Laterite': ['Tea', 'Coffee', 'Cashew'],
    'peat': ['Rice', 'Vegetables'],
    'Read': ['Millets', 'Groundnut', 'Pulses'],
    'Yellow': ['Maize', 'Peanuts', 'Pulses']
}

# Soil type prediction
def predict_soil_type(image_path):
    img = Image.open(image_path).resize((150, 150))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = cnn_model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    soil_label = list(soil_dict.keys())[predicted_class_index]
    return soil_label

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        result = ""
        soil_type = None

        # Check if image uploaded
        if 'soil_image' in request.files:
            file = request.files['soil_image']
            if file.filename != '':
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(image_path)
                soil_type = predict_soil_type(image_path)

        # Get parameters
        N = request.form.get('Nitrogen')
        P = request.form.get('Phosphorus')
        K = request.form.get('Potassium')
        temp = request.form.get('Temperature')
        humidity = request.form.get('Humidity')
        ph = request.form.get('pH')
        rainfall = request.form.get('Rainfall')

        inputs_available = all([N, P, K, temp, humidity, ph, rainfall])

        if inputs_available:
            # convert to floats
            N = float(N)
            P = float(P)
            K = float(K)
            temp = float(temp)
            humidity = float(humidity)
            ph = float(ph)
            rainfall = float(rainfall)

            # Validations
            valid_ranges = {
                "Nitrogen": (0, 200),
                "Phosphorus": (0, 200),
                "Potassium": (0, 250),
                "Temperature": (7, 50),
                "Humidity": (10, 100),
                "pH": (3, 12),
                "Rainfall": (10, 300)
            }
            if not (valid_ranges["Nitrogen"][0] <= N <= valid_ranges["Nitrogen"][1] and
                    valid_ranges["Phosphorus"][0] <= P <= valid_ranges["Phosphorus"][1] and
                    valid_ranges["Potassium"][0] <= K <= valid_ranges["Potassium"][1] and
                    valid_ranges["Temperature"][0] <= temp <= valid_ranges["Temperature"][1] and
                    valid_ranges["Humidity"][0] <= humidity <= valid_ranges["Humidity"][1] and
                    valid_ranges["pH"][0] <= ph <= valid_ranges["pH"][1] and
                    valid_ranges["Rainfall"][0] <= rainfall <= valid_ranges["Rainfall"][1]):
                return render_template('index.html', result="Invalid input: No suitable crop found for these conditions.")

            # Predict using Random Forest
            features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
            features_scaled = ms.transform(features)
            prediction = rfc_model.predict(features_scaled)[0]
            crop_name = crop_dict.get(prediction, "Unknown Crop")

            if soil_type:
                result = f"{crop_name.title()}"
            else:
                result = f"{crop_name.title()}"

        elif soil_type:
            # Only image given
            crops = soil_crop_mapping.get(soil_type, ["No data available"])
            result = f"🧱 Predicted Soil Type: {soil_type}<br>🌾 Compatible Crops: {', '.join(crops)}"
        else:
            result = "⚠️ Please upload an image or enter valid parameters!"

    except Exception as e:
        result = f"An error occurred: {str(e)}"

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
