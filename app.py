import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = 'models/video_classifier_model.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'

model = None
label_encoder = None

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        print("Model and label encoder loaded successfully.")
    except Exception as e:
        print(f"Error loading model or encoder: {e}")
else:
    print("Model or label encoder files not found. Please train the model.")


def predict_gender(image_path):
    if model is None or label_encoder is None:
        return "❌ Model not found. Please train the model first."

    try:
        image = cv2.imread(image_path)
        if image is not None:
            image_resized = cv2.resize(image, (64, 64)).flatten().reshape(1, -1)
            prediction = model.predict(image_resized)
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            return predicted_label
        else:
            return "❌ Image could not be loaded."
    except Exception as e:
        return f"❌ An error occurred during prediction: {e}"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                predicted_gender = predict_gender(file_path)

                return render_template("index.html", image=filename, result=predicted_gender)
            except Exception as e:
                return f"An error occurred: {e}"

    return render_template("index.html", image=None, result=None)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)