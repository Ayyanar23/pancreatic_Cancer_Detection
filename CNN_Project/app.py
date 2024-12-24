from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Path to your model
model_path = "p:/Anusha/Images/pancreatic_cancer_cnn.h5"

# Open the model file to check the Keras version
with h5py.File(model_path, "r") as f:
    config = f.attrs.get("keras_version")
    print(f"Keras Version in File: {config}")

# Load the model
model = load_model(model_path)
print("Model loaded successfully!")

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def home():
    results = []  # List to store results for all uploaded files

    if request.method == "POST":
        if "files" not in request.files:
            return render_template("index.html", error="No files uploaded")
        
        files = request.files.getlist("files")
        if not files or files[0].filename == "":
            return render_template("index.html", error="No files selected")

        for file in files:
            try:
                # Save the file to the uploads directory
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)

                # Preprocess the image
                image = Image.open(file_path).convert("RGB").resize((128, 128))
                image_array = np.array(image) / 255.0  # Normalize
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
                
                # Make a prediction
                prediction = model.predict(image_array)
                prediction_result = "Positive for Pancreatic Cancer" if prediction[0][0] > 0.5 else "Negative for Pancreatic Cancer"

                # Store the result for this file
                results.append({
                    "filename": file.filename,
                    "prediction": prediction_result,
                    "image_path": file_path
                })

            except Exception as e:
                return render_template("index.html", error=f"Error processing {file.filename}: {str(e)}")

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
