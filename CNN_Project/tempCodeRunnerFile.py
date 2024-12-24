from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from PIL import Image
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

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
    prediction_result = None
    prediction_accuracy = None
    image_path = None
    visualization_path = None
    confusion_matrix_path = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")
        
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")
        
        # Save the file to the uploads directory
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        image_path = file_path  # Store the path for displaying the image

        # Preprocess the image
        try:
            image = Image.open(file_path).convert("RGB").resize((128, 128))
            image_array = np.array(image) / 255.0  # Normalize
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            
            # Make a prediction
            prediction = model.predict(image_array)
            prediction_result = "Positive for Pancreatic Cancer" if prediction[0][0] > 0.5 else "Negative for Pancreatic Cancer"

            # Calculate prediction accuracy
            # Assuming you have ground truth (for example purposes, 0.75 accuracy used)
            y_true = np.array([0])  # Placeholder for true label
            y_pred = np.array([1 if prediction[0][0] > 0.5 else 0])
            prediction_accuracy = accuracy_score(y_true, y_pred)

            # Create a Matplotlib confusion matrix plot with a mappable object
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            plt.xticks([0, 1])
            plt.yticks([0, 1])
            plt.colorbar(cax)  # Colorbar for the confusion matrix
            confusion_matrix_path = os.path.join(UPLOAD_FOLDER, "confusion_matrix.png")
            plt.savefig(confusion_matrix_path)
            plt.close()

            # Create a visualization of the image with prediction result
            plt.figure(figsize=(6, 4))
            plt.imshow(image)
            plt.title(f"Prediction: {prediction_result}")
            plt.axis('off')
            visualization_path = os.path.join(UPLOAD_FOLDER, "visualization.png")
            plt.savefig(visualization_path)
            plt.close()

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html", 
                           prediction=prediction_result, 
                           prediction_accuracy=prediction_accuracy, 
                           image_path=image_path, 
                           visualization_path=visualization_path, 
                           confusion_matrix_path=confusion_matrix_path)

if __name__ == "__main__":
    app.run(debug=True)
