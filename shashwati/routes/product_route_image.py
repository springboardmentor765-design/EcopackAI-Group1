from flask import Blueprint, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

product_bp = Blueprint("product", __name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "Models", "product_classifier.h5")

model = load_model(model_path)

CLASS_NAMES = [
    "BalletFlat", "Belt", "Blazer", "Bracelet", "Coat",
    "Dress", "Earrings", "FlipFlops", "HandbagLuggage",
    "Hats"
]

@product_bp.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Product Image API is running"})

@product_bp.route("/predict", methods=["POST"])
def predict_product():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    img_path = "temp.jpg"
    img_file.save(img_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    os.remove(img_path)

    return jsonify({
        "product_type": predicted_class
    })
