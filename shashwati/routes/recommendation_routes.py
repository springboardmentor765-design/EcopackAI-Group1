from flask import Blueprint, request, jsonify
import pandas as pd
import joblib
import os

recommend_bp = Blueprint('recommend', __name__)

# Model feature columns
feature_cols = [
    "tensile_strength_mpa",
    "weight_capacity_kg",
    "biodegradability_score",
    "recyclability_percent",
    "moisture_barrier_grade",
    "heat_resistance_c",
    "flexibility_score",
    "derived_material_co2_factor",
    "material_type_Bamboo Fiber",
    "material_type_Bioplastic (PLA)",
    "material_type_Mushroom Mycelium",
    "material_type_Recycled Cardboard"
]

# Initialize models as None
cost_model = None
co2_model = None

def load_models():
    """Load models only when needed (lazy loading)"""
    global cost_model, co2_model
    if cost_model is None or co2_model is None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        cost_model = joblib.load(os.path.join(BASE_DIR, 'cost_model.pkl'))
        co2_model = joblib.load(os.path.join(BASE_DIR, 'co2_model.pkl'))
        print("Models loaded successfully")  # For Railway logs

@recommend_bp.route("/api", methods=["GET", "POST"])
def recommend_material():
    load_models()  # Ensure models are loaded before prediction

    if request.method == "GET":
        return jsonify({"message": "Send a POST request with JSON data to get predictions!"})

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        materials = [
            "Bamboo Fiber",
            "Bioplastic (PLA)",
            "Mushroom Mycelium",
            "Recycled Cardboard"
        ]

        results = []

        for material in materials:
            mapped_data = {
                "tensile_strength_mpa": data.get("weight", 0) * 5,
                "weight_capacity_kg": data.get("weight", 0),
                "biodegradability_score": 8,
                "recyclability_percent": 80,
                "moisture_barrier_grade": 5,
                "heat_resistance_c": 100,
                "flexibility_score": 1 if data.get("fragile", 0) else 5,
                "derived_material_co2_factor": 2.5,
            }

            # One-hot encoding
            for col in feature_cols:
                if col.startswith("material_type_"):
                    mapped_data[col] = 1 if col.endswith(material) else 0

            df = pd.DataFrame([{col: mapped_data.get(col, 0) for col in feature_cols}])

            cost_pred = float(cost_model.predict(df)[0])
            co2_pred = float(co2_model.predict(df)[0])

            results.append({
                "material": material,
                "predicted_cost": cost_pred,
                "predicted_co2": co2_pred
            })

        best_material = min(results, key=lambda x: x["predicted_co2"])

        return jsonify({
            "recommendation": best_material,
            "all_predictions": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@recommend_bp.route("/", methods=["GET"])
def index():
    return "Material Recommendation API is running."