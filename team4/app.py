from flask import Flask, render_template, request, jsonify
from utils.model import model_manager
from utils.db import get_db_connection
import os

app = Flask(__name__)

# --- Routes ---

@app.route("/", methods=["GET"])
def home():
    """Renders the landing page."""
    return render_template('home.html')

@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Renders the main frontend dashboard."""
    return render_template('index.html')

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    API Endpoint to get material recommendations.
    Expected JSON: { "weight_capacity": 5.0, "category": "Food", "fragility_score": 8, ... }
    """
    try:
        user_input = request.get_json()
        
        # Determine recommendations based on physics/stats
        recommendations = model_manager.predict_all(user_input)
        
        return jsonify({
            "status": "success",
            "recommendations": recommendations,
            "input_summary": user_input # Echo back for UI verification
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/materials", methods=["GET"])
def get_materials():
    """Returns list of available materials from the model."""
    try:
        materials = model_manager.get_available_materials()
        return jsonify({"status": "success", "materials": materials})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/db-check", methods=["GET"])
def check_db():
    """Utility to check DB connection."""
    conn = get_db_connection()
    if conn:
        conn.close()
        return jsonify({"status": "connected", "message": "PostgreSQL connection successful."})
    else:
        return jsonify({"status": "error", "message": "Could not connect to database."}), 503

if __name__ == "__main__":
    app.run(debug=True, port=5000)
