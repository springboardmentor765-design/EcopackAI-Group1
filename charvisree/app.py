from flask import Flask, jsonify, request, render_template, send_file
from db_config import get_db_connection
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import io
from openpyxl import Workbook
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from flask import send_file
import io
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import pagesizes


app = Flask(__name__)

API_TOKEN = "ecopackai123"

CATEGORY_MAP = {
    1: "food",
    2: "pharmaceutical",
    3: "cosmetics",
    4: "electronics",
    5: "apparel",
    6: "industrial",
    7: "automotive",
    8: "luxury",
    9: "agriculture",
    10: "logistics"
}

# ===============================
# LOAD ML ASSETS
# ===============================
model = joblib.load("material_recommender_model.joblib")
scaler = joblib.load("ml_feature_scaler.joblib")
encoder = joblib.load("category_encoder.joblib")
materials = pd.read_csv("data/materials_ml_clean.csv")

NUMERIC_FEATURE_ORDER = [
    "product_weight_g",
    "product_volume_cm3",
    "fragility_level",
    "moisture_sensitivity",
    "temperature_sensitivity",
    "shelf_life_days",
    "price_inr",
    "strength_mpa",
    "weight_capacity_kg",
    "moisture_barrier",
    "temp_resistance",
    "rigidity",
    "biodegradability_score",
    "recyclability_pct",
    "load_ratio",
    "environmental_pressure",
    "protection_score",
    "sustainability_index",
    "shelf_life_stress"
]

# ===============================
# HOME
# ===============================
@app.route("/")
def home():
    return render_template("index.html", categories=CATEGORY_MAP)


# ===============================
# RECOMMENDATION ENGINE
# ===============================
@app.route("/recommend", methods=["POST"])
def recommend_material():

    if request.headers.get("X-From-UI") != "true":
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    data = request.get_json()

    try:
        required = [
            "product_id", "category_id", "weight_g", "volume_cm3",
            "fragility", "moisture_sensitivity",
            "temperature_sensitivity", "shelf_life_days", "price_inr"
        ]

        for f in required:
            if f not in data:
                return jsonify({"status": "error", "message": f"{f} required"}), 400

        if data["category_id"] not in CATEGORY_MAP:
            return jsonify({"status": "error", "message": "Invalid category"}), 400

        # Safe conversion
        weight_g = float(data["weight_g"])
        volume_cm3 = float(data["volume_cm3"])
        fragility = float(data["fragility"])
        moisture = float(data["moisture_sensitivity"])
        temperature = float(data["temperature_sensitivity"])
        shelf_life = float(data["shelf_life_days"])
        price = float(data["price_inr"])
        weight_kg = weight_g / 1000.0

        # Encode category
        cat_text = CATEGORY_MAP[data["category_id"]]
        cat_enc = int(encoder.transform([cat_text])[0])

        # Filter materials by weight
        filtered = materials[
            materials["weight_capacity_kg"] * 1000 >= weight_g
        ].copy()

        if filtered.empty:
            return jsonify({"status": "error", "message": "No feasible material found"})

        rows = []

        for _, m in filtered.iterrows():

            capacity = max(float(m["weight_capacity_kg"]), 0.0001)
            load_ratio = weight_kg / capacity

            environmental_pressure = (
                moisture * float(m["moisture_barrier"]) +
                temperature * float(m["temp_resistance"])
            )

            protection_score = fragility * float(m["rigidity"])

            sustainability_index = (
                float(m["biodegradability_score"]) * 0.6 +
                float(m["recyclability_pct"]) * 0.4
            )

            shelf_life_stress = shelf_life / 365.0

            rows.append([
                weight_g,
                volume_cm3,
                fragility,
                moisture,
                temperature,
                shelf_life,
                price,
                float(m["strength_mpa"]),
                float(m["weight_capacity_kg"]),
                float(m["moisture_barrier"]),
                float(m["temp_resistance"]),
                float(m["rigidity"]),
                float(m["biodegradability_score"]),
                float(m["recyclability_pct"]),
                load_ratio,
                environmental_pressure,
                protection_score,
                sustainability_index,
                shelf_life_stress
            ])

        # ML Prediction
        X = pd.DataFrame(rows, columns=NUMERIC_FEATURE_ORDER)
        X_scaled = scaler.transform(X)
        X_final = np.hstack([np.full((X_scaled.shape[0], 1), cat_enc), X_scaled])
        probs = model.predict_proba(X_final)[:, 1]

        # Baseline (LDPE Plastic)
        baseline_row = materials[
            materials["material_type"] == "LDPE Plastic"
        ]

        baseline = baseline_row.iloc[0] if not baseline_row.empty else filtered.iloc[0]

        baseline_cost = weight_kg * float(baseline["cost_inr_per_kg"])
        baseline_co2 = weight_kg * float(baseline["co2_emission_per_kg"])

        results = []

        for i, (_, m) in enumerate(filtered.iterrows()):

            total_cost = weight_kg * float(m["cost_inr_per_kg"])
            total_co2 = weight_kg * float(m["co2_emission_per_kg"])

            co2_reduction = 0 if baseline_co2 == 0 else ((baseline_co2 - total_co2) / baseline_co2) * 100
            cost_savings = 0 if baseline_cost == 0 else ((baseline_cost - total_cost) / baseline_cost) * 100

            co2_reduction = float(max(min(co2_reduction, 60), -50))
            cost_savings = float(max(min(cost_savings, 60), -50))

            raw_prob = float(probs[i])
            adjusted_prob = float(0.6 * raw_prob + 0.2)

            results.append({
                "material_type": str(m["material_type"]),
                "Suitability_Prob": round(adjusted_prob, 4),

                "cost_inr_per_kg": float(m["cost_inr_per_kg"]),
                "co2_emission_per_kg": float(m["co2_emission_per_kg"]),
                "recyclability_pct": float(m["recyclability_pct"]),
                "biodegradability_score": float(m["biodegradability_score"]),

                "total_cost_inr": round(total_cost, 2),
                "total_co2_kg": round(total_co2, 3),
                "co2_reduction_pct": round(co2_reduction, 2),
                "cost_savings_pct": round(cost_savings, 2)
            })

        results = sorted(results, key=lambda x: x["Suitability_Prob"], reverse=True)[:5]

        # Save to DB
        conn = get_db_connection()
        cur = conn.cursor()
        import time

        for r in results:
            product_id = int(time.time() * 1000)
            cur.execute("""
                INSERT INTO recommendations (
                    product_id, material_type, suitability_prob,
                    total_cost_inr, total_co2_kg,
                    co2_reduction_pct, cost_savings_pct,
                    created_at
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                product_id,
                r["material_type"],
                r["Suitability_Prob"],
                r["total_cost_inr"],
                r["total_co2_kg"],
                r["co2_reduction_pct"],
                r["cost_savings_pct"],
                datetime.now()
            ))

        conn.commit()
        cur.close()
        conn.close()

        return jsonify({
            "status": "success",
            "recommended_material": results[0]["material_type"],
            "recommendations": results
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ===============================
# DASHBOARD DATA API
# ===============================
@app.route("/api/dashboard-data")
def dashboard_data():
    conn = get_db_connection()
    cur = conn.cursor()

    # Total recommendations
    cur.execute("SELECT COUNT(*) FROM recommendations")
    total = cur.fetchone()[0]

    # Average CO2 reduction
    cur.execute("SELECT AVG(co2_reduction_pct) FROM recommendations")
    avg_co2 = cur.fetchone()[0] or 0

    # Average cost savings
    cur.execute("SELECT AVG(cost_savings_pct) FROM recommendations")
    avg_cost = cur.fetchone()[0] or 0

    # Material usage distribution
    cur.execute("""
        SELECT material_type, COUNT(*) 
        FROM recommendations 
        GROUP BY material_type
    """)
    materials = cur.fetchall()

    # Monthly growth
    cur.execute("""
        SELECT DATE_FORMAT(created_at, '%Y-%m') AS month,
               COUNT(*) 
        FROM recommendations
        GROUP BY month
        ORDER BY month
    """)
    monthly = cur.fetchall()

    cur.close()
    conn.close()

    return jsonify({
        "total": total,
        "avg_co2": round(avg_co2, 2),
        "avg_cost": round(avg_cost, 2),
        "materials": materials,
        "monthly": monthly
    })


# ===============================
# EXPORT EXCEL
# ===============================
@app.route("/export/excel")
def export_excel():

    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM recommendations", conn)
    conn.close()

    output = io.BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="EcoPackAI_Sustainability_Report.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ===============================
# EXPORT PDF
# ===============================
@app.route("/export/pdf")
def export_pdf():

    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM recommendations", conn)
    conn.close()

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, height - 50, "EcoPackAI Sustainability Report")

    pdf.setFont("Helvetica", 10)
    y = height - 80

    for _, row in df.iterrows():
        line = f"{row['material_type']} | CO2 Red: {row['co2_reduction_pct']}% | Cost Save: {row['cost_savings_pct']}%"
        pdf.drawString(50, y, line)
        y -= 15
        if y < 40:
            pdf.showPage()
            pdf.setFont("Helvetica", 10)
            y = height - 50

    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="EcoPackAI_Sustainability_Report.pdf",
        mimetype="application/pdf"
    )


if __name__ == "__main__":
    app.run()
