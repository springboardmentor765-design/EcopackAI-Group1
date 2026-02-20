from flask import Flask, request, jsonify, render_template
import pandas as pd
import matplotlib
import warnings

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import joblib

from flask import send_file  

from fpdf import FPDF
import os


plt.rcParams.update({'figure.max_open_warning': 0})

app = Flask(__name__)



# Load dataset and feature engineering


material_df = pd.read_csv("finalmaterialsdataset.csv")


material_df['Co2_impact_index'] = material_df['CO2_Emission_Score'] * (1 - material_df['Recyclability_Percent'] / 100)

material_df["Cost_Normalized"] = (

    (material_df["Cost_per_unit"] - material_df["Cost_per_unit"].min()) /

    (material_df["Cost_per_unit"].max() - material_df["Cost_per_unit"].min())
)

material_df["Cost_Efficiency_Index"] = 1 - material_df["Cost_Normalized"]


FEATURES = [

    "Tensile_Strength_MPa",

    "Weight_Capacity_kg",

    "Moisture_Barrier_Grade",

    "Biodegradability_Score",

    "Recyclability_Percent"

]



# Load pre-trained models


cost_model = joblib.load("cost_model.pkl")

co2_model = joblib.load("co2_model.pkl")


# Ensure compatibility with different xgboost/sklearn versions: some saved

# XGBModel objects may be missing attributes that sklearn's `get_params`

# expects (e.g. `gpu_id`). Add safe defaults to avoid AttributeError during

# prediction calls.

def _ensure_xgb_compat(model):

    try:

        # Only adjust objects that look like XGBoost estimators
        name = model.__class__.__name__

        if name.startswith("XGB") or "xgboost" in str(model.__class__).lower():

            if not hasattr(model, "gpu_id"):

                setattr(model, "gpu_id", None)

            if not hasattr(model, "tree_method"):

                setattr(model, "tree_method", None)

            # newer/older xgboost wrappers may expect a `predictor` attribute

            if not hasattr(model, "predictor"):

                setattr(model, "predictor", None)

            # sklearn compatibility

            if not hasattr(model, "n_jobs") and hasattr(model, "nthread"):

                try:

                    setattr(model, "n_jobs", getattr(model, "nthread"))

                except Exception:

                    setattr(model, "n_jobs", None)

    except Exception:
        pass


_ensure_xgb_compat(cost_model)

_ensure_xgb_compat(co2_model)



# Routes


@app.route("/")

def home():

    # Serve frontend HTML

    return render_template("index.html")


@app.route("/recommend", methods=["POST"])

def recommend():

    try:

        data = request.get_json()

        product_weight_kg = data.get("product_weight_kg", 0)


        # Strict feasibility: only materials that can handle product weight

        feasable_material = (material_df["Weight_Capacity_kg"] >= product_weight_kg)

        feasable_material_df = material_df[feasable_material].copy()


        # If no material can fully support the weight, fall back to best-effort

        fallback_message = None

        if feasable_material_df.empty:

            fallback_message = (

                f"No material has Weight_Capacity_kg >= {product_weight_kg} kg. "

                "Returning best-effort recommendations (materials will be overloaded)."
            )

            feasable_material_df = material_df.copy()


        # Predictions

        # Ensure correct feature order and column names
        x = feasable_material_df[FEATURES].copy()

        # Make absolutely sure columns match model training order
        x = x[FEATURES]

        feasable_material_df["Co2_impact_index_pred"] = co2_model.predict(x)
        feasable_material_df["cost_efficiency_pred"] = cost_model.predict(x)


        # Capacity utilization

        feasable_material_df["capacity_utilization"] = product_weight_kg / feasable_material_df["Weight_Capacity_kg"]


        # Safe normalization helper

        def _safe_normalize(series):

            denom = series.max() - series.min()

            if denom == 0 or pd.isna(denom):

                return pd.Series(0.5, index=series.index)

            return (series - series.min()) / denom


        # Normalization (guard against zero-division when all values equal)

        feasable_material_df["co2_norm"] = _safe_normalize(feasable_material_df["Co2_impact_index_pred"])

        feasable_material_df["cost_norm"] = _safe_normalize(feasable_material_df["cost_efficiency_pred"])

        feasable_material_df["util_norm"] = _safe_normalize(feasable_material_df["capacity_utilization"])


        # Suitability score

        feasable_material_df["suitability_score"] = (

            (0.5 * (1 - feasable_material_df["co2_norm"])) +

            (0.4 * (1 - feasable_material_df["cost_norm"])) +

            (0.1 * feasable_material_df["util_norm"])
        )


        # Final ranking
        top_materials = (

            feasable_material_df.sort_values(by="suitability_score", ascending=False)

            .drop_duplicates(subset="Material_Type", keep="first")

            .head()
        )


        result = top_materials[[

            "Material_Type",

            "Co2_impact_index_pred",

            "cost_efficiency_pred",

            "suitability_score"

        ]].to_dict(orient="records")

        resp = {

            "product_input": data,

            "recommendations": result

        }

        if fallback_message:

            resp["message"] = fallback_message

        return jsonify(resp)

    except Exception as e:

        import traceback

        traceback.print_exc()

        return jsonify({"error": "prediction_failed", "message": str(e)}), 500
        

def generate_charts(results):
    base_path = os.path.dirname(__file__)

    materials = [r["Material_Type"] for r in results]
    cost = [r["cost_efficiency_pred"] for r in results]
    co2 = [r["Co2_impact_index_pred"] for r in results]
    suitability = [r["suitability_score"] for r in results]

    # ---------- Cost Efficiency ----------
    plt.figure(figsize=(8, 4))
    plt.bar(materials, cost, color="green")
    plt.title("Cost Efficiency")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(base_path, "cost_chart.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    # ---------- CO2 Impact ----------
    plt.figure(figsize=(8, 4))
    plt.plot(materials, co2, marker="o", color="red")
    plt.title("CO₂ Impact")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(base_path, "co2_chart.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()

    # ---------- Suitability ----------
    plt.figure(figsize=(8, 4))
    plt.bar(materials, suitability, color="orange")
    plt.title("Suitability Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(base_path, "suitability_chart.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()


@app.route("/download", methods=["POST"])
def download_report():
    data = request.get_json()

    if not data or "recommendations" not in data:
        return jsonify({
            "error": "No data received",
            "message": "Please generate recommendations first."
        }), 400

    results = data["recommendations"]

    base_path = os.path.dirname(os.path.abspath(__file__))

    # Generate charts
    generate_charts(results)

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "EcoPackAI Recommendation Report", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Top 5 Recommended Packaging Materials:", ln=True)
    pdf.ln(3)

    pdf.set_font("Arial", "B", 11)
    pdf.cell(45, 8, "Material", 1)
    pdf.cell(40, 8, "Cost Efficiency", 1)
    pdf.cell(35, 8, "CO2 Impact", 1)
    pdf.cell(35, 8, "Suitability", 1)
    pdf.ln()

    pdf.set_font("Helvetica", size=11)
    for row in results:
        pdf.cell(45, 8, row["Material_Type"], 1)
        pdf.cell(40, 8, f'{row["cost_efficiency_pred"]:.2f}', 1)
        pdf.cell(35, 8, f'{row["Co2_impact_index_pred"]:.2f}', 1)
        pdf.cell(35, 8, f'{row["suitability_score"]:.2f}', 1)
        pdf.ln()

    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Analytics & Visual Insights", ln=True)
    pdf.ln(5)

    pdf.image(os.path.join(base_path, "cost_chart.png"), x=10, y=30, w=190)
    pdf.add_page()
    pdf.image(os.path.join(base_path, "co2_chart.png"), x=10, y=30, w=190)
    pdf.add_page()
    pdf.image(os.path.join(base_path, "suitability_chart.png"), x=10, y=30, w=190)

    file_path = os.path.join(base_path, "EcoPackAI_Report.pdf")
    pdf.output(file_path)

    return send_file(file_path, as_attachment=True)

# Run app


if __name__ == "__main__":

    app.run(host="0.0.0.0",port=5000)