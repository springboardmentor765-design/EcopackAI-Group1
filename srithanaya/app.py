from flask import Flask, render_template, request, send_file
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import os

# PDF Libraries
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# ================= DATABASE CONNECTION =================
engine = create_engine("postgresql://postgres:abcd@localhost/ecopack_db")

# Load tables
materials = pd.read_sql("SELECT * FROM materials", engine)
products = pd.read_sql("SELECT * FROM products", engine)

# ================= HOME PAGE =================
@app.route("/")
def home():
    categories = products["category"].unique()
    return render_template("index.html", categories=categories)

# ================= RECOMMENDATION PAGE =================
@app.route("/recommend", methods=["POST"])
def recommend():
    category = request.form["category"]
    weight = float(request.form["weight"])

    # Convert numeric columns safely
    materials["co2_emission_score"] = pd.to_numeric(materials["co2_emission_score"], errors="coerce")
    materials["cost_per_unit"] = pd.to_numeric(materials["cost_per_unit"], errors="coerce")

    # Sustainability Score Formula (Mentor Friendly)
    materials["score"] = 100 / (materials["co2_emission_score"] + materials["cost_per_unit"])

    # Group by material type
    mat_avg = materials.groupby("material_type")[["score"]].mean().reset_index()

    # Top 3 Sustainable Materials
    top3 = mat_avg.sort_values("score", ascending=False).head(3)

    # Create static exports folder
    os.makedirs("static/exports", exist_ok=True)

    # BAR CHART
    plt.figure()
    plt.bar(top3["material_type"], top3["score"])
    plt.title("Top 3 Sustainable Materials Score")
    plt.xlabel("Material")
    plt.ylabel("Score")
    plt.savefig("static/exports/bar.png")
    plt.close()

    # PIE CHART
    plt.figure()
    plt.pie(top3["score"], labels=top3["material_type"], autopct="%1.1f%%")
    plt.title("Material Sustainability Distribution")
    plt.savefig("static/exports/pie.png")
    plt.close()

    return render_template("dashboard.html", recs=top3.to_dict(orient="records"))

# ================= DASHBOARD TEST PAGE =================
@app.route("/dashboard")
def dashboard():
    recs = [
        {"material_type": "Paper", "score": 85},
        {"material_type": "Bioplastic", "score": 78},
        {"material_type": "Glass", "score": 65}
    ]
    return render_template("dashboard.html", recs=recs)

# ================= EXPORT PDF =================
@app.route("/export_pdf")
def export_pdf():
    os.makedirs("exports", exist_ok=True)
    file_path = "exports/EcoPackAI_Report.pdf"

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    elements = []

    # Title
    elements.append(Paragraph("EcoPackAI Sustainable Packaging Recommendation Report", styles["Heading1"]))
    elements.append(Spacer(1, 12))

    # Add Recommended Materials
    elements.append(Paragraph("Top Recommended Materials:", styles["Heading2"]))

    top_materials = materials["material_type"].unique()[:3]
    for mat in top_materials:
        elements.append(Paragraph(f"- {mat}", styles["Normal"]))

    elements.append(Spacer(1, 12))

    # Add Charts
    if os.path.exists("static/exports/bar.png"):
        elements.append(Paragraph("Bar Chart:", styles["Heading2"]))
        elements.append(Image("static/exports/bar.png", width=400, height=250))

    if os.path.exists("static/exports/pie.png"):
        elements.append(Paragraph("Pie Chart:", styles["Heading2"]))
        elements.append(Image("static/exports/pie.png", width=400, height=250))

    doc.build(elements)

    return send_file(file_path, as_attachment=True)

# ================= RUN APP =================
if __name__ == "__main__":
    app.run(debug=False)
