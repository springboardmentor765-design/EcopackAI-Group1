from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=14)

pdf.cell(0, 10, "EcoPackAI Recommendation Report", ln=True)
pdf.ln(5)
pdf.set_font("Arial", size=12)

pdf.multi_cell(0, 8,
"""This report contains AI-based recommendations
for sustainable packaging materials.

Top 5 materials are selected based on:
- Cost efficiency
- CO2 impact
- Suitability score
""")

pdf.output("EcoPackAI_Report.pdf")
