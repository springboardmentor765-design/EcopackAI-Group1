// ======================================
// SECTION SWITCHING
// ======================================
function showSection(section) {
    document.querySelectorAll(".section")
        .forEach(sec => sec.classList.add("d-none"));

    document.getElementById(section + "Section")
        .classList.remove("d-none");
}


// ======================================
// STORE LAST RECOMMENDATION (NEW - ADDED)
// ======================================
let lastRecommendationData = null;


// ======================================
// RECOMMENDATION FORM
// ======================================
const ecoForm = document.getElementById("ecoForm");

if (ecoForm) {

    ecoForm.addEventListener("submit", async function (e) {
        e.preventDefault();

        const spinner = document.getElementById("loadingSpinner");
        const bestDiv = document.getElementById("bestResult");
        const comparisonSection = document.getElementById("comparisonSection");

        spinner.classList.remove("d-none");
        bestDiv.classList.add("d-none");
        comparisonSection.classList.add("d-none");

        const payload = {
            product_id: Date.now(),
            category_id: parseInt(document.getElementById("category_id").value),
            weight_g: parseFloat(document.getElementById("weight_g").value),
            volume_cm3: parseFloat(document.getElementById("volume_cm3").value),
            fragility: parseInt(document.getElementById("fragility").value),
            moisture_sensitivity: parseInt(document.getElementById("moisture").value),
            temperature_sensitivity: parseInt(document.getElementById("temperature").value),
            shelf_life_days: parseInt(document.getElementById("shelf_life").value),
            price_inr: parseFloat(document.getElementById("price").value)
        };

        try {
            const res = await fetch("/recommend", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-From-UI": "true"
                },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            spinner.classList.add("d-none");

            if (data.status !== "success") {
                alert(data.message || "Recommendation failed");
                return;
            }

            lastRecommendationData = data.recommendations;

            displayRecommendation(data);

        } catch (err) {
            spinner.classList.add("d-none");
            console.error(err);
            alert("Server error. Please try again.");
        }
    });
}


// ======================================
// DISPLAY RECOMMENDATION RESULTS
// ======================================
function displayRecommendation(data) {

    const bestDiv = document.getElementById("bestResult");
    bestDiv.classList.remove("d-none");

    const bestMaterial = data.recommended_material;
    const bestObj = data.recommendations.find(
        r => r.material_type === bestMaterial
    );

    const probPercent = (bestObj.Suitability_Prob * 100).toFixed(1);

    let confidenceLevel = "";
    if (probPercent >= 80) confidenceLevel = "Very High Confidence";
    else if (probPercent >= 65) confidenceLevel = "High Confidence";
    else if (probPercent >= 50) confidenceLevel = "Moderate Confidence";
    else confidenceLevel = "Low Confidence";

    let reasons = [];

    if (bestObj.co2_reduction_pct >= 40)
        reasons.push("Significant CO₂ reduction");

    if (bestObj.cost_savings_pct >= 30)
        reasons.push("Strong cost savings");

    if (bestObj.recyclability_pct >= 80)
        reasons.push("Highly recyclable");

    if (bestObj.biodegradability_score >= 8)
        reasons.push("High biodegradability");

    if (reasons.length === 0)
        reasons.push("Balanced sustainability and cost performance");

    const reasonText = reasons.join(", ");

    bestDiv.innerHTML = `
        🏆 Best Recommended Material: 
        <b>${bestMaterial}</b>
        <br>
        📊 Model Confidence: <b>${probPercent}%</b> (${confidenceLevel})
        <div class="progress mt-2" style="height: 8px;">
            <div class="progress-bar bg-success"
                 style="width: ${probPercent}%">
            </div>
        </div>
        <br>
        💡 Why this material? <b>${reasonText}</b>
    `;

    const tableBody = document.getElementById("comparisonTableBody");
    tableBody.innerHTML = "";

    data.recommendations.forEach((r, index) => {

        const suitabilityPercent = (r.Suitability_Prob * 100).toFixed(1);

        const highlightStyle = r.material_type === bestMaterial
            ? 'style="background-color:#e8f5e9;font-weight:bold;"'
            : '';

        const row = `
            <tr ${highlightStyle}>
                <td>${index + 1}</td>
                <td>${r.material_type}</td>
                <td>${suitabilityPercent}%</td>
                <td>${r.cost_inr_per_kg ?? "-"}</td>
                <td>${r.co2_emission_per_kg ?? "-"}</td>
                <td>${r.recyclability_pct ?? "-"}</td>
                <td>${r.biodegradability_score ?? "-"}</td>
                <td>₹${r.total_cost_inr}</td>
                <td>${r.total_co2_kg} kg</td>
                <td>${r.co2_reduction_pct}%</td>
                <td>${r.cost_savings_pct}%</td>
            </tr>
        `;

        tableBody.innerHTML += row;
    });

    document.getElementById("comparisonSection")
        .classList.remove("d-none");
}


// ======================================
// DASHBOARD ADVANCED VERSION
// ======================================

let suitabilityChart = null;
let co2Chart = null;
let costChart = null;
let radarChart = null;

async function loadDashboard() {

    if (!lastRecommendationData) {
    alert("Generate recommendation first.");
    return;
}

// ==============================
// FILL DASHBOARD TABLE
// ==============================
const dashTable = document.getElementById("dashboardTableBody");
dashTable.innerHTML = "";

lastRecommendationData.forEach((r, index) => {

    dashTable.innerHTML += `
        <tr>
            <td>${index + 1}</td>
            <td>${r.material_type}</td>
            <td>${(r.Suitability_Prob * 100).toFixed(1)}</td>
            <td>${r.cost_inr_per_kg}</td>
            <td>${r.co2_emission_per_kg}</td>
            <td>${r.recyclability_pct}</td>
            <td>${r.biodegradability_score}</td>
            <td>${r.total_cost_inr}</td>
            <td>${r.total_co2_kg}</td>
            <td>${r.co2_reduction_pct}</td>
            <td>${r.cost_savings_pct}</td>
        </tr>
    `;
});

const labels = lastRecommendationData.map(m => m.material_type);
const suitability = lastRecommendationData.map(m => m.Suitability_Prob * 100);
const co2Reduction = lastRecommendationData.map(m => m.co2_reduction_pct);
const costSavings = lastRecommendationData.map(m => m.cost_savings_pct);
const totalCost = lastRecommendationData.map(m => m.total_cost_inr);
const totalCO2 = lastRecommendationData.map(m => m.total_co2_kg);
const recyclability = lastRecommendationData.map(m => m.recyclability_pct);
const biodegradability = lastRecommendationData.map(m => m.biodegradability_score);

// Destroy previous charts if exist
if (suitabilityChart) suitabilityChart.destroy();
if (co2Chart) co2Chart.destroy();
if (costChart) costChart.destroy();

// ==============================
// 1️⃣ Suitability Horizontal
// ==============================
suitabilityChart = new Chart(
    document.getElementById("chartSuitability"),
    {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Suitability (%)",
                data: suitability,
                backgroundColor: "#198754"
            }]
        },
        options: {
            indexAxis: "y",
            plugins: {
                tooltip: {
                    callbacks: {
                        label: ctx => ctx.raw.toFixed(1) + "%"
                    }
                }
            }
        }
    }
);

// ==============================
// 2️⃣ CO₂ Reduction
// ==============================
co2Chart = new Chart(
    document.getElementById("chartCO2Reduction"),
    {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "CO₂ Reduction (%)",
                data: co2Reduction,
                backgroundColor: "#0d6efd"
            }]
        }
    }
);

// ==============================
// 3️⃣ Cost Savings Line
// ==============================
costChart = new Chart(
    document.getElementById("chartCostSavings"),
    {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Cost Savings (%)",
                data: costSavings,
                borderColor: "#ffc107",
                fill: false,
                tension: 0.3
            }]
        }
    }
);

// ==============================
// 4️⃣ Total Cost
// ==============================
new Chart(
    document.getElementById("chartTotalCost"),
    {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Total Cost (₹)",
                data: totalCost,
                backgroundColor: "#6f42c1"
            }]
        }
    }
);

// ==============================
// 5️⃣ Total CO₂
// ==============================
new Chart(
    document.getElementById("chartTotalCO2"),
    {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Total CO₂ (kg)",
                data: totalCO2,
                backgroundColor: "#dc3545"
            }]
        }
    }
);

// ==============================
// 6️⃣ Radar (Best Material Profile)
// ==============================
const best = lastRecommendationData[0];

new Chart(
    document.getElementById("chartEnvironmental"),
    {
        type: "radar",
        data: {
            labels: [
                "Suitability",
                "CO₂ Reduction",
                "Cost Savings",
                "Recyclability",
                "Biodegradability"
            ],
            datasets: [{
                label: best.material_type,
                data: [
                    best.Suitability_Prob * 100,
                    best.co2_reduction_pct,
                    best.cost_savings_pct,
                    best.recyclability_pct,
                    best.biodegradability_score * 10
                ],
                backgroundColor: "rgba(25,135,84,0.2)",
                borderColor: "#198754"
            }]
        }
    }
);

    // Executive Summary
    const bestMaterial = labels[bestIndex];
    const summaryText = `
        Switching to ${bestMaterial} improves sustainability by 
        ${co2Reduction[bestIndex].toFixed(1)}% CO₂ reduction and 
        ${costSavings[bestIndex].toFixed(1)}% cost savings.
    `;

    const summaryDiv = document.getElementById("executiveSummary");
    if (summaryDiv) summaryDiv.innerHTML = summaryText;
}
// ======================================
// EXPORT DASHBOARD TO PDF
// ======================================
function exportDashboardPDF() {

    const dashboard = document.getElementById("dashboardSection");

    if (!dashboard) {
        alert("Dashboard not found!");
        return;
    }

    html2canvas(dashboard, { scale: 2 }).then(canvas => {

        const imgData = canvas.toDataURL("image/png");

        const { jsPDF } = window.jspdf;
        const pdf = new jsPDF("p", "mm", "a4");

        const imgWidth = 210;
        const pageHeight = 297;
        const imgHeight = canvas.height * imgWidth / canvas.width;

        let heightLeft = imgHeight;
        let position = 0;

        pdf.addImage(imgData, "PNG", 0, position, imgWidth, imgHeight);
        heightLeft -= pageHeight;

        while (heightLeft > 0) {
            position = heightLeft - imgHeight;
            pdf.addPage();
            pdf.addImage(imgData, "PNG", 0, position, imgWidth, imgHeight);
            heightLeft -= pageHeight;
        }

        pdf.save("EcoPackAI_Dashboard.pdf");

    }).catch(error => {
        console.error("PDF Export Error:", error);
        alert("Error generating PDF. Check console.");
    });
}
