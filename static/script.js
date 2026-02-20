let lastPredictionResults = null;
console.log("Script.js loaded successfully.");
const ecoFacts = [
  "Recycling one ton of cardboard saves over 9 cubic yards of landfill space.",
  "82% of consumers are willing to pay more for products with sustainable packaging.",
  "Compostable packaging can break down in as little as 90 days under industrial composting conditions.",
  "Switching from plastic to paper packaging can reduce carbon emissions by up to 60%.",
  "Reusable packaging systems can cut packaging waste by more than 80% in e-commerce logistics.",
  "Packaging made from bamboo or sugarcane bagasse is renewable and compostable.",
  "Recycled PET (rPET) packaging uses up to 75% less energy compared to virgin plastic production.",
  "Eco-friendly packaging reduces shipping weight, which lowers fuel consumption and CO₂ emissions."
];

function showRandomEcoFact() {
  const factBox = document.getElementById("ecoFactBox");
  const randomFact = ecoFacts[Math.floor(Math.random() * ecoFacts.length)];
  factBox.innerHTML = "🌍 Eco Fact: " + randomFact;
}

async function getPrediction() {
  const weight = parseFloat(document.getElementById("productWeight").value);
  const category = document.getElementById("productType").value;

  if (!weight || weight <= 0 || !category) {
    alert("Please enter a valid weight and select a category.");
    return;
  }

  try {
    const response = await fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        product_weight_kg: weight,
        product_category: category
      })
    });

    const data = await response.json();
    lastPredictionResults = data;  // Store for report download
    const resultsDiv = document.getElementById("results");
    resultsDiv.innerHTML = "";
    // Show server-side messages (e.g., fallback notice when weight exceeds capacity)
    if (data.message && !data.message.toLowerCase().includes("feature names")) {
    resultsDiv.innerHTML = `<p class="notice" style="color:orange">${data.message}</p>`;
    }

    if (data.recommendations && data.recommendations.length > 0) {
      let tableHTML = `
        <table>
          <tr>
            <th><img src="/static/Material.png" alt="Material">Material</th>
            <th><img src="/static/Cost.png" alt="Cost">Cost Efficiency</th>
            <th><img src="/static/co2.png" alt="CO2">CO₂ Impact</th>
            <th><img src="/static/Score.png" alt="Score">Suitability Score</th>
          </tr>
      `;

      data.recommendations.forEach(item => {
        tableHTML += `
          <tr>
            <td>${item.Material_Type}</td>
            <td>${item.cost_efficiency_pred.toFixed(3)}</td>
            <td>${item.Co2_impact_index_pred.toFixed(3)}</td>
            <td>${item.suitability_score.toFixed(3)}</td>
          </tr>
        `;
      });

      tableHTML += `</table>`;
      resultsDiv.innerHTML += tableHTML;
      showRandomEcoFact();

      // --- Render charts dynamically ---
      renderCharts(data.recommendations);

      // --- Auto show charts panel after prediction ---
      document.querySelector(".container").classList.add("show-charts");

    } else {
      const msg = data.message || "No suitable materials found.";
      resultsDiv.innerHTML = `<p>${msg}</p>`;
    }
  } catch (err) {
    console.error("Prediction error:", err);
    document.getElementById("results").innerHTML = `<p style="color:red">Error fetching predictions.</p>`;
  }
}

setInterval(showRandomEcoFact, 10000);

// --- Toggle layout when "Show Charts" is clicked ---
document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("showChartsBtn");
  if (btn) {
    btn.addEventListener("click", () => {
      const container = document.querySelector(".container");
      container.classList.toggle("show-charts");  // toggle on/off
    });
  }
});

// --- Chart rendering with Plotly ---
function renderCharts(recommendations) {
  const materials = recommendations.map(r => r.Material_Type);
  const costEff = recommendations.map(r => r.cost_efficiency_pred);
  const co2Impact = recommendations.map(r => r.Co2_impact_index_pred);
  const suitability = recommendations.map(r => r.suitability_score);

  // Bar chart - Cost Efficiency
  Plotly.newPlot("barChart", [{
    x: materials,
    y: costEff,
    type: "bar",
    marker: {color: ["green","teal","orange","grey"]}
  }], {title: "Cost Efficiency"});

  // Line chart - CO2 Impact
  Plotly.newPlot("lineChart", [{
    x: materials,
    y: co2Impact,
    type: "scatter",
    mode: "lines+markers",
    line: {color: "red"}
  }], {title: "CO₂ Impact"});

  // Pie chart - Suitability Score
  Plotly.newPlot("pieChart", [{
    labels: materials,
    values: suitability,
    type: "pie",
    marker: {colors: ["green","blue","orange"]}
  }], {title: "Suitability Score Distribution"});
}
function downloadReport() {
  if (!lastPredictionResults || !lastPredictionResults.recommendations) {
    alert("Please generate recommendations first.");
    return;
  }

  fetch("/download", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      recommendations: lastPredictionResults.recommendations
    })
  })
  .then(response => response.blob())
  .then(blob => {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "EcoPackAI_Report.pdf";
    document.body.appendChild(a);
    a.click();
    a.remove();
  })
  .catch(error => {
    console.error("Download error:", error);
    alert("Failed to download report");
  });
}