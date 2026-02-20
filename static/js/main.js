// Global Sidebar Toggle Function (Must be outside DOMContentLoaded)
function toggleSidebar() {
    const sidebar = document.getElementById("sidebarUI");
    if (sidebar) {
        sidebar.classList.toggle("active");
    } else {
        console.error("Sidebar element not found!");
    }
}

// Chart Instances
let mainChartInstance = null;
let radarChartInstance = null;
let pieChartInstance = null;

// Store results for PDF generation
let lastResults = null;
let lastFormData = null;

document.addEventListener('DOMContentLoaded', function() {
    
    const form = document.getElementById('predictionForm');
    
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Show Loading
            const btn = this.querySelector('button[type="submit"]');
            const originalText = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            btn.disabled = true;

            // Collect Data
            const formData = {
                weight_capacity: parseFloat(document.getElementById('weight').value),
                category: document.getElementById('category').value,
                fragility_score: parseInt(document.getElementById('fragilityScore').value),
                shelf_life_days: parseInt(document.getElementById('shelfLife').value),
                dimensions: {
                    l: parseFloat(document.getElementById('dimL').value),
                    w: parseFloat(document.getElementById('dimW').value),
                    h: parseFloat(document.getElementById('dimH').value)
                }
            };

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                const data = await response.json();

                if (data.status === 'success') {
                    lastResults = data.recommendations;
                    lastFormData = formData;
                    updateUI(data.recommendations);
                    toggleSidebar();
                } else {
                    alert('Error: ' + data.message);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Failed to connect to AI server.');
            } finally {
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        });
    }
});

function updateUI(results) {
    // Hide placeholder, show results
    document.getElementById('initialState').classList.add('d-none');
    document.getElementById('resultsSection').classList.remove('d-none');

    // Show download button
    document.getElementById('downloadReportBtn').classList.remove('d-none');

    const top = results[0];

    // Update Top Card
    document.getElementById('topName').innerText = top.material_type;
    document.getElementById('topScore').innerText = top.suitability_score;
    document.getElementById('topCO2').innerText = top.predicted_co2;
    document.getElementById('topCost').innerText = top.predicted_cost_efficiency;

    // Update Table
    const tbody = document.getElementById('rankingTableBody');
    tbody.innerHTML = '';
    results.forEach((item, index) => {
        const row = `
            <tr>
                <td><span class="badge bg-${index === 0 ? 'success' : 'secondary'} rounded-pill">${index + 1}</span></td>
                <td class="fw-bold">${item.material_type}</td>
                <td>
                    <div class="d-flex align-items-center">
                        <span class="me-2">${item.suitability_score}</span>
                        <div class="progress flex-grow-1" style="height: 6px;">
                            <div class="progress-bar bg-success" role="progressbar" style="width: ${item.suitability_score}%"></div>
                        </div>
                    </div>
                </td>
                <td>${item.biodegradability} / 100</td>
                <td>${item.predicted_co2} units</td>
            </tr>
        `;
        tbody.innerHTML += row;
    });

    // Render Charts
    renderMainChart(results.slice(0, 5));
    renderRadarChart(top);
    renderPieChart(results.slice(0, 5));
}

function renderMainChart(data) {
    const ctx = document.getElementById('mainChart').getContext('2d');
    const labels = data.map(d => d.material_type);
    
    if (mainChartInstance) mainChartInstance.destroy();

    mainChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                { label: 'Cost Efficiency', data: data.map(d => d.predicted_cost_efficiency), backgroundColor: '#10b981', borderRadius: 5, yAxisID: 'y' },
                { label: 'CO₂ Impact', data: data.map(d => d.predicted_co2), backgroundColor: '#3b82f6', borderRadius: 5, yAxisID: 'y1' }
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: { type: 'linear', display: true, position: 'left', grid: { display: false } },
                y1: { type: 'linear', display: true, position: 'right', grid: { display: false } },
                x: { grid: { display: false } }
            },
            plugins: { legend: { position: 'bottom' } }
        }
    });
}

function renderRadarChart(item) {
    const ctx = document.getElementById('radarChart').getContext('2d');
    if (radarChartInstance) radarChartInstance.destroy();

    radarChartInstance = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Biodegradability', 'Recyclability', 'Tensile Str.', 'Moisture Barrier', 'Cost Eff.'],
            datasets: [{
                label: item.material_type,
                data: [
                    item.biodegradability,
                    item.recyclability,
                    Math.min(100, item.tensile_strength * 1.5), 
                    85, // Dummy visual for Moisture
                    Math.min(100, item.predicted_cost_efficiency * 5)
                ],
                backgroundColor: 'rgba(16, 185, 129, 0.2)',
                borderColor: '#10b981',
                pointBackgroundColor: '#10b981'
            }]
        },
        options: {
            scales: { r: { suggestMin: 0, suggestedMax: 100 } },
            plugins: { legend: { display: false } }
        }
    });
}

function renderPieChart(data) {
    const ctx = document.getElementById('pieChart').getContext('2d');
    if (pieChartInstance) pieChartInstance.destroy();

    pieChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: data.map(d => d.material_type),
            datasets: [{
                data: data.map(d => d.suitability_score),
                backgroundColor: [
                    '#10b981', '#34d399', '#6ee7b7', '#a7f3d0', '#d1fae5'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            plugins: { 
                legend: { position: 'bottom', labels: { boxWidth: 10 } } 
            },
            cutout: '70%'
        }
    });
}

// PDF Report Generation
async function generatePDFReport() {
    if (!lastResults || lastResults.length === 0) {
        alert('Run an analysis first before downloading a report.');
        return;
    }

    const btn = document.getElementById('downloadReportBtn');
    const originalHTML = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Generating...';
    btn.classList.add('generating');

    try {
        const { jsPDF } = window.jspdf;
        const pdf = new jsPDF('p', 'mm', 'a4');
        const pageWidth = pdf.internal.pageSize.getWidth();
        const margin = 15;
        const contentWidth = pageWidth - margin * 2;
        let y = margin;

        // -- Page 1: Header --
        pdf.setFillColor(5, 150, 105);
        pdf.rect(0, 0, pageWidth, 40, 'F');

        pdf.setFont('helvetica', 'bold');
        pdf.setFontSize(22);
        pdf.setTextColor(255, 255, 255);
        pdf.text('EcoPack AI', pageWidth / 2, 18, { align: 'center' });

        pdf.setFontSize(11);
        pdf.setFont('helvetica', 'normal');
        pdf.text('Sustainable Packaging Analysis Report', pageWidth / 2, 28, { align: 'center' });

        const now = new Date();
        pdf.setFontSize(8);
        pdf.text('Generated: ' + now.toLocaleDateString() + ' ' + now.toLocaleTimeString(), pageWidth / 2, 35, { align: 'center' });

        y = 50;

        // -- Input Summary --
        pdf.setTextColor(5, 150, 105);
        pdf.setFontSize(14);
        pdf.setFont('helvetica', 'bold');
        pdf.text('Input Summary', margin, y);
        y += 2;
        pdf.setDrawColor(209, 250, 229);
        pdf.setLineWidth(0.5);
        pdf.line(margin, y, pageWidth - margin, y);
        y += 8;

        pdf.setTextColor(31, 41, 55);
        pdf.setFontSize(10);
        pdf.setFont('helvetica', 'normal');

        if (lastFormData) {
            const inputs = [
                ['Category', lastFormData.category || 'General'],
                ['Weight', (lastFormData.weight_capacity || 1.0) + ' kg'],
                ['Fragility', (lastFormData.fragility_score || 5) + ' / 10'],
                ['Shelf Life', (lastFormData.shelf_life_days || 30) + ' days'],
            ];
            if (lastFormData.dimensions) {
                inputs.push(['Dimensions', lastFormData.dimensions.l + ' x ' + lastFormData.dimensions.w + ' x ' + lastFormData.dimensions.h + ' cm']);
            }

            inputs.forEach(([label, value]) => {
                pdf.setFont('helvetica', 'bold');
                pdf.text(label + ':', margin + 5, y);
                pdf.setFont('helvetica', 'normal');
                pdf.text(String(value), margin + 45, y);
                y += 6;
            });
        }
        y += 5;

        // -- Top Recommendation --
        const top = lastResults[0];
        pdf.setTextColor(5, 150, 105);
        pdf.setFontSize(14);
        pdf.setFont('helvetica', 'bold');
        pdf.text('Top Recommendation', margin, y);
        y += 2;
        pdf.line(margin, y, pageWidth - margin, y);
        y += 5;

        // Green card background
        pdf.setFillColor(236, 253, 245);
        pdf.roundedRect(margin, y, contentWidth, 30, 3, 3, 'F');

        pdf.setTextColor(5, 150, 105);
        pdf.setFontSize(18);
        pdf.setFont('helvetica', 'bold');
        pdf.text(top.material_type, pageWidth / 2, y + 12, { align: 'center' });

        pdf.setFontSize(10);
        pdf.setTextColor(75, 85, 99);
        pdf.setFont('helvetica', 'normal');
        const topLine = 'Score: ' + top.suitability_score + '  |  CO\u2082: ' + top.predicted_co2 + '  |  Efficiency: ' + top.predicted_cost_efficiency;
        pdf.text(topLine, pageWidth / 2, y + 22, { align: 'center' });
        y += 40;

        // -- Charts Section --
        pdf.setTextColor(5, 150, 105);
        pdf.setFontSize(14);
        pdf.setFont('helvetica', 'bold');
        pdf.text('Analysis Charts', margin, y);
        y += 2;
        pdf.line(margin, y, pageWidth - margin, y);
        y += 8;

        // Capture charts as images
        const chartIds = ['mainChart', 'radarChart', 'pieChart'];
        const chartLabels = ['Cost vs Impact', 'Material Profile', 'Top Contenders'];
        const chartImages = [];

        for (const id of chartIds) {
            const canvas = document.getElementById(id);
            if (canvas) {
                chartImages.push(canvas.toDataURL('image/png', 1.0));
            }
        }

        // Place first chart (bar) wide
        if (chartImages.length > 0) {
            const chartW = contentWidth;
            const chartH = 55;
            pdf.setFontSize(9);
            pdf.setTextColor(75, 85, 99);
            pdf.text(chartLabels[0], margin, y);
            y += 3;
            pdf.addImage(chartImages[0], 'PNG', margin, y, chartW, chartH);
            y += chartH + 8;
        }

        // Place radar and pie side by side
        if (chartImages.length >= 3) {
            const halfW = (contentWidth - 5) / 2;
            const chartH = 55;

            // Check if we need a new page
            if (y + chartH + 10 > pdf.internal.pageSize.getHeight() - margin) {
                pdf.addPage();
                y = margin;
            }

            pdf.setFontSize(9);
            pdf.setTextColor(75, 85, 99);
            pdf.text(chartLabels[1], margin, y);
            pdf.text(chartLabels[2], margin + halfW + 5, y);
            y += 3;
            pdf.addImage(chartImages[1], 'PNG', margin, y, halfW, chartH);
            pdf.addImage(chartImages[2], 'PNG', margin + halfW + 5, y, halfW, chartH);
            y += chartH + 10;
        }

        // -- Ranking Table --
        // Check if we need a new page
        if (y + 40 > pdf.internal.pageSize.getHeight() - margin) {
            pdf.addPage();
            y = margin;
        }

        pdf.setTextColor(5, 150, 105);
        pdf.setFontSize(14);
        pdf.setFont('helvetica', 'bold');
        pdf.text('Material Ranking', margin, y);
        y += 2;
        pdf.line(margin, y, pageWidth - margin, y);
        y += 8;

        // Table header
        const colWidths = [15, 50, 30, 40, 35];
        const headers = ['#', 'Material', 'Score', 'Biodegradability', 'CO\u2082 Impact'];

        pdf.setFillColor(236, 253, 245);
        pdf.rect(margin, y - 4, contentWidth, 8, 'F');
        pdf.setFontSize(9);
        pdf.setFont('helvetica', 'bold');
        pdf.setTextColor(5, 150, 105);

        let xPos = margin + 2;
        headers.forEach((h, i) => {
            pdf.text(h, xPos, y);
            xPos += colWidths[i];
        });
        y += 7;

        // Table rows
        pdf.setFont('helvetica', 'normal');
        pdf.setTextColor(31, 41, 55);
        pdf.setFontSize(9);

        lastResults.forEach((item, index) => {
            if (y > pdf.internal.pageSize.getHeight() - 20) {
                pdf.addPage();
                y = margin;
            }

            // Alternate row color
            if (index % 2 === 0) {
                pdf.setFillColor(249, 250, 251);
                pdf.rect(margin, y - 4, contentWidth, 7, 'F');
            }

            // Highlight top material
            if (index === 0) {
                pdf.setFillColor(236, 253, 245);
                pdf.rect(margin, y - 4, contentWidth, 7, 'F');
                pdf.setFont('helvetica', 'bold');
            } else {
                pdf.setFont('helvetica', 'normal');
            }

            xPos = margin + 2;
            pdf.text(String(index + 1), xPos, y);
            xPos += colWidths[0];
            pdf.text(item.material_type, xPos, y);
            xPos += colWidths[1];
            pdf.text(String(item.suitability_score), xPos, y);
            xPos += colWidths[2];
            pdf.text(item.biodegradability + ' / 100', xPos, y);
            xPos += colWidths[3];
            pdf.text(String(item.predicted_co2), xPos, y);
            y += 7;
        });

        // -- Footer --
        y = pdf.internal.pageSize.getHeight() - 15;
        pdf.setDrawColor(229, 231, 235);
        pdf.line(margin, y - 5, pageWidth - margin, y - 5);
        pdf.setFontSize(8);
        pdf.setTextColor(156, 163, 175);
        pdf.text('EcoPack AI - Sustainable Packaging Intelligence', pageWidth / 2, y, { align: 'center' });
        pdf.text('Report generated on ' + now.toLocaleDateString(), pageWidth / 2, y + 4, { align: 'center' });

        // Save PDF
        pdf.save('EcoPackAI_Report_' + now.toISOString().slice(0, 10) + '.pdf');

    } catch (error) {
        console.error('PDF generation error:', error);
        alert('Could not generate PDF. Please try again.');
    } finally {
        btn.innerHTML = originalHTML;
        btn.classList.remove('generating');
    }
}
