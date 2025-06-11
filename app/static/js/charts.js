// Global risk data storage
let globalRiskData = {};

// Chart.js global settings
Chart.defaults.font.family = "'Comic Neue', 'Helvetica', 'Arial', sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(236, 72, 153, 0.8)';

// Main function to create all charts
function createCharts(data) {
    // Show visualization area
    document.getElementById('visualizations').style.display = 'block';
    
    // Clean up any existing charts
    destroyCharts();
    
    // Calculate consistent risk values
    calculateConsistentRiskValues(data);
    
    // Create the charts
    createRiskGauge(data);
    createMedicationRisksChart(data);
    createSideEffectsChart(data);
    
    // Create anatomy visualization if side effects are available
    if (data.side_effects && data.side_effects.length > 0) {
        createAnatomyVisualization(data.side_effects);
    }
}

// Calculate consistent risk values for all medications
function calculateConsistentRiskValues(data) {
    globalRiskData = {};
    
    data.results.forEach((med, index) => {
        // Base probability value
        let baseProb = parseFloat(med.probability);
        
        // Apply small variations to make chart visually interesting
        let adjustedProb = baseProb;
        if (index > 0) {
            // Small variation for visual appeal
            const variation = (Math.random() * 0.1) - 0.05;
            adjustedProb = Math.max(0.15, Math.min(0.95, baseProb + variation));
        }
        
        // Determine risk level
        let riskLevel = determineRiskLevel(adjustedProb);
        
        // Calculate risk color
        const riskColor = getRiskColor(riskLevel, adjustedProb);
        
        // Store in global data
        globalRiskData[med.medication] = {
            medication: med.medication,
            originalProbability: baseProb,
            adjustedProbability: adjustedProb,
            riskLevel: riskLevel,
            riskColor: riskColor,
            recommendation: med.recommendation,
            alternatives: med.alternative_medications || []
        };
    });
    
    // Update the prediction results table
    updatePredictionResultsTable();
}

// Risk level determination helper
function determineRiskLevel(probability) {
    if (probability > 0.60) {
        return 'High';
    } else if (probability > 0.40) {
        return 'Medium';
    } else {
        return 'Low';
    }
}

// Update the prediction results table with calculated risk values
function updatePredictionResultsTable() {
    const resultRows = document.querySelectorAll('#resultsContent table tbody tr');
    
    resultRows.forEach(row => {
        const medicationCell = row.querySelector('td:first-child');
        const probabilityCell = row.querySelector('td:nth-child(3)');
        const riskLevelCell = row.querySelector('td:nth-child(2)');
        
        if (medicationCell) {
            const medication = medicationCell.textContent.trim();
            if (globalRiskData[medication]) {
                const riskInfo = globalRiskData[medication];
                
                // Update probability
                if (probabilityCell) {
                    probabilityCell.textContent = Math.round(riskInfo.adjustedProbability * 100) + '%';
                }
                
                // Update risk level
                if (riskLevelCell) {
                    const riskColorClass = riskInfo.riskLevel === 'Low' ? 'text-success' : 
                                          riskInfo.riskLevel === 'Medium' ? 'text-warning' : 'text-danger';
                    
                    riskLevelCell.innerHTML = `<strong class="${riskColorClass}">${riskInfo.riskLevel}</strong>`;
                }
            }
        }
    });
}

// Get color based on risk level
function getRiskColor(risk, probability = null) {
    // Base risk colors
    const baseRiskColors = {
        'Low': '#28a745',     // Green
        'Medium': '#ffc107',  // Yellow
        'High': '#fd7e14',    // Orange
        'Very High': '#dc3545' // Red
    };
    
    // More precise color gradation if probability provided
    if (probability !== null) {
        if (risk === 'Low') {
            // Green shades
            const greenValue = Math.round(175 - probability * 30);
            return `rgb(40, ${greenValue}, 69)`;
        } else if (risk === 'Medium') {
            // Yellow-orange shades
            const redValue = Math.round(255 - probability * 20);
            const greenValue = Math.round(193 - probability * 40);
            return `rgb(${redValue}, ${greenValue}, 7)`;
        } else if (risk === 'High' || risk === 'Very High') {
            // Orange-red shades
            const redValue = Math.round(220 + probability * 35);
            const greenValue = Math.round(65 - probability * 45);
            return `rgb(${redValue}, ${greenValue}, 20)`;
        }
    }
    
    return baseRiskColors[risk] || '#6c757d'; // Default gray
}

// Create the risk gauge chart
function createRiskGauge(data) {
    // Calculate highest risk from all medications
    let highestRiskProbability = 0;
    let overallRiskLevel = 'Low';
    let riskiestMedication = null;
    
    // Use global risk data
    data.results.forEach(med => {
        const riskInfo = globalRiskData[med.medication];
        if (riskInfo && riskInfo.adjustedProbability > highestRiskProbability) {
            highestRiskProbability = riskInfo.adjustedProbability;
            overallRiskLevel = riskInfo.riskLevel;
            riskiestMedication = med;
        }
    });
    
    // Determine risk color
    const riskColor = getRiskColor(overallRiskLevel, highestRiskProbability);
    
    const ctx = document.getElementById('riskGauge').getContext('2d');
    const chart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Risk Level', ''],
            datasets: [{
                data: [highestRiskProbability * 100, 100 - (highestRiskProbability * 100)],
                backgroundColor: [
                    riskColor,
                    '#f9e9f3'
                ],
                borderWidth: 0
            }]
        },
        options: {
            circumference: 180,
            rotation: 270,
            cutout: '70%',
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.dataIndex === 0) {
                                return [
                                    `Risk Level: ${overallRiskLevel}`,
                                    `Probability: ${Math.round(highestRiskProbability * 100)}%`,
                                    `Medication: ${riskiestMedication?.medication || 'Unknown'}`,
                                    `Click for more information`
                                ];
                            }
                            return '';
                        }
                    }
                }
            },
            onClick: () => {
                if (riskiestMedication) {
                    showRiskDetails(data.results, riskiestMedication, overallRiskLevel, highestRiskProbability);
                }
            }
        }
    });
}

// Create medication risks comparison chart
function createMedicationRisksChart(data) {
    const labels = data.results.map(med => med.medication);
    
    // Use global risk data
    const probabilities = data.results.map(med => 
        globalRiskData[med.medication].adjustedProbability * 100
    );
    
    const backgroundColors = data.results.map(med => 
        globalRiskData[med.medication].riskColor
    );
    
    const ctx = document.getElementById('medicationRisksChart').getContext('2d');
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Risk Percentage',
                data: probabilities,
                backgroundColor: backgroundColors,
                borderColor: backgroundColors,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Risk Percentage (%)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Medications'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const index = context.dataIndex;
                            const medRisk = globalRiskData[labels[index]];
                            return [
                                `Risk: ${Math.round(medRisk.adjustedProbability * 100)}%`, 
                                `Click for more details`
                            ];
                        }
                    }
                }
            },
            onClick: (e, elements) => {
                if (elements.length > 0) {
                    const index = elements[0].index;
                    const medication = data.results[index];
                    showMedicationDetails(medication);
                }
            }
        }
    });
}

// Create side effects chart with mock data
function createSideEffectsChart(data) {
    // Mock side effects data
    const sideEffects = generateMockSideEffectsData(data.results);
    
    // Get top 10 side effects by probability
    sideEffects.sort((a, b) => b.probability - a.probability);
    const topEffects = sideEffects.slice(0, 10);
    
    const labels = topEffects.map(effect => `${effect.name} (${effect.medication})`);
    const probabilities = topEffects.map(effect => effect.probability * 100);
    
    // Side effect colors based on medication risk
    const backgroundColors = topEffects.map(effect => {
        const medRiskInfo = globalRiskData[effect.medication];
        return medRiskInfo ? medRiskInfo.riskColor : '#f472b6';
    });
    
    const ctx = document.getElementById('sideEffectsChart').getContext('2d');
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability',
                data: probabilities,
                backgroundColor: backgroundColors,
                borderColor: backgroundColors,
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            scales: {
                x: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Probability (%)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Side Effects'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const index = context.dataIndex;
                            const detail = topEffects[index];
                            return [
                                `Probability: ${Math.round(detail.probability * 100)}%`,
                                `Severity: ${detail.severity}`,
                                `Medication: ${detail.medication}`
                            ];
                        }
                    }
                }
            }
        }
    });
}

// Generate mock side effects data
function generateMockSideEffectsData(medicationResults) {
    // Common side effects by medication
    const sideEffectsMap = {
        'aspirin': [
            { name: 'Stomach Upset', severity: 'Medium' },
            { name: 'Bleeding', severity: 'High' },
            { name: 'Heartburn', severity: 'Low' }
        ],
        'ibuprofen': [
            { name: 'Stomach Pain', severity: 'Medium' },
            { name: 'Dizziness', severity: 'Low' },
            { name: 'Headache', severity: 'Low' }
        ],
        'amoxicillin': [
            { name: 'Diarrhea', severity: 'Medium' },
            { name: 'Rash', severity: 'Medium' },
            { name: 'Nausea', severity: 'Low' }
        ],
        'metformin': [
            { name: 'Digestive Issues', severity: 'Medium' },
            { name: 'Vitamin B12 Deficiency', severity: 'Medium' },
            { name: 'Lactic Acidosis', severity: 'High' }
        ],
        'lisinopril': [
            { name: 'Dry Cough', severity: 'Medium' },
            { name: 'Dizziness', severity: 'Low' },
            { name: 'Fatigue', severity: 'Low' }
        ],
        'atorvastatin': [
            { name: 'Muscle Pain', severity: 'Medium' },
            { name: 'Liver Damage', severity: 'High' },
            { name: 'Digestive Problems', severity: 'Low' }
        ],
        'fluoxetine': [
            { name: 'Insomnia', severity: 'Medium' },
            { name: 'Nausea', severity: 'Low' },
            { name: 'Headache', severity: 'Low' }
        ],
        'omeprazole': [
            { name: 'Headache', severity: 'Low' },
            { name: 'Diarrhea', severity: 'Low' },
            { name: 'Abdominal Pain', severity: 'Medium' }
        ]
    };
    
    let allEffects = [];
    
    // For each selected medication
    medicationResults.forEach(med => {
        const medicationName = med.medication.toLowerCase();
        const baseProb = parseFloat(med.probability);
        
        // Find matching side effects or use a default
        const sideEffectsForMed = sideEffectsMap[medicationName] || [
            { name: 'Unknown Side Effect', severity: 'Medium' }
        ];
        
        // Add probability based on medication risk and side effect severity
        sideEffectsForMed.forEach((effect, index) => {
            // Adjust probability based on severity
            let probability = baseProb;
            if (effect.severity === 'High') {
                probability *= 0.9; // High severity side effects are slightly less probable
            } else if (effect.severity === 'Low') {
                probability *= 0.7; // Low severity side effects are more common
            }
            
            // Add small variation
            probability = Math.max(0.3, Math.min(0.9, probability * (1 - (index * 0.15))));
            
            allEffects.push({
                name: effect.name,
                medication: med.medication,
                severity: effect.severity,
                probability: probability
            });
        });
    });
    
    return allEffects;
}

// Show risk details modal
function showRiskDetails(results, riskiestMedication, overallRiskLevel, highestRiskProbability) {
    const modal = new bootstrap.Modal(document.getElementById('riskDetailsModal'));
    const modalContent = document.getElementById('riskDetailsContent');
    
    // Risk factors (placeholder)
    let riskFactors = [
        "Patient age may influence medication tolerance",
        "Multiple medications can cause interactions",
        `${riskiestMedication.medication} has a higher risk profile`
    ];
    
    modalContent.innerHTML = `
        <div class="card mb-4">
            <div class="card-body text-center">
                <h5 class="mb-0">Overall Risk: ${overallRiskLevel} (${Math.round(highestRiskProbability * 100)}%)</h5>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card mb-3">
                    <div class="card-header bg-pink text-white">
                        <h6 class="mb-0">Risk Factors</h6>
                    </div>
                    <div class="card-body">
                        <ul class="list-group list-group-flush">
                            ${riskFactors.map(factor => `<li class="list-group-item">${factor}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-pink text-white">
                        <h6 class="mb-0">Highest Risk Medication</h6>
                    </div>
                    <div class="card-body">
                        <h5 class="card-title">${riskiestMedication.medication}</h5>
                        <p class="card-text">
                            ${riskiestMedication.recommendation || 'No specific recommendations available'}
                        </p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card mt-3">
            <div class="card-header bg-pink text-white">
                <h6 class="mb-0">Medication Risk Summary</h6>
            </div>
            <div class="card-body">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>Medication</th>
                            <th>Risk Level</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${results.map(med => {
                            const riskInfo = globalRiskData[med.medication];
                            const riskClass = riskInfo.riskLevel === 'Low' ? 'success' : 
                                             riskInfo.riskLevel === 'Medium' ? 'warning' : 'danger';
                            return `
                                <tr>
                                    <td>${med.medication}</td>
                                    <td><span class="text-${riskClass}">${riskInfo.riskLevel}</span></td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        </div>
    `;
    
    modal.show();
}

// Show medication details
function showMedicationDetails(medication) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('medicationModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'medicationModal';
        modal.className = 'modal fade';
        modal.setAttribute('tabindex', '-1');
        modal.setAttribute('aria-hidden', 'true');
        
        modal.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="medicationModalTitle">Medication Details</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body" id="medicationModalContent"></div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
    
    // Prepare modal content
    const modalTitle = document.getElementById('medicationModalTitle') || modal.querySelector('.modal-title');
    const modalContent = document.getElementById('medicationModalContent') || modal.querySelector('.modal-body');
    
    modalTitle.textContent = medication.medication;
    
    const riskInfo = globalRiskData[medication.medication];
    const riskLevel = riskInfo.riskLevel;
    const riskClass = riskLevel === 'Low' ? 'success' : riskLevel === 'Medium' ? 'warning' : 'danger';
    
    // Format alternative medications correctly
    let alternativesHtml = '<p class="text-muted">No alternatives provided</p>';
    
    if (medication.alternative_medications && medication.alternative_medications.length > 0) {
        // Determine if we're dealing with categorized alternatives or simple array
        if (medication.alternative_medications[0] && 
            (medication.alternative_medications[0].category || 
             medication.alternative_medications[0].category_name)) {
            // We have categorized alternatives (with category and alternatives properties)
            alternativesHtml = '';
            
            // Process each category
            medication.alternative_medications.forEach(category => {
                const categoryName = category.category_name || category.category || "Alternatives";
                
                alternativesHtml += `<h6 class="mt-3">${categoryName}</h6>`;
                
                if (category.alternatives && category.alternatives.length > 0) {
                    alternativesHtml += '<ul class="list-group mt-2">';
                    
                    // Process each alternative in this category
                    category.alternatives.forEach(alt => {
                        let altText = "";
                        
                        // If it's an object with name property
                        if (typeof alt === 'object' && alt !== null) {
                            if (alt.name) {
                                altText = alt.name;
                                
                                // Add dosage if available
                                if (alt.dosage) {
                                    altText += ` (${alt.dosage})`;
                                }
                                
                                // Add description if available
                                if (alt.description) {
                                    altText += ` - ${alt.description}`;
                                }
                            } else {
                                // Fallback if name is not available
                                altText = JSON.stringify(alt);
                            }
                        } else if (typeof alt === 'string') {
                            // Simple string
                            altText = alt;
                        }
                        
                        alternativesHtml += `<li class="list-group-item">${altText}</li>`;
                    });
                    
                    alternativesHtml += '</ul>';
                } else {
                    alternativesHtml += '<p class="text-muted">No alternatives in this category</p>';
                }
            });
        } else if (typeof medication.alternative_medications[0] === 'string') {
            // Simple array of strings
            alternativesHtml = `
                <ul class="list-group mt-2">
                    ${medication.alternative_medications.map(alt => `
                        <li class="list-group-item">${alt}</li>
                    `).join('')}
                </ul>
            `;
        } else {
            // Array of objects (non-categorized)
            alternativesHtml = `
                <ul class="list-group mt-2">
                    ${medication.alternative_medications.map(alt => {
                        // Try to find a usable string property
                        let altText = "";
                        
                        if (typeof alt === 'object' && alt !== null) {
                            if (alt.name) {
                                altText = alt.name;
                                if (alt.dosage) altText += ` (${alt.dosage})`;
                                if (alt.description) altText += ` - ${alt.description}`;
                            } else {
                                // Just display first property we can find
                                for (const key in alt) {
                                    if (typeof alt[key] === 'string') {
                                        altText = alt[key];
                                        break;
                                    }
                                }
                                
                                // If we couldn't find a string property
                                if (!altText) {
                                    altText = "Alternative medication option";
                                }
                            }
                        } else {
                            altText = String(alt);
                        }
                        
                        return `<li class="list-group-item">${altText}</li>`;
                    }).join('')}
                </ul>
            `;
        }
    }
    
    modalContent.innerHTML = `
        <div class="card">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h5>Risk Assessment</h5>
                    <span class="badge bg-${riskClass}">${riskLevel}</span>
                </div>
                <hr>
                <p><strong>Probability:</strong> ${Math.round(riskInfo.adjustedProbability * 100)}%</p>
                <p><strong>Recommendation:</strong> ${medication.recommendation}</p>
                
                <div class="mt-3">
                    <h6>Alternative Medications</h6>
                    ${alternativesHtml}
                </div>
            </div>
        </div>
    `;
    
    // Show modal
    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();
}

// Clean up existing charts
function destroyCharts() {
    ['riskGauge', 'medicationRisksChart', 'sideEffectsChart'].forEach(chartId => {
        const chartInstance = Chart.getChart(chartId);
        if (chartInstance) {
            chartInstance.destroy();
        }
    });
}

// Create anatomy visualization with side effects
function createAnatomyVisualization(sideEffects) {
    // Clear existing hotspots
    const hotspotsContainer = document.getElementById('anatomyHotspots');
    if (!hotspotsContainer) return;
    
    hotspotsContainer.innerHTML = '';
    
    // Clear existing body systems list
    const bodySystemsContainer = document.getElementById('bodySystems');
    if (bodySystemsContainer) {
        bodySystemsContainer.innerHTML = '';
    }
    
    // Reset selected body part title
    const bodyPartTitle = document.getElementById('selectedBodyPartTitle');
    if (bodyPartTitle) {
        bodyPartTitle.textContent = 'Select a body part to see side effects';
    }
    
    // Clear body part side effects
    const bodyPartSideEffects = document.getElementById('bodyPartSideEffects');
    if (bodyPartSideEffects) {
        bodyPartSideEffects.innerHTML = '<p class="text-muted">Click on a body part or system to see possible side effects.</p>';
    }
    
    // Define body parts and their positions (with more accurate coordinates)
    const bodyParts = [
        { id: 'head', name: 'Head', x: 50, y: 7, width: 12, height: 12, system: 'Neurological' },
        { id: 'chest', name: 'Chest', x: 50, y: 22, width: 18, height: 10, system: 'Cardiovascular' },
        { id: 'abdomen', name: 'Abdomen', x: 50, y: 35, width: 16, height: 12, system: 'Gastrointestinal' },
        { id: 'arms_right', name: 'Right Arm', x: 71, y: 25, width: 12, height: 18, system: 'Musculoskeletal' },
        { id: 'arms_left', name: 'Left Arm', x: 29, y: 25, width: 12, height: 18, system: 'Musculoskeletal' },
        { id: 'legs_right', name: 'Right Leg', x: 57, y: 65, width: 10, height: 25, system: 'Musculoskeletal' },
        { id: 'legs_left', name: 'Left Leg', x: 43, y: 65, width: 10, height: 25, system: 'Musculoskeletal' }
    ];
    
    // Define system to body map
    const systemToBodyMap = {
        'Neurological': ['head'],
        'Cardiovascular': ['chest'],
        'Respiratory': ['chest'],
        'Gastrointestinal': ['abdomen'],
        'Musculoskeletal': ['arms_right', 'arms_left', 'legs_right', 'legs_left'],
        'Dermatological': ['head', 'chest', 'abdomen', 'arms_right', 'arms_left', 'legs_right', 'legs_left'],
        'Immunological': ['whole_body'],
        'Endocrine': ['abdomen'],
        'Renal': ['abdomen'],
        'Hematological': ['whole_body']
    };
    
    // Group side effects by body system
    const systemEffects = {};
    
    // Map side effects to body systems
    sideEffects.forEach(effect => {
        // Determine the body system based on effect name
        let system = mapEffectToBodySystem(effect.name);
        
        // Initialize the system if it doesn't exist
        if (!systemEffects[system]) {
            systemEffects[system] = [];
        }
        
        // Add the effect to the system
        systemEffects[system].push(effect);
    });
    
    // Create system list
    if (bodySystemsContainer) {
        Object.keys(systemEffects).sort().forEach(system => {
            const systemItem = document.createElement('a');
            systemItem.href = '#';
            systemItem.className = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';
            systemItem.textContent = system;
            
            // Add badge with count
            const badge = document.createElement('span');
            badge.className = 'badge bg-pink rounded-pill';
            badge.textContent = systemEffects[system].length;
            systemItem.appendChild(badge);
            
            // Add click event to show side effects for this system
            systemItem.addEventListener('click', (e) => {
                e.preventDefault();
                showSideEffectsForSystem(system, systemEffects[system]);
            });
            
            bodySystemsContainer.appendChild(systemItem);
        });
    }
    
    // Create hotspots on the anatomy image
    const container = document.getElementById('humanAnatomyContainer');
    if (!container) return;
    
    const imgWidth = container.offsetWidth;
    const imgHeight = container.offsetHeight;
    
    bodyParts.forEach(part => {
        // Check if there are effects for this body part
        let hasEffects = false;
        const systems = Object.keys(systemToBodyMap).filter(sys => 
            systemToBodyMap[sys].includes(part.id) && systemEffects[sys]
        );
        
        if (systems.length > 0) {
            hasEffects = true;
        }
        
        if (hasEffects) {
            // Create hotspot element with fixed pixel size for better circles
            const hotspot = document.createElement('div');
            hotspot.className = 'anatomy-hotspot';
            
            // Calculate size based on body part (fixed pixel size instead of percentage)
            let size;
            if (part.id === 'head') {
                size = 80;
            } else if (part.id === 'chest') {
                size = 120;
            } else if (part.id === 'abdomen') {
                size = 110;
            } else if (part.id.includes('arms')) {
                size = 90;
            } else if (part.id.includes('legs')) {
                size = 100;
            }
            
            // Position is still percentage-based for responsiveness
            hotspot.style.left = `calc(${part.x}% - ${size/2}px)`;
            hotspot.style.top = `calc(${part.y}% - ${size/2}px)`;
            hotspot.style.width = `${size}px`;
            hotspot.style.height = `${size}px`;
            hotspot.title = part.name;
            
            // Add click event
            hotspot.addEventListener('click', () => {
                // Collect all side effects for this body part
                let allEffects = [];
                systems.forEach(sys => {
                    allEffects = allEffects.concat(systemEffects[sys]);
                });
                
                // Show side effects for this body part
                showSideEffectsForBodyPart(part.name, allEffects);
            });
            
            hotspotsContainer.appendChild(hotspot);
        }
    });
}

// Map effect name to body system
function mapEffectToBodySystem(effectName) {
    effectName = effectName.toLowerCase();
    
    // Neurological symptoms
    if (effectName.includes('head') || effectName.includes('dizz') || 
        effectName.includes('seizure') || effectName.includes('migraine') || 
        effectName.includes('ache') || effectName.includes('confusion') || 
        effectName.includes('sleep') || effectName.includes('insomnia') || 
        effectName.includes('anxiety') || effectName.includes('depression')) {
        return 'Neurological';
    }
    
    // Cardiovascular symptoms
    if (effectName.includes('heart') || effectName.includes('blood pressure') || 
        effectName.includes('hypertension') || effectName.includes('hypotension') || 
        effectName.includes('cardiac') || effectName.includes('chest pain')) {
        return 'Cardiovascular';
    }
    
    // Respiratory symptoms
    if (effectName.includes('breath') || effectName.includes('cough') || 
        effectName.includes('lung') || effectName.includes('respiratory')) {
        return 'Respiratory';
    }
    
    // Gastrointestinal symptoms
    if (effectName.includes('stomach') || effectName.includes('nausea') || 
        effectName.includes('vomit') || effectName.includes('diarrhea') || 
        effectName.includes('constipation') || effectName.includes('digest') || 
        effectName.includes('abdominal') || effectName.includes('bowel') || 
        effectName.includes('intestin')) {
        return 'Gastrointestinal';
    }
    
    // Musculoskeletal symptoms
    if (effectName.includes('muscle') || effectName.includes('joint') || 
        effectName.includes('bone') || effectName.includes('weakness') || 
        effectName.includes('arthritis') || effectName.includes('pain')) {
        return 'Musculoskeletal';
    }
    
    // Dermatological symptoms
    if (effectName.includes('skin') || effectName.includes('rash') || 
        effectName.includes('itch') || effectName.includes('derm')) {
        return 'Dermatological';
    }
    
    // Default to "Other" if no match
    return 'Other';
}

// Show side effects for a specific body part
function showSideEffectsForBodyPart(partName, effects) {
    const bodyPartTitle = document.getElementById('selectedBodyPartTitle');
    if (bodyPartTitle) {
        bodyPartTitle.textContent = `Side Effects: ${partName}`;
    }
    
    const bodyPartSideEffects = document.getElementById('bodyPartSideEffects');
    if (bodyPartSideEffects) {
        if (effects.length === 0) {
            bodyPartSideEffects.innerHTML = '<p class="text-muted">No specific side effects identified for this area.</p>';
            return;
        }
        
        // Sort by severity (High to Low)
        effects.sort((a, b) => {
            const severityOrder = { 'High': 0, 'Medium': 1, 'Low': 2 };
            return severityOrder[a.severity] - severityOrder[b.severity];
        });
        
        // Create list of effects
        let html = '<ul class="list-group">';
        
        effects.forEach(effect => {
            const severityClass = effect.severity === 'High' ? 'danger' : 
                                 effect.severity === 'Medium' ? 'warning' : 'success';
            
            html += `
                <li class="list-group-item">
                    <div class="d-flex justify-content-between align-items-center">
                        <span>${effect.name}</span>
                        <span class="badge bg-${severityClass} rounded-pill">${effect.severity}</span>
                    </div>
                    <small class="text-muted">Medication: ${effect.medication}</small>
                </li>
            `;
        });
        
        html += '</ul>';
        bodyPartSideEffects.innerHTML = html;
    }
}

// Show side effects for a specific system
function showSideEffectsForSystem(systemName, effects) {
    // Just reuse the body part function with system name
    showSideEffectsForBodyPart(systemName + ' System', effects);
} 