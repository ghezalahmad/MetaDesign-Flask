// ==========================================================
//  DASHBOARD.JS — FINAL VERSION WITH ACTIVE LEARNING PLOTS
// ==========================================================

document.addEventListener("DOMContentLoaded", () => {
    // ----- DOM ELEMENTS -----
    const csvUpload = document.getElementById("csv-upload");
    const uploadButton = document.getElementById("upload-button");
    const datasetTableBody = document.getElementById("dataset-table-body");

    const inputColumns = document.getElementById("input-columns");
    const aprioriColumns = document.getElementById("apriori-columns");
    const targetContainer = document.getElementById("target-properties-container");

    const modelSelect = document.getElementById("model-select");
    const modelHelpButton = document.getElementById("model-help-button"); 
    const curiositySlider = document.getElementById("curiosity-slider");
    const curiosityValueDisplay = document.getElementById("curiosity-value-display"); 
    const curiosityGuidanceText = document.getElementById("curiosity-guidance-text"); 

    const runButton = document.getElementById("run-experiment-button");
    const resultsSection = document.getElementById("results-section");
    const resultsTableContainer = document.getElementById("results-table-container");
    const addTargetButton = document.getElementById("add-target-property-button");

    // Existing Plot Divs
    const tsnePlotDiv = document.getElementById("tsne-plot");
    const scatterPlotDiv = document.getElementById("scatter-plot");
    
    // NEW Plot Divs
    const uncertaintyPlotDiv = document.getElementById("uncertainty-plot");
    const historyPlotDiv = document.getElementById("history-plot");
    const utilitySurfacePlotDiv = document.getElementById("utility-surface-plot");
    const utilitySurfaceMessage = document.getElementById("utility-surface-message");


    // ----- STATE VARIABLES -----
    let allColumns = []; 
    let uploadedDatasets = []; 
    let experimentData = null;
    let resultsDataTable = null; 

    // ----- MODEL DESCRIPTIONS (Keep as is) -----
    const MODEL_INFO = {
        'pinn': {
            name: 'Physics Informed Neural Network (PINN)',
            description: '<strong>Integrates known physical laws</strong> into the loss function. Best for **sparse datasets** where physical constraints are well-defined. Good generalization, but requires theoretical setup.',
            warning: 'Best used when you have a **strong theoretical foundation** (known physics) governing material properties.'
        },
        'lolopy': {
            name: 'lolo Random Forest (AI model)',
            description: 'An optimized **Random Forest** focusing on fast, accurate predictions. Robust to overfitting and excellent for feature importance. Provides a strong, reliable baseline.',
            warning: 'Generally robust, but may struggle with very high-dimensional or extremely non-linear data.'
        },
        'dkl': {
            name: 'Deep Kernel Learning (DKL)',
            description: 'Combines the feature-learning of a Neural Network (NN) with the **uncertainty quantification of a Gaussian Process (GP)**. Provides superior predictive power with reliable uncertainty estimates.',
            warning: 'Computationally more expensive than standard RF or NN models, but provides superior **uncertainty metrics** crucial for active learning.'
        },
        'rf': {
            name: 'Random Forest (RF)',
            description: 'A classic ensemble method. Highly stable, robust to outliers, and requires minimal preprocessing. A great choice for fast, **reliable predictions** on standard data.',
            warning: 'Does **not extrapolate well** outside the range of the training data. Be cautious when exploring new regions.'
        },
        'gp': {
            name: 'Gaussian Process (GP)',
            description: 'Non-parametric model that provides accurate **uncertainty estimates (error bars)**, which are critical for effective exploration. Works best with **smaller, high-quality datasets**.',
            warning: 'Scalability is a major limitation; computation time increases **cubically** with data points. Not suitable for datasets exceeding a few thousand entries.'
        },
        'maml': {
            name: 'Model-Agnostic Meta-Learning (MAML)',
            description: 'Designed to train a model\'s initial parameters to **quickly adapt to new, unseen tasks** (datasets) with minimal training data. Useful for diverse sets of small, related materials tasks.',
            warning: 'Requires prior training on **multiple related datasets** (tasks) to be effective.'
        },
        'reptile': {
            name: 'Reptile (Meta-Learning)',
            description: 'Iteratively moves the model towards an initialization that **generalizes well across different tasks**. Simpler and often faster than MAML for few-shot learning.',
            warning: 'Similar to MAML, performs best when leveraging knowledge gained from **multiple prior tasks** or datasets.'
        },
        'protonet': {
            name: 'Prototypical Networks (ProtoNet)',
            description: 'Primarily used for **few-shot classification** (e.g., categorizing new materials with limited examples). Learns a metric space around a "prototype" vector for each class.',
            warning: 'Best suited for **classification problems**. Less ideal for continuous regression tasks.'
        },
        'ensemble': {
            name: 'Ensemble Model',
            description: 'Combines predictions from **multiple different base models** (e.g., RF, GP, DKL) for higher accuracy and robustness. Leverages the "wisdom of the crowd."',
            warning: 'Can be slower to train and predict due to running multiple models concurrently. Provides **high reliability**.'
        }
    };


    // ==========================================================
    //  Utility helpers (Keep as is)
    // ==========================================================
    function createOption(value) {
        const opt = document.createElement("option");
        opt.value = value;
        opt.textContent = value;
        return opt;
    }

    function setSelectOptions(select, options, preserved = []) {
        select.innerHTML = "";
        options.forEach(c => {
            const opt = createOption(c);
            if (preserved.includes(c)) {
                opt.selected = true;
            }
            select.appendChild(opt);
        });
    }

    function getSelected(select) {
        return Array.from(select.selectedOptions).map(opt => opt.value);
    }

    // ==========================================================
    //  HELP FEATURE LOGIC (Keep as is)
    // ==========================================================
    
    function updateModelPopover(selectedValue) {
        const info = MODEL_INFO[selectedValue];
        const popover = bootstrap.Popover.getInstance(modelHelpButton);

        if (info && popover) {
            const content = `
                <div class="p-1">
                    <p class="mb-1">${info.description}</p>
                    <p class="small text-danger mb-0"><strong>Warning:</strong> ${info.warning}</p>
                </div>
            `;
            modelHelpButton.setAttribute('data-bs-original-title', info.name);
            modelHelpButton.setAttribute('data-bs-content', content);
            
            popover.dispose(); 
            new bootstrap.Popover(modelHelpButton, { 
                html: true,
                sanitize: false,
                trigger: 'focus' 
            });
        }
    }

    modelSelect.addEventListener("change", (event) => {
        const selectedValue = event.target.value;
        updateModelPopover(selectedValue);
        modelHelpButton.click(); 
        modelHelpButton.focus();
    });

    updateModelPopover(modelSelect.value);

    curiositySlider.addEventListener("input", () => {
        const value = parseFloat(curiositySlider.value);
        curiosityValueDisplay.textContent = value.toFixed(1);

        if (value < -1.0) {
            curiosityGuidanceText.textContent = "Heavy EXPLOITATION: System prioritizes materials predicted to have the highest performance, ignoring model uncertainty. Useful for confirming known optima, but high-risk.";
        } else if (value >= -1.0 && value < -0.2) {
            curiosityGuidanceText.textContent = "Focused EXPLOITATION: Prioritizes high-performing, well-understood regions. Use this when you are confident in your model's predictions.";
        } else if (value >= -0.2 && value <= 0.2) {
            curiosityGuidanceText.textContent = "Balanced Approach: Equal weighting between performance (Exploit) and uncertainty (Explore). A good default for a mix of optimization and learning.";
        } else if (value > 0.2 && value <= 1.0) {
            curiosityGuidanceText.textContent = "Focused EXPLORATION: System prioritizes regions where the model is most uncertain, seeking new data to improve predictions. Use this when data is sparse or incomplete.";
        } else if (value > 1.0) {
            curiosityGuidanceText.textContent = "Heavy EXPLORATION: Strongly focused on uncertainty. Will primarily seek information in completely unknown areas, essential for discovering novel material classes.";
        }
    });
    
    curiositySlider.dispatchEvent(new Event('input'));


    // ==========================================================
    //  DATASET MANAGEMENT (Keep as is)
    // ==========================================================

    uploadButton.addEventListener("click", () => {
        const file = csvUpload.files[0];
        if (!file) {
            alert("Please select a CSV file.");
            return;
        }

        const formData = new FormData();
        formData.append("dataset", file);

        fetch("/upload", { method: "POST", body: formData })
            .then(r => r.json())
            .then(data => {
                if (!data.success) return alert("Upload error: " + data.error);
                
                const newDataset = {
                    filename: data.filename,
                    columns: data.columns,
                    isActive: false 
                };
                uploadedDatasets.push(newDataset);
                
                const newIndex = uploadedDatasets.length - 1;
                addDatasetRow(newDataset, newIndex);

                if (uploadedDatasets.filter(d => d.isActive).length === 0) {
                    handleDatasetSelection(newIndex);
                }
            });
    });

    function addDatasetRow(dataset, index) {
        const tr = document.createElement("tr");
        tr.dataset.index = index;
        
        tr.innerHTML = `
            <td><button class="btn btn-sm btn-danger delete-btn">Delete</button></td>
            <td>${dataset.filename}</td>
            <td>${dataset.columns.join(", ")}</td>
            <td><button class="btn btn-sm btn-primary select-btn">Select</button></td>
        `;
        
        datasetTableBody.appendChild(tr);

        tr.querySelector(".select-btn").addEventListener("click", () => {
            handleDatasetSelection(index);
        });
        
        updateTableRowStyle(index, dataset.isActive);
    }

    function handleDatasetSelection(index) {
        uploadedDatasets.forEach((d, i) => {
            d.isActive = (i === index);
            updateTableRowStyle(i, d.isActive);
        });

        const selectedDataset = uploadedDatasets[index];
        allColumns = selectedDataset.columns;
        
        console.log(`✅ Dataset '${selectedDataset.filename}' selected. Available columns updated.`);

        inputColumns.innerHTML = '';
        aprioriColumns.innerHTML = '';
        targetContainer.innerHTML = '';
        
        populateInitialSelectors();
    }
    
    function updateTableRowStyle(index, isActive) {
        const row = datasetTableBody.querySelector(`tr[data-index="${index}"]`);
        const selectBtn = row?.querySelector(".select-btn");
        
        if (row) {
            if (isActive) {
                row.classList.add('table-success', 'fw-bold');
                if (selectBtn) {
                    selectBtn.textContent = 'Selected';
                    selectBtn.classList.remove('btn-primary');
                    selectBtn.classList.add('btn-secondary');
                    selectBtn.disabled = true;
                }
            } else {
                row.classList.remove('table-success', 'fw-bold');
                if (selectBtn) {
                    selectBtn.textContent = 'Select';
                    selectBtn.classList.remove('btn-secondary');
                    selectBtn.classList.add('btn-primary');
                    selectBtn.disabled = false;
                }
            }
        }
    }

    // ==========================================================
    //  COLUMN SELECTORS & TARGET PROPERTY GROUPS (Keep as is)
    // ==========================================================

    function populateInitialSelectors() {
        setSelectOptions(inputColumns, allColumns);
        setSelectOptions(aprioriColumns, allColumns);
    }

    inputColumns.addEventListener("change", updateAprioriOptions);

    function updateAprioriOptions() {
        if (allColumns.length === 0) return;
        
        const selectedInputs = getSelected(inputColumns);
        const selectedTargets = Array.from(
            document.querySelectorAll("select[name='target_columns']")
        ).map(sel => sel.value);

        const available = allColumns.filter(
            c => !selectedInputs.includes(c) && !selectedTargets.includes(c)
        );

        setSelectOptions(
            aprioriColumns,
            available,
            getSelected(aprioriColumns)
        );
    }

    addTargetButton.addEventListener("click", () => addTargetGroup());

    function addTargetGroup() {
        if (allColumns.length === 0) {
            alert("Please upload and select a dataset first.");
            return;
        }

        const wrapper = document.createElement("div");
        wrapper.classList.add("mb-3", "target-group");

        const availableColumns = allColumns.filter(
            c => !getSelected(inputColumns).includes(c)
        );
        
        const currentTargets = Array.from(document.querySelectorAll(".target-group select[name='target_columns']"))
                                    .map(sel => sel.value);
        
        const optionsHTML = availableColumns
            .filter(c => !currentTargets.includes(c))
            .map(c => `<option value="${c}">${c}</option>`)
            .join("");
            
        if (optionsHTML === "" && availableColumns.length > 0) {
             alert("All available columns are already assigned as target properties.");
             return;
        } else if (availableColumns.length === 0) {
             alert("No columns left to be assigned as target properties.");
             return;
        }


        wrapper.innerHTML = `
            <div class="input-group">
                <select class="form-select" name="target_columns">
                    ${optionsHTML}
                </select>
                <input type="number" class="form-control" name="weights" value="1.0" step="0.1">
                <select class="form-select" name="max_or_min">
                    <option value="max">Maximize</option>
                    <option value="min">Minimize</option>
                </select>
                <button class="btn btn-danger" type="button">Remove</button>
            </div>
        `;

        wrapper.querySelector(".btn-danger").addEventListener("click", () => {
            wrapper.remove();
            updateAprioriOptions();
        });

        wrapper.querySelector("select[name='target_columns']")
            .addEventListener("change", updateAprioriOptions);

        targetContainer.appendChild(wrapper);
        updateAprioriOptions();
    }

    function collectTargetConfig() {
        return Array.from(document.querySelectorAll(".target-group")).map(g => ({
            name: g.querySelector("select[name='target_columns']").value,
            weight: parseFloat(g.querySelector("input[name='weights']").value),
            optimization: g.querySelector("select[name='max_or_min']").value
        }));
    }

    // ==========================================================
    //  RUN EXPERIMENT 
    // ==========================================================

    runButton.addEventListener("click", runExperiment);

    // Add this at the beginning of your runExperiment function

    function runExperiment() {
        const selectedInputs = getSelected(inputColumns);
        const targets = collectTargetConfig();

        if (selectedInputs.length === 0 || targets.length === 0) {
            return alert("Please select input and target columns.");
        }
        
        const activeDataset = uploadedDatasets.find(d => d.isActive);
        if (!activeDataset) {
            return alert("No dataset is currently selected. Please select one from the table.");
        }

        const payload = {
            model: modelSelect.value,
            curiosity: curiositySlider.value,
            dataset_filename: activeDataset.filename,
            input_columns: selectedInputs,
            target_columns: targets
        };

        console.log("🚀 Sending experiment request:", payload);

        // === IMPROVED LOADING INDICATOR ===
        runButton.disabled = true;
        runButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
        
        // Show detailed progress
        resultsSection.style.display = "block";
        resultsTableContainer.innerHTML = `
            <div class="alert alert-info text-center">
                <div class="spinner-border mb-3" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <h5 id="progress-title">Running Experiment...</h5>
                <div class="progress mb-3" style="height: 25px;">
                    <div id="progress-bar" class="progress-bar progress-bar-striped progress-bar-animated" 
                        role="progressbar" style="width: 100%">
                        Processing...
                    </div>
                </div>
                <p class="mb-2"><strong>Current Phase:</strong> <span id="progress-phase">Training model</span></p>
                <small class="text-muted">Please wait, this may take 20-40 seconds for large datasets.</small>
            </div>
        `;

        const startTime = Date.now();

        fetch("/run-experiment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
            .then(r => {
                const duration = ((Date.now() - startTime) / 1000).toFixed(1);
                console.log(`⏱️ Server responded in ${duration}s`);
                
                // Update progress
                document.getElementById('progress-phase').textContent = 'Receiving data...';
                
                if (!r.ok) {
                    throw new Error(`HTTP ${r.status}: ${r.statusText}`);
                }
                
                return r.json();
            })
            .then(data => {
                console.log("📦 Data received, size:", JSON.stringify(data).length, "characters");
                
                // Update progress
                document.getElementById('progress-phase').textContent = 'Parsing response...';
                
                // Use setTimeout to allow UI to update before heavy processing
                setTimeout(() => {
                    processExperimentResults(data);
                }, 100);
            })
            .catch(err => {
                console.error("💥 Error:", err);
                runButton.disabled = false;
                runButton.innerHTML = 'Run Experiment';
                resultsTableContainer.innerHTML = `
                    <div class="alert alert-danger">
                        <strong>Network Error:</strong> ${err.message}<br>
                        <small>Check the browser console and Flask terminal for details.</small>
                    </div>
                `;
            });
    }

    // SEPARATE FUNCTION: Process results without blocking UI
    function processExperimentResults(data) {
        console.log("🎨 Starting to render results...");
        
        try {
            // Re-enable button first
            runButton.disabled = false;
            runButton.innerHTML = 'Run Experiment';
            
            if (!data.success) {
                console.error("❌ Experiment error:", data.error);
                resultsTableContainer.innerHTML = `
                    <div class="alert alert-danger">
                        <strong>Error:</strong> ${data.error}
                    </div>
                `;
                return;
            }

            experimentData = data;
            
            // Update progress
            document.getElementById('progress-title').textContent = 'Rendering visualizations...';
            document.getElementById('progress-phase').textContent = 'Creating plots...';
            
            // Render in stages to prevent UI freeze
            renderResultsProgressively(data);
            
        } catch (err) {
            console.error("💥 Error processing results:", err);
            resultsTableContainer.innerHTML = `
                <div class="alert alert-danger">
                    <strong>Processing Error:</strong> ${err.message}
                </div>
            `;
        }
    }

    // PROGRESSIVE RENDERING: Render one plot at a time with longer delays
    function renderResultsProgressively(data) {
        console.log("📊 Progressive rendering started");
        
        // Stage 1: Show results section
        resultsSection.style.display = "block";
        
        // Stage 2: Render table (fastest)
        setTimeout(() => {
            console.log("📊 Rendering table...");
            renderTable(data.results_table);
            
            // Stage 3: Render scatter plot (small, fast)
            setTimeout(() => {
                console.log("📊 Rendering scatter plot...");
                drawPlot("scatter-plot", data.target_scatter_figure);
                
                // Stage 4: Render TSNE plot (medium, 2000 points)
                setTimeout(() => {
                    console.log("📊 Rendering TSNE plot...");
                    drawPlot("tsne-plot", data.tsne_figure);
                    
                    // Stage 5: Render remaining plots (give more time)
                    setTimeout(() => {
                        console.log("📊 Rendering uncertainty plot...");
                        drawPlot("uncertainty-plot", data.uncertainty_plot);
                        
                        setTimeout(() => {
                            console.log("📊 Rendering history plot...");
                            drawPlot("history-plot", data.history_plot);
                            
                            setTimeout(() => {
                                console.log("📊 Rendering utility surface plot...");
                                drawPlot("utility-surface-plot", data.utility_surface_plot);
                                
                                if (data.prediction_error_plot) {
                                    setTimeout(() => {
                                        drawPlot("prediction-error-plot", data.prediction_error_plot);
                                        console.log("✅ All plots rendered!");
                                    }, 200);
                                } else {
                                    console.log("✅ All plots rendered!");
                                }
                            }, 300);  // Longer delay for utility surface
                        }, 200);
                    }, 200);
                }, 200);
            }, 150);
        }, 100);
    }

    // RENDER TABLE SEPARATELY
    function renderTable(tableHtml) {
        if (resultsDataTable) {
            resultsDataTable.destroy();
            resultsDataTable = null;
        }
        
        resultsTableContainer.innerHTML = tableHtml;
        
        const newTable = resultsTableContainer.querySelector("table");
        if (newTable) {
            if (!newTable.classList.contains('table')) {
                newTable.classList.add('table', 'table-striped', 'w-100');
            }
            
            resultsDataTable = new DataTable(newTable, {
                paging: true,
                searching: true,
                ordering: true,
                info: true,
                responsive: true,
                deferRender: true,  // Performance optimization
                dom: 'lfrtipB',
                buttons: [
                    {
                        extend: 'csv',
                        text: '<i class="bi bi-file-earmark-arrow-down"></i> Download CSV',
                        className: 'btn-sm btn-primary ms-2'
                    },
                    {
                        extend: 'excel',
                        text: '<i class="bi bi-file-earmark-excel"></i> Download Excel',
                        className: 'btn-sm btn-success ms-2'
                    }
                ]
            });
            console.log("✅ DataTable initialized");
        }
    }

    // GENERIC PLOT HELPER (unchanged)
    function drawPlot(divId, figData) {
        const div = document.getElementById(divId);
        if (!div) {
            console.warn(`⚠️ Plot div not found: ${divId}`);
            return;
        }

        // Clean up previous plot
        if (window.Plotly) {
            Plotly.purge(div);
        }

        // Handle empty data
        if (!figData || !figData.data || figData.data.length === 0) {
            if (divId === 'utility-surface-plot') {
                const msg = document.getElementById("utility-surface-message");
                if (msg) msg.style.display = 'block';
            } else {
                div.innerHTML = `<div class='alert alert-warning d-flex align-items-center justify-content-center' style='height:100%'>No data available</div>`;
            }
            return;
        }

        // Handle string input (legacy)
        let plotData = figData;
        if (typeof figData === 'string') {
            try {
                plotData = JSON.parse(figData);
            } catch (e) {
                console.error(`Failed to parse JSON for ${divId}:`, e);
                return;
            }
        }

        console.log(`📈 Drawing ${divId}...`);

        try {
            Plotly.newPlot(div, plotData.data, plotData.layout, {
                responsive: true,
                displayModeBar: true
            });
            
            if (divId === 'utility-surface-plot') {
                const msg = document.getElementById("utility-surface-message");
                if (msg) msg.style.display = 'none';
            }
            
            console.log(`✅ ${divId} drawn successfully`);

        } catch (err) {
            console.error(`💥 Error drawing ${divId}:`, err);
            div.innerHTML = `<div class='alert alert-danger'>Plot Error: ${err.message}</div>`;
        }
}

    // ==========================================================
    //  RESPONSIVE RESIZING
    // ==========================================================

    window.addEventListener("resize", () => {
        const plots = ["tsne-plot", "scatter-plot", "uncertainty-plot", "history-plot", "utility-surface-plot", "prediction-error-plot"];
        
        plots.forEach(id => {
            const div = document.getElementById(id);
            if (div && div.data) {
                Plotly.Plots.resize(div);
            }
        });

        if (resultsDataTable) {
             resultsDataTable.columns.adjust().draw();
        }
    });
});