// ==========================================================
//  DASHBOARD.JS — FINAL VERSION WITH DATASET SELECTION AND HELP FEATURES
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
    const modelHelpButton = document.getElementById("model-help-button"); // NEW: Reference to the help button
    const curiositySlider = document.getElementById("curiosity-slider");
    const curiosityValueDisplay = document.getElementById("curiosity-value-display"); 
    const curiosityGuidanceText = document.getElementById("curiosity-guidance-text"); 

    const runButton = document.getElementById("run-experiment-button");
    const resultsSection = document.getElementById("results-section");
    const resultsTableContainer = document.getElementById("results-table-container");
    const addTargetButton = document.getElementById("add-target-property-button");

    const tsnePlotDiv = document.getElementById("tsne-plot");
    const scatterPlotDiv = document.getElementById("scatter-plot");


    // ----- STATE VARIABLES -----
    let allColumns = []; // Columns of the currently active dataset
    let uploadedDatasets = []; // Array to store all uploaded datasets: {filename, columns, isActive}
    let experimentData = null;
    let resultsDataTable = null; // Variable for DataTables instance

    // ----- MODEL DESCRIPTIONS -----
    // NOTE: HTML is included for better formatting in the popover
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
    //  Utility helpers (Functions remain the same)
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
    //  HELP FEATURE LOGIC (UPDATED FOR POPOVER)
    // ==========================================================
    
    // Function to update the Popover content
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
            // Update the popover title and content attributes
            modelHelpButton.setAttribute('data-bs-original-title', info.name);
            modelHelpButton.setAttribute('data-bs-content', content);
            
            // Re-initialize or update if needed (Popovers handle updates internally when attributes change)
            popover.dispose(); // Dispose the old instance
            new bootstrap.Popover(modelHelpButton, { // Create a new one with updated data
                html: true,
                sanitize: false,
                trigger: 'focus' 
            });
        }
    }

    /**
     * Shows the model information pop-up when a new model is selected.
     */
    modelSelect.addEventListener("change", (event) => {
        const selectedValue = event.target.value;
        updateModelPopover(selectedValue);
        // Automatically click the help button to show the popover on change
        modelHelpButton.click(); 
        modelHelpButton.focus();
    });

    // Initialize the popover content for the default selected model on load
    updateModelPopover(modelSelect.value);

    /**
     * Provides dynamic guidance for the Curiosity slider.
     */
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
    
    // Trigger initial guidance text on load
    curiositySlider.dispatchEvent(new Event('input'));


    // ==========================================================
    //  DATASET MANAGEMENT (UPLOAD, SELECTION, DISPLAY) 
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
    //  COLUMN SELECTORS 
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

    // ==========================================================
    //  TARGET PROPERTY GROUPS 
    // ==========================================================

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

        fetch("/run-experiment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
            .then(r => r.json())
            .then(data => {
                console.log("📦 Received response:", data);
                
                if (!data.success) {
                    console.error("❌ Experiment error:", data.error);
                    return alert("Error: " + data.error);
                }

                experimentData = data;
                renderResults(data);
            })
            .catch(err => {
                console.error("💥 Network error:", err);
                alert("Network error — see console.");
            });
    }

    // ==========================================================
    //  RENDER RESULTS + PLOTS 
    // ==========================================================

    function renderResults(data) {
        console.log("🎨 Rendering results...");
        
        resultsSection.style.display = "block";

        if (resultsDataTable) {
            resultsDataTable.destroy();
            resultsDataTable = null;
            resultsTableContainer.innerHTML = '';
            console.log("ℹ️ Previous DataTable instance destroyed.");
        }

        resultsTableContainer.innerHTML = data.results_table; 
        
        const newTable = resultsTableContainer.querySelector("table"); 
        
        if (newTable) {
            if (!newTable.classList.contains('table')) {
                newTable.classList.add('table', 'table-striped', 'w-100');
            }
            
            console.log("Initializing DataTable on the results table...");
            
            resultsDataTable = new DataTable(newTable, {
                paging: true,
                searching: true,
                ordering: true,
                info: true,
                responsive: true,
                dom: 'lfrtipB', 
                buttons: [
                    {
                        extend: 'csv',
                        text: '<i class="bi bi-file-earmark-arrow-down"></i> Download CSV',
                        className: 'btn-sm btn-primary ms-2', 
                        exportOptions: {
                            modifier: {
                                page: 'current' 
                            }
                        }
                    },
                    {
                        extend: 'excel',
                        text: '<i class="bi bi-file-earmark-excel"></i> Download Excel',
                        className: 'btn-sm btn-success ms-2', 
                        exportOptions: {
                            modifier: {
                                page: 'current'
                            }
                        }
                    }
                ]
            });
            console.log("✅ DataTable initialized successfully.");
        } else {
            console.warn("Table element not found inside resultsTableContainer.");
        }
        
        drawTSNE(data.tsne_figure);
        drawScatter(data.target_scatter_figure);
        if (data.prediction_error_plot) {
            drawPredictionError(data.prediction_error_plot);
        }
    }

    function drawPredictionError(fig) {
        console.log("📊 Drawing prediction error plot...");
        const predErrorPlotDiv = document.getElementById("prediction-error-plot");

        try {
            if (window.Plotly) Plotly.purge(predErrorPlotDiv); 
            if (typeof fig === "string") fig = JSON.parse(fig);

            if (!fig || !fig.data || fig.data.length === 0) {
                predErrorPlotDiv.innerHTML = 
                    "<div class='alert alert-warning'>Empty Prediction Error plot.</div>";
                return;
            }

            Plotly.newPlot(predErrorPlotDiv, fig.data, fig.layout, {responsive: true});
            
        } catch (err) {
            console.error("💥 Prediction Error plotting error:", err);
            predErrorPlotDiv.innerHTML = 
                `<div class='alert alert-danger'>Error: ${err.message}</div>`;
        }
    }

    function drawTSNE(fig) {
        console.log("📊 Drawing t-SNE plot...");
        
        try {
            if (window.Plotly) Plotly.purge(tsnePlotDiv); 
            if (typeof fig === "string") fig = JSON.parse(fig);

            if (!fig || !fig.data || fig.data.length === 0) {
                tsnePlotDiv.innerHTML = 
                    "<div class='alert alert-warning'>Empty t-SNE figure.</div>";
                return;
            }

            Plotly.newPlot(tsnePlotDiv, fig.data, fig.layout, {responsive: true});
            
        } catch (err) {
            console.error("💥 t-SNE plotting error:", err);
            tsnePlotDiv.innerHTML = 
                `<div class='alert alert-danger'>Error: ${err.message}</div>`;
        }
    }

    function drawScatter(fig) {
        console.log("📊 Drawing scatter plot...");
        
        try {
            if (window.Plotly) Plotly.purge(scatterPlotDiv); 
            if (typeof fig === "string") fig = JSON.parse(fig);

            if (!fig || !fig.data || fig.data.length === 0) {
                scatterPlotDiv.innerHTML = 
                    "<div class='alert alert-warning'>Empty scatter plot.</div>";
                return;
            }

            Plotly.newPlot(scatterPlotDiv, fig.data, fig.layout, {responsive: true});
            
        } catch (err) {
            console.error("💥 Scatter plotting error:", err);
            scatterPlotDiv.innerHTML = 
                `<div class='alert alert-danger'>Error: ${err.message}</div>`;
        }
    }


    // ==========================================================
    //  RESPONSIVE RESIZING
    // ==========================================================

    window.addEventListener("resize", () => {
        if (tsnePlotDiv && tsnePlotDiv.data) {
            Plotly.Plots.resize(tsnePlotDiv);
        }
        if (scatterPlotDiv && scatterPlotDiv.data) {
            Plotly.Plots.resize(scatterPlotDiv);
        }
        if (resultsDataTable) {
             resultsDataTable.columns.adjust().draw();
        }
    });
});