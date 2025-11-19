// ==========================================================
//  DASHBOARD.JS — FINAL VERSION WITH DATASET SELECTION
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
    const curiositySlider = document.getElementById("curiosity-slider");
    const curiosityValue = document.getElementById("curiosity-value");

    const runButton = document.getElementById("run-experiment-button");

    const tsnePlotDiv = document.getElementById("tsne-plot");
    const scatterPlotDiv = document.getElementById("scatter-plot");

    const resultsSection = document.getElementById("results-section");
    const resultsTableContainer = document.getElementById("results-table-container");

    const addTargetButton = document.getElementById("add-target-property-button");

    // ----- STATE VARIABLES -----
    let allColumns = []; // Columns of the currently active dataset
    let uploadedDatasets = []; // Array to store all uploaded datasets: {filename, columns, isActive}
    let experimentData = null;
    let resultsDataTable = null; // Variable for DataTables instance

    // ==========================================================
    //  Utility helpers
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
                
                // Store the new dataset
                const newDataset = {
                    filename: data.filename,
                    columns: data.columns,
                    isActive: false 
                };
                uploadedDatasets.push(newDataset);
                
                // Add the row to the table
                const newIndex = uploadedDatasets.length - 1;
                addDatasetRow(newDataset, newIndex);

                // Automatically select the first uploaded dataset if none is active
                if (uploadedDatasets.filter(d => d.isActive).length === 0) {
                    handleDatasetSelection(newIndex);
                }
            });
    });

    /**
     * Creates a table row for a new dataset and adds a Select button.
     */
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

        // Attach event listener to the Select button
        tr.querySelector(".select-btn").addEventListener("click", () => {
            handleDatasetSelection(index);
        });
        
        // Initial styling based on activity
        updateTableRowStyle(index, dataset.isActive);
    }

    /**
     * Handles the selection of a new active dataset.
     */
    function handleDatasetSelection(index) {
        // 1. Update the active status in the array and update styles
        uploadedDatasets.forEach((d, i) => {
            d.isActive = (i === index);
            updateTableRowStyle(i, d.isActive);
        });

        const selectedDataset = uploadedDatasets[index];
        allColumns = selectedDataset.columns;
        
        console.log(`✅ Dataset '${selectedDataset.filename}' selected. Available columns updated.`);

        // 2. Clear existing configuration selectors and repopulate them
        // Note: Target groups are removed to prevent mixing columns from different datasets
        inputColumns.innerHTML = '';
        aprioriColumns.innerHTML = '';
        targetContainer.innerHTML = '';
        
        populateInitialSelectors();
    }
    
    /**
     * Updates the visual style of a table row based on its active state.
     */
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
        // Only run if a dataset is loaded
        if (allColumns.length === 0) return;
        
        const selectedInputs = getSelected(inputColumns);
        const selectedTargets = Array.from(
            document.querySelectorAll("select[name='target_columns']")
        ).map(sel => sel.value);

        // Filters columns not used as Input or Target
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
        // Only allow adding if a dataset is loaded
        if (allColumns.length === 0) {
            alert("Please upload and select a dataset first.");
            return;
        }

        const wrapper = document.createElement("div");
        wrapper.classList.add("mb-3", "target-group");

        // Available columns are those not selected as Input
        const availableColumns = allColumns.filter(
            c => !getSelected(inputColumns).includes(c)
        );
        
        // Find columns already used as targets to avoid duplicates in the dropdown
        const currentTargets = Array.from(document.querySelectorAll(".target-group select[name='target_columns']"))
                                    .map(sel => sel.value);
        
        const optionsHTML = availableColumns
            .filter(c => !currentTargets.includes(c)) // Filter out already assigned targets
            .map(c => `<option value="${c}">${c}</option>`)
            .join("");
            
        if (optionsHTML === "" && availableColumns.length > 0) {
             // Happens if all available columns are already assigned as targets
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
            updateAprioriOptions(); // Recalculate apriori options after removal
        });

        wrapper.querySelector("select[name='target_columns']")
            .addEventListener("change", updateAprioriOptions);

        targetContainer.appendChild(wrapper);
        updateAprioriOptions(); // Recalculate apriori options after adding
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

    curiositySlider.addEventListener("input", () => {
        curiosityValue.textContent = curiositySlider.value;
    });

    runButton.addEventListener("click", runExperiment);

    function runExperiment() {
        const selectedInputs = getSelected(inputColumns);
        const targets = collectTargetConfig();

        if (selectedInputs.length === 0 || targets.length === 0) {
            return alert("Please select input and target columns.");
        }
        
        // Find the filename of the currently active dataset to send to the backend
        const activeDataset = uploadedDatasets.find(d => d.isActive);
        if (!activeDataset) {
             return alert("No dataset is currently selected. Please select one from the table.");
        }


        const payload = {
            model: modelSelect.value,
            curiosity: curiositySlider.value,
            dataset_filename: activeDataset.filename, // Send the filename to the backend
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

        // --- DataTables Initialization ---
        
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
        // Note: The backend must ensure that the prediction_error_plot is included in 'data'
        // For now, we assume it's included and render it if available.
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