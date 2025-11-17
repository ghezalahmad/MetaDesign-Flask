// ==========================================================
//  DASHBOARD.JS — CLEAN, STABLE, PRODUCTION VERSION
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

    let allColumns = [];
    let experimentData = null;

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
    //  DATASET UPLOAD + AUTOLOAD
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
                allColumns = data.columns;
                addDatasetRow(data.filename, data.columns);
                populateInitialSelectors();
            });
    });

    function addDatasetRow(filename, columns) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td><button class="btn btn-sm btn-danger">Delete</button></td>
            <td>${filename}</td>
            <td>${columns.join(", ")}</td>
        `;
        datasetTableBody.appendChild(tr);
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
        const wrapper = document.createElement("div");
        wrapper.classList.add("mb-3", "target-group");

        const availableColumns = allColumns.filter(
            c => !getSelected(inputColumns).includes(c)
        );

        wrapper.innerHTML = `
            <div class="input-group">
                <select class="form-select" name="target_columns">
                    ${availableColumns.map(c => `<option value="${c}">${c}</option>`).join("")}
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

        const payload = {
            model: modelSelect.value,
            curiosity: curiositySlider.value,
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

                // Debug the received figures
                console.log("🔍 t-SNE figure data length:", 
                    data.tsne_figure?.data?.[0]?.x?.length || "MISSING");
                console.log("🔍 Scatter figure data length:", 
                    data.target_scatter_figure?.data?.[0]?.x?.length || "MISSING");

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
        resultsTableContainer.innerHTML = data.results_table;

        drawTSNE(data.tsne_figure);
        drawScatter(data.target_scatter_figure);
    }

    function drawTSNE(fig) {
        console.log("📊 Drawing t-SNE plot...");
        
        try {
            // Handle both string and object
            if (typeof fig === "string") {
                console.log("⚠️ t-SNE figure is string, parsing...");
                fig = JSON.parse(fig);
            }

            console.log("t-SNE figure structure:", {
                hasData: !!fig?.data,
                dataLength: fig?.data?.length,
                firstTraceLength: fig?.data?.[0]?.x?.length
            });

            if (!fig || !fig.data || fig.data.length === 0) {
                console.error("❌ Empty t-SNE figure");
                tsnePlotDiv.innerHTML = 
                    "<div class='alert alert-warning'>Empty t-SNE figure.</div>";
                return;
            }

            if (!fig.data[0].x || fig.data[0].x.length === 0) {
                console.error("❌ t-SNE trace has no data points");
                tsnePlotDiv.innerHTML = 
                    "<div class='alert alert-warning'>No data points in t-SNE trace.</div>";
                return;
            }

            console.log("✅ Plotting t-SNE with", fig.data[0].x.length, "points");
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
            // Handle both string and object
            if (typeof fig === "string") {
                console.log("⚠️ Scatter figure is string, parsing...");
                fig = JSON.parse(fig);
            }

            console.log("Scatter figure structure:", {
                hasData: !!fig?.data,
                dataLength: fig?.data?.length,
                firstTraceLength: fig?.data?.[0]?.x?.length
            });

            if (!fig || !fig.data || fig.data.length === 0) {
                console.error("❌ Empty scatter figure");
                scatterPlotDiv.innerHTML = 
                    "<div class='alert alert-warning'>Empty scatter plot.</div>";
                return;
            }

            if (!fig.data[0].x || fig.data[0].x.length === 0) {
                console.error("❌ Scatter trace has no data points");
                scatterPlotDiv.innerHTML = 
                    "<div class='alert alert-warning'>No data points in scatter trace.</div>";
                return;
            }

            console.log("✅ Plotting scatter with", fig.data[0].x.length, "points");
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
    });
});