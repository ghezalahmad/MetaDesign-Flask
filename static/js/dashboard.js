document.addEventListener('DOMContentLoaded', function() {
    const csvUpload = document.getElementById('csv-upload');
    const uploadButton = document.getElementById('upload-button');
    const datasetTableBody = document.getElementById('dataset-table-body');
    const inputColumns = document.getElementById('input-columns');
    const targetPropertiesContainer = document.getElementById('target-properties-container');
    const aprioriColumns = document.getElementById('apriori-columns');
    const modelSelect = document.getElementById('model-select');
    const curiositySlider = document.getElementById('curiosity-slider');
    const curiosityValue = document.getElementById('curiosity-value');
    const runExperimentButton = document.getElementById('run-experiment-button');
    const tsnePlot = document.getElementById('tsne-plot');
    const scatterPlot = document.getElementById('scatter-plot');

    let allColumns = [];

    // --- Auto-load dataset from URL if provided ---
    function autoLoadDataset() {
        const urlParams = new URLSearchParams(window.location.search);
        const filename = urlParams.get('ds');
        if (filename) {
            fetch(`/set-filepath-from-url?filename=${encodeURIComponent(filename)}`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        allColumns = data.columns;
                        updateDatasetTable(data.filename, data.columns);
                        populateColumnSelectors(data.columns);
                        const cardTitle = document.querySelector('.card-title');
                        if (cardTitle) {
                            cardTitle.insertAdjacentHTML(
                                'afterend',
                                `<div class="alert alert-success" role="alert">Loaded dataset: ${data.filename}</div>`
                            );
                        }
                    } else {
                        console.error('Error auto-loading dataset: ' + data.error);
                    }
                });
        }
    }
    autoLoadDataset();

    // --- Upload dataset ---
    uploadButton.addEventListener('click', function() {
        const file = csvUpload.files[0];
        if (!file) {
            alert('Please select a file to upload.');
            return;
        }

        const formData = new FormData();
        formData.append('dataset', file);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                allColumns = data.columns;
                updateDatasetTable(data.filename, data.columns);
                populateColumnSelectors(data.columns);
            } else {
                alert('Error uploading file: ' + data.error);
            }
        });
    });

    // --- Update curiosity label ---
    curiositySlider.addEventListener('input', function() {
        curiosityValue.textContent = this.value;
    });

    // --- Run experiment ---
    runExperimentButton.addEventListener('click', function() {
        const selectedInputColumns = Array.from(inputColumns.selectedOptions).map(opt => opt.value);
        const selectedTargetColumns = getTargetColumnConfig();

        if (selectedInputColumns.length === 0 || selectedTargetColumns.length === 0) {
            alert('Please select at least one input and one target column.');
            return;
        }

        const experimentConfig = {
            model: modelSelect.value,
            curiosity: curiositySlider.value,
            input_columns: selectedInputColumns,
            target_columns: selectedTargetColumns,
        };

        fetch('/run-experiment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(experimentConfig)
        })
        .then(response => {
            console.log("Received response from /run-experiment:", response);
            return response.json();
        })
        .then(data => {
            console.log("Received data from /run-experiment:", data);
            if (data.success) {
                displayResults(data);
            } else {
                alert('Error running experiment: ' + data.error);
                console.error('Error running experiment:', data.error);
            }
        })
        .catch(error => {
            console.error('Fetch error:', error);
            alert('A network error occurred. Check the console for details.');
        });
    });

    // --- Dataset table ---
    function updateDatasetTable(filename, columns) {
        const newRow = document.createElement('tr');
        newRow.innerHTML = `
            <td><button class="btn btn-sm btn-danger">Delete</button></td>
            <td>${filename}</td>
            <td>${columns.join(', ')}</td>
        `;
        datasetTableBody.appendChild(newRow);
    }

    // --- Populate initial selectors (input + apriori only) ---
    function populateColumnSelectors(columns) {
        updateOptions(inputColumns, columns, []);
        updateOptions(aprioriColumns, columns, []);
    }

    // --- Handle cascading selector updates ---
    function updateCascadingSelectors() {
        const selectedInputs = Array.from(inputColumns.selectedOptions).map(opt => opt.value);

        // Collect selected targets from all dynamic target groups
        const selectedTargets = Array.from(
            document.querySelectorAll('select[name="target_columns"]')
        ).map(sel => sel.value);

        // Filter available columns for apriori
        const availableForApriori = allColumns.filter(
            col => !selectedInputs.includes(col) && !selectedTargets.includes(col)
        );

        updateOptions(
            aprioriColumns,
            availableForApriori,
            Array.from(aprioriColumns.selectedOptions).map(opt => opt.value)
        );
    }

    // --- Update options in a <select> element ---
    function updateOptions(selectElement, options, selectedValues) {
        selectElement.innerHTML = '';
        options.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            if (selectedValues.includes(col)) {
                option.selected = true;
            }
            selectElement.appendChild(option);
        });
    }

    inputColumns.addEventListener('change', updateCascadingSelectors);

    // --- Collect dynamic target configuration ---
    function getTargetColumnConfig() {
        const config = [];
        const targetGroups = document.querySelectorAll('.target-group');
        targetGroups.forEach(group => {
            const select = group.querySelector('select[name="target_columns"]');
            const weight = group.querySelector('input[name="weights"]');
            const optimization = group.querySelector('select[name="max_or_min"]');
            if (select && select.value) {
                config.push({
                    name: select.value,
                    weight: parseFloat(weight.value),
                    optimization: optimization.value
                });
            }
        });
        return config;
    }

    // --- Add a new target property group dynamically ---
    function addTargetProperty() {
        const newTargetGroup = document.createElement('div');
        newTargetGroup.classList.add('mb-3', 'target-group');
        const availableColumns = allColumns.filter(
            col => !Array.from(inputColumns.selectedOptions).map(opt => opt.value).includes(col)
        );

        let options = '';
        availableColumns.forEach(col => {
            options += `<option value="${col}">${col}</option>`;
        });

        newTargetGroup.innerHTML = `
            <div class="input-group">
                <select class="form-select" name="target_columns">${options}</select>
                <input type="number" class="form-control" name="weights" value="1.0" step="0.1">
                <select class="form-select" name="max_or_min">
                    <option value="max">Maximize</option>
                    <option value="min">Minimize</option>
                </select>
                <button class="btn btn-danger" type="button" onclick="this.parentElement.parentElement.remove()">Remove</button>
            </div>
        `;
        targetPropertiesContainer.appendChild(newTargetGroup);

        newTargetGroup
            .querySelector('select[name="target_columns"]')
            .addEventListener('change', updateCascadingSelectors);
    }

    const addTargetButton = document.getElementById('add-target-property-button');
    addTargetButton.addEventListener('click', addTargetProperty);

    // --- Display experiment results and plots ---
    function displayResults(data) {
        // keep reference for optional plots
        window.experimentData = data;

        console.log("✅ Experiment data received:", data);

        const resultsSection = document.getElementById('results-section');
        const resultsTableContainer = document.getElementById('results-table-container');
        resultsTableContainer.innerHTML = data.results_table;
        resultsSection.style.display = 'block';

        // --- t-SNE plot ---
        try {
            const tsneFig = typeof data.tsne_figure === "string"
                ? JSON.parse(data.tsne_figure)
                : data.tsne_figure;
            console.log("t-SNE Figure JSON:", JSON.stringify(tsneFig, null, 2));

            if (tsneFig && tsneFig.data && tsneFig.data.length) {
                Plotly.newPlot('tsne-plot', tsneFig.data, tsneFig.layout);
            } else {
                document.getElementById('tsne-plot').innerHTML =
                    "<div class='alert alert-warning'>t-SNE JSON empty or invalid.</div>";
            }
        } catch (err) {
            console.error("Error processing t-SNE plot:", err);
        }

        // --- Scatter plot ---
        try {
            const scatterFig = typeof data.target_scatter_figure === "string"
                ? JSON.parse(data.target_scatter_figure)
                : data.target_scatter_figure;
            console.log("Scatter Figure JSON:", JSON.stringify(scatterFig, null, 2));

            if (scatterFig && scatterFig.data && scatterFig.data.length) {
                Plotly.newPlot('scatter-plot', scatterFig.data, scatterFig.layout);
            } else {
                document.getElementById('scatter-plot').innerHTML =
                    "<div class='alert alert-warning'>Scatter JSON empty or invalid.</div>";
            }
        } catch (err) {
            console.error("Error processing scatter plot:", err);
        }
    }



        // --- Optional additional plots ---
            // --- Optional additional plots ---
        const additionalPlotContainer = document.getElementById('additional-plot-container');
        const plotRadios = document.querySelectorAll('input[name="plot-select"]');

        plotRadios.forEach(radio => {
            radio.addEventListener('change', function() {
                additionalPlotContainer.innerHTML = '';
                if (this.checked) {
                    const plotDiv = document.createElement('div');
                    plotDiv.id = `${this.value}-plot`;
                    additionalPlotContainer.appendChild(plotDiv);

                    // We'll handle optional plots only if returned by backend
                    if (window.experimentData) {
                        const data = window.experimentData;
                        if (this.value === 'parallel-coordinates' && data.parallel_coordinates_data) {
                            Plotly.newPlot(plotDiv, [data.parallel_coordinates_data], {
                                title: 'Parallel Coordinates Plot'
                            });
                        } else if (this.value === 'correlation-heatmap' && data.correlation_heatmap_data) {
                            Plotly.newPlot(plotDiv, [{
                                z: data.correlation_heatmap_data.z,
                                x: data.correlation_heatmap_data.x,
                                y: data.correlation_heatmap_data.y,
                                type: 'heatmap',
                                colorscale: 'Viridis'
                            }], {
                                title: 'Correlation Heatmap'
                            });
                        }
                    }
                }
            });
        });
    });

