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
    const resultsTable = document.createElement('div');
    const tsnePlot = document.getElementById('tsne-plot');
    const scatterPlot = document.getElementById('scatter-plot');

    let allColumns = [];

    function autoLoadDataset() {
        const urlParams = new URLSearchParams(window.location.search);
        const filename = urlParams.get('ds');
        if (filename) {
            // Set the session filepath for the backend
            fetch(`/set-filepath-from-url?filename=${encodeURIComponent(filename)}`, { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if(data.success) {
                    allColumns = data.columns;
                    updateDatasetTable(data.filename, data.columns);
                    populateColumnSelectors(data.columns);
                    const cardTitle = document.querySelector('.card-title');
                    if(cardTitle) {
                        cardTitle.insertAdjacentHTML('afterend', `<div class="alert alert-success" role="alert">Loaded dataset: ${data.filename}</div>`);
                    }
                } else {
                     alert('Error auto-loading dataset: ' + data.error);
                }
            });
        }
    }
    autoLoadDataset();

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

    curiositySlider.addEventListener('input', function() {
        curiosityValue.textContent = this.value;
    });

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
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayResults(data);
            } else {
                alert('Error running experiment: ' + data.error);
            }
        });
    });

    function updateDatasetTable(filename, columns) {
        const newRow = document.createElement('tr');
        newRow.innerHTML = `
            <td><button class="btn btn-sm btn-danger">Delete</button></td>
            <td>${filename}</td>
            <td>${columns.join(', ')}</td>
        `;
        datasetTableBody.appendChild(newRow);
    }

    function populateColumnSelectors(columns) {
        // Initial population
        updateOptions(inputColumns, columns, []);
        updateOptions(targetColumns, columns, []);
        updateOptions(aprioriColumns, columns, []);
    }

    function updateCascadingSelectors() {
        const selectedInputs = Array.from(inputColumns.selectedOptions).map(opt => opt.value);
        const selectedTargets = Array.from(targetColumns.selectedOptions).map(opt => opt.value);

        const availableForTargets = allColumns.filter(col => !selectedInputs.includes(col));
        updateOptions(targetColumns, availableForTargets, selectedTargets);

        const availableForApriori = allColumns.filter(col => !selectedInputs.includes(col) && !selectedTargets.includes(col));
        updateOptions(aprioriColumns, availableForApriori, Array.from(aprioriColumns.selectedOptions).map(opt => opt.value));
    }

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

    function getTargetColumnConfig() {
        const config = [];
        const targetGroups = document.querySelectorAll('.target-group');
        targetGroups.forEach(group => {
            const select = group.querySelector('select[name="target_columns"]');
            const weight = group.querySelector('input[name="weights"]');
            const optimization = group.querySelector('select[name="max_or_min"]');
            if (select.value) {
                config.push({
                    name: select.value,
                    weight: parseFloat(weight.value),
                    optimization: optimization.value
                });
            }
        });
        return config;
    }

    function addTargetProperty() {
        const index = targetPropertiesContainer.children.length;
        const newTargetGroup = document.createElement('div');
        newTargetGroup.classList.add('mb-3', 'target-group');
        const availableColumns = allColumns.filter(col => !Array.from(inputColumns.selectedOptions).map(opt => opt.value).includes(col));

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
        newTargetGroup.querySelector('select[name="target_columns"]').addEventListener('change', updateCascadingSelectors);
    }

    const addTargetButton = document.getElementById('add-target-property-button');
    addTargetButton.addEventListener('click', addTargetProperty);

    function displayResults(data) {
        const resultsSection = document.getElementById('results-section');
        const resultsTableContainer = document.getElementById('results-table-container');

        resultsTableContainer.innerHTML = data.results_table;
        resultsSection.style.display = 'block';

        if (data.tsne_plot_json) {
            const tsneFig = JSON.parse(data.tsne_plot_json);
            Plotly.newPlot('tsne-plot', tsneFig.data, tsneFig.layout);
        }
        if (data.target_scatter_json) {
            const scatterFig = JSON.parse(data.target_scatter_json);
            Plotly.newPlot('scatter-plot', scatterFig.data, scatterFig.layout);
        }

        const additionalPlotContainer = document.getElementById('additional-plot-container');
        const plotRadios = document.querySelectorAll('input[name="plot-select"]');

        plotRadios.forEach(radio => {
            radio.addEventListener('change', function() {
                additionalPlotContainer.innerHTML = '';
                if (this.checked) {
                    const plotDiv = document.createElement('div');
                    plotDiv.id = `${this.value}-plot`;
                    additionalPlotContainer.appendChild(plotDiv);

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
            });
        });
    }
});
