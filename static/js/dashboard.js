document.addEventListener('DOMContentLoaded', function() {
    const csvUpload = document.getElementById('csv-upload');
    const uploadButton = document.getElementById('upload-button');
    const datasetTableBody = document.getElementById('dataset-table-body');
    const inputColumns = document.getElementById('input-columns');
    const targetColumns = document.getElementById('target-columns');
    const aprioriColumns = document.getElementById('apriori-columns');
    const modelSelect = document.getElementById('model-select');
    const curiositySlider = document.getElementById('curiosity-slider');
    const curiosityValue = document.getElementById('curiosity-value');
    const runExperimentButton = document.getElementById('run-experiment-button');
    const resultsTable = document.createElement('div');
    const tsnePlot = document.getElementById('tsne-plot');
    const scatterPlot = document.getElementById('scatter-plot');

    let allColumns = [];

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
        [inputColumns, targetColumns, aprioriColumns].forEach(selector => {
            selector.innerHTML = '';
            columns.forEach(col => {
                const option = document.createElement('option');
                option.value = col;
                option.textContent = col;
                selector.appendChild(option);
            });
        });
    }

    function getTargetColumnConfig() {
        const config = [];
        Array.from(targetColumns.selectedOptions).forEach(option => {
            config.push({
                name: option.value,
                weight: 1.0, // Default weight
                optimization: 'max' // Default optimization
            });
        });
        return config;
    }

    function displayResults(data) {
        resultsTable.innerHTML = data.results_table;
        document.querySelector('.col-md-8 h3').insertAdjacentElement('afterend', resultsTable);

        if (data.tsne_data) {
            Plotly.newPlot('tsne-plot', [{
                x: data.tsne_data.x,
                y: data.tsne_data.y,
                mode: 'markers',
                type: 'scatter'
            }], {
                title: 't-SNE Visualization'
            });
        }

        if (data.scatter_data) {
            Plotly.newPlot('scatter-plot', [{
                x: data.scatter_data.x,
                y: data.scatter_data.y,
                text: data.scatter_data.labels,
                mode: 'markers',
                type: 'scatter'
            }], {
                title: 'Scatter Plot',
                xaxis: { title: 'Target Property' },
                yaxis: { title: 'Uncertainty' }
            });
        }
    }
});
