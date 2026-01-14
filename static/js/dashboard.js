// ==========================================================
//  DASHBOARD.JS — FINAL VERSION WITH ACTIVE LEARNING PLOTS
//  + LOCALSTORAGE CACHING & RESTORE
// ==========================================================

// Debug mode - only log to console in development
const DEBUG = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
const log = (...args) => { if (DEBUG) console.log(...args); };
const warn = (...args) => { if (DEBUG) console.warn(...args); };

// Helper function to show error modal popup
function showErrorModal(title, message, allErrors = []) {
    // Remove any existing modal
    const existingModal = document.getElementById('error-modal');
    if (existingModal) existingModal.remove();

    // Build error list HTML if there are multiple errors
    let errorListHtml = '';
    if (allErrors.length > 1) {
        errorListHtml = '<ul class="mb-0 mt-2">';
        allErrors.forEach(err => {
            errorListHtml += `<li>${err}</li>`;
        });
        errorListHtml += '</ul>';
    }

    // Create modal HTML
    const modalHtml = `
        <div class="modal fade" id="error-modal" tabindex="-1" aria-hidden="true">
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content bg-white">
                    <div class="modal-header bg-danger text-white">
                        <h5 class="modal-title">
                            <i class="bi bi-exclamation-triangle me-2"></i>${title}
                        </h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body text-dark">
                        <p class="mb-0">${message}</p>
                        ${errorListHtml}
                    </div>
                    <div class="modal-footer bg-light">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Add to DOM and show
    document.body.insertAdjacentHTML('beforeend', modalHtml);
    const modal = new bootstrap.Modal(document.getElementById('error-modal'));
    modal.show();

    // Clean up when hidden
    document.getElementById('error-modal').addEventListener('hidden.bs.modal', function () {
        this.remove();
    });
}

document.addEventListener("DOMContentLoaded", () => {
    // ----- DOM ELEMENTS -----
    const csvUpload = document.getElementById("csv-upload");
    const uploadButton = document.getElementById("upload-button");
    const datasetTableBody = document.getElementById("dataset-table-body");

    const inputColumns = document.getElementById("input-columns");
    const targetContainer = document.getElementById("target-properties-container");
    const aprioriContainer = document.getElementById("apriori-properties-container");
    const addAprioriButton = document.getElementById("add-apriori-property-button");

    const modelSelect = document.getElementById("model-select");
    const modelHelpButton = document.getElementById("model-help-button");
    const curiositySlider = document.getElementById("curiosity-slider");
    const curiosityValueDisplay = document.getElementById("curiosity-value-display");
    const curiosityGuidanceText = document.getElementById("curiosity-guidance-text");

    const runButton = document.getElementById("run-experiment-button");
    const runLLMButton = document.getElementById("run-llm-experiment-button");
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
    // ----- STATE VARIABLES -----
    let allColumns = [];
    let uploadedDatasets = [];
    let experimentData = null;
    let resultsDataTable = null;
    let shapashData = null; // Store explainability data
    let lastExperimentConfig = null; // Store last experiment config for Excel export

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
        'rl': {
            name: 'Reinforcement Learning (RL)',
            description: 'Uses **PPO (Proximal Policy Optimization)** to learn optimal sample selection strategies. Combines prediction accuracy improvement AND discovery of high-performing materials. **Improves with each cycle** as it learns from session history.',
            warning: 'Best suited for **iterative active learning campaigns**. Initial selections may be less optimal, but performance improves as the agent learns from feedback.'
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
    //  ACTIVE LEARNING SETTINGS LOGIC
    // ==========================================================
    const modeRadios = document.getElementsByName('al_mode');
    const llmPanel = document.getElementById('llm-settings-panel');
    const hybridPanel = document.getElementById('hybrid-settings-panel');
    const mlOptionsPanel = document.getElementById('ml-options-panel');
    const llmOptionsPanel = document.getElementById('llm-options-panel');

    function updateSettingsVisibility() {
        // Find checked mode
        let mode = "ML_MODE";
        for (const r of modeRadios) {
            if (r.checked) mode = r.value;
        }

        if (mode === "ML_MODE") {
            // ML Only mode: Show ML options, hide LLM settings
            if (llmPanel) llmPanel.style.display = "none";
            if (hybridPanel) hybridPanel.style.display = "none";
            if (mlOptionsPanel) mlOptionsPanel.style.display = "block";
            if (llmOptionsPanel) llmOptionsPanel.style.display = "none";
        } else if (mode === "LLM_AGENT_MODE") {
            // LLM Only mode: Show LLM settings and LLM options, hide ML options
            if (llmPanel) llmPanel.style.display = "block";
            if (hybridPanel) hybridPanel.style.display = "none";
            if (mlOptionsPanel) mlOptionsPanel.style.display = "none";
            if (llmOptionsPanel) llmOptionsPanel.style.display = "block";
        } else if (mode === "HYBRID_MODE") {
            // Hybrid mode: Show both LLM and ML options
            if (llmPanel) llmPanel.style.display = "block";
            if (hybridPanel) hybridPanel.style.display = "block";
            if (mlOptionsPanel) mlOptionsPanel.style.display = "block";
            if (llmOptionsPanel) llmOptionsPanel.style.display = "none";
        }
    }

    // Load settings from server on page load
    function loadSettings() {
        fetch('/api/settings')
            .then(r => r.json())
            .then(data => {
                if (data.success && data.settings) {
                    const s = data.settings;

                    // Apply mode
                    if (s.active_learning_mode) {
                        for (const r of modeRadios) {
                            r.checked = (r.value === s.active_learning_mode);
                        }
                    }

                    // Apply prompt style
                    if (s.prompt_style) {
                        const styleRadios = document.getElementsByName('prompt_style');
                        for (const r of styleRadios) {
                            r.checked = (r.value === s.prompt_style);
                        }
                    }

                    // Apply hybrid weights
                    if (s.hybrid_weights) {
                        const wLlm = document.getElementById('w_llm');
                        const wMl = document.getElementById('w_ml');
                        if (wLlm) wLlm.value = s.hybrid_weights.w_llm || 0.5;
                        if (wMl) wMl.value = s.hybrid_weights.w_ml || 0.5;
                    }

                    // Apply Ollama model selection
                    const modelSelect = document.getElementById('ollama_model');
                    if (modelSelect && s.ollama_model) {
                        modelSelect.value = s.ollama_model;
                    }

                    // Apply LLM provider selection
                    if (s.llm_provider) {
                        const providerRadios = document.getElementsByName('llm_provider');
                        for (const r of providerRadios) {
                            r.checked = (r.value === s.llm_provider);
                        }
                    }

                    // Apply Mistral API key
                    const apiKeyInput = document.getElementById('mistral_api_key');
                    if (apiKeyInput && s.mistral_api_key) {
                        apiKeyInput.value = s.mistral_api_key;
                    }

                    // Restore previously uploaded dataset
                    if (s.current_dataset && s.current_dataset_columns) {
                        const existingDataset = uploadedDatasets.find(d => d.filename === s.current_dataset);
                        if (!existingDataset) {
                            const restoredDataset = {
                                filename: s.current_dataset,
                                columns: s.current_dataset_columns,
                                isActive: true
                            };
                            uploadedDatasets.push(restoredDataset);
                            const newIndex = uploadedDatasets.length - 1;
                            addDatasetRow(restoredDataset, newIndex);
                            handleDatasetSelection(newIndex);
                            console.log("📂 Restored dataset:", s.current_dataset);
                        }
                    }

                    updateSettingsVisibility();
                    console.log("✅ Settings loaded from server");
                }
            })
            .catch(err => console.warn("⚠️ Could not load settings:", err));
    }

    // Save settings to server
    function saveSettings() {
        let mode = "ML_MODE";
        for (const r of modeRadios) {
            if (r.checked) mode = r.value;
        }

        let promptStyle = "parameter-format";
        const styleRadios = document.getElementsByName('prompt_style');
        for (const r of styleRadios) {
            if (r.checked) promptStyle = r.value;
        }

        // Get LLM provider
        let llmProvider = 'ollama';
        const providerRadios = document.getElementsByName('llm_provider');
        for (const r of providerRadios) {
            if (r.checked) llmProvider = r.value;
        }

        // Get LLM strategy
        let llmStrategy = 'balanced';
        const strategyRadios = document.getElementsByName('llm_strategy');
        for (const r of strategyRadios) {
            if (r.checked) llmStrategy = r.value;
        }

        const settings = {
            active_learning_mode: mode,
            prompt_style: promptStyle,
            llm_provider: llmProvider,
            llm_strategy: llmStrategy,
            ollama_model: document.getElementById('ollama_model')?.value || 'mistral:latest',
            mistral_api_key: document.getElementById('mistral_api_key')?.value || '',
            hybrid_weights: {
                w_llm: parseFloat(document.getElementById('w_llm')?.value || 0.5),
                w_ml: parseFloat(document.getElementById('w_ml')?.value || 0.5)
            }
        };

        fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    console.log("✅ Settings saved");
                } else {
                    console.warn("⚠️ Failed to save settings:", data.error);
                }
            })
            .catch(err => console.error("❌ Error saving settings:", err));
    }

    // Attach save handlers to settings controls
    modeRadios.forEach(r => r.addEventListener('change', () => {
        updateSettingsVisibility();
        saveSettings();
        validateApiKey();
    }));

    document.getElementsByName('prompt_style').forEach(r =>
        r.addEventListener('change', saveSettings)
    );

    document.getElementById('w_llm')?.addEventListener('change', saveSettings);
    document.getElementById('w_ml')?.addEventListener('change', saveSettings);

    // Ollama model and Mistral API key handlers
    document.getElementById('ollama_model')?.addEventListener('change', saveSettings);
    document.getElementById('mistral_api_key')?.addEventListener('change', () => {
        saveSettings();
        validateCloudApiKey();
    });

    // Provider toggle handler
    const providerRadios = document.getElementsByName('llm_provider');
    const ollamaSettings = document.getElementById('ollama-settings');
    const mistralCloudSettings = document.getElementById('mistral-cloud-settings');

    function updateProviderVisibility() {
        let provider = 'ollama';
        for (const r of providerRadios) {
            if (r.checked) provider = r.value;
        }

        if (provider === 'ollama') {
            if (ollamaSettings) ollamaSettings.style.display = 'flex';
            if (mistralCloudSettings) mistralCloudSettings.style.display = 'none';
            checkOllamaStatus();
        } else {
            if (ollamaSettings) ollamaSettings.style.display = 'none';
            if (mistralCloudSettings) mistralCloudSettings.style.display = 'flex';
            validateCloudApiKey();
        }
    }

    providerRadios.forEach(r => r.addEventListener('change', () => {
        updateProviderVisibility();
        saveSettings();
    }));

    // Check Ollama status
    function checkOllamaStatus() {
        const statusDiv = document.getElementById('ollama-status');
        if (!statusDiv) return;

        fetch('http://localhost:11434/api/tags', { method: 'GET' })
            .then(response => {
                if (response.ok) return response.json();
                throw new Error('Ollama not responding');
            })
            .then(data => {
                const models = data.models?.map(m => m.name) || [];
                statusDiv.className = 'alert alert-success py-2 mt-2';
                statusDiv.innerHTML = `<span>✅</span> Ollama running! (${models.length} models)`;
            })
            .catch(err => {
                statusDiv.className = 'alert alert-warning py-2 mt-2';
                statusDiv.innerHTML = `<span>⚠️</span> Start Ollama: <code>ollama serve</code>`;
            });
    }

    // Validate Cloud API key
    function validateCloudApiKey() {
        const apiKey = document.getElementById('mistral_api_key')?.value || '';
        const warning = document.getElementById('api-key-warning');
        if (!apiKey.trim()) {
            if (warning) warning.style.display = 'block';
        } else {
            if (warning) warning.style.display = 'none';
        }
    }

    // Toggle API key visibility
    document.getElementById('toggle-api-key')?.addEventListener('click', () => {
        const input = document.getElementById('mistral_api_key');
        const icon = document.querySelector('#toggle-api-key i');
        if (input.type === 'password') {
            input.type = 'text';
            icon?.classList.replace('bi-eye', 'bi-eye-slash');
        } else {
            input.type = 'password';
            icon?.classList.replace('bi-eye-slash', 'bi-eye');
        }
    });

    // Check Ollama status when LLM panel becomes visible
    modeRadios.forEach(r => r.addEventListener('change', () => {
        const mode = r.value;
        if (mode === 'LLM_AGENT_MODE' || mode === 'HYBRID_MODE') {
            setTimeout(updateProviderVisibility, 100);
        }
    }));

    // Load settings on page init
    loadSettings();

    // Check for design space file in URL parameter
    checkForDesignSpaceParam();

    function checkForDesignSpaceParam() {
        const urlParams = new URLSearchParams(window.location.search);
        const dsFilename = urlParams.get('ds');

        if (!dsFilename) return;

        console.log(`📊 Loading design space from URL: ${dsFilename}`);

        // Fetch the design space file info from the server
        fetch(`/api/design-space-info?filename=${encodeURIComponent(dsFilename)}`)
            .then(r => r.json())
            .then(data => {
                if (!data.success) {
                    console.warn('⚠️ Could not load design space:', data.error);
                    return;
                }

                // Check if already loaded
                const existingDataset = uploadedDatasets.find(d => d.filename === dsFilename);
                if (existingDataset) {
                    const index = uploadedDatasets.indexOf(existingDataset);
                    handleDatasetSelection(index);
                    console.log(`✅ Design space '${dsFilename}' already loaded, selected.`);
                    return;
                }

                // Add as new dataset
                const newDataset = {
                    filename: dsFilename,
                    columns: data.columns,
                    isActive: true,
                    isDesignSpace: true
                };
                uploadedDatasets.push(newDataset);

                const newIndex = uploadedDatasets.length - 1;
                addDatasetRow(newDataset, newIndex);
                handleDatasetSelection(newIndex);

                console.log(`✅ Design space '${dsFilename}' loaded with ${data.columns.length} columns`);

                // Show a brief success message
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert alert-success alert-dismissible fade show';
                alertDiv.innerHTML = `
                    <strong>Design Space Loaded!</strong> ${dsFilename} with ${data.columns.length} columns.
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                `;
                document.querySelector('.container-fluid')?.prepend(alertDiv);

                // Auto-dismiss after 5 seconds
                setTimeout(() => alertDiv.remove(), 5000);
            })
            .catch(err => {
                console.error('❌ Error loading design space:', err);
            });
    }

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
            <td>
                <div class="btn-group btn-group-sm" role="group">
                    <button class="btn btn-outline-success select-btn" title="Select this dataset">
                        <i class="bi bi-check-circle"></i>
                    </button>
                    <a href="/data/${dataset.filename}" download class="btn btn-outline-primary download-btn" title="Download dataset">
                        <i class="bi bi-download"></i>
                    </a>
                    <button class="btn btn-outline-danger delete-btn" title="Delete dataset">
                        <i class="bi bi-trash"></i>
                    </button>
                </div>
            </td>
            <td>${dataset.filename}</td>
            <td class="text-truncate" style="max-width: 400px;">${dataset.columns.join(", ")}</td>
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
        aprioriContainer.innerHTML = '';
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
                    selectBtn.innerHTML = '<i class="bi bi-check-circle-fill"></i>';
                    selectBtn.classList.remove('btn-outline-success');
                    selectBtn.classList.add('btn-success');
                    selectBtn.disabled = true;
                    selectBtn.title = 'Currently selected';
                }
            } else {
                row.classList.remove('table-success', 'fw-bold');
                if (selectBtn) {
                    selectBtn.innerHTML = '<i class="bi bi-check-circle"></i>';
                    selectBtn.classList.remove('btn-success');
                    selectBtn.classList.add('btn-outline-success');
                    selectBtn.disabled = false;
                    selectBtn.title = 'Select this dataset';
                }
            }
        }
    }

    // ==========================================================
    //  COLUMN SELECTORS & TARGET PROPERTY GROUPS (Keep as is)
    // ==========================================================

    function populateInitialSelectors() {
        setSelectOptions(inputColumns, allColumns);
    }

    inputColumns.addEventListener("change", updateAvailableColumns);

    function updateAvailableColumns() {
        // Just trigger re-evaluation when inputs change
        // A-priori now uses dynamic groups like targets
    }

    function getAvailableAprioriColumns() {
        if (allColumns.length === 0) return [];

        const selectedInputs = getSelected(inputColumns);
        const selectedTargets = Array.from(
            document.querySelectorAll("select[name='target_columns']")
        ).map(sel => sel.value);
        const selectedApriori = Array.from(
            document.querySelectorAll("select[name='apriori_columns']")
        ).map(sel => sel.value);

        return allColumns.filter(
            c => !selectedInputs.includes(c) &&
                !selectedTargets.includes(c) &&
                !selectedApriori.includes(c)
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
        });

        wrapper.querySelector("select[name='target_columns']")
            .addEventListener("change", () => { });

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
    //  A-PRIORI PROPERTY GROUPS (with min/max like targets)
    // ==========================================================

    addAprioriButton.addEventListener("click", () => addAprioriGroup());

    function addAprioriGroup() {
        if (allColumns.length === 0) {
            alert("Please upload and select a dataset first.");
            return;
        }

        const wrapper = document.createElement("div");
        wrapper.classList.add("mb-3", "apriori-group");

        const availableColumns = getAvailableAprioriColumns();

        if (availableColumns.length === 0) {
            alert("No columns available for a-priori properties. All columns are assigned to inputs, targets, or other a-priori.");
            return;
        }

        const optionsHTML = availableColumns
            .map(c => `<option value="${c}">${c}</option>`)
            .join("");

        wrapper.innerHTML = `
            <div class="input-group">
                <select class="form-select" name="apriori_columns">
                    ${optionsHTML}
                </select>
                <input type="number" class="form-control" name="apriori_weights" value="1.0" step="0.1" title="Weight">
                <select class="form-select" name="apriori_max_or_min">
                    <option value="max">Maximize</option>
                    <option value="min">Minimize</option>
                </select>
                <button class="btn btn-danger" type="button">Remove</button>
            </div>
        `;

        wrapper.querySelector(".btn-danger").addEventListener("click", () => {
            wrapper.remove();
        });

        aprioriContainer.appendChild(wrapper);
    }

    function collectAprioriConfig() {
        return Array.from(document.querySelectorAll(".apriori-group")).map(g => ({
            name: g.querySelector("select[name='apriori_columns']").value,
            weight: parseFloat(g.querySelector("input[name='apriori_weights']").value),
            optimization: g.querySelector("select[name='apriori_max_or_min']").value
        }));
    }
    // ==========================================================

    runButton.addEventListener("click", runExperiment);

    // LLM Run button uses same experiment logic (mode is determined by settings)
    if (runLLMButton) {
        runLLMButton.addEventListener("click", runExperiment);
    }

    // LLM Strategy toggle - update description
    const llmStrategyRadios = document.getElementsByName('llm_strategy');
    const llmStrategyDesc = document.getElementById('llm-strategy-description');

    llmStrategyRadios.forEach(radio => {
        radio.addEventListener('change', () => {
            const descriptions = {
                'balanced': 'Balanced: LLM decides the best approach based on context.',
                'explore': 'Explore: Prioritize diversity and unexplored regions.',
                'exploit': 'Exploit: Focus on refining high-performing candidates.'
            };
            if (llmStrategyDesc) {
                llmStrategyDesc.textContent = descriptions[radio.value] || '';
            }
            saveSettings();
        });
    });

    // Clear Trajectory button handler
    const clearTrajectoryBtn = document.getElementById('clear-trajectory-btn');
    if (clearTrajectoryBtn) {
        clearTrajectoryBtn.addEventListener('click', () => {
            if (confirm('Clear trajectory history? This will reset the exploration path visualization.')) {
                fetch('/api/trajectory', { method: 'DELETE' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            // Clear the plots
                            const trajectoryPlot = document.getElementById('trajectory-plot');
                            const distancePlot = document.getElementById('distance-plot');
                            if (trajectoryPlot) trajectoryPlot.innerHTML = '<p class="text-muted text-center mt-5">No trajectory data. Run experiments to see the path.</p>';
                            if (distancePlot) distancePlot.innerHTML = '';

                            // Reset stats
                            document.getElementById('stat-iterations').textContent = '0';
                            document.getElementById('stat-distance').textContent = '0.00';

                            console.log('✅ Trajectory cleared');
                        }
                    })
                    .catch(err => console.error('Failed to clear trajectory:', err));
            }
        });
    }


    function runExperiment() {
        const selectedInputs = getSelected(inputColumns);
        const targets = collectTargetConfig();
        const apriori = collectAprioriConfig();

        if (selectedInputs.length === 0 || targets.length === 0) {
            return alert("Please select input and target columns.");
        }

        const activeDataset = uploadedDatasets.find(d => d.isActive);
        if (!activeDataset) {
            return alert("No dataset is currently selected. Please select one from the table.");
        }

        // Collect Settings
        let mode = "ML_MODE";
        for (const r of modeRadios) {
            if (r.checked) mode = r.value;
        }

        // Prompt Style
        let promptStyle = "parameter-format";
        const styleRadios = document.getElementsByName('prompt_style');
        for (const r of styleRadios) {
            if (r.checked) promptStyle = r.value;
        }

        const w_llm = parseFloat(document.getElementById('w_llm')?.value || 0.5);
        const w_ml = parseFloat(document.getElementById('w_ml')?.value || 0.5);
        const acquisitionFunc = document.getElementById('acquisition-select')?.value || 'webslamd';
        const batchSize = parseInt(document.getElementById('batch-size-select')?.value || 1);

        const payload = {
            model: modelSelect.value,
            curiosity: curiositySlider.value,
            dataset_filename: activeDataset.filename,
            input_columns: selectedInputs,
            target_columns: targets,
            apriori_columns: apriori,
            acquisition_function: acquisitionFunc,
            active_learning_mode: mode,
            prompt_style: promptStyle,
            hybrid_weights: { w_llm, w_ml },
            batch_size: batchSize
        };

        // Store config for Excel export with metadata
        lastExperimentConfig = {
            ...payload,
            timestamp: new Date().toISOString(),
            dataset_filename: activeDataset.filename
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
                const phaseEl = document.getElementById('progress-phase');
                if (phaseEl) phaseEl.textContent = 'Receiving data...';

                if (!r.ok) {
                    throw new Error(`HTTP ${r.status}: ${r.statusText}`);
                }

                return r.json();
            })
            .then(data => {
                console.log("📦 Data received, size:", JSON.stringify(data).length, "characters");

                const phaseEl = document.getElementById('progress-phase');
                if (phaseEl) phaseEl.textContent = 'Parsing response...';

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
            // Reset button UI
            runButton.disabled = false;
            runButton.innerHTML = 'Run Experiment';

            if (!data.success) {
                // Show error in a proper modal popup
                const errorType = data.error_type || 'error';
                const errorMsg = data.error || 'Unknown error';

                // Check if it's a preprocessing error (like column validation)
                if (errorType === 'preprocessing') {
                    showErrorModal(
                        'Validation Error',
                        errorMsg,
                        data.all_errors || []
                    );
                } else {
                    showErrorModal('Experiment Error', errorMsg, []);
                }
                return;
            }

            // Store in global state
            experimentData = data;

            // ✅ Also cache in localStorage so we can restore after navigation
            try {
                localStorage.setItem('metadesign_last_results', JSON.stringify(data));
                console.log("💾 Cached experiment results to localStorage");
            } catch (e) {
                console.warn("⚠️ Could not cache results in localStorage:", e);
            }

            // Update progress text
            const progressTitle = document.getElementById('progress-title');
            const progressPhase = document.getElementById('progress-phase');
            if (progressTitle) progressTitle.textContent = 'Rendering visualizations...';
            if (progressPhase) progressPhase.textContent = 'Creating plots...';

            // Render all plots and table
            renderResultsProgressively(data);

        } catch (err) {
            console.error("💥 Error processing results:", err);
            resultsTableContainer.innerHTML = `<div class="alert alert-danger"><strong>Processing Error:</strong> ${err.message}</div>`;
        }
    }

    // PROGRESSIVE RENDERING: Render one plot at a time with delays
    function renderResultsProgressively(data) {
        console.log("📊 Progressive rendering started");

        // Stage 1: Show results section
        resultsSection.style.display = "block";

        // Stage 2: Render table (fastest)
        setTimeout(() => {
            console.log("📊 Rendering table...");
            renderTable(data.results_table);

            // Stage 3: Render scatter plot
            setTimeout(() => {
                console.log("📊 Rendering scatter plot...");
                drawPlot("scatter-plot", data.target_scatter_figure);

                // Stage 4: Render TSNE plot
                setTimeout(() => {
                    console.log("📊 Rendering TSNE plot...");
                    drawPlot("tsne-plot", data.tsne_figure);

                    // Stage 5: Render remaining plots
                    setTimeout(() => {
                        console.log("📊 Rendering uncertainty plot...");
                        drawPlot("uncertainty-plot", data.uncertainty_plot);

                        setTimeout(() => {
                            console.log("📊 Rendering history plot...");
                            drawPlot("history-plot", data.history_plot);

                            setTimeout(() => {
                                console.log("📊 Rendering utility surface plot...");
                                drawPlot("utility-surface-plot", data.utility_surface_plot);

                                // Render trajectory plots
                                setTimeout(() => {
                                    console.log("📊 Rendering trajectory plots...");
                                    if (data.trajectory_plot) {
                                        drawPlot("trajectory-plot", data.trajectory_plot);
                                    }
                                    if (data.distance_plot) {
                                        drawPlot("distance-plot", data.distance_plot);
                                    }

                                    // Update trajectory stats
                                    if (data.trajectory_summary) {
                                        const stats = data.trajectory_summary;
                                        document.getElementById('stat-iterations').textContent = stats.total_iterations || 0;
                                        document.getElementById('stat-distance').textContent = (stats.total_distance || 0).toFixed(2);
                                    }

                                    console.log("✅ All plots rendered!");

                                    // Store new analysis plots for radio button handlers
                                    if (experimentData) {
                                        experimentData.feature_importance_plot = data.feature_importance_plot;
                                        experimentData.prediction_actual_plot = data.prediction_actual_plot;
                                    }

                                    // Setup event handlers for Model Analysis radio buttons
                                    setupModelAnalysisRadios(data);

                                }, 200);
                            }, 300);
                        }, 200);
                    }, 200);
                }, 200);
            }, 150);
        }, 100);
    }

    // Setup Model Analysis radio button event handlers
    function setupModelAnalysisRadios(data) {
        const featureImportanceRadio = document.getElementById('feature-importance-radio');
        const predictionActualRadio = document.getElementById('prediction-actual-radio');
        const plotContainer = document.getElementById('additional-plot-container');

        if (featureImportanceRadio) {
            featureImportanceRadio.addEventListener('change', function () {
                if (this.checked && data.feature_importance_plot) {
                    plotContainer.innerHTML = '<div id="feature-importance-plot" style="width:100%;height:400px;"></div>';
                    setTimeout(() => {
                        Plotly.newPlot('feature-importance-plot', data.feature_importance_plot.data, data.feature_importance_plot.layout, { responsive: true });
                    }, 100);
                }
            });
        }

        if (predictionActualRadio) {
            predictionActualRadio.addEventListener('change', function () {
                if (this.checked && data.prediction_actual_plot) {
                    plotContainer.innerHTML = '<div id="prediction-actual-plot" style="width:100%;height:500px;"></div>';
                    setTimeout(() => {
                        Plotly.newPlot('prediction-actual-plot', data.prediction_actual_plot.data, data.prediction_actual_plot.layout, { responsive: true });
                    }, 100);
                }
            });
        }
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

            // Add a header row for column filters
            const thead = newTable.querySelector('thead');
            if (thead) {
                const filterRow = document.createElement('tr');
                filterRow.id = 'filter-row';
                filterRow.classList.add('bg-light');
                const headerCells = thead.querySelectorAll('th');
                headerCells.forEach(() => {
                    filterRow.appendChild(document.createElement('th'));
                });
                thead.appendChild(filterRow);
            }

            resultsDataTable = new DataTable(newTable, {
                paging: true,
                searching: true,
                ordering: true,  // User can still click columns to re-sort
                order: [],  // PRESERVE SERVER-SIDE ORDER - don't apply initial sort
                info: true,
                responsive: true,
                deferRender: true,  // Performance optimization
                dom: 'lfrtipB',
                orderCellsTop: true,  // Keep sorting on first header row
                // Highlight rows where "Selected for Testing" is True
                createdRow: function (row, data, dataIndex) {
                    // Find the column index for "Selected for Testing"
                    const headers = Array.from(newTable.querySelectorAll('thead th')).map(th => th.textContent.trim());
                    const selectedColIdx = headers.findIndex(h => h.toLowerCase().includes('selected for testing'));

                    if (selectedColIdx >= 0 && data[selectedColIdx]) {
                        const cellValue = String(data[selectedColIdx]).toLowerCase().trim();
                        if (cellValue === 'true' || cellValue === '1' || cellValue === 'yes') {
                            row.classList.add('table-primary', 'fw-bold');
                            row.style.backgroundColor = '#cfe2ff';  // Light blue highlight
                            row.style.borderLeft = '4px solid #0d6efd';  // Blue left border
                        }
                    }
                },
                buttons: [
                    {
                        extend: 'csv',
                        text: '<i class="bi bi-file-earmark-arrow-down"></i> Download CSV',
                        className: 'btn-sm btn-primary ms-2'
                    },
                    {
                        text: '<i class="bi bi-file-earmark-excel"></i> Download Excel (with Metadata)',
                        className: 'btn-sm btn-success ms-2',
                        action: function (e, dt, node, config) {
                            exportExcelWithMetadata(dt);
                        }
                    }
                ],
                initComplete: function () {
                    // Add column filters like Excel
                    const api = this.api();
                    const filterRowCells = document.querySelectorAll('#filter-row th');

                    api.columns().every(function (colIdx) {
                        const column = this;
                        const header = column.header().textContent.trim();
                        const filterCell = filterRowCells[colIdx];

                        if (!filterCell) return;

                        // Check if column has numeric data
                        let isNumeric = true;
                        column.data().each(function (d) {
                            if (d !== null && d !== '' && isNaN(parseFloat(d))) {
                                isNumeric = false;
                            }
                        });

                        // For columns with few unique values, use dropdown
                        const uniqueValues = [...new Set(column.data().toArray())].filter(v => v !== null && v !== '');

                        if (uniqueValues.length <= 10 && uniqueValues.length > 0) {
                            // Dropdown filter for categorical/boolean columns
                            const select = document.createElement('select');
                            select.classList.add('form-select', 'form-select-sm');
                            select.style.minWidth = '80px';
                            select.innerHTML = '<option value="">All</option>';

                            uniqueValues.sort().forEach(function (val) {
                                const displayVal = String(val).length > 15 ? String(val).substring(0, 12) + '...' : val;
                                select.innerHTML += `<option value="${val}">${displayVal}</option>`;
                            });

                            select.addEventListener('change', function () {
                                const val = this.value;
                                column.search(val ? '^' + escapeRegex(val) + '$' : '', true, false).draw();
                            });

                            filterCell.appendChild(select);
                        } else if (isNumeric && uniqueValues.length > 0) {
                            // Range inputs for numeric columns
                            const container = document.createElement('div');
                            container.classList.add('d-flex', 'gap-1');
                            container.innerHTML = `
                                <input type="number" class="form-control form-control-sm filter-min" placeholder="Min" style="width:60px;">
                                <input type="number" class="form-control form-control-sm filter-max" placeholder="Max" style="width:60px;">
                            `;

                            const minInput = container.querySelector('.filter-min');
                            const maxInput = container.querySelector('.filter-max');

                            const filterNumeric = function () {
                                const min = parseFloat(minInput.value) || -Infinity;
                                const max = parseFloat(maxInput.value) || Infinity;

                                // Custom filter function
                                $.fn.dataTable.ext.search.push(function (settings, data, dataIndex) {
                                    const val = parseFloat(data[colIdx]) || 0;
                                    return val >= min && val <= max;
                                });

                                api.draw();

                                // Remove filter after draw to avoid accumulation
                                $.fn.dataTable.ext.search.pop();
                            };

                            minInput.addEventListener('change', filterNumeric);
                            maxInput.addEventListener('change', filterNumeric);

                            filterCell.appendChild(container);
                        } else {
                            // Text search for other columns
                            const input = document.createElement('input');
                            input.type = 'text';
                            input.classList.add('form-control', 'form-control-sm');
                            input.placeholder = 'Search...';
                            input.style.minWidth = '70px';

                            input.addEventListener('keyup', function () {
                                if (column.search() !== this.value) {
                                    column.search(this.value).draw();
                                }
                            });

                            filterCell.appendChild(input);
                        }
                    });
                }
            });
            console.log("✅ DataTable initialized with column filters");
        }
    }

    // Helper function to escape regex special characters
    function escapeRegex(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    // =========================================================
    //  EXCEL EXPORT WITH EXPERIMENT METADATA
    // =========================================================
    function exportExcelWithMetadata(dt) {
        // Check if XLSX library is available
        if (typeof XLSX === 'undefined') {
            // Fallback to regular export if XLSX not loaded
            console.warn("XLSX library not loaded, using basic export");
            dt.button('.buttons-csv').trigger();
            return;
        }

        try {
            // Create new workbook
            const wb = XLSX.utils.book_new();

            // === Sheet 1: Results ===
            // Get table data from DataTable
            const headers = [];
            dt.columns().header().each(function (th) {
                headers.push(th.textContent.trim());
            });

            const tableData = [headers];
            dt.rows().data().each(function (row) {
                tableData.push(Array.from(row));
            });

            const wsResults = XLSX.utils.aoa_to_sheet(tableData);
            XLSX.utils.book_append_sheet(wb, wsResults, "Results");

            // === Sheet 2: Experiment Info ===
            const configData = [
                ["Experiment Configuration", ""],
                ["", ""],
                ["Timestamp", lastExperimentConfig?.timestamp || new Date().toISOString()],
                ["Dataset", lastExperimentConfig?.dataset_filename || "Unknown"],
                ["", ""],
                ["Model Settings", ""],
                ["Model", lastExperimentConfig?.model || "Unknown"],
                ["Acquisition Function", lastExperimentConfig?.acquisition_function || "webslamd"],
                ["Curiosity", lastExperimentConfig?.curiosity || 0],
                ["Active Learning Mode", lastExperimentConfig?.active_learning_mode || "ML_MODE"],
                ["", ""],
                ["Target Properties", ""],
            ];

            // Add target properties
            if (lastExperimentConfig?.target_columns) {
                lastExperimentConfig.target_columns.forEach((target, idx) => {
                    configData.push([
                        `Target ${idx + 1}: ${target.name}`,
                        `Weight: ${target.weight}, Optimization: ${target.optimization}`
                    ]);
                });
            }

            // Add a-priori properties if any
            if (lastExperimentConfig?.apriori_columns && lastExperimentConfig.apriori_columns.length > 0) {
                configData.push(["", ""]);
                configData.push(["A-Priori Properties", ""]);
                lastExperimentConfig.apriori_columns.forEach((apriori, idx) => {
                    configData.push([
                        `A-Priori ${idx + 1}: ${apriori.name}`,
                        `Weight: ${apriori.weight}, Optimization: ${apriori.optimization}`
                    ]);
                });
            }

            // Add input columns
            if (lastExperimentConfig?.input_columns) {
                configData.push(["", ""]);
                configData.push(["Input Features", lastExperimentConfig.input_columns.join(", ")]);
            }

            const wsConfig = XLSX.utils.aoa_to_sheet(configData);
            // Set column widths
            wsConfig['!cols'] = [{ wch: 30 }, { wch: 50 }];
            XLSX.utils.book_append_sheet(wb, wsConfig, "Experiment Info");

            // Generate filename with timestamp
            const now = new Date();
            const timestamp = now.toISOString().slice(0, 19).replace(/[:-]/g, '');
            const filename = `experiment_results_${timestamp}.xlsx`;

            // Download
            XLSX.writeFile(wb, filename);
            console.log("✅ Excel exported with metadata:", filename);

        } catch (error) {
            console.error("Excel export error:", error);
            alert("Excel export failed. Please try CSV export instead.");
        }
    }

    // =========================================================
    //  EXPERIMENT HISTORY FUNCTIONS (Enhanced)
    // =========================================================

    // Store all experiments for filtering
    let allExperiments = [];
    let selectedExperimentIds = new Set();

    async function loadExperimentHistory() {
        try {
            const response = await fetch('/api/experiments');
            const data = await response.json();

            const tbody = document.getElementById('experiment-history-body');
            if (!tbody) return;

            if (!data.success || !data.experiments || data.experiments.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" class="text-muted text-center py-4">No experiments logged yet. Run an experiment to start tracking.</td></tr>';
                updateExperimentCounts(0, 0);
                allExperiments = [];
                return;
            }

            allExperiments = data.experiments;
            applyFiltersAndRender();

            console.log(`✅ Loaded ${data.experiments.length} experiments`);

        } catch (error) {
            console.error("Error loading experiment history:", error);
        }
    }

    function applyFiltersAndRender() {
        const filterModel = document.getElementById('filter-model')?.value || '';
        const filterStatus = document.getElementById('filter-status')?.value || '';
        const filterAcquisition = document.getElementById('filter-acquisition')?.value || '';

        let filtered = allExperiments;

        if (filterModel) {
            filtered = filtered.filter(exp => {
                const model = exp.metrics?.model || exp.name?.split('_')[0] || '';
                return model.toLowerCase().includes(filterModel.toLowerCase());
            });
        }

        if (filterStatus) {
            filtered = filtered.filter(exp => exp.status === filterStatus);
        }

        if (filterAcquisition) {
            filtered = filtered.filter(exp => {
                const acq = exp.metrics?.acquisition || exp.config?.acquisition || 'webslamd';
                return acq.toLowerCase().includes(filterAcquisition.toLowerCase());
            });
        }

        renderExperimentTable(filtered);
        updateExperimentCounts(filtered.length, selectedExperimentIds.size);
    }

    function renderExperimentTable(experiments) {
        const tbody = document.getElementById('experiment-history-body');
        if (!tbody) return;

        if (experiments.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-muted text-center py-4">No experiments match the current filters.</td></tr>';
            return;
        }

        tbody.innerHTML = experiments.map(exp => {
            const timestamp = formatTimestamp(exp.start_time);
            const model = exp.metrics?.model || exp.name?.split('_')[0] || 'Unknown';
            const acquisition = exp.metrics?.acquisition || exp.config?.acquisition || 'webslamd';
            const utilityMax = exp.metrics?.utility_max ? exp.metrics.utility_max.toFixed(2) : '-';
            const isChecked = selectedExperimentIds.has(exp.id) ? 'checked' : '';

            const statusBadge = exp.status === 'completed'
                ? '<span class="badge bg-success">✓</span>'
                : exp.status === 'failed'
                    ? '<span class="badge bg-danger">✗</span>'
                    : '<span class="badge bg-warning">⏳</span>';

            // Generate mini sparkline SVG
            const sparkline = generateSparkline(exp.metrics?.utility_history || []);

            return `
                <tr data-exp-id="${exp.id}">
                    <td>
                        <input type="checkbox" class="form-check-input exp-checkbox" 
                               data-exp-id="${exp.id}" ${isChecked}
                               onchange="toggleExperimentSelection('${exp.id}', this.checked)">
                    </td>
                    <td><small>${timestamp}</small></td>
                    <td><strong>${model}</strong></td>
                    <td><small>${acquisition}</small></td>
                    <td>${statusBadge}</td>
                    <td><strong>${utilityMax}</strong></td>
                    <td>${sparkline}</td>
                    <td>
                        <div class="btn-group btn-group-sm">
                            <button class="btn btn-outline-info btn-sm" onclick="showExperimentDetail('${exp.id}')" title="View Details">
                                <i class="bi bi-eye"></i>
                            </button>
                            <button class="btn btn-outline-danger btn-sm" onclick="deleteExperiment('${exp.id}')" title="Delete">
                                <i class="bi bi-trash"></i>
                            </button>
                        </div>
                    </td>
                </tr>
            `;
        }).join('');
    }

    function generateSparkline(history) {
        if (!history || history.length < 2) {
            return '<span class="text-muted">-</span>';
        }

        // Normalize values to 0-20 range for SVG
        const min = Math.min(...history);
        const max = Math.max(...history);
        const range = max - min || 1;

        const width = 60;
        const height = 20;
        const points = history.map((val, i) => {
            const x = (i / (history.length - 1)) * width;
            const y = height - ((val - min) / range) * height;
            return `${x},${y}`;
        }).join(' ');

        const trend = history[history.length - 1] > history[0] ? '#198754' : '#dc3545';

        return `<svg width="${width}" height="${height}" style="vertical-align: middle;">
            <polyline points="${points}" fill="none" stroke="${trend}" stroke-width="1.5"/>
            <circle cx="${width}" cy="${height - ((history[history.length - 1] - min) / range) * height}" r="2" fill="${trend}"/>
        </svg>`;
    }

    function updateExperimentCounts(total, selected) {
        const countEl = document.getElementById('experiment-count');
        const selectedEl = document.getElementById('selected-count');
        if (countEl) countEl.textContent = total;
        if (selectedEl) selectedEl.textContent = selected;

        // Enable/disable buttons based on selection
        const compareBtn = document.getElementById('compare-selected-btn');
        const deleteBtn = document.getElementById('delete-selected-btn');
        if (compareBtn) compareBtn.disabled = selected < 2;
        if (deleteBtn) deleteBtn.disabled = selected === 0;
    }

    window.toggleExperimentSelection = function (expId, isSelected) {
        if (isSelected) {
            selectedExperimentIds.add(expId);
        } else {
            selectedExperimentIds.delete(expId);
        }
        updateExperimentCounts(allExperiments.length, selectedExperimentIds.size);
    };

    // Select All checkbox handler
    document.getElementById('select-all-experiments')?.addEventListener('change', function () {
        const checkboxes = document.querySelectorAll('.exp-checkbox');
        checkboxes.forEach(cb => {
            cb.checked = this.checked;
            const expId = cb.dataset.expId;
            if (this.checked) {
                selectedExperimentIds.add(expId);
            } else {
                selectedExperimentIds.delete(expId);
            }
        });
        updateExperimentCounts(allExperiments.length, selectedExperimentIds.size);
    });

    // Filter change handlers
    ['filter-model', 'filter-status', 'filter-acquisition'].forEach(id => {
        document.getElementById(id)?.addEventListener('change', applyFiltersAndRender);
    });

    // Clear filters button
    document.getElementById('clear-filters-btn')?.addEventListener('click', () => {
        document.getElementById('filter-model').value = '';
        document.getElementById('filter-status').value = '';
        document.getElementById('filter-acquisition').value = '';
        applyFiltersAndRender();
    });

    // Compare Selected button
    document.getElementById('compare-selected-btn')?.addEventListener('click', async () => {
        if (selectedExperimentIds.size < 2) {
            alert('Select at least 2 experiments to compare');
            return;
        }

        const experiments = allExperiments.filter(e => selectedExperimentIds.has(e.id));
        showExperimentComparison(experiments);
    });

    function showExperimentComparison(experiments) {
        const content = document.getElementById('experiment-compare-content');
        if (!content) return;

        const headers = experiments.map(exp => {
            const model = exp.metrics?.model || exp.name?.split('_')[0] || 'Unknown';
            return `<th class="text-center">${model}<br><small class="text-muted">${formatTimestamp(exp.start_time)}</small></th>`;
        }).join('');

        const compareRows = [
            { label: 'Model', key: exp => exp.metrics?.model || exp.name?.split('_')[0] || '-' },
            { label: 'Acquisition', key: exp => exp.metrics?.acquisition || exp.config?.acquisition || 'webslamd' },
            { label: 'Curiosity', key: exp => exp.config?.curiosity?.toFixed(2) || '-' },
            { label: 'Max Utility', key: exp => exp.metrics?.utility_max?.toFixed(3) || '-' },
            { label: 'Mean Utility', key: exp => exp.metrics?.utility_mean?.toFixed(3) || '-' },
            { label: 'Candidates', key: exp => exp.metrics?.num_candidates || '-' },
            { label: 'Status', key: exp => exp.status || '-' },
            { label: 'Duration', key: exp => calculateDuration(exp.start_time, exp.end_time) },
        ];

        const rows = compareRows.map(row => {
            const cells = experiments.map(exp => `<td class="text-center">${row.key(exp)}</td>`).join('');
            return `<tr><td><strong>${row.label}</strong></td>${cells}</tr>`;
        }).join('');

        content.innerHTML = `
            <table class="table table-bordered table-hover">
                <thead class="table-light">
                    <tr>
                        <th>Metric</th>
                        ${headers}
                    </tr>
                </thead>
                <tbody>
                    ${rows}
                </tbody>
            </table>
            <div class="alert alert-info mt-3">
                <i class="bi bi-info-circle"></i> 
                <strong>Tip:</strong> Higher Max Utility indicates better candidate selection.
            </div>
        `;

        const modal = new bootstrap.Modal(document.getElementById('experimentCompareModal'));
        modal.show();
    }

    // Delete Selected button
    document.getElementById('delete-selected-btn')?.addEventListener('click', async () => {
        if (selectedExperimentIds.size === 0) return;

        if (!confirm(`Delete ${selectedExperimentIds.size} experiment(s)? This cannot be undone.`)) {
            return;
        }

        for (const expId of selectedExperimentIds) {
            await deleteExperimentById(expId, false);
        }

        selectedExperimentIds.clear();
        loadExperimentHistory();
    });

    window.deleteExperiment = async function (expId) {
        if (!confirm('Delete this experiment? This cannot be undone.')) {
            return;
        }
        await deleteExperimentById(expId, true);
    };

    async function deleteExperimentById(expId, reload = true) {
        try {
            const response = await fetch(`/api/experiments/${expId}`, { method: 'DELETE' });
            const data = await response.json();
            if (data.success) {
                console.log(`✅ Deleted experiment: ${expId}`);
                if (reload) loadExperimentHistory();
            } else {
                console.error(`Failed to delete experiment: ${expId}`);
            }
        } catch (error) {
            console.error(`Error deleting experiment ${expId}:`, error);
        }
    }

    // Make this function global for button onclick
    window.showExperimentDetail = async function (expId) {
        try {
            const response = await fetch(`/api/experiments/${expId}`);
            const data = await response.json();

            if (!data.success) {
                alert('Could not load experiment details');
                return;
            }

            const exp = data.experiment;
            const content = document.getElementById('experiment-detail-content');

            content.innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <h6><i class="bi bi-info-circle"></i> Basic Info</h6>
                        <table class="table table-sm">
                            <tr><td><strong>ID</strong></td><td>${exp.id || '-'}</td></tr>
                            <tr><td><strong>Timestamp</strong></td><td>${formatTimestamp(exp.start_time)}</td></tr>
                            <tr><td><strong>Status</strong></td><td>${exp.status || '-'}</td></tr>
                            <tr><td><strong>Duration</strong></td><td>${calculateDuration(exp.start_time, exp.end_time)}</td></tr>
                        </table>
                    </div>
                    <div class="col-md-6">
                        <h6><i class="bi bi-gear"></i> Configuration</h6>
                        <table class="table table-sm">
                            <tr><td><strong>Model</strong></td><td>${exp.config?.model || '-'}</td></tr>
                            <tr><td><strong>Acquisition</strong></td><td>${exp.config?.acquisition || 'webslamd'}</td></tr>
                            <tr><td><strong>Curiosity</strong></td><td>${exp.config?.curiosity || '-'}</td></tr>
                            <tr><td><strong>Targets</strong></td><td>${exp.config?.num_targets || '-'}</td></tr>
                        </table>
                    </div>
                </div>
                <hr>
                <h6><i class="bi bi-graph-up"></i> Metrics</h6>
                <div class="row">
                    <div class="col-md-4">
                        <div class="card text-center">
                            <div class="card-body py-2">
                                <div class="fs-4 text-primary">${exp.metrics?.utility_max?.toFixed(3) || '-'}</div>
                                <small class="text-muted">Max Utility</small>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card text-center">
                            <div class="card-body py-2">
                                <div class="fs-4 text-info">${exp.metrics?.utility_mean?.toFixed(3) || '-'}</div>
                                <small class="text-muted">Mean Utility</small>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card text-center">
                            <div class="card-body py-2">
                                <div class="fs-4 text-success">${exp.metrics?.num_candidates || '-'}</div>
                                <small class="text-muted">Candidates</small>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Show the modal
            const modal = new bootstrap.Modal(document.getElementById('experimentDetailModal'));
            modal.show();

        } catch (error) {
            console.error("Error showing experiment detail:", error);
            alert('Error loading experiment details');
        }
    };

    function formatTimestamp(isoString) {
        if (!isoString) return '-';
        try {
            const date = new Date(isoString);
            return date.toLocaleString('en-US', {
                month: 'short', day: 'numeric',
                hour: '2-digit', minute: '2-digit'
            });
        } catch (e) {
            return isoString;
        }
    }

    function calculateDuration(start, end) {
        if (!start || !end) return '-';
        try {
            const startDate = new Date(start);
            const endDate = new Date(end);
            const diffMs = endDate - startDate;
            const diffSecs = Math.round(diffMs / 1000);
            if (diffSecs < 60) return `${diffSecs}s`;
            const diffMins = Math.round(diffSecs / 60);
            return `${diffMins}m ${diffSecs % 60}s`;
        } catch (e) {
            return '-';
        }
    }

    // Event listener for refresh button
    const refreshHistoryBtn = document.getElementById('refresh-history-btn');
    if (refreshHistoryBtn) {
        refreshHistoryBtn.addEventListener('click', loadExperimentHistory);
    }

    // Load experiment history on page load
    loadExperimentHistory();

    // GENERIC PLOT HELPER (unchanged except logs)
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
    //  RESTORE FROM LOCALSTORAGE ON DASHBOARD LOAD
    // ==========================================================
    function restoreLastResultsFromCache() {
        try {
            const cached = localStorage.getItem('metadesign_last_results');
            if (!cached) {
                console.log("ℹ️ No cached experiment results found.");
                return;
            }

            const parsed = JSON.parse(cached);
            if (!parsed || !parsed.success) {
                console.log("ℹ️ Cached results invalid or unsuccessful.");
                return;
            }

            console.log("♻️ Restoring experiment results from localStorage cache...");
            experimentData = parsed;

            // Make sure results section is visible
            if (resultsSection) {
                resultsSection.style.display = "block";
            }

            // Re-render everything from the cached data
            renderResultsProgressively(parsed);
        } catch (e) {
            console.warn("⚠️ Failed to restore results from cache:", e);
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

    // ✅ Try to restore last results when the dashboard page loads
    restoreLastResultsFromCache();

    // ==========================================================
    //  RESULTS PAGE INTEGRATION
    // ==========================================================

    let selectedSamplesForResults = new Set();
    let currentDatasetPath = null;

    // Load projects for dropdown
    async function loadResultsProjects() {
        try {
            const response = await fetch('/api/results/projects');
            const data = await response.json();

            const select = document.getElementById('results-project-select');
            if (!select) return;

            // Preserve the first two options
            const existingOptions = select.innerHTML;
            select.innerHTML = '<option value="">-- Select project --</option><option value="new">+ Create New Project</option>';

            data.projects.forEach(p => {
                const option = document.createElement('option');
                option.value = p.id;
                option.textContent = `${p.name} (${p.cycle_count} cycles)`;
                select.appendChild(option);
            });
        } catch (err) {
            console.error('Error loading projects:', err);
        }
    }

    // Project selector change handler
    document.getElementById('results-project-select')?.addEventListener('change', function () {
        const newProjectInput = document.getElementById('new-project-input');
        if (this.value === 'new') {
            newProjectInput.style.display = 'block';
        } else {
            newProjectInput.style.display = 'none';
        }
        updateSendButtonState();
    });

    // Update send button state based on selection
    function updateSendButtonState() {
        const sendBtn = document.getElementById('send-to-results-btn');
        const projectSelect = document.getElementById('results-project-select');
        const newProjectName = document.getElementById('new-project-name-input');

        if (!sendBtn) return;

        const hasProject = projectSelect.value && projectSelect.value !== '' &&
            (projectSelect.value !== 'new' || newProjectName.value.trim() !== '');
        const hasSamples = selectedSamplesForResults.size > 0;

        sendBtn.disabled = !(hasProject && hasSamples);
        document.getElementById('selected-samples-count').textContent = selectedSamplesForResults.size;
    }

    // New project name input handler
    document.getElementById('new-project-name-input')?.addEventListener('input', updateSendButtonState);

    // Send to Results button click
    document.getElementById('send-to-results-btn')?.addEventListener('click', async function () {
        const projectSelect = document.getElementById('results-project-select');
        const newProjectName = document.getElementById('new-project-name-input');

        let projectId = projectSelect.value;

        // Create new project if needed
        if (projectId === 'new') {
            const name = newProjectName.value.trim();
            if (!name) {
                alert('Please enter a project name');
                return;
            }

            if (!currentDatasetPath) {
                alert('No active dataset. Please ensure a dataset is loaded.');
                return;
            }

            try {
                const response = await fetch('/api/results/projects', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: name,
                        dataset_path: currentDatasetPath
                    })
                });

                const data = await response.json();
                if (!data.success) {
                    alert(data.error || 'Failed to create project');
                    return;
                }

                projectId = data.project.id;
            } catch (err) {
                console.error('Error creating project:', err);
                alert('Error creating project');
                return;
            }
        }

        // Collect selected samples
        const samples = [];
        selectedSamplesForResults.forEach(rowData => {
            samples.push(rowData);
        });

        if (samples.length === 0) {
            alert('No samples selected');
            return;
        }

        // Collect target and a-priori column names for lab results
        const labResultColumns = [];

        // Get target columns from the dashboard
        document.querySelectorAll('.target-group select[name="target_columns"]').forEach(sel => {
            if (sel.value) labResultColumns.push(sel.value);
        });

        // Get a-priori columns from the dashboard  
        document.querySelectorAll('.apriori-group select[name="apriori_columns"]').forEach(sel => {
            if (sel.value) labResultColumns.push(sel.value);
        });

        console.log('📋 Lab result columns:', labResultColumns);

        // Send to Results API
        try {
            const response = await fetch('/api/results/cycles', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    project_id: parseInt(projectId),
                    samples: samples,
                    lab_result_columns: labResultColumns
                })
            });

            const data = await response.json();

            if (data.success) {
                alert(`Created Cycle ${data.cycle.cycle_number} with ${data.cycle.sample_count} samples!`);

                // Clear selection
                selectedSamplesForResults.clear();
                updateSendButtonState();

                // Redirect to Results page
                window.location.href = '/results';
            } else {
                alert(data.error || 'Failed to create cycle');
            }
        } catch (err) {
            console.error('Error creating cycle:', err);
            alert('Error sending samples to Results');
        }
    });

    // Override renderTable to add checkboxes
    const originalRenderTable = renderTable;
    renderTable = function (tableHtml) {
        // Store dataset path for project creation
        const activeDataset = uploadedDatasets.find(d => d.isActive);
        if (activeDataset) {
            // Construct full path from filename (datasets are stored in data/ directory)
            currentDatasetPath = `data/${activeDataset.filename}`;
            console.log('📂 Current dataset path:', currentDatasetPath);
        }

        // Call original renderTable WITHOUT modifications
        originalRenderTable(tableHtml);

        // Show send to results section
        document.getElementById('send-to-results-section').style.display = 'block';
        loadResultsProjects();

        // Setup row click selection after a small delay to ensure DataTable is ready
        setTimeout(() => {
            setupRowClickSelection();
        }, 100);
    };

    function setupRowClickSelection() {
        const tbody = resultsTableContainer.querySelector('tbody');
        if (!tbody || tbody.dataset.selectionInitialized) return;

        tbody.dataset.selectionInitialized = 'true';

        // Add click handler to all body rows
        tbody.addEventListener('click', (e) => {
            const row = e.target.closest('tr');
            if (!row) return;

            // Toggle selection
            const isSelected = row.classList.toggle('table-success');

            if (isSelected) {
                addSampleToSelection(row);
            } else {
                removeSampleFromSelection(row);
            }

            updateSendButtonState();
        });

        console.log('✅ Row click selection enabled - click rows to select for Results');
    }

    function addSampleToSelection(row) {
        const cells = row.querySelectorAll('td');
        const rowData = {};

        // Get column headers
        const table = row.closest('table');
        const headers = table.querySelectorAll('thead th');

        // Extract row data
        cells.forEach((cell, idx) => {
            if (headers[idx]) {
                const colName = headers[idx].textContent.trim();
                const value = cell.textContent.trim();
                rowData[colName] = isNaN(parseFloat(value)) ? value : parseFloat(value);
            }
        });

        // Get IDX_SAMPLE - check multiple possible column names (case variations)
        // Priority: Idx_Sample > IDX_SAMPLE > idx_sample, then fallback to Row number
        const idxSample = rowData['Idx_Sample'] || rowData['IDX_SAMPLE'] || rowData['idx_sample'] ||
            rowData['IdxSample'] || rowData['Row number'] || rowData['index'] || 0;

        console.log('📌 Sample selected with idx_sample:', idxSample, 'from row data:', rowData);

        // Separate predictions from input data
        const predictions = {};
        const inputData = {};

        for (const [key, value] of Object.entries(rowData)) {
            if (key.startsWith('Predicted_') || key.includes('Uncertainty') || key === 'Utility' || key === 'Novelty') {
                predictions[key] = value;
            } else {
                inputData[key] = value;
            }
        }

        // Use a unique key for the Map
        const sampleKey = String(idxSample);

        // Store in a Map for easier lookup
        if (!window.selectedSamplesMap) {
            window.selectedSamplesMap = new Map();
        }

        window.selectedSamplesMap.set(sampleKey, {
            idx_sample: idxSample,
            row_data: inputData,
            predictions: predictions
        });

        // Sync to Set
        selectedSamplesForResults.clear();
        window.selectedSamplesMap.forEach(v => selectedSamplesForResults.add(v));
    }

    function removeSampleFromSelection(row) {
        const cells = row.querySelectorAll('td');
        const table = row.closest('table');
        const headers = table.querySelectorAll('thead th');

        // Find IDX_SAMPLE column - check multiple case variations
        let idxSample = 0;
        cells.forEach((cell, idx) => {
            if (headers[idx]) {
                const colName = headers[idx].textContent.trim();
                if (colName === 'Idx_Sample' || colName === 'IDX_SAMPLE' || colName === 'idx_sample' ||
                    colName === 'IdxSample' || colName === 'Row number' || colName === 'index') {
                    idxSample = parseFloat(cell.textContent.trim());
                }
            }
        });

        const sampleKey = String(idxSample);

        if (window.selectedSamplesMap) {
            window.selectedSamplesMap.delete(sampleKey);
        }

        // Sync to Set
        selectedSamplesForResults.clear();
        if (window.selectedSamplesMap) {
            window.selectedSamplesMap.forEach(v => selectedSamplesForResults.add(v));
        }
    }
});
