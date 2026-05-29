# Meta-Design Dashboard (Flask Version)

This is a Flask-based version of the Meta-Design Dashboard, a tool for accelerating materials discovery and design through the use of advanced machine learning models.

## Decision Intelligence Features

The dashboard includes a model-agnostic Decision Intelligence layer that runs after every successful recommendation engine output. It applies to classic ML, hybrid ML+LLM, and LLM-only modes because it post-processes the shared `/run-experiment` result table.

The seven added capabilities are:

1. **Multi-objective Pareto analysis**: marks Pareto-front candidates and exposes `Pareto_Front` and `Pareto_Rank` in the results.
2. **Constraint-aware recommendation**: optional target and a-priori thresholds become feasibility checks through `Constraint_Feasible`, `Constraint_Violations`, and `Constraint_Count`.
3. **Uncertainty, trust, and OOD diagnostics**: combines model uncertainty with distance from labelled training data to produce `OOD_Risk`, `Trust_Score`, and `Trust_Flag`.
4. **Human-in-the-loop oversight**: force-include and reject row controls are applied after model scoring, so lab judgement can override any ML, hybrid, or LLM-only recommendation.
5. **Cost and fidelity awareness**: optional cost and fidelity columns produce `Experiment_Cost`, `Fidelity_Level`, `Cost_Penalty`, and `Cost_Adjusted_Utility`.
6. **Batch decision scoring**: recommendations are ranked by `Decision_Score`, combining utility, Pareto status, trust, feasibility, cost, and oversight rules.
7. **Reproducibility manifest and dependency hardening**: each run reports dataset/model/config/package metadata, and requirements pin compatible `numpy`/`scikit-learn` versions for `lolopy`, `shapash`, and related tooling.

The results page also includes Decision Intelligence plots for Pareto trade-offs, trust diagnostics, selected batch recommendations, and cost/fidelity awareness. The t-SNE graph window can be redrawn by selected parameters, including `TSNE_X` vs `TSNE_Y`, color fields, and population overlays.

## Public Demo and Session Workflow

MetaDesign is safe for public demos because uploaded datasets, generated design spaces, settings, projects, scenarios, cycles, and lab results are scoped to the current browser session. They are not shown to other users.

- Use **Load Demo Dataset** on the Home or Material Discovery page to copy the curated example dataset into the current session.
- Use **Save Session** on the Results page before closing the browser. The exported JSON archive contains the session datasets, scenario/project metadata, cycles, selected samples, predictions, and lab results.
- Use **Upload Session** on the Results page to restore a saved JSON archive into a new browser session.
- Generated design spaces and uploaded datasets both remain available across Home, Scenario, Material Discovery, and Results inside the same session.

The bundled demo file lives at `data/examples/metadesign_demo_cement.csv`. To replace it with a different public example, add a clean CSV to `data/examples/` and update the demo filename in `app/api/uploads.py`.

## Active Learning and BO Behavior

The acquisition functions are implemented as design-space ranking tools. They do not mathematically guarantee discovery of the true global optimum, but they prioritize the next experiment using the current surrogate model, target directions, uncertainty, and observed data.

- `Curiosity = 0` is pure exploitation: candidates are ranked mainly by predicted objective quality.
- `Curiosity > 0` adds exploration pressure: candidates with higher model uncertainty can move up the ranking.
- WEBSLAMD, UCB, Expected Improvement, and Thompson Sampling normalize target scales and respect maximize/minimize directions before combining objectives.
- Classic ML and hybrid ML+LLM modes pass the selected acquisition function into the ML surrogate path. Hybrid mode then fuses normalized ML acquisition utility with the LLM semantic proposal score.
- LLM-only mode can recommend from the design space through semantic matching and the shared Decision Intelligence layer, but true Bayesian optimization uncertainty requires the ML or hybrid surrogate path.

## LLM Providers

MetaDesign supports local and cloud LLM modes:

- **Local Ollama**: no API key; the user runs Ollama on their own machine.
- **Mistral API**: user-provided key for the current run.
- **OpenAI API**: user-provided key for the current run.
- **Claude/Anthropic API**: user-provided key for the current run.

Cloud API keys entered in the UI are used only for the current run request and are not saved in session settings.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/meta-design-flask.git
    cd meta-design-flask
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create local environment settings:**
    ```bash
    cp .env.example .env
    ```

    Replace `SECRET_KEY` in `.env` with a long random value before running the app.

## Running the Application

1.  **Set the Flask application entry point:**
    ```bash
    export FLASK_APP=app.py  # On Windows, use `set FLASK_APP=app.py`
    ```

2.  **Run the Flask development server:**
    ```bash
    flask run
    ```

The application will be available at `http://127.0.0.1:5000`.

## Hugging Face Space Deployment

The public demo can run as a Docker Space. The Space must listen on port `7860`, and the Docker command should start the Flask app through Gunicorn.

After each push to Hugging Face:

1. Wait for the Space rebuild to finish.
2. Open the Space logs and confirm Gunicorn starts without import errors.
3. Open the app and click **Load Demo Dataset**.
4. Run Material Discovery with Lolopy using the demo dataset. With 2-7 labelled rows, the app should show the early-cycle fallback warning instead of a Java/lolopy stack trace.
5. Send samples to Results, click **Save Session**, then upload the exported JSON to confirm the session restores correctly.
