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
