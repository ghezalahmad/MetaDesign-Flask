"""Shared decision-analysis layer for all experiment modes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        parsed = float(value)
        if np.isnan(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def _normalize(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).astype(float).values
    if len(arr) == 0:
        return arr
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if abs(max_val - min_val) < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - min_val) / (max_val - min_val)


def _parse_row_list(value: Any) -> set[str]:
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = str(value or "").replace("\n", ",").split(",")
    return {str(item).strip() for item in raw_items if str(item).strip()}


def _row_value(value: Any) -> str:
    parsed = _as_float(value)
    if parsed is not None and float(parsed).is_integer():
        return str(int(parsed))
    return str(value)


class DecisionAnalyzer:
    """Model-agnostic post-processing for active-learning recommendations."""

    @classmethod
    def apply(
        cls,
        results_df: pd.DataFrame,
        source_df: pd.DataFrame,
        input_columns: list[str],
        target_configs: list[dict],
        apriori_configs: list[dict] | None,
        config: dict,
    ) -> tuple[pd.DataFrame, dict]:
        if results_df is None or results_df.empty:
            return results_df, cls._empty_analysis(config)

        df = results_df.copy()
        source = source_df.copy() if source_df is not None else pd.DataFrame()
        settings = config.get("decision_settings", {}) or {}
        target_columns = [t.get("name") for t in target_configs or [] if t.get("name")]
        batch_size = max(1, int(config.get("batch_size", 1) or 1))

        cls._ensure_identity_columns(df)
        cls._add_constraints(df, target_configs, apriori_configs or [])
        cls._add_pareto_columns(df, target_configs)
        cls._add_trust_columns(df, source, input_columns, target_columns)
        cls._add_fidelity_columns(df, settings)
        cls._add_decision_score(df, settings)
        cls._apply_human_oversight(df, settings, batch_size)

        analysis = cls._build_analysis(df, source, input_columns, target_configs, settings, config)
        return df, analysis

    @staticmethod
    def _ensure_identity_columns(df: pd.DataFrame) -> None:
        if "Row number" not in df.columns:
            if "Idx_Sample" in df.columns:
                df["Row number"] = df["Idx_Sample"]
            else:
                df["Row number"] = df.index + 1

    @staticmethod
    def _row_ids(df: pd.DataFrame) -> pd.Series:
        ids = df["Row number"].apply(_row_value)
        if "Idx_Sample" in df.columns:
            ids = ids + "|" + df["Idx_Sample"].apply(_row_value)
        return ids

    @classmethod
    def _add_constraints(cls, df: pd.DataFrame, target_configs: list[dict], apriori_configs: list[dict]) -> None:
        constraints = []
        for item in list(target_configs or []) + list(apriori_configs or []):
            threshold = _as_float(item.get("threshold"))
            name = item.get("name")
            if name and threshold is not None and name in df.columns:
                constraints.append({
                    "name": name,
                    "threshold": threshold,
                    "optimization": item.get("optimization", "max"),
                })

        feasible = pd.Series(True, index=df.index)
        violations: list[str] = []

        for idx, row in df.iterrows():
            failed = []
            for constraint in constraints:
                value = _as_float(row.get(constraint["name"]))
                if value is None:
                    failed.append(f"{constraint['name']} missing")
                    continue
                if constraint["optimization"] == "min":
                    passes = value <= constraint["threshold"]
                    op = "<="
                else:
                    passes = value >= constraint["threshold"]
                    op = ">="
                if not passes:
                    failed.append(f"{constraint['name']} {op} {constraint['threshold']}")
            feasible.loc[idx] = len(failed) == 0
            violations.append("; ".join(failed))

        df["Constraint_Feasible"] = feasible.astype(bool)
        df["Constraint_Violations"] = violations
        df["Constraint_Count"] = [0 if not item else len(item.split("; ")) for item in violations]

    @classmethod
    def _add_pareto_columns(cls, df: pd.DataFrame, target_configs: list[dict]) -> None:
        target_configs = [t for t in target_configs or [] if t.get("name") in df.columns]
        if not target_configs:
            df["Pareto_Front"] = False
            df["Pareto_Rank"] = 2
            return

        values = []
        valid_mask = pd.Series(True, index=df.index)
        for target in target_configs:
            series = pd.to_numeric(df[target["name"]], errors="coerce")
            valid_mask &= series.notna()
            adjusted = series.astype(float)
            if target.get("optimization", "max") == "min":
                adjusted = -adjusted
            values.append(adjusted.values)

        matrix = np.vstack(values).T if values else np.empty((len(df), 0))
        pareto = np.zeros(len(df), dtype=bool)

        valid_positions = np.where(valid_mask.values)[0]
        for pos in valid_positions:
            candidate = matrix[pos]
            dominated = False
            for other in valid_positions:
                if other == pos:
                    continue
                other_values = matrix[other]
                if np.all(other_values >= candidate) and np.any(other_values > candidate):
                    dominated = True
                    break
            pareto[pos] = not dominated

        df["Pareto_Front"] = pareto
        df["Pareto_Rank"] = np.where(pareto, 1, 2)

    @classmethod
    def _add_trust_columns(
        cls,
        df: pd.DataFrame,
        source: pd.DataFrame,
        input_columns: list[str],
        target_columns: list[str],
    ) -> None:
        if "Uncertainty" in df.columns:
            unc_norm = _normalize(df["Uncertainty"])
        else:
            df["Uncertainty"] = 0.0
            unc_norm = np.zeros(len(df))

        train_df = source.dropna(subset=target_columns) if target_columns else source
        usable_inputs = [c for c in input_columns or [] if c in df.columns and c in train_df.columns]

        ood = np.zeros(len(df), dtype=float)
        if usable_inputs and len(train_df) >= 2:
            try:
                train_x = train_df[usable_inputs].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                cand_x = df[usable_inputs].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                scaler = StandardScaler()
                train_scaled = scaler.fit_transform(train_x.values)
                cand_scaled = scaler.transform(cand_x.values)
                distances = []
                for row in cand_scaled:
                    distances.append(float(np.min(np.linalg.norm(train_scaled - row, axis=1))))
                ood = _normalize(distances)
            except Exception:
                ood = np.zeros(len(df), dtype=float)

        trust = 1.0 - (0.55 * unc_norm + 0.45 * ood)
        trust = np.clip(trust, 0.0, 1.0)

        df["OOD_Risk"] = ood
        df["Trust_Score"] = trust
        df["Trust_Flag"] = pd.cut(
            trust,
            bins=[-0.01, 0.4, 0.7, 1.01],
            labels=["Low", "Medium", "High"],
        ).astype(str)

    @classmethod
    def _add_fidelity_columns(cls, df: pd.DataFrame, settings: dict) -> None:
        cost_column = settings.get("cost_column")
        fidelity_column = settings.get("fidelity_column")

        if cost_column and cost_column in df.columns:
            cost = pd.to_numeric(df[cost_column], errors="coerce")
            fallback = float(cost.median()) if cost.notna().any() else 1.0
            df["Experiment_Cost"] = cost.fillna(fallback).clip(lower=1e-9)
        else:
            df["Experiment_Cost"] = 1.0

        if fidelity_column and fidelity_column in df.columns:
            df["Fidelity_Level"] = df[fidelity_column].fillna("unspecified").astype(str)
        else:
            df["Fidelity_Level"] = "standard"

        cost_norm = _normalize(df["Experiment_Cost"])
        df["Cost_Penalty"] = cost_norm

    @classmethod
    def _add_decision_score(cls, df: pd.DataFrame, settings: dict) -> None:
        utility = _normalize(df["Utility"]) if "Utility" in df.columns else np.zeros(len(df))
        pareto_bonus = df.get("Pareto_Front", False).astype(float).values * 0.2
        trust = pd.to_numeric(df.get("Trust_Score", 0.5), errors="coerce").fillna(0.5).values
        cost_penalty = pd.to_numeric(df.get("Cost_Penalty", 0.0), errors="coerce").fillna(0.0).values

        score = (0.62 * utility) + pareto_bonus + (0.18 * trust) - (0.12 * cost_penalty)

        if settings.get("prefer_feasible", True):
            feasible = df.get("Constraint_Feasible", True).astype(bool).values
            score = np.where(feasible, score, score - 1.0)

        df["Decision_Score"] = np.round(score, 6)
        df["Cost_Adjusted_Utility"] = np.round(utility / np.sqrt(1.0 + cost_penalty), 6)

    @classmethod
    def _apply_human_oversight(cls, df: pd.DataFrame, settings: dict, batch_size: int) -> None:
        force_rows = _parse_row_list(settings.get("force_rows"))
        reject_rows = _parse_row_list(settings.get("reject_rows"))
        row_ids = cls._row_ids(df)

        forced = row_ids.apply(lambda value: any(item in value.split("|") for item in force_rows))
        rejected = row_ids.apply(lambda value: any(item in value.split("|") for item in reject_rows))

        eligible = pd.Series(True, index=df.index)
        if "is_train_data" in df.columns:
            eligible = ~df["is_train_data"].astype(bool)

        if "Selected for Testing" not in df.columns:
            df["Selected for Testing"] = False
        df["Selected for Testing"] = False
        df["Decision_Action"] = "Reserve"

        df.loc[rejected, "Decision_Action"] = "Rejected by oversight"
        df.loc[forced & ~rejected, "Decision_Action"] = "Force include"

        selected_indices = list(df[forced & ~rejected & eligible].index)
        remaining_slots = max(batch_size - len(selected_indices), 0)

        if remaining_slots > 0:
            candidates = df[eligible & ~rejected & ~forced].sort_values("Decision_Score", ascending=False)
            selected_indices.extend(list(candidates.head(remaining_slots).index))

        df.loc[selected_indices, "Selected for Testing"] = True
        df.loc[df.index.isin(selected_indices) & ~forced, "Decision_Action"] = "Recommended"

    @classmethod
    def _build_analysis(
        cls,
        df: pd.DataFrame,
        source: pd.DataFrame,
        input_columns: list[str],
        target_configs: list[dict],
        settings: dict,
        config: dict,
    ) -> dict:
        selected = df[df.get("Selected for Testing", False) == True]
        feasible = int(df.get("Constraint_Feasible", pd.Series(dtype=bool)).sum())
        pareto = int(df.get("Pareto_Front", pd.Series(dtype=bool)).sum())

        return {
            "summary": {
                "candidate_count": int(len(df)),
                "selected_count": int(len(selected)),
                "pareto_count": pareto,
                "feasible_count": feasible,
                "low_trust_count": int((df.get("Trust_Flag", "") == "Low").sum()),
                "mean_trust": round(float(pd.to_numeric(df.get("Trust_Score", 0), errors="coerce").mean()), 3),
                "mean_uncertainty": round(float(pd.to_numeric(df.get("Uncertainty", 0), errors="coerce").mean()), 3),
            },
            "selected_batch": cls._records(selected, limit=25),
            "oversight": {
                "force_rows": sorted(_parse_row_list(settings.get("force_rows"))),
                "reject_rows": sorted(_parse_row_list(settings.get("reject_rows"))),
                "notes": settings.get("oversight_notes", ""),
            },
            "fidelity": {
                "cost_column": settings.get("cost_column") or None,
                "fidelity_column": settings.get("fidelity_column") or None,
                "levels": sorted(df["Fidelity_Level"].dropna().astype(str).unique().tolist()),
            },
            "manifest": cls._manifest(config, source, input_columns, target_configs),
            "plots": {
                "pareto": cls._pareto_plot(df, target_configs),
                "trust": cls._trust_plot(df),
                "batch": cls._batch_plot(selected),
                "fidelity": cls._fidelity_plot(df),
            },
        }

    @staticmethod
    def _records(df: pd.DataFrame, limit: int = 25) -> list[dict]:
        if df is None or df.empty:
            return []
        columns = [
            "Row number", "Idx_Sample", "Decision_Action", "Decision_Score", "Utility",
            "Pareto_Front", "Constraint_Feasible", "Trust_Flag", "OOD_Risk",
            "Experiment_Cost", "Fidelity_Level",
        ]
        existing = [c for c in columns if c in df.columns]
        return json.loads(df[existing].head(limit).to_json(orient="records"))

    @staticmethod
    def _manifest(config: dict, source: pd.DataFrame, input_columns: list[str], target_configs: list[dict]) -> dict:
        packages = {}
        for package in ["numpy", "pandas", "scikit-learn", "flask", "plotly", "lolopy"]:
            try:
                packages[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                packages[package] = "not installed"

        return {
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "dataset_filename": config.get("dataset_filename"),
            "row_count": int(len(source)) if source is not None else 0,
            "model": config.get("model"),
            "mode": config.get("active_learning_mode", "ML_MODE"),
            "acquisition_function": config.get("acquisition_function"),
            "batch_size": int(config.get("batch_size", 1) or 1),
            "input_columns": input_columns,
            "target_columns": target_configs,
            "packages": packages,
        }

    @staticmethod
    def _pareto_plot(df: pd.DataFrame, target_configs: list[dict]) -> dict:
        targets = [t.get("name") for t in target_configs or [] if t.get("name") in df.columns]
        if not targets:
            return {"data": [], "layout": {"title": "No target columns available"}}

        x_col = targets[0]
        y_col = targets[1] if len(targets) > 1 else "Decision_Score"
        plot_df = df.dropna(subset=[x_col, y_col]).copy()
        if plot_df.empty:
            return {"data": [], "layout": {"title": "No Pareto data available"}}

        fig = go.Figure()
        for name, group in [("Other candidates", plot_df[~plot_df["Pareto_Front"]]), ("Pareto front", plot_df[plot_df["Pareto_Front"]])]:
            if group.empty:
                continue
            fig.add_trace(go.Scatter(
                x=group[x_col].tolist(),
                y=group[y_col].tolist(),
                mode="markers",
                name=name,
                marker={
                    "size": np.where(group.get("Selected for Testing", False), 13, 8).tolist(),
                    "color": group["Decision_Score"].tolist(),
                    "colorscale": "Viridis",
                    "showscale": name == "Pareto front",
                    "line": {"color": "#1f2937", "width": 0.6},
                },
                customdata=group[["Row number", "Decision_Score", "Trust_Flag"]].values.tolist(),
                hovertemplate="Row: %{customdata[0]}<br>Decision: %{customdata[1]:.3f}<br>Trust: %{customdata[2]}<extra></extra>",
            ))
        fig.update_layout(title="Pareto Trade-Off View", xaxis_title=x_col, yaxis_title=y_col, template="plotly_white", height=420)
        return fig.to_dict()

    @staticmethod
    def _trust_plot(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"data": [], "layout": {"title": "No trust data available"}}
        fig = go.Figure(go.Scatter(
            x=pd.Series(df.get("OOD_Risk", 0)).tolist(),
            y=pd.Series(df.get("Uncertainty", 0)).tolist(),
            mode="markers",
            marker={
                "size": np.where(df.get("Selected for Testing", False), 13, 8).tolist(),
                "color": pd.Series(df.get("Trust_Score", 0)).tolist(),
                "colorscale": "RdYlGn",
                "showscale": True,
                "colorbar": {"title": "Trust"},
            },
            customdata=df[["Row number", "Trust_Flag", "Decision_Score"]].values.tolist(),
            hovertemplate="Row: %{customdata[0]}<br>Trust: %{customdata[1]}<br>Decision: %{customdata[2]:.3f}<extra></extra>",
        ))
        fig.update_layout(title="Trust Diagnostics", xaxis_title="Out-of-distribution risk", yaxis_title="Uncertainty", template="plotly_white", height=420)
        return fig.to_dict()

    @staticmethod
    def _batch_plot(selected: pd.DataFrame) -> dict:
        if selected.empty:
            return {"data": [], "layout": {"title": "No selected batch"}}
        labels = selected["Row number"].astype(str).tolist()
        fig = go.Figure(go.Bar(
            x=labels,
            y=selected["Decision_Score"].tolist(),
            marker={"color": selected["Trust_Score"].tolist(), "colorscale": "Cividis", "showscale": True},
            customdata=selected[["Decision_Action", "Constraint_Feasible", "Trust_Flag"]].values.tolist(),
            hovertemplate="Action: %{customdata[0]}<br>Feasible: %{customdata[1]}<br>Trust: %{customdata[2]}<extra></extra>",
        ))
        fig.update_layout(title="Selected Batch Recommendation", xaxis_title="Row", yaxis_title="Decision score", template="plotly_white", height=360)
        return fig.to_dict()

    @staticmethod
    def _fidelity_plot(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"data": [], "layout": {"title": "No fidelity data available"}}
        fig = go.Figure(go.Scatter(
            x=df["Experiment_Cost"].tolist(),
            y=df["Decision_Score"].tolist(),
            mode="markers",
            marker={
                "size": np.where(df.get("Selected for Testing", False), 13, 8).tolist(),
                "color": pd.Series(df.get("Cost_Adjusted_Utility", 0)).tolist(),
                "colorscale": "Bluered",
                "showscale": True,
                "colorbar": {"title": "Cost-adjusted"},
            },
            text=df["Fidelity_Level"].tolist(),
            customdata=df[["Row number", "Fidelity_Level", "Experiment_Cost"]].values.tolist(),
            hovertemplate="Row: %{customdata[0]}<br>Fidelity: %{customdata[1]}<br>Cost: %{customdata[2]}<extra></extra>",
        ))
        fig.update_layout(title="Cost and Fidelity Awareness", xaxis_title="Experiment cost", yaxis_title="Decision score", template="plotly_white", height=360)
        return fig.to_dict()

    @staticmethod
    def _empty_analysis(config: dict) -> dict:
        return {
            "summary": {
                "candidate_count": 0,
                "selected_count": 0,
                "pareto_count": 0,
                "feasible_count": 0,
                "low_trust_count": 0,
                "mean_trust": 0,
                "mean_uncertainty": 0,
            },
            "selected_batch": [],
            "oversight": {},
            "fidelity": {},
            "manifest": {
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "model": config.get("model"),
                "mode": config.get("active_learning_mode", "ML_MODE"),
            },
            "plots": {},
        }
