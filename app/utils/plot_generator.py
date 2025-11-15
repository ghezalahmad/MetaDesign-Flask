
import json
import logging
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from plotly.utils import PlotlyJSONEncoder

UNCERTAINTY_COLUMN_PREFIX = "Uncertainty ("


class PlotGenerator:
    """Generate t-SNE and scatter plots for SLAMD-style visualization."""
    PlotlyJSONEncoder = PlotlyJSONEncoder

    # ---------- PUBLIC API ----------

    @classmethod
    def create_target_scatter_plot(cls, plot_df: pd.DataFrame, target_columns):
        """Build SLAMD-style scatter or scatter-matrix plots for selected target columns."""
        if plot_df is None or plot_df.empty:
            return json.dumps({}, cls=plotly.utils.PlotlyJSONEncoder)

        if "Row number" not in plot_df.columns:
            plot_df['Row number'] = plot_df.index

        target_columns = [
            c for c in target_columns
            if c in plot_df.columns and pd.api.types.is_numeric_dtype(plot_df[c])
        ]
        if not target_columns:
            logging.warning("No valid numeric target columns found in DataFrame for scatter plot.")
            return json.dumps({}, cls=plotly.utils.PlotlyJSONEncoder)

        # --- One target: simple scatter Utility vs target ---
        if len(target_columns) == 1:
            target = target_columns[0]
            fig = px.scatter(
                plot_df,
                x=target,
                y="Utility",
                color="Utility",
                symbol="is_train_data",
                custom_data=["Row number"],
                color_continuous_scale="Turbo",
                title="Scatter Plot of Target Properties"
            )

        # --- Multi-target scatter matrix ---
        else:
            fig = px.scatter_matrix(
                plot_df,
                dimensions=target_columns,
                color="Utility",
                symbol="is_train_data",
                custom_data=["Row number"],
                color_continuous_scale="Turbo",
                title="Scatter Matrix of Target Properties"
            )
            fig.update_traces(diagonal_visible=False)

        cls._apply_light_layout(fig)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    @classmethod
    def create_tsne_input_space_plot(cls, plot_df: pd.DataFrame, input_columns):
        """Build SLAMD-style t-SNE plot using numeric input columns."""
        if plot_df is None or plot_df.empty:
            return json.dumps({}, cls=plotly.utils.PlotlyJSONEncoder)

        if "Row number" not in plot_df.columns:
            plot_df['Row number'] = plot_df.index

        valid_inputs = [c for c in input_columns if c in plot_df.columns]
        if not valid_inputs:
            return json.dumps({}, cls=plotly.utils.PlotlyJSONEncoder)

        features = plot_df[valid_inputs].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
        if features.empty or features.shape[1] < 1:
            return json.dumps({}, cls=plotly.utils.PlotlyJSONEncoder)

        idx = features.index
        perplexity = max(5, min(20, len(features) - 1))
        tsne = TSNE(
            n_components=2, verbose=1, perplexity=perplexity,
            max_iter=350, random_state=42, init="pca", learning_rate=100
        )
        ts = tsne.fit_transform(features.values)

        tsne_df = pd.DataFrame({
            "t-SNE-1": ts[:, 0],
            "t-SNE-2": ts[:, 1],
            "Utility": plot_df.loc[idx, "Utility"].values,
            "Row number": plot_df.loc[idx, "Row number"].values
        }, index=idx)

        tsne_df["is_train_data"] = plot_df.loc[idx, "is_train_data"].values if "is_train_data" in plot_df.columns else False

        fig = px.scatter(
            tsne_df, x="t-SNE-1", y="t-SNE-2",
            color="Utility",
            symbol="is_train_data",
            color_continuous_scale="Turbo",
            custom_data=["Row number"],
            title="t-SNE Visualization of Material Space",
        )

        fig.update_traces(
            marker=dict(size=9, line=dict(width=0.6, color="black")),
            hovertemplate=(
                "Row number: %{customdata[0]}<br>Utility: %{marker.color:.2f}"
            )
        )

        fig.update_layout(
            legend_title_text="",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=900
        )

        cls._apply_light_layout(fig)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # ---------- INTERNAL HELPERS ----------

    @staticmethod
    def _apply_light_layout(fig):
        """Bright layout — visible on all backgrounds."""
        fig.update_layout(
            plot_bgcolor="#f7f7f7",
            paper_bgcolor="#f7f7f7",
            font=dict(color="black"),
            title_font=dict(color="black"),
            xaxis=dict(showgrid=True, gridcolor="#d0d0d0", zerolinecolor="#c0c0c0"),
            yaxis=dict(showgrid=True, gridcolor="#d0d0d0", zerolinecolor="#c0c0c0"),
            legend=dict(
                bgcolor="rgba(255,255,255,0.6)",
                font=dict(color="black"),
                bordercolor="#ccc",
                borderwidth=0
            ),
            margin=dict(l=60, r=40, t=80, b=60)
        )
