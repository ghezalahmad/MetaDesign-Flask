import json
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

        target_columns = [
            c for c in target_columns
            if c in plot_df.columns and pd.api.types.is_numeric_dtype(plot_df[c])
        ]
        if not target_columns:
            return json.dumps({}, cls=plotly.utils.PlotlyJSONEncoder)

        # --- One target: simple scatter Utility vs target ---
        if len(target_columns) == 1:
            target = target_columns[0]
            fig = go.Figure()
            fig.add_trace(cls._create_scatter_trace(
                x=plot_df[target],
                y=plot_df["Utility"],
                color=plot_df["Utility"],
                customdata=plot_df.get("Row number"),
                error_x=cls._uncert_col(plot_df, target)
            ))
            fig.update_xaxes(title_text=target)
            fig.update_yaxes(title_text="Utility")
            fig.update_layout(title="Scatter Plot of Target Properties")

        # --- Multi-target scatter matrix ---
        else:
            n = len(target_columns)
            fig = make_subplots(
                rows=n-1, cols=n-1, start_cell="top-left",
                horizontal_spacing=0.02, vertical_spacing=0.02,
                shared_xaxes=True, shared_yaxes=True
            )
            fig.update_layout(title="Scatter Matrix of Target Properties", showlegend=False)

            row_idx, col_idx = np.tril_indices(n=n-1, k=0)
            row_idx += 1
            col_idx += 1

            for r, c in zip(row_idx, col_idx):
                xcol = target_columns[c - 1]
                ycol = target_columns[r]

                fig.add_trace(
                    cls._create_scatter_trace(
                        x=plot_df[xcol],
                        y=plot_df[ycol],
                        color=plot_df["Utility"],
                        customdata=plot_df.get("Row number"),
                        error_x=cls._uncert_col(plot_df, xcol),
                        error_y=cls._uncert_col(plot_df, ycol)
                    ),
                    row=r, col=c
                )
                if r == (n - 1):
                    fig.update_xaxes(title_text=xcol, row=r, col=c)
                if c == 1:
                    fig.update_yaxes(title_text=ycol, row=r, col=c)

        cls._apply_dark_layout(fig)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    @classmethod
    def create_tsne_input_space_plot(cls, plot_df: pd.DataFrame, input_columns):
        """Build SLAMD-style t-SNE plot using numeric input columns."""
        if plot_df is None or plot_df.empty:
            return json.dumps({}, cls=plotly.utils.PlotlyJSONEncoder)

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
            "Utility": plot_df.loc[idx, "Utility"].values
        }, index=idx)

        tsne_df["is_train_data"] = plot_df.loc[idx, "is_train_data"].values if "is_train_data" in plot_df.columns else False
        if "Row number" in plot_df.columns:
            tsne_df["Row number"] = plot_df.loc[idx, "Row number"].values

        fig = px.scatter(
            tsne_df, x="t-SNE-1", y="t-SNE-2",
            color="Utility", symbol="is_train_data",
            color_continuous_scale="Turbo",
            custom_data=["Row number"] if "Row number" in tsne_df.columns else None,
            title="t-SNE Visualization of Material Space",
            symbol_sequence=["circle", "cross"], render_mode="svg"
        )

        fig.update_traces(
            marker=dict(size=9, line=dict(width=0.6, color="white")),
            hovertemplate=(
                "Row number: %{customdata}<br>Utility: %{marker.color:.2f}"
                if "Row number" in tsne_df.columns
                else "Utility: %{marker.color:.2f}"
            )
        )

        fig.update_layout(
            legend_title_text="",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            height=900
        )

        cls._apply_dark_layout(fig)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # ---------- INTERNAL HELPERS ----------

    @staticmethod
    def _apply_dark_layout(fig):
        """Dark layout for high visibility on dark backgrounds."""
        fig.update_layout(
            plot_bgcolor="#212529",
            paper_bgcolor="#212529",
            font=dict(color="white"),
            title_font=dict(color="white"),
            xaxis=dict(showgrid=True, gridcolor="#444", zerolinecolor="#555"),
            yaxis=dict(showgrid=True, gridcolor="#444", zerolinecolor="#555"),
            legend=dict(
                bgcolor="rgba(0,0,0,0.5)",
                font=dict(color="white"),
                bordercolor="#888",
                borderwidth=1
            ),
            margin=dict(l=60, r=40, t=80, b=60)
        )

    @staticmethod
    def _uncert_col(df: pd.DataFrame, colname: str):
        """Return uncertainty array for a column if present, else None."""
        key = f"{UNCERTAINTY_COLUMN_PREFIX}{colname})"
        return df[key] if key in df.columns else None

    @staticmethod
    def _create_scatter_trace(x=None, y=None, color=None, customdata=None, error_x=None, error_y=None):
        """Create a scatter trace compatible with dark themes."""
        return go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=9,
                color=color if color is not None else "cyan",
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(
                    title=dict(text="Utility", font=dict(color="white")),
                    tickfont=dict(color="white")
                ),
                line=dict(width=0.6, color="white")
            ),
            customdata=customdata,
            error_x=dict(type="data", array=error_x, color="gray", thickness=1) if error_x is not None else None,
            error_y=dict(type="data", array=error_y, color="gray", thickness=1) if error_y is not None else None,
            hoverlabel=dict(bgcolor="#343a40", font=dict(color="white")),
            hovertemplate=(
                "Row number: %{customdata}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Utility: %{marker.color:.2f}"
                if customdata is not None
                else "X: %{x:.2f}<br>Y: %{y:.2f}<br>Utility: %{marker.color:.2f}"
            ),
            name=""
        )

    @classmethod
    def _select_error_col_if_available(cls, plot_df, column_name=None):
        """Return the uncertainty column if available, else None."""
        error_column = plot_df.get(f"{UNCERTAINTY_COLUMN_PREFIX}{column_name})")
        return error_column if error_column is not None else None
