import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE


class PlotGenerator:

    # ======================================================
    #   TARGET SCATTER PLOT
    # ======================================================
    @classmethod
    def create_target_scatter_plot(cls, df: pd.DataFrame, target_columns):

        if df is None or df.empty:
            print("⚠ SCATTER: empty dataframe")
            return {}

        df = df.copy()

        # Row number
        if "Row number" not in df.columns:
            df["Row number"] = df.index

        # Utility must exist
        if "Utility" not in df.columns:
            print("⚠ SCATTER: Utility missing → created zeros")
            df["Utility"] = 0.0

        df["Utility"] = pd.to_numeric(df["Utility"], errors="coerce").fillna(0)

        # Determine x axis
        if target_columns and target_columns[0] in df.columns:
            x_col = target_columns[0]
        else:
            numerics = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "Utility"]
            x_col = numerics[0] if numerics else "Utility"

        df[x_col] = pd.to_numeric(df[x_col], errors="coerce").fillna(0)

        print("✅ create_target_scatter_plot:", x_col, "vs Utility")

        # Convert to native Python types BEFORE creating figure
        x_values = df[x_col].astype(float).tolist()
        y_values = df["Utility"].astype(float).tolist()
        utility_values = df["Utility"].astype(float).tolist()
        row_numbers = df["Row number"].astype(int).tolist()

        # Create figure using graph_objects for better control
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            marker=dict(
                size=8,
                color=utility_values,
                colorscale='Turbo',
                showscale=True,
                colorbar=dict(title="Utility"),
                line=dict(color='black', width=0.5)
            ),
            customdata=[[row] for row in row_numbers],
            hovertemplate=(
                "Row: %{customdata[0]}<br>" +
                f"{x_col}: %{{x:.2f}}<br>" +
                "Utility: %{y:.2f}<br>" +
                "<extra></extra>"
            ),
            name=''
        ))

        fig.update_layout(
            title=f"{x_col} vs Utility",
            xaxis_title=x_col,
            yaxis_title="Utility",
            plot_bgcolor="#f7f7f7",
            paper_bgcolor="#f7f7f7",
            font=dict(color="black"),
            margin=dict(l=60, r=40, t=60, b=60),
            xaxis=dict(showgrid=True, gridcolor="#cccccc"),
            yaxis=dict(showgrid=True, gridcolor="#cccccc"),
        )

        return fig.to_dict()


    # ======================================================
    #   t-SNE PLOT
    # ======================================================
    @classmethod
    def create_tsne_input_space_plot(cls, df: pd.DataFrame, input_columns):

        if df is None or df.empty:
            print("⚠ TSNE: empty dataframe")
            return {}

        df = df.copy()

        # Row number
        if "Row number" not in df.columns:
            df["Row number"] = df.index

        # Utility
        if "Utility" not in df.columns:
            print("⚠ TSNE: Utility missing → created zeros")
            df["Utility"] = 0.0

        df["Utility"] = pd.to_numeric(df["Utility"], errors="coerce").fillna(0)

        # Numeric input features
        valid_inputs = [
            c for c in (input_columns or [])
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not valid_inputs:
            valid_inputs = [
                c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and c not in ["Utility"]
            ]

        if not valid_inputs:
            print("⚠ TSNE: no numeric inputs found.")
            return {}

        features = df[valid_inputs].astype(float)
        n = len(features)

        if n < 3:
            print("⚠ TSNE: n<3 → cannot compute t-SNE")
            return {}

        perplexity = max(5, min(30, n - 1))
        print(f"✅ TSNE: running t-SNE({features.shape}) perplexity={perplexity}")

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=350,
            init="pca",
            random_state=42,
            learning_rate=80,
            verbose=1
        )

        emb = tsne.fit_transform(features.values)

        # Convert to native Python types IMMEDIATELY
        tsne_x = emb[:, 0].astype(float).tolist()
        tsne_y = emb[:, 1].astype(float).tolist()
        utility_values = df["Utility"].astype(float).tolist()
        row_numbers = df["Row number"].astype(int).tolist()

        print(f"✅ TSNE: Converted {len(tsne_x)} points to Python lists")

        # Create figure using graph_objects for better control
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=tsne_x,
            y=tsne_y,
            mode='markers',
            marker=dict(
                size=7,
                color=utility_values,
                colorscale='Turbo',
                showscale=True,
                colorbar=dict(title="Utility"),
                line=dict(color='black', width=0.5)
            ),
            customdata=[[row] for row in row_numbers],
            hovertemplate=(
                "Row: %{customdata[0]}<br>" +
                "t-SNE-1: %{x:.2f}<br>" +
                "t-SNE-2: %{y:.2f}<br>" +
                "Utility: %{marker.color:.2f}<br>" +
                "<extra></extra>"
            ),
            name=''
        ))

        fig.update_layout(
            title="t-SNE Material Space",
            xaxis_title="t-SNE-1",
            yaxis_title="t-SNE-2",
            plot_bgcolor="#f7f7f7",
            paper_bgcolor="#f7f7f7",
            font=dict(color="black"),
            margin=dict(l=60, r=40, t=60, b=60),
            xaxis=dict(showgrid=True, gridcolor="#cccccc"),
            yaxis=dict(showgrid=True, gridcolor="#cccccc"),
        )

        fig_dict = fig.to_dict()

        # Debug
        print("DEBUG TSNE: traces =", len(fig_dict['data']))
        if fig_dict['data'] and len(fig_dict['data']) > 0:
            trace_len = len(fig_dict['data'][0].get('x', []))
            print("DEBUG TSNE: points =", trace_len)
            if trace_len >= 5:
                print("DEBUG TSNE sample:", fig_dict['data'][0]['x'][:5])
            else:
                print("DEBUG TSNE: NOT ENOUGH POINTS IN TRACE!")
                print("DEBUG TSNE: Full x data:", fig_dict['data'][0].get('x'))

        return fig_dict