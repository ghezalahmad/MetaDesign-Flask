import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px  # ADDED for SLAMD-style plots
from sklearn.manifold import TSNE

class PlotGenerator:

    # ======================================================
    #   TSNE CALCULATION (SLAMD-EXACT WITH NORMALIZATION)
    # ======================================================
    @classmethod
    def _run_tsne(cls, df: pd.DataFrame, input_columns):
        """
        Calculate TSNE coordinates EXACTLY like SLAMD.
        CRITICAL: SLAMD standardizes features before TSNE!
        """
        if df is None or df.empty:
            print("⚠️ TSNE: empty dataframe")
            return df
            
        df = df.copy()
        
        # SLAMD APPROACH: Exclude specific columns, keep everything else
        exclude_columns = ['Row number', 'Utility', 'is_train_data', 'Uncertainty']
        
        # Also exclude any uncertainty columns
        exclude_columns.extend([col for col in df.columns if col.startswith('Uncertainty (')])
        
        # Get all numeric columns that aren't excluded
        feature_columns = [col for col in df.columns 
                          if col not in exclude_columns 
                          and pd.api.types.is_numeric_dtype(df[col])]
        
        # Need at least 1 feature and 3 samples for TSNE
        if not feature_columns or len(df) < 3:
            print(f"⚠️ TSNE: Insufficient data (features={len(feature_columns)}, samples={len(df)})")
            df['tsne-2d-one'] = 0.0
            df['tsne-2d-two'] = 0.0
            return df
        
        print(f"✅ TSNE: Running on {len(feature_columns)} features × {len(df)} samples")
        print(f"   Features: {feature_columns}")
        
        try:
            # Prepare feature matrix
            tsne_input_df = df[feature_columns].fillna(0).astype(float)
            
            # CRITICAL: SLAMD STANDARDIZES DATA BEFORE TSNE
            # This is why their clusters are so tight!
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            tsne_input_scaled = scaler.fit_transform(tsne_input_df.values)
            
            print(f"✅ TSNE: Data standardized (mean=0, std=1)")
            
            # SLAMD exact parameters
            perplexity = min(20, len(df) - 1)
            
            tsne = TSNE(
                n_components=2,
                verbose=1,
                perplexity=perplexity,
                max_iter=350,
                random_state=42,
                init='pca',
                learning_rate=100
            )
            
            # Run TSNE on STANDARDIZED data
            tsne_result = tsne.fit_transform(tsne_input_scaled)
            
            # Add results to dataframe (SLAMD naming)
            df['tsne-2d-one'] = tsne_result[:, 0]
            df['tsne-2d-two'] = tsne_result[:, 1]
            
            print(f"✅ TSNE: Successfully calculated coordinates")
            print(f"   Range X: [{tsne_result[:, 0].min():.2f}, {tsne_result[:, 0].max():.2f}]")
            print(f"   Range Y: [{tsne_result[:, 1].min():.2f}, {tsne_result[:, 1].max():.2f}]")
            
        except Exception as e:
            print(f"❌ TSNE: Calculation failed - {str(e)}")
            import traceback
            traceback.print_exc()
            df['tsne-2d-one'] = 0.0
            df['tsne-2d-two'] = 0.0
        
        return df


    @classmethod
    def create_target_scatter_plot(cls, df: pd.DataFrame, target_columns):
        if df is None or df.empty:
            print("⚠️ SCATTER: empty dataframe")
            return {'data': [], 'layout': {'title': 'No data available'}}

        df = df.copy()

        # Row number
        if "Row number" not in df.columns:
            df["Row number"] = df.index

        # Utility must exist
        if "Utility" not in df.columns:
            df["Utility"] = 0.0
        df["Utility"] = pd.to_numeric(df["Utility"], errors="coerce").fillna(0)

        # Determine x axis
        if target_columns and target_columns[0] in df.columns:
            x_col = target_columns[0]
        else:
            numerics = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "Utility"]
            x_col = numerics[0] if numerics else "Utility"

        df[x_col] = pd.to_numeric(df[x_col], errors="coerce").fillna(0)

        # Convert to native Python types (CRITICAL for JSON serialization)
        x_values = [float(x) for x in df[x_col].values]
        y_values = [float(y) for y in df["Utility"].values]
        utility_values = [float(u) for u in df["Utility"].values]
        row_numbers = [int(r) for r in df["Row number"].values]

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
            hovertemplate="Row: %{customdata[0]}<br>" + f"{x_col}: %{{x:.2f}}<br>Utility: %{{y:.2f}}<extra></extra>",
            name=''
        ))

        fig.update_layout(
            title=f"{x_col} vs Utility",
            xaxis_title=x_col,
            yaxis_title="Utility",
            plot_bgcolor="#f7f7f7",
            paper_bgcolor="#f7f7f7",
            margin=dict(l=60, r=40, t=60, b=60),
            xaxis=dict(showgrid=True, gridcolor="#cccccc"),
            yaxis=dict(showgrid=True, gridcolor="#cccccc"),
        )

        return fig.to_dict()


    @classmethod
    def create_tsne_input_space_plot(cls, df: pd.DataFrame, input_columns):
        """
        Create TSNE plot following SLAMD's visual style.
        Expects df to already have 'tsne-2d-one' and 'tsne-2d-two' columns.
        OPTIMIZED: Handles large datasets efficiently.
        """
        if df is None or df.empty:
            return {'data': [], 'layout': {'title': 'No data available'}}

        df = df.copy()

        if "Row number" not in df.columns:
            df["Row number"] = df.index
        if "Utility" not in df.columns:
            df["Utility"] = 0.0
        df["Utility"] = pd.to_numeric(df["Utility"], errors="coerce").fillna(0)

        # CHECK: Do we have TSNE coords?
        if 'tsne-2d-one' not in df.columns or 'tsne-2d-two' not in df.columns:
            print("⚠️ TSNE PLOT: Missing coordinates")
            return {'data': [], 'layout': {'title': 'TSNE coordinates missing'}}
        
        # Convert to native Python types (CRITICAL for JSON serialization)
        tsne_x = [float(x) for x in df['tsne-2d-one'].values]
        tsne_y = [float(y) for y in df['tsne-2d-two'].values]
        utility_values = [float(u) for u in df["Utility"].values]
        row_numbers = [int(r) for r in df["Row number"].values]

        # Separate train vs test data
        is_train = df.get('is_train_data', pd.Series([True] * len(df)))
        
        fig = go.Figure()
        
        # Plot test/unknown points (suggestions)
        test_indices = [i for i, val in enumerate(is_train) if not val]
        if test_indices:
            fig.add_trace(go.Scatter(
                x=[tsne_x[i] for i in test_indices],
                y=[tsne_y[i] for i in test_indices],
                mode='markers',
                name='Suggestions',
                marker=dict(
                    size=8,
                    color=[utility_values[i] for i in test_indices],
                    colorscale='Turbo',
                    showscale=True,
                    colorbar=dict(title="Utility"),
                    line=dict(color='black', width=0.5),
                    symbol='cross'
                ),
                customdata=[[row_numbers[i]] for i in test_indices],
                hovertemplate="Row: %{customdata[0]}<br>t-SNE-1: %{x:.2f}<br>t-SNE-2: %{y:.2f}<br>Utility: %{marker.color:.2f}<extra></extra>"
            ))
        
        # Plot training points
        train_indices = [i for i, val in enumerate(is_train) if val]
        if train_indices:
            fig.add_trace(go.Scatter(
                x=[tsne_x[i] for i in train_indices],
                y=[tsne_y[i] for i in train_indices],
                mode='markers',
                name='Training Data',
                marker=dict(
                    size=7,
                    color=[utility_values[i] for i in train_indices],
                    colorscale='Turbo',
                    showscale=False,
                    line=dict(color='black', width=0.5),
                    symbol='circle'
                ),
                customdata=[[row_numbers[i]] for i in train_indices],
                hovertemplate="Row: %{customdata[0]}<br>t-SNE-1: %{x:.2f}<br>t-SNE-2: %{y:.2f}<br>Utility: %{marker.color:.2f}<extra></extra>"
            ))

        fig.update_layout(
            title="t-SNE Material Space: Training Data and Suggestions",
            xaxis_title="t-SNE-1",
            yaxis_title="t-SNE-2",
            plot_bgcolor="#f7f7f7",
            paper_bgcolor="#f7f7f7",
            margin=dict(l=60, r=40, t=60, b=60),
            height=1000,
            xaxis=dict(showgrid=True, gridcolor="#cccccc"),
            yaxis=dict(showgrid=True, gridcolor="#cccccc"),
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=0.01
            )
        )

        return fig.to_dict()


    # ======================================================
    #   UNCERTAINTY PLOT (OPTIMIZED)
    # ======================================================
    @classmethod
    def create_uncertainty_plot(cls, df: pd.DataFrame, target_columns):
        """X=Prediction, Y=Uncertainty, Color=Utility"""
        if df is None or df.empty:
            return {'data': [], 'layout': {'title': 'No data available'}}

        df = df.copy()
        
        target_col = target_columns[0] if target_columns else None
        unc_col = 'Uncertainty' if 'Uncertainty' in df.columns else 'uncertainty'
        
        if not target_col or target_col not in df.columns or unc_col not in df.columns:
            print("⚠️ UNCERTAINTY: Missing required columns")
            return {'data': [], 'layout': {'title': 'Missing data columns'}}

        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df[unc_col] = pd.to_numeric(df[unc_col], errors='coerce')
        df['Utility'] = pd.to_numeric(df.get('Utility', 0), errors='coerce').fillna(0)
        
        plot_df = df.dropna(subset=[target_col, unc_col])
        if plot_df.empty:
            return {'data': [], 'layout': {'title': 'No valid data'}}

        # Convert to native Python types
        x_vals = [float(x) for x in plot_df[target_col].values]
        y_vals = [float(y) for y in plot_df[unc_col].values]
        c_vals = [float(c) for c in plot_df['Utility'].values]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            marker=dict(
                size=8,
                color=c_vals,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Utility"),
                line=dict(color='gray', width=0.5)
            ),
            hovertemplate=f"{target_col}: %{{x:.2f}}<br>Uncertainty: %{{y:.2f}}<br>Utility: %{{marker.color:.2f}}<extra></extra>"
        ))

        fig.update_layout(
            title="Prediction vs. Uncertainty",
            xaxis_title=f"Predicted {target_col}",
            yaxis_title="Uncertainty (Std Dev)",
            plot_bgcolor="#f7f7f7",
            paper_bgcolor="#f7f7f7",
            margin=dict(l=60, r=40, t=60, b=60),
            xaxis=dict(showgrid=True, gridcolor="#cccccc"),
            yaxis=dict(showgrid=True, gridcolor="#cccccc"),
        )
        
        return fig.to_dict()


    # ======================================================
    #   OPTIMIZATION HISTORY PLOT (OPTIMIZED)
    # ======================================================
    @classmethod
    def create_optimization_history_plot(cls, df: pd.DataFrame, target_columns):
        """X=Row Number, Y=Cumulative Max of Target"""
        if df is None or df.empty:
            return {'data': [], 'layout': {'title': 'No data available'}}
            
        df = df.copy()
        target_col = target_columns[0] if target_columns else None
        
        if not target_col or target_col not in df.columns:
            return {'data': [], 'layout': {'title': 'Missing target column'}}
            
        if "Row number" not in df.columns:
            df["Row number"] = range(1, len(df) + 1)
            
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        history = df.dropna(subset=[target_col]).sort_values('Row number')
        
        if history.empty:
            return {'data': [], 'layout': {'title': 'No valid data'}}

        cum_best = history[target_col].cummax()
        
        # Convert to native Python types
        x_vals = [int(x) for x in history['Row number'].values]
        y_raw = [float(y) for y in history[target_col].values]
        y_best = [float(y) for y in cum_best.values]

        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_raw,
            mode='markers',
            name='Observed',
            marker=dict(color='rgba(150, 150, 150, 0.6)', size=5)
        ))
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_best,
            mode='lines',
            name='Best Found',
            line=dict(color='green', width=3)
        ))

        fig.update_layout(
            title="Optimization History",
            xaxis_title="Iteration (Row #)",
            yaxis_title=f"Best {target_col}",
            plot_bgcolor="#f7f7f7",
            paper_bgcolor="#f7f7f7",
            margin=dict(l=60, r=40, t=60, b=60),
            xaxis=dict(showgrid=True, gridcolor="#cccccc"),
            yaxis=dict(showgrid=True, gridcolor="#cccccc"),
        )
        
        return fig.to_dict()


    # ======================================================
    #   UTILITY SURFACE PLOT (OPTIMIZED - SCATTER MODE)
    # ======================================================
    @classmethod
    def create_utility_surface_plot(cls, df: pd.DataFrame, input_columns=None):
        """
        Visualizes Utility over the TSNE space.
        OPTIMIZED: Uses scatter plot instead of contour for better performance.
        """
        if df is None or df.empty:
            return {'data': [], 'layout': {'title': 'No data available'}}
            
        if 'tsne-2d-one' not in df.columns or 'tsne-2d-two' not in df.columns:
            print("⚠️ SURFACE: Missing TSNE coordinates")
            return {'data': [], 'layout': {'title': 'TSNE coordinates required'}}
            
        df = df.copy()
        df['Utility'] = pd.to_numeric(df.get('Utility', 0), errors='coerce').fillna(0)
        
        # Convert to native Python types
        x_vals = [float(x) for x in df['tsne-2d-one'].values]
        y_vals = [float(y) for y in df['tsne-2d-two'].values]
        z_vals = [float(z) for z in df['Utility'].values]
        
        fig = go.Figure()
        
        # Use scatter plot with color gradient (MUCH faster than contour)
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            marker=dict(
                size=10,
                color=z_vals,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Utility"),
                opacity=0.7,
                line=dict(color='white', width=0.5)
            ),
            hovertemplate="t-SNE-1: %{x:.2f}<br>t-SNE-2: %{y:.2f}<br>Utility: %{marker.color:.2f}<extra></extra>",
            name=''
        ))

        fig.update_layout(
            title="Utility Landscape (TSNE Space)",
            xaxis_title="t-SNE-1",
            yaxis_title="t-SNE-2",
            plot_bgcolor="#f7f7f7",
            paper_bgcolor="#f7f7f7",
            margin=dict(l=60, r=40, t=60, b=60),
            xaxis=dict(showgrid=True, gridcolor="#cccccc"),
            yaxis=dict(showgrid=True, gridcolor="#cccccc"),
        )
        
        return fig.to_dict()