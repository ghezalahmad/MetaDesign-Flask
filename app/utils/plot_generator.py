import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px  # ADDED for SLAMD-style plots
from sklearn.manifold import TSNE

class PlotGenerator:


    # Cache to store t-SNE results: key -> DataFrame with tsne columns
    _tsne_cache = {}

    # ======================================================
    #   TSNE CALCULATION (SLAMD-EXACT WITH NORMALIZATION)
    # ======================================================
    @classmethod
    def _run_tsne(cls, df: pd.DataFrame, input_columns, cache_key=None):
        """
        Calculate TSNE coordinates EXACTLY like SLAMD.
        CRITICAL: SLAMD standardizes features before TSNE!
        
        Args:
            df: DataFrame containing the data
            input_columns: List of input column names
            cache_key: Optional key for caching the result (e.g., filename + timestamp)
        """
        if df is None or df.empty:
            print("⚠️ TSNE: empty dataframe")
            return df
            
        # Check cache if key is provided
        if cache_key and cache_key in cls._tsne_cache:
            print(f"✅ TSNE: Using cached coordinates for key: {cache_key}")
            cached_data = cls._tsne_cache[cache_key]
            
            # We need to merge the cached TSNE coordinates back into the current df
            # The current df might have different utility/predictions, but the rows (and their order) 
            # for TSNE purposes (input space) should be the same if the cache key is valid.
            
            # It's safest to rely on Row number or index for merging, 
            # but usually the df passed here is the full dataset in the same order.
            
            if len(df) == len(cached_data):
                df = df.copy()
                df['tsne-2d-one'] = cached_data['tsne-2d-one'].values
                df['tsne-2d-two'] = cached_data['tsne-2d-two'].values
                return df
            else:
                print("⚠️ TSNE: Cache mismatch in length, re-calculating...")
            
        df = df.copy()
        
        # WEBSLAMD APPROACH: Use ONLY input_columns (features), NOT targets
        # This is critical - if targets are included, labelled rows cluster together
        # because predicted rows have NaN values that get filled to 0
        
        # Filter to only columns that exist in the dataframe
        feature_columns = [col for col in input_columns if col in df.columns]
        
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
        
        if cache_key:
            # Store only the necessary columns to save memory
            cls._tsne_cache[cache_key] = df[['tsne-2d-one', 'tsne-2d-two']].copy()
            print(f"✅ TSNE: Cached results for key: {cache_key}")

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
    def create_tsne_input_space_plot(cls, df: pd.DataFrame, input_columns, mode=None):
        """
        Create TSNE plot following SLAMD's visual style.
        Expects df to already have 'tsne-2d-one' and 'tsne-2d-two' columns.
        
        Args:
            df: DataFrame with data
            input_columns: List of input column names
            mode: Optional - "ML_MODE", "LLM_AGENT_MODE", or "HYBRID_MODE"
                  If not provided, will attempt to detect from data
        """
        if df is None or df.empty:
            return {'data': [], 'layout': {'title': 'No data available'}}

        df = df.copy()

        if "Row number" not in df.columns:
            df["Row number"] = df.index
        if "Utility" not in df.columns:
            df["Utility"] = 0.0
        df["Utility"] = pd.to_numeric(df["Utility"], errors="coerce").fillna(0)

        if 'tsne-2d-one' not in df.columns or 'tsne-2d-two' not in df.columns:
            print("⚠️ TSNE PLOT: Missing coordinates")
            return {'data': [], 'layout': {'title': 'TSNE coordinates missing'}}
        
        # Determine visualization mode
        if mode is not None:
            # Use explicitly provided mode
            is_llm_mode = (mode == "LLM_AGENT_MODE")
            print(f"📊 TSNE: Using explicit mode: {mode}")
        else:
            # Fallback: detect from data (Predicted column is all NaN)
            is_llm_mode = False
            if 'Predicted' in df.columns:
                predicted = pd.to_numeric(df['Predicted'], errors='coerce')
                if predicted.isna().all():
                    is_llm_mode = True
            else:
                is_llm_mode = True
        
        tsne_x = [float(x) for x in df['tsne-2d-one'].values]
        tsne_y = [float(y) for y in df['tsne-2d-two'].values]
        utility_values = [float(u) for u in df["Utility"].values]
        row_numbers = [int(r) for r in df["Row number"].values]

        fig = go.Figure()
        
        if is_llm_mode:
            print("📊 TSNE: Using LLM mode visualization")
            is_train = df.get('is_train_data', pd.Series([False] * len(df)))
            selected = df.get('Selected for Testing', pd.Series([False] * len(df)))
            
            # Unlabeled candidates (gray)
            unlabeled_idx = [i for i in range(len(df)) 
                           if not is_train.iloc[i] and not selected.iloc[i]]
            if unlabeled_idx:
                fig.add_trace(go.Scatter(
                    x=[tsne_x[i] for i in unlabeled_idx],
                    y=[tsne_y[i] for i in unlabeled_idx],
                    mode='markers', name='Unlabeled Candidates',
                    marker=dict(size=6, color='#CCCCCC', symbol='cross',
                               line=dict(color='#999999', width=0.5)),
                    customdata=[[row_numbers[i]] for i in unlabeled_idx],
                    hovertemplate="Row: %{customdata[0]}<br>t-SNE-1: %{x:.2f}<br>t-SNE-2: %{y:.2f}<extra>Candidate</extra>"
                ))
            
            # Labeled experiments (green)
            labeled_idx = [i for i in range(len(df)) if is_train.iloc[i]]
            if labeled_idx:
                fig.add_trace(go.Scatter(
                    x=[tsne_x[i] for i in labeled_idx],
                    y=[tsne_y[i] for i in labeled_idx],
                    mode='markers', name='Labeled Experiments',
                    marker=dict(size=10, color='#28a745', symbol='circle',
                               line=dict(color='#1e7e34', width=1)),
                    customdata=[[row_numbers[i]] for i in labeled_idx],
                    hovertemplate="Row: %{customdata[0]}<br>t-SNE-1: %{x:.2f}<br>t-SNE-2: %{y:.2f}<extra>Labeled</extra>"
                ))
            
            # LLM-selected point (red star)
            selected_idx = [i for i in range(len(df)) if selected.iloc[i]]
            if selected_idx:
                fig.add_trace(go.Scatter(
                    x=[tsne_x[i] for i in selected_idx],
                    y=[tsne_y[i] for i in selected_idx],
                    mode='markers', name='LLM Selected',
                    marker=dict(size=18, color='#dc3545', symbol='star',
                               line=dict(color='#721c24', width=2)),
                    customdata=[[row_numbers[i]] for i in selected_idx],
                    hovertemplate="Row: %{customdata[0]}<br>t-SNE-1: %{x:.2f}<br>t-SNE-2: %{y:.2f}<extra>🤖 LLM Selected</extra>"
                ))
            
            title = "t-SNE Material Space: LLM Agent Selection"
        else:
            # Standard ML mode - WEBSLAMD style
            is_train = df.get('is_train_data', pd.Series([True] * len(df)))
            
            # WEBSLAMD: Predicted points (circle) first, then Labelled (cross)
            predicted_indices = [i for i, val in enumerate(is_train) if not val]
            if predicted_indices:
                fig.add_trace(go.Scatter(
                    x=[tsne_x[i] for i in predicted_indices],
                    y=[tsne_y[i] for i in predicted_indices],
                    mode='markers', name='Predicted',
                    marker=dict(size=7, color=[utility_values[i] for i in predicted_indices],
                               colorscale='Plasma', showscale=True,
                               colorbar=dict(title="Utility"),
                               symbol='circle'),
                    customdata=[[row_numbers[i]] for i in predicted_indices],
                    hovertemplate="Row: %{customdata[0]}<br>t-SNE-1: %{x:.2f}<br>t-SNE-2: %{y:.2f}<br>Utility: %{marker.color:.2f}<extra></extra>"
                ))
            
            # WEBSLAMD: Labelled points (cross/plus)
            labelled_indices = [i for i, val in enumerate(is_train) if val]
            if labelled_indices:
                fig.add_trace(go.Scatter(
                    x=[tsne_x[i] for i in labelled_indices],
                    y=[tsne_y[i] for i in labelled_indices],
                    mode='markers', name='Labelled',
                    marker=dict(size=8, color=[utility_values[i] for i in labelled_indices],
                               colorscale='Plasma', showscale=False,
                               symbol='cross', line=dict(width=1)),
                    customdata=[[row_numbers[i]] for i in labelled_indices],
                    hovertemplate="Row: %{customdata[0]}<br>t-SNE-1: %{x:.2f}<br>t-SNE-2: %{y:.2f}<br>Utility: %{marker.color:.2f}<extra></extra>"
                ))
            
            title = "Materials data in t-SNE coordinates: train data and targets"

        # WEBSLAMD-style layout
        fig.update_layout(
            title=title,
            xaxis_title="t-SNE-1", yaxis_title="t-SNE-2",
            plot_bgcolor="lavender", paper_bgcolor="lavender",
            margin=dict(l=60, r=40, t=60, b=60), height=1000,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01)
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

    # ======================================================
    #   TRAJECTORY VISUALIZATION (LLM-AL Paper Feature)
    # ======================================================
    @classmethod
    def create_trajectory_plot(cls, df: pd.DataFrame, trajectory_data: dict, input_columns):
        """
        Visualize exploration trajectory overlaid on TSNE space.
        
        Shows the path the active learning agent takes through the feature space,
        as described in the LLM-AL paper.
        """
        if df is None or df.empty:
            return {'data': [], 'layout': {'title': 'No data available'}}
            
        if 'tsne-2d-one' not in df.columns or 'tsne-2d-two' not in df.columns:
            return {'data': [], 'layout': {'title': 'TSNE coordinates required'}}
        
        trajectory = trajectory_data.get('trajectory', [])
        
        if len(trajectory) < 1:
            return {'data': [], 'layout': {'title': 'No trajectory data yet'}}
        
        fig = go.Figure()
        
        # Background: all points (gray, transparent)
        tsne_x = [float(x) for x in df['tsne-2d-one'].values]
        tsne_y = [float(y) for y in df['tsne-2d-two'].values]
        
        fig.add_trace(go.Scatter(
            x=tsne_x, y=tsne_y,
            mode='markers',
            name='Design Space',
            marker=dict(size=4, color='#CCCCCC', opacity=0.3),
            hoverinfo='skip'
        ))
        
        # Get trajectory point coordinates
        # Match trajectory points to TSNE coordinates
        traj_x, traj_y, traj_labels = [], [], []
        
        for point in trajectory:
            row_idx = point.get('row_index')
            if row_idx is not None and row_idx in df.index:
                traj_x.append(float(df.loc[row_idx, 'tsne-2d-one']))
                traj_y.append(float(df.loc[row_idx, 'tsne-2d-two']))
                traj_labels.append(f"Iter {point['iteration']} ({point['mode']})")
        
        if len(traj_x) >= 2:
            # Draw trajectory path (lines)
            fig.add_trace(go.Scatter(
                x=traj_x, y=traj_y,
                mode='lines',
                name='Trajectory Path',
                line=dict(color='#0077B6', width=2, dash='solid'),
                hoverinfo='skip'
            ))
        
        if len(traj_x) >= 1:
            # Draw trajectory points with gradient colors
            colors = list(range(len(traj_x)))
            
            fig.add_trace(go.Scatter(
                x=traj_x, y=traj_y,
                mode='markers+text',
                name='Selected Points',
                marker=dict(
                    size=15,
                    color=colors,
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title="Iteration"),
                    line=dict(color='#023E8A', width=2)
                ),
                text=[str(i+1) for i in range(len(traj_x))],
                textposition='top center',
                textfont=dict(size=10, color='#023E8A'),
                customdata=traj_labels,
                hovertemplate="%{customdata}<br>t-SNE-1: %{x:.2f}<br>t-SNE-2: %{y:.2f}<extra></extra>"
            ))
        
        # Highlight start and end
        if len(traj_x) >= 1:
            fig.add_trace(go.Scatter(
                x=[traj_x[0]], y=[traj_y[0]],
                mode='markers',
                name='Start',
                marker=dict(size=20, color='#2ECC71', symbol='circle',
                           line=dict(color='#1E8449', width=2))
            ))
        
        if len(traj_x) >= 2:
            fig.add_trace(go.Scatter(
                x=[traj_x[-1]], y=[traj_y[-1]],
                mode='markers',
                name='Current',
                marker=dict(size=20, color='#E74C3C', symbol='star',
                           line=dict(color='#922B21', width=2))
            ))
        
        total_distance = trajectory_data.get('total_distance', 0)
        
        fig.update_layout(
            title=f"Exploration Trajectory (Cumulative Distance: {total_distance:.2f})",
            xaxis_title="t-SNE-1",
            yaxis_title="t-SNE-2",
            plot_bgcolor="#f7f7f7",
            paper_bgcolor="#f7f7f7",
            margin=dict(l=60, r=40, t=60, b=60),
            xaxis=dict(showgrid=True, gridcolor="#cccccc"),
            yaxis=dict(showgrid=True, gridcolor="#cccccc"),
            legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01)
        )
        
        return fig.to_dict()

    @classmethod
    def create_distance_plot(cls, trajectory_data: dict):
        """
        Plot cumulative distance traveled over iterations.
        
        Implements visualization from LLM-AL paper showing how different
        algorithms navigate the search space.
        """
        trajectory = trajectory_data.get('trajectory', [])
        distances = trajectory_data.get('cumulative_distances', [])
        
        if len(trajectory) < 2:
            return {'data': [], 'layout': {'title': 'Need at least 2 iterations'}}
        
        iterations = [p['iteration'] for p in trajectory]
        modes = [p['mode'] for p in trajectory]
        
        # Create hover text
        hover_text = [f"Iter {i}: {m}<br>Distance: {d:.3f}" 
                     for i, m, d in zip(iterations, modes, distances)]
        
        fig = go.Figure()
        
        # Main line
        fig.add_trace(go.Scatter(
            x=iterations,
            y=distances,
            mode='lines+markers',
            name='Cumulative Distance',
            line=dict(color='#0077B6', width=3),
            marker=dict(size=10, color='#0077B6'),
            text=hover_text,
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title="Cumulative Distance in Feature Space",
            xaxis_title="Iteration",
            yaxis_title="Cumulative Distance (Standardized)",
            plot_bgcolor="#f7f7f7",
            paper_bgcolor="#f7f7f7",
            margin=dict(l=60, r=40, t=60, b=60),
            xaxis=dict(showgrid=True, gridcolor="#cccccc", dtick=1),
            yaxis=dict(showgrid=True, gridcolor="#cccccc"),
        )
        
        return fig.to_dict()