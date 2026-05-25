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
    def _run_tsne(
        cls,
        df: pd.DataFrame,
        input_columns,
        cache_key=None,
        perplexity=None,
        max_iter=350,
        learning_rate=100,
        random_state=42,
        scaling="standard",
    ):
        """
        Calculate TSNE coordinates EXACTLY like SLAMD.
        CRITICAL: SLAMD standardizes features before TSNE!
        
        Args:
            df: DataFrame containing the data
            input_columns: List of input column names
            cache_key: Optional key for caching the result (e.g., filename + timestamp)
            perplexity: Optional effective-neighbor setting. Defaults to min(20, n - 1).
            max_iter: Maximum t-SNE iterations.
            learning_rate: Gradient descent step size.
            random_state: Seed for stable embeddings.
            scaling: standard, robust, or none.
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
            
            # CRITICAL: SLAMD STANDARDIZES DATA BEFORE TSNE by default.
            if scaling == "robust":
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                tsne_input_scaled = scaler.fit_transform(tsne_input_df.values)
                print("✅ TSNE: Data robust-scaled (median/IQR)")
            elif scaling == "none":
                tsne_input_scaled = tsne_input_df.values
                print("✅ TSNE: Data scaling skipped")
            else:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                tsne_input_scaled = scaler.fit_transform(tsne_input_df.values)
                print(f"✅ TSNE: Data standardized (mean=0, std=1)")

            effective_perplexity = perplexity if perplexity is not None else min(20, len(df) - 1)
            effective_perplexity = max(2, min(float(effective_perplexity), max(2, len(df) - 1)))
            effective_iter = max(250, int(max_iter))
            effective_learning_rate = max(2.0, float(learning_rate))
            
            tsne = TSNE(
                n_components=2,
                verbose=1,
                perplexity=effective_perplexity,
                max_iter=effective_iter,
                random_state=int(random_state),
                init='pca',
                learning_rate=effective_learning_rate
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
            
            # Debug: Check column values
            print(f"📊 TSNE DEBUG: 'Selected for Testing' column exists: {'Selected for Testing' in df.columns}")
            if 'Selected for Testing' in df.columns:
                print(f"📊 TSNE DEBUG: Selected column dtype: {df['Selected for Testing'].dtype}")
                print(f"📊 TSNE DEBUG: Selected column values (unique): {df['Selected for Testing'].unique()}")
                print(f"📊 TSNE DEBUG: Selected True count: {(df['Selected for Testing'] == True).sum()}")
                print(f"📊 TSNE DEBUG: Selected 'True' string count: {(df['Selected for Testing'] == 'True').sum()}")
            
            # Handle both boolean and string "True"/"False"
            if selected.dtype == object:
                selected = selected.astype(str).str.lower() == 'true'
                print(f"📊 TSNE DEBUG: Converted string to bool, selected count: {selected.sum()}")
            
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
            margin=dict(l=60, r=40, t=60, b=60), height=600,
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
        """Plot measured target values already present in the uploaded dataset."""
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
            title="Observed Optimization History",
            xaxis_title="Dataset row with measured target",
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
    
    # Mode colors for consistent visualization
    MODE_COLORS = {
        'ML_MODE': {'primary': '#0077B6', 'secondary': '#00B4D8', 'name': 'ML Only'},
        'LLM_AGENT_MODE': {'primary': '#6F42C1', 'secondary': '#9775FA', 'name': 'LLM Agent'},
        'HYBRID_MODE': {'primary': '#198754', 'secondary': '#40C057', 'name': 'Hybrid'},
    }
    
    @classmethod
    def create_trajectory_plot(cls, df: pd.DataFrame, trajectory_data: dict, input_columns):
        """
        Visualize exploration trajectory overlaid on TSNE space with COLOR-CODED MODES.
        
        Each mode (ML, LLM, Hybrid) is shown in a different color:
        - ML Only: Blue (#0077B6)
        - LLM Agent: Purple (#6F42C1)
        - Hybrid: Green (#198754)
        
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
        
        # Group trajectory points by mode for color-coding
        mode_trajectories = {}
        for point in trajectory:
            mode = point.get('mode', 'ML_MODE')
            if mode not in mode_trajectories:
                mode_trajectories[mode] = []
            
            row_idx = point.get('row_index')
            if row_idx is not None and row_idx in df.index:
                mode_trajectories[mode].append({
                    'x': float(df.loc[row_idx, 'tsne-2d-one']),
                    'y': float(df.loc[row_idx, 'tsne-2d-two']),
                    'iteration': point['iteration'],
                    'mode': mode,
                    'utility': point.get('utility', 0)
                })
        
        # Draw the full path (all points in order) with connecting lines
        all_traj_x, all_traj_y = [], []
        for point in trajectory:
            row_idx = point.get('row_index')
            if row_idx is not None and row_idx in df.index:
                all_traj_x.append(float(df.loc[row_idx, 'tsne-2d-one']))
                all_traj_y.append(float(df.loc[row_idx, 'tsne-2d-two']))
        
        if len(all_traj_x) >= 2:
            # Draw connecting path (thin gray line)
            fig.add_trace(go.Scatter(
                x=all_traj_x, y=all_traj_y,
                mode='lines',
                name='Full Path',
                line=dict(color='#888888', width=1, dash='dot'),
                hoverinfo='skip',
                showlegend=False
            ))
        
        # Draw mode-specific points with their colors
        for mode, points in mode_trajectories.items():
            if not points:
                continue
                
            mode_info = cls.MODE_COLORS.get(mode, {'primary': '#888888', 'secondary': '#AAAAAA', 'name': mode})
            color = mode_info['primary']
            name = mode_info['name']
            
            xs = [p['x'] for p in points]
            ys = [p['y'] for p in points]
            iterations = [p['iteration'] for p in points]
            utilities = [p['utility'] for p in points]
            
            # Draw trajectory lines for this mode (segments only between consecutive same-mode points)
            if len(xs) >= 2:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode='lines',
                    name=f'{name} Path',
                    line=dict(color=color, width=2),
                    hoverinfo='skip',
                    showlegend=False
                ))
            
            # Draw points with mode color
            hover_text = [f"Iter {i}<br>{name}<br>Utility: {u:.3f}" 
                         for i, u in zip(iterations, utilities)]
            
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode='markers+text',
                name=f'{name} ({len(points)} pts)',
                marker=dict(
                    size=14,
                    color=color,
                    line=dict(color='white', width=2),
                    opacity=0.9
                ),
                text=[str(i) for i in iterations],
                textposition='top center',
                textfont=dict(size=9, color=color),
                hovertext=hover_text,
                hoverinfo='text'
            ))
        
        # Highlight start and current end points
        if len(all_traj_x) >= 1:
            first_point = trajectory[0]
            first_mode = first_point.get('mode', 'ML_MODE')
            first_color = cls.MODE_COLORS.get(first_mode, {}).get('primary', '#2ECC71')
            
            fig.add_trace(go.Scatter(
                x=[all_traj_x[0]], y=[all_traj_y[0]],
                mode='markers',
                name='Start',
                marker=dict(size=22, color=first_color, symbol='circle',
                           line=dict(color='white', width=3)),
                hoverinfo='name'
            ))
        
        if len(all_traj_x) >= 2:
            last_point = trajectory[-1]
            last_mode = last_point.get('mode', 'ML_MODE')
            last_color = cls.MODE_COLORS.get(last_mode, {}).get('primary', '#E74C3C')
            
            fig.add_trace(go.Scatter(
                x=[all_traj_x[-1]], y=[all_traj_y[-1]],
                mode='markers',
                name='Current',
                marker=dict(size=22, color=last_color, symbol='star',
                           line=dict(color='white', width=3)),
                hoverinfo='name'
            ))
        
        # Summary info
        total_distance = trajectory_data.get('total_distance', 0)
        modes_used = list(mode_trajectories.keys())
        mode_summary = ' | '.join([f"{cls.MODE_COLORS.get(m, {}).get('name', m)}: {len(mode_trajectories[m])}" 
                                   for m in modes_used])
        
        fig.update_layout(
            title=f"Trajectory by Mode ({mode_summary})<br><sub>Total Distance: {total_distance:.2f}</sub>",
            xaxis_title="t-SNE-1",
            yaxis_title="t-SNE-2",
            plot_bgcolor="#f7f7f7",
            paper_bgcolor="#f7f7f7",
            margin=dict(l=60, r=40, t=80, b=60),
            xaxis=dict(showgrid=True, gridcolor="#cccccc"),
            yaxis=dict(showgrid=True, gridcolor="#cccccc"),
            legend=dict(
                yanchor='top', y=0.99, xanchor='left', x=0.01,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#CCCCCC',
                borderwidth=1
            )
        )
        
        return fig.to_dict()

    @classmethod
    def create_distance_plot(cls, trajectory_data: dict):
        """
        Plot cumulative distance traveled over iterations with COLOR-CODED modes.
        
        Implements visualization from LLM-AL paper showing how different
        algorithms navigate the search space.
        """
        trajectory = trajectory_data.get('trajectory', [])
        distances = trajectory_data.get('cumulative_distances', [])
        
        if len(trajectory) < 2:
            return {'data': [], 'layout': {'title': 'Need at least 2 iterations'}}
        
        iterations = [p['iteration'] for p in trajectory]
        modes = [p['mode'] for p in trajectory]
        
        # Map modes to colors
        marker_colors = []
        for mode in modes:
            mode_info = cls.MODE_COLORS.get(mode, {'primary': '#888888'})
            marker_colors.append(mode_info['primary'])
        
        # Create hover text with mode names
        hover_text = []
        for i, m, d in zip(iterations, modes, distances):
            mode_name = cls.MODE_COLORS.get(m, {}).get('name', m)
            hover_text.append(f"Iter {i}: {mode_name}<br>Distance: {d:.3f}")
        
        fig = go.Figure()
        
        # Line connecting all points (gray)
        fig.add_trace(go.Scatter(
            x=iterations,
            y=distances,
            mode='lines',
            name='Path',
            line=dict(color='#AAAAAA', width=2),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Points colored by mode
        fig.add_trace(go.Scatter(
            x=iterations,
            y=distances,
            mode='markers',
            name='Iterations',
            marker=dict(
                size=12,
                color=marker_colors,
                line=dict(color='white', width=2)
            ),
            text=hover_text,
            hoverinfo='text'
        ))
        
        # Add legend entries for each mode used
        modes_used = list(set(modes))
        for mode in modes_used:
            mode_info = cls.MODE_COLORS.get(mode, {'primary': '#888888', 'name': mode})
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                name=mode_info['name'],
                marker=dict(size=10, color=mode_info['primary']),
                showlegend=True
            ))
        
        fig.update_layout(
            title="Cumulative Distance by Mode",
            xaxis_title="Iteration",
            yaxis_title="Cumulative Distance (Standardized)",
            plot_bgcolor="#f7f7f7",
            paper_bgcolor="#f7f7f7",
            margin=dict(l=60, r=40, t=60, b=60),
            xaxis=dict(showgrid=True, gridcolor="#cccccc", dtick=1),
            yaxis=dict(showgrid=True, gridcolor="#cccccc"),
            legend=dict(
                yanchor='bottom', y=0.01, xanchor='right', x=0.99,
                bgcolor='rgba(255,255,255,0.9)'
            )
        )
        
        return fig.to_dict()

    @classmethod
    def create_feature_importance_plot(cls, feature_importances: dict, input_columns: list):
        """
        Create a horizontal bar chart showing feature importance.
        
        Args:
            feature_importances: Dict mapping feature names to importance scores
            input_columns: List of input column names
        
        Returns:
            Plotly figure as dict
        """
        # Sort by importance
        sorted_features = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)
        features = [f[0] for f in sorted_features]
        importances = [f[1] for f in sorted_features]
        
        # Create color scale (positive = green, negative = red)
        colors = ['#198754' if imp >= 0 else '#dc3545' for imp in importances]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=features,
            x=importances,
            orientation='h',
            marker=dict(
                color=colors,
                opacity=0.8
            ),
            text=[f'{imp:.3f}' for imp in importances],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='Feature Importance',
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            template='plotly_white',
            height=max(400, len(features) * 25),  # Dynamic height based on features
            margin=dict(l=150, r=50, t=50, b=50),
            yaxis=dict(
                autorange='reversed'  # Most important at top
            )
        )
        
        return fig.to_dict()
    
    @classmethod
    def create_prediction_actual_plot(cls, df: pd.DataFrame, target_columns: list):
        """
        Create scatter plot of predicted vs actual values with perfect prediction line.
        
        Shows how well the model predictions match actual values for labeled samples.
        
        Args:
            df: DataFrame with labeled samples (must have both actual and predicted values)
            target_columns: List of target column names
        
        Returns:
            Plotly figure as dict
        """
        fig = go.Figure()
        
        colors = ['#0077B6', '#198754', '#FF6B35', '#6F42C1']
        
        for i, target_col in enumerate(target_columns):
            pred_col = f'Predicted_{target_col}'
            
            # Filter for samples with both actual and predicted values
            if pred_col in df.columns:
                valid_mask = df[target_col].notna() & df[pred_col].notna()
                actual = df.loc[valid_mask, target_col].values.tolist()  # Convert to list for JSON
                predicted = df.loc[valid_mask, pred_col].values.tolist()  # Convert to list for JSON
                
                if len(actual) > 0:
                    # Scatter points
                    fig.add_trace(go.Scatter(
                        x=actual,
                        y=predicted,
                        mode='markers',
                        name=target_col,
                        marker=dict(
                            color=colors[i % len(colors)],
                            size=10,
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=f'<b>{target_col}</b><br>Actual: %{{x:.2f}}<br>Predicted: %{{y:.2f}}<extra></extra>'
                    ))
                    
                    # Calculate R² score
                    if len(actual) > 1:
                        actual_arr = np.array(actual)
                        predicted_arr = np.array(predicted)
                        ss_res = ((actual_arr - predicted_arr) ** 2).sum()
                        ss_tot = ((actual_arr - actual_arr.mean()) ** 2).sum()
                        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        
                        # Add annotation with R²
                        fig.add_annotation(
                            x=0.02, y=0.98 - i * 0.08,
                            xref='paper', yref='paper',
                            text=f'{target_col}: R² = {r2:.3f}',
                            showarrow=False,
                            font=dict(size=12, color=colors[i % len(colors)]),
                            bgcolor='white',
                            bordercolor=colors[i % len(colors)],
                            borderwidth=1
                        )
        
        # Add perfect prediction line (y = x)
        if len(fig.data) > 0:
            all_values = []
            for trace in fig.data:
                all_values.extend(trace.x)
                all_values.extend(trace.y)
            min_val = min(all_values)
            max_val = max(all_values)
            
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='#888', dash='dash', width=2),
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=dict(
                text='Prediction vs Actual',
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title='Actual Value',
            yaxis_title='Predicted Value',
            template='plotly_white',
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(
                yanchor='bottom', y=0.01, xanchor='right', x=0.99,
                bgcolor='rgba(255,255,255,0.9)'
            )
        )
        
        return fig.to_dict()
