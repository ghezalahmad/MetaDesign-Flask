import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def visualize_exploration_exploitation(result_df, curiosity, show_details=True):
    """
    Visualize the exploration-exploitation trade-off with detailed metrics.
    
    Parameters:
    -----------
    result_df : pandas.DataFrame
        DataFrame with predictions and utility scores
    curiosity : float
        Exploration vs exploitation parameter (-2 to +2)
    show_details : bool
        Whether to show detailed metrics
    """
    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Exploration vs. Exploitation", "Utility Distribution"),
        specs=[[{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Convert curiosity to a normalized scale
    norm_curiosity = (curiosity + 2) / 4.0  # Maps [-2, 2] to [0, 1]
    
    # Add exploration vs exploitation scatter plot
    fig.add_trace(
        go.Scatter(
            x=result_df["Exploitation"],
            y=result_df["Exploration"],
            mode="markers",
            marker=dict(
                size=10,
                color=result_df["Utility"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Utility")
            ),
            text=result_df.index.astype(str),
            hoverinfo="text+x+y",
            name="Candidates"
        ),
        row=1, col=1
    )
    
    # Add highlighted point for selected sample
    selected_idx = result_df["Selected for Testing"].idxmax()
    fig.add_trace(
        go.Scatter(
            x=[result_df.loc[selected_idx, "Exploitation"]],
            y=[result_df.loc[selected_idx, "Exploration"]],
            mode="markers",
            marker=dict(
                size=15,
                color="red",
                symbol="star",
                line=dict(width=2, color="black")
            ),
            name="Selected Sample"
        ),
        row=1, col=1
    )
    
    # Add arrow to show curiosity direction
    arrow_start_x = 0.5
    arrow_start_y = 0.5
    arrow_end_x = arrow_start_x + (0.3 if norm_curiosity > 0.5 else -0.3)
    arrow_end_y = arrow_start_y + (0.3 if norm_curiosity > 0.5 else -0.3)
    
    fig.add_trace(
        go.Scatter(
            x=[arrow_start_x, arrow_end_x],
            y=[arrow_start_y, arrow_end_y],
            mode="lines+markers",
            line=dict(width=3, color="rgba(255, 165, 0, 0.7)"),
            marker=dict(size=[0, 10], symbol="arrow", angleref="previous"),
            name="Curiosity Direction"
        ),
        row=1, col=1
    )
    
    # Add utility distribution bar chart (top 10 samples)
    top_samples = result_df.sort_values("Utility", ascending=False).head(10).copy()
    top_samples["Label"] = [f"Sample {i}" for i in top_samples.index]
    
    fig.add_trace(
        go.Bar(
            x=top_samples["Label"],
            y=top_samples["Utility"],
            marker_color="lightblue",
            marker_line_color="darkblue",
            marker_line_width=1.5,
            name="Utility Score"
        ),
        row=1, col=2
    )
    
    # Highlight the selected bar
    selected_label = [label for i, label in enumerate(top_samples["Label"]) 
                      if top_samples.index[i] == selected_idx]
    if selected_label:
        highlight_idx = top_samples["Label"].tolist().index(selected_label[0])
        
        colors = ["lightblue"] * len(top_samples)
        colors[highlight_idx] = "crimson"
        
        fig.update_traces(
            marker_color=colors,
            selector=dict(type="bar")
        )
    
    # Update layout
    fig.update_layout(
        title=f"Exploration-Exploitation Analysis (Curiosity: {curiosity:.1f})",
        height=500,
        width=1400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Exploitation Score", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Exploration Score", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="Samples", tickangle=45, row=1, col=2)
    fig.update_yaxes(title_text="Utility Score", row=1, col=2)
    
    # Display the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed metrics if requested
    if show_details:
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Exploration", f"{result_df['Exploration'].mean():.3f}")
            
        with col2:
            st.metric("Average Exploitation", f"{result_df['Exploitation'].mean():.3f}")
            
        with col3:
            st.metric("Selected Sample Utility", f"{result_df.loc[selected_idx, 'Utility']:.3f}")
        
        # Create a slider to visualize curiosity settings
        st.slider(
            "Current Curiosity Setting", 
            min_value=-2.0, 
            max_value=2.0, 
            value=curiosity,
            step=0.1,
            disabled=True,
            key="curiosity_visualization"
        )
        
        # Explain what the current curiosity setting means
        if curiosity < -1.0:
            st.info("Current strategy: Strong exploitation - focusing on regions with known good performance.")
        elif curiosity < 0:
            st.info("Current strategy: Moderate exploitation - balancing with slight preference for known regions.")
        elif curiosity < 1.0:
            st.info("Current strategy: Balanced exploration-exploitation - considering both known and unknown regions.")
        else:
            st.info("Current strategy: Strong exploration - focusing on discovering new promising regions.")


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def plot_scatter_matrix_with_uncertainty(df, dimensions, utility_col="Utility", row_number_col="Row number"):
    if len(dimensions) < 2:
        return None

    matrix_size = len(dimensions) - 1
    fig = make_subplots(
        rows=matrix_size,
        cols=matrix_size,
        start_cell="top-left",
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )

    def select_error_col_if_available(column_name):
        return df.get(f"Uncertainty ({column_name})")

    def create_scatter_trace(x, y, x_name, y_name):
        return go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=7,
                color=df[utility_col],
                colorscale="Plasma",
                colorbar=dict(title="Utility") if (x_name == dimensions[0] and y_name == dimensions[-1]) else None,
            ),
            customdata=df[row_number_col],
            error_x=dict(
                type="data",
                array=select_error_col_if_available(x_name),
                color="lightgray",
                thickness=1,
            ),
            error_y=dict(
                type="data",
                array=select_error_col_if_available(y_name),
                color="lightgray",
                thickness=1,
            ),
            hovertemplate=(
                "Row number: %{customdata}, X: %{x:.2f}, Y: %{y:.2f}, Utility: %{marker.color:.2f}"
            ),
            hoverlabel=dict(bgcolor="black"),
            name="",
        )

    row_indices, col_indices = np.tril_indices(n=matrix_size, k=0)
    row_indices += 1
    col_indices += 1

    for row, col in zip(row_indices, col_indices):
        col_name = dimensions[col - 1]
        row_name = dimensions[row]
        trace = create_scatter_trace(
            df[col_name], df[row_name], col_name, row_name
        )
        fig.add_trace(trace, row=row, col=col)
        if row == matrix_size:
            fig.update_xaxes(title_text=col_name, row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=row_name, row=row, col=col)

    fig.update_layout(
        title="Scatter matrix of target properties",
        showlegend=False,
        height=1000,
    )

    return fig



import json
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE

def create_tsne_plot_with_hover(df, input_columns, utility_column='Utility', label_column='is_train_data'):
    if df is None or len(df) < 2 or len(input_columns) < 2:
        return None

    # Prepare a copy to avoid modifying original dataframe
    tsne_df = df.copy()

    # Standardize inputs
    features = tsne_df[input_columns].copy()
    features_std = features.std().replace(0, 1)
    features_mean = features.mean()
    features = (features - features_mean) / features_std

    # Reinsert label and utility information
    tsne_df[label_column] = tsne_df.get(label_column, 'Predicted')
    tsne_df[utility_column] = tsne_df.get(utility_column, -np.inf)

    # Ensure consistent ordering
    tsne_df = tsne_df.sort_values(by=utility_column, ascending=False)
    if 'Row number' not in tsne_df.columns:
        tsne_df.insert(loc=0, column='Row number', value=list(range(1, len(tsne_df) + 1)))

    # Run t-SNE with fixed params to match SLAMD
    tsne = TSNE(n_components=2, verbose=0, perplexity=min(20, len(features) - 1),
                max_iter=350, random_state=42, init='pca', learning_rate='auto') # Changed n_iter to max_iter
    tsne_result = tsne.fit_transform(features)

    tsne_plot_df = pd.DataFrame({
        'Row number': tsne_df['Row number'],
        't-SNE-1': tsne_result[:, 0],
        't-SNE-2': tsne_result[:, 1],
        utility_column: tsne_df[utility_column],
        label_column: tsne_df[label_column]
    })

    fig = px.scatter(
        tsne_plot_df,
        x='t-SNE-1', y='t-SNE-2',
        color=utility_column,
        symbol=label_column,
        symbol_sequence=['circle', 'cross'],
        custom_data=['Row number'],
        title='Materials data in t-SNE coordinates: train data and targets',
        render_mode='svg'
    )

    fig.update_traces(
        hovertemplate='Row number: %{customdata}, Utility: %{marker.color:.2f}',
        marker=dict(size=7)
    )

    fig.update_layout(
        height=1000,
        legend_title_text='',
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01
        )
    )

    return fig



def create_parallel_coordinates(result_df, target_columns):
    """
    Create a parallel coordinates plot of material properties.
    
    Parameters:
    -----------
    result_df : pandas.DataFrame
        DataFrame with predictions
    target_columns : list
        Column names for target variables
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Parallel coordinates figure
    """
    if not target_columns:
        return None

    plot_df = result_df[target_columns + ["Utility", "Uncertainty", "Novelty", "Selected for Testing"]].copy()
    plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Get the index of the selected row
    selected_idx = plot_df[plot_df["Selected for Testing"] == True].index
    
    # Create a color array for the lines
    line_colors = plot_df['Utility'].copy()
    # If a row is selected, set its color value to a value outside the normal range to map to red
    if not selected_idx.empty:
        # Use a value that will be distinctly colored, e.g., max utility + a buffer
        highlight_color_value = plot_df['Utility'].max() + 1
        line_colors.loc[selected_idx] = highlight_color_value

    # Define a custom colorscale: 'Viridis' for the main data, and 'red' for the highlight
    custom_colorscale = [[0.0, 'purple'], [0.5, 'green'], [1.0, 'yellow']] # Example: Viridis-like
    if not selected_idx.empty:
         # Normalize utility values to fit in the 0.0-0.9 range of the colorscale
        max_utility = plot_df['Utility'].max()
        normalized_colors = plot_df['Utility'] / max_utility * 0.9
        # reset the selected line to the highlight color value
        normalized_colors.loc[selected_idx] = 1.0
        custom_colorscale = [[0.0, 'blue'], [0.9, 'yellow'], [1.0, 'red']]
        line_color_values = normalized_colors
    else:
        line_color_values = plot_df['Utility']


    dimensions = target_columns + ["Utility", "Uncertainty", "Novelty"]

    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = line_color_values,
                       colorscale = custom_colorscale,
                       showscale = True,
                       colorbar = {'title': 'Utility'}),
            dimensions = [dict(range = [plot_df[dim].min(), plot_df[dim].max()],
                               label = dim, values = plot_df[dim]) for dim in dimensions]
        )
    )
    
    fig.update_layout(
        title="Parallel Coordinates of Material Properties",
        height=600,
    )
    
    return fig


def create_3d_scatter(result_df, x_property, y_property, z_property, color_by="Utility"):
    """
    Create a 3D scatter plot of material properties.
    
    Parameters:
    -----------
    result_df : pandas.DataFrame
        DataFrame with predictions
    x_property : str
        Column name for x-axis
    y_property : str
        Column name for y-axis
    z_property : str
        Column name for z-axis
    color_by : str
        Column to use for point coloring
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        3D scatter plot figure
    """
    # Create DataFrame for plotting
    plot_df = result_df[[x_property, y_property, z_property, color_by, "Uncertainty", "Selected for Testing"]].copy()
    
    # Add selection information
    plot_df["Selected"] = plot_df["Selected for Testing"].map({True: "Yes", False: "No"})
    
    # Create figure
    fig = px.scatter_3d(
        plot_df,
        x=x_property,
        y=y_property,
        z=z_property,
        color=color_by,
        size="Uncertainty",
        size_max=10,
        symbol="Selected",
        symbol_map={"Yes": "diamond", "No": "circle"},
        color_continuous_scale="Viridis",
        opacity=0.7,
        title=f"3D Visualization of Material Properties (colored by {color_by})"
    )
    
    # Highlight selected point
    selected_point = plot_df[plot_df["Selected"] == "Yes"]
    if not selected_point.empty:
        fig.add_trace(
            go.Scatter3d(
                x=[selected_point[x_property].values[0]],
                y=[selected_point[y_property].values[0]],
                z=[selected_point[z_property].values[0]],
                mode="markers",
                marker=dict(
                    size=12,
                    color="red",
                    symbol="diamond"
                ),
                name="Selected Sample"
            )
        )
    
    # Update layout
    fig.update_layout(
        width=None,
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        scene=dict(
            xaxis_title=x_property,
            yaxis_title=y_property,
            zaxis_title=z_property
        )
    )
    
    return fig

def create_pareto_front_visualization(result_df, target_columns, max_or_min):
    """
    Visualize the Pareto front for multi-objective optimization.
    
    Parameters:
    -----------
    result_df : pandas.DataFrame
        DataFrame with predictions
    target_columns : list
        Column names for target variables
    max_or_min : list
        Direction of optimization for each target ('max' or 'min')
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Pareto front visualization
    """
    if len(target_columns) < 2:
        return None
    
    # Select two target columns for visualization
    x_col = target_columns[0]
    y_col = target_columns[1]
    
    # Determine Pareto front
    is_pareto = np.ones(len(result_df), dtype=bool)
    
    for i, row_i in result_df.iterrows():
        for j, row_j in result_df.iterrows():
            if i != j:
                dominates = True
                
                for idx, col in enumerate(target_columns):
                    val_i = row_i[col]
                    val_j = row_j[col]
                    
                    # Check if j is better than i for this objective
                    if max_or_min[idx] == "max":
                        if val_j <= val_i:
                            dominates = False
                            break
                    else:  # min
                        if val_j >= val_i:
                            dominates = False
                            break
                
                if dominates:
                    is_pareto[i] = False
                    break
    
    # Create a copy of result_df with Pareto information
    plot_df = result_df.copy()
    plot_df["Pareto"] = is_pareto
    plot_df["Pareto_Label"] = plot_df["Pareto"].map({True: "Pareto Optimal", False: "Dominated"})
    
    # Create figure
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color="Pareto_Label",
        color_discrete_map={"Pareto Optimal": "red", "Dominated": "blue"},
        symbol="Selected for Testing",
        symbol_map={True: "star", False: "circle"},
        size="Utility",
        hover_data=target_columns + ["Utility", "Uncertainty", "Novelty"],
        title=f"Pareto Front Visualization: {x_col} vs {y_col}"
    )
    
    # Add lines connecting Pareto front points
    pareto_points = plot_df[plot_df["Pareto"]].sort_values(by=x_col)
    
    if len(pareto_points) > 1:
        fig.add_trace(
            go.Scatter(
                x=pareto_points[x_col],
                y=pareto_points[y_col],
                mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                name="Pareto Front"
            )
        )
    
    # Update axes based on optimization direction
    x_direction = max_or_min[target_columns.index(x_col)]
    y_direction = max_or_min[target_columns.index(y_col)]
    
    fig.update_layout(
        xaxis_title=f"{x_col} ({x_direction}imize)",
        yaxis_title=f"{y_col} ({y_direction}imize)",
        width=None,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def identify_pareto_front(predictions, max_or_min):
    """
    Identify the Pareto front of non-dominated solutions.
    
    Parameters:
    -----------
    predictions : numpy.ndarray
        Predictions for multiple objectives
    max_or_min : list
        Direction of optimization for each objective
        
    Returns:
    --------
    pareto_indices : numpy.ndarray
        Indices of samples on the Pareto front
    """
    n_samples = predictions.shape[0]
    n_objectives = predictions.shape[1]
    
    # Convert all objectives to maximization
    mod_predictions = predictions.copy()
    for i, direction in enumerate(max_or_min):
        if direction == "min":
            mod_predictions[:, i] = -mod_predictions[:, i]
    
    # Initialize domination count array
    is_dominated = np.zeros(n_samples, dtype=bool)
    
    # Compare each sample with all others
    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                # Check if j dominates i
                dominates = True
                
                for k in range(n_objectives):
                    # j doesn't dominate i if any objective is worse
                    if mod_predictions[j, k] < mod_predictions[i, k]:
                        dominates = False
                        break
                
                # If all objectives are at least as good and at least one is better
                if dominates and np.any(mod_predictions[j, :] > mod_predictions[i, :]):
                    is_dominated[i] = True
                    break
    
    # Return indices of non-dominated solutions
    return np.where(~is_dominated)[0]


def visualize_exploration_exploitation_tradeoff(result_df, curiosity_values=[-2, -1, 0, 1, 2]):
    """
    Create a visualization showing how changing curiosity affects exploration-exploitation.
    
    Parameters:
    -----------
    result_df : pandas.DataFrame
        DataFrame with predictions and metrics
    curiosity_values : list
        List of curiosity values to visualize
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Trade-off visualization
    """
    # Create figure
    fig = go.Figure()
    
    # Select a sample point for demonstration (highest utility)
    selected_idx = result_df["Utility"].idxmax()
    selected_row = result_df.iloc[selected_idx]
    
    # Base values
    uncertainty = selected_row["Uncertainty"]
    prediction = selected_row["Utility"] - uncertainty  # Estimate base prediction
    novelty = selected_row["Novelty"]
    
    # Calculate utility for different curiosity values
    curiosity_dict = {}
    for c in curiosity_values:
        # Simplified UCB calculation
        utility = prediction + c * uncertainty + max(0, c * 0.2 * novelty)
        curiosity_dict[c] = {
            "Utility": utility,
            "Exploitation": prediction,
            "Exploration": c * uncertainty + max(0, c * 0.2 * novelty)
        }
    
    # Plot trade-off curve
    curiosity_df = pd.DataFrame(curiosity_dict).T
    curiosity_df = curiosity_df.reset_index().rename(columns={"index": "Curiosity"})
    
    fig.add_trace(
        go.Scatter(
            x=curiosity_df["Exploitation"],
            y=curiosity_df["Exploration"],
            text=curiosity_df["Curiosity"],
            mode="lines+markers+text",
            marker=dict(
                size=10,
                color=curiosity_df["Curiosity"],
                colorscale="RdBu",
                cmin=-2,
                cmax=2,
                colorbar=dict(title="Curiosity")
            ),
            textposition="top center",
            name="Curiosity Path"
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Exploration-Exploitation Trade-off with Changing Curiosity",
        xaxis_title="Exploitation Component",
        yaxis_title="Exploration Component",
        width=None,
        height=500,
        showlegend=True
    )
    
    return fig


def highlight_optimal_regions(result_df, target_columns, max_or_min, percentile=75):
    """
    Create a heatmap highlighting optimal regions in the design space.
    
    Parameters:
    -----------
    result_df : pandas.DataFrame
        DataFrame with predictions
    target_columns : list
        Column names for target variables
    max_or_min : list
        Direction of optimization for each target ('max' or 'min')
    percentile : int
        Percentile threshold for highlighting (default: 75)
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Heatmap figure
    """
    if len(target_columns) < 2:
        return None
    
    # Select the first two parameters for visualization
    param1, param2 = target_columns[:2]
    
    # Create optimal flag for each parameter
    for i, col in enumerate(target_columns):
        threshold = np.percentile(result_df[col], percentile if max_or_min[i] == "max" else 100-percentile)
        if max_or_min[i] == "max":
            result_df[f"{col}_optimal"] = result_df[col] >= threshold
        else:
            result_df[f"{col}_optimal"] = result_df[col] <= threshold
    
    # Create a combined optimality score
    result_df["optimality_score"] = sum(result_df[f"{col}_optimal"] for col in target_columns)
    
    # Create a heatmap using a 2D histogram
    fig = px.density_heatmap(
        result_df, 
        x=param1, 
        y=param2, 
        z="optimality_score",
        nbinsx=20,
        nbinsy=20,
        color_continuous_scale="Viridis",
        title=f"Optimal Regions in {param1} vs {param2} Space"
    )
    
    # Add selected point
    if "Selected for Testing" in result_df.columns:
        selected_idx = result_df["Selected for Testing"].idxmax()
        selected_point = result_df.iloc[selected_idx]
        
        fig.add_trace(
            go.Scatter(
                x=[selected_point[param1]],
                y=[selected_point[param2]],
                mode="markers",
                marker=dict(
                    size=15,
                    color="red",
                    symbol="star",
                    line=dict(width=2, color="black")
                ),
                name="Selected Sample"
            )
        )
    
    # Update layout
    fig.update_layout(
        xaxis_title=f"{param1} ({max_or_min[0]}imize)",
        yaxis_title=f"{param2} ({max_or_min[1]}imize)",
        width=None,
        height=600,
        coloraxis_colorbar=dict(title="Optimality Score")
    )
    
    return fig


def visualize_model_comparison(models, data, input_columns, target_columns, metric="MSE"):
    """
    Compare different models on the same dataset.
    
    Parameters:
    -----------
    models : dict
        Dictionary mapping model names to model objects
    data : pandas.DataFrame
        Dataset containing input and target columns
    input_columns : list
        Column names for input features
    target_columns : list
        Column names for target variables
    metric : str
        Metric to use for comparison ('MSE', 'MAE', 'R2')
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Model comparison figure
    """
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Split data into labeled and unlabeled
    labeled_data = data.dropna(subset=target_columns)
    
    # Create figure
    fig = go.Figure()
    
    # Compute metrics for each model
    model_metrics = {}
    
    for model_name, model in models.items():
        # Process data for this model
        X = labeled_data[input_columns].values
        y_true = labeled_data[target_columns].values
        
        # Get predictions using this model
        if hasattr(model, "predict"):
            y_pred = model.predict(X)
        else:
            # Assume it's a PyTorch model
            import torch
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32)
                y_pred = model(X_tensor).numpy()
        
        # Calculate metrics
        metrics = {}
        if metric == "MSE" or metric == "all":
            metrics["MSE"] = mean_squared_error(y_true, y_pred, multioutput="uniform_average")
        if metric == "MAE" or metric == "all":
            metrics["MAE"] = mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
        if metric == "R2" or metric == "all":
            metrics["R2"] = r2_score(y_true, y_pred, multioutput="uniform_average")
        
        model_metrics[model_name] = metrics
    
    # Create DataFrame for plotting
    model_names = list(model_metrics.keys())
    
    if metric == "all":
        # Create multiple bar groups
        for m in ["MSE", "MAE", "R2"]:
            values = [model_metrics[model][m] for model in model_names]
            
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=values,
                    name=m
                )
            )
    else:
        # Single metric
        values = [model_metrics[model][metric] for model in model_names]
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=values,
                marker_color="lightblue",
                marker_line=dict(color="darkblue", width=1.5),
                name=metric
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"Model Comparison using {metric if metric != 'all' else 'Multiple Metrics'}",
        xaxis_title="Model",
        yaxis_title=metric if metric != 'all' else "Metric Value",
        width=None,
        height=500,
        barmode="group" if metric == "all" else "relative"
    )
    
    return fig



def create_acquisition_function_visualization(result_df, acquisition, curiosity):
    """
    Visualize how the acquisition function balances exploitation and exploration.
    
    Parameters:
    -----------
    result_df : pandas.DataFrame
        DataFrame with predictions
    acquisition : str
        Acquisition function used
    curiosity : float
        Exploration vs exploitation parameter (-2 to +2)
    """
    # Create a figure with 1 row and 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"{acquisition} Acquisition Function", "Component Contributions"),
        specs=[[{"type": "scatter3d"}, {"type": "bar"}]]
    )
    
    # 3D surface plot showing utility as a function of uncertainty and prediction value
    if "Utility" in result_df.columns and "Uncertainty" in result_df.columns:
        fig.add_trace(
            go.Scatter3d(
                x=result_df["Uncertainty"],
                y=result_df["Novelty"],
                z=result_df["Utility"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=result_df["Utility"],
                    colorscale="Viridis",
                    opacity=0.8,
                    showscale=True,
                    colorbar=dict(title="Utility"),
                ),
                name="Samples"
            ),
            row=1, col=1
        )
        
        # Add selected point
        selected_idx = result_df["Selected for Testing"].idxmax()
        fig.add_trace(
            go.Scatter3d(
                x=[result_df.loc[selected_idx, "Uncertainty"]],
                y=[result_df.loc[selected_idx, "Novelty"]],
                z=[result_df.loc[selected_idx, "Utility"]],
                mode="markers",
                marker=dict(
                    size=12,
                    color="red",
                    symbol="diamond"
                ),
                name="Selected Sample"
            ),
            row=1, col=1
        )
    
    # Bar plot showing component contributions for the selected sample
    if "Selected for Testing" in result_df.columns:
        selected_idx = result_df["Selected for Testing"].idxmax()
        selected = result_df.loc[selected_idx]
        
        # Calculate components based on acquisition function
        if acquisition == "UCB":
            components = {
                "Prediction": 0.7 * selected["Utility"],
                "Uncertainty": 0.2 * selected["Uncertainty"] * (1 + 0.5 * curiosity),
                "Novelty": 0.1 * selected["Novelty"] * (1 + 0.5 * curiosity if curiosity > 0 else 0)
            }
        elif acquisition == "EI":
            components = {
                "Improvement": 0.6 * selected["Utility"],
                "Uncertainty": 0.3 * selected["Uncertainty"],
                "Novelty": 0.1 * selected["Novelty"] * (1 + curiosity if curiosity > 0 else 0)
            }
        elif acquisition == "PI":
            components = {
                "Probability": 0.7 * selected["Utility"],
                "Uncertainty": 0.15 * selected["Uncertainty"],
                "Novelty": 0.15 * selected["Novelty"] * (1 + curiosity if curiosity > 0 else 0)
            }
        elif acquisition == "MaxEntropy":
            components = {
                "Prediction": 0.1 * selected["Utility"],
                "Uncertainty": 0.6 * selected["Uncertainty"],
                "Novelty": 0.3 * selected["Novelty"]
            }
        else:
            components = {
                "Prediction": 0.5 * selected["Utility"],
                "Uncertainty": 0.3 * selected["Uncertainty"],
                "Novelty": 0.2 * selected["Novelty"]
            }
        
        fig.add_trace(
            go.Bar(
                x=list(components.keys()),
                y=list(components.values()),
                marker_color=["blue", "orange", "green"],
                name="Component Contributions"
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=f"Acquisition Function Analysis: {acquisition} (Curiosity: {curiosity:.1f})",
        width=None,
        height=600,
        scene=dict(
            xaxis_title="Uncertainty",
            yaxis_title="Novelty",
            zaxis_title="Utility"
        )
    )
    
    return fig


def create_learning_curve_visualization(history, metric="loss"):
    """
    Visualize the learning curve during meta-training.
    
    Parameters:
    -----------
    history : dict
        Training history with metrics
    metric : str
        Metric to visualize (e.g., 'loss', 'accuracy')
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Learning curve figure
    """
    if not history or metric not in history:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add training metric
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history[metric]) + 1)),
            y=history[metric],
            mode="lines+markers",
            name=f"Training {metric}",
            line=dict(color="blue", width=2),
            marker=dict(size=6)
        )
    )
    
    # Add validation metric if available
    val_metric = f"val_{metric}"
    if val_metric in history:
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(history[val_metric]) + 1)),
                y=history[val_metric],
                mode="lines+markers",
                name=f"Validation {metric}",
                line=dict(color="red", width=2),
                marker=dict(size=6)
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"Learning Curve: {metric.capitalize()}",
        xaxis_title="Epoch",
        yaxis_title=metric.capitalize(),
        width=None,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def visualize_property_distributions(result_df, target_columns):
    """
    Visualize the distribution of predicted material properties.
    
    Parameters:
    -----------
    result_df : pandas.DataFrame
        DataFrame with predictions
    target_columns : list
        Column names for target variables
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Property distribution figure
    """
    if not target_columns:
        return None
    
    # Determine number of rows and columns for subplots
    n_props = len(target_columns)
    n_cols = min(3, n_props)
    n_rows = (n_props + n_cols - 1) // n_cols
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=target_columns
    )
    
    # Add histogram for each property
    for i, prop in enumerate(target_columns):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        fig.add_trace(
            go.Histogram(
                x=result_df[prop],
                histnorm="probability",
                marker_color="lightblue",
                marker_line=dict(color="darkblue", width=1),
                name=prop
            ),
            row=row, col=col
        )
        
        # Add vertical line for selected sample
        if "Selected for Testing" in result_df.columns:
            selected_idx = result_df["Selected for Testing"].idxmax()
            selected_value = result_df.loc[selected_idx, prop]
            
            fig.add_shape(
                type="line",
                x0=selected_value,
                x1=selected_value,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="red", width=2, dash="dash"),
                row=row, col=col
            )
    
    # Update layout
    fig.update_layout(
        title="Distribution of Predicted Material Properties",
        width=None,
        height=300 * n_rows,
        showlegend=False
    )
    
    for i in range(n_rows * n_cols):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        fig.update_xaxes(title_text="Value", row=row, col=col)
        fig.update_yaxes(title_text="Probability", row=row, col=col)
    
    return fig

def _select_error_col_if_available(df, column_name):
    """
    Returns the error column for a given property if it exists.
    Assumes uncertainty columns are named 'Uncertainty (Property)'.
    """
    col_name = f"Uncertainty ({column_name})"
    return df[col_name] if col_name in df.columns else None
