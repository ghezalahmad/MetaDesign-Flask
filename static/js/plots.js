//- Plotly configuration
var config = {
    'displayModeBar': true,
    'displaylogo': false,
    'modeBarButtonsToRemove': [
        'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
        'autoScale2d', 'resetScale2d', 'hoverClosestCartesian',
        'hoverCompareCartesian', 'toggleSpikelines'
    ]
}

//- Plotly layout for t-SNE plot
var layout_tsne = {
    'xaxis': {
        'title': 'tsne-1',
        'zeroline': true,
        'gridcolor': '#444444',
    },
    'yaxis': {
        'title': 'tsne-2',
        'zeroline': true,
        'gridcolor': '#444444',
    },
    'plot_bgcolor': '#2a2a2a',
    'paper_bgcolor': '#2a2a2a',
    'font': {
        'color': '#ffffff'
    },
    'showlegend': true,
    'legend': {
        'x': 1,
        'y': 1,
        'xanchor': 'right',
        'yanchor': 'top',
        'bgcolor': 'rgba(0,0,0,0)',
        'bordercolor': '#ffffff',
        'borderwidth': 1
    }
}

function create_scatter_plot(scatter_data) {
    var traces = [];
    var categories = [...new Set(scatter_data.categories)];
    var colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];

    categories.forEach((cat, i) => {
        var cat_data = scatter_data.points.filter(p => p.category === cat);
        traces.push({
            x: cat_data.map(p => p.x),
            y: cat_data.map(p => p.y),
            mode: 'markers',
            type: 'scatter',
            name: cat,
            marker: {
                color: colors[i % colors.length],
                size: 10
            }
        });
    });

    Plotly.newPlot('scatter-plot', traces, layout_scatter, config);
}

function create_tsne_plot(tsne_data) {
    var traces = [];
    var categories = [...new Set(tsne_data.categories)];
    var colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'];

    categories.forEach((cat, i) => {
        var cat_data = tsne_data.points.filter(p => p.category === cat);
        traces.push({
            x: cat_data.map(p => p.x),
            y: cat_data.map(p => p.y),
            mode: 'markers',
            type: 'scatter',
            name: cat,
            marker: {
                color: colors[i % colors.length],
                size: 10
            }
        });
    });

    Plotly.newPlot('tsne-plot', traces, layout_tsne, config);
}

//- Plotly layout for scatter plot
var layout_scatter = {
    'xaxis': {
        'title': 'Target Property',
        'zeroline': true,
        'gridcolor': '#444444',
    },
    'yaxis': {
        'title': 'Uncertainty',
        'zeroline': true,
        'gridcolor': '#444444',
    },
    'plot_bgcolor': '#2a2a2a',
    'paper_bgcolor': '#2a2a2a',
    'font': {
        'color': '#ffffff'
    },
    'showlegend': true,
    'legend': {
        'x': 1,
        'y': 1,
        'xanchor': 'right',
        'yanchor': 'top',
        'bgcolor': 'rgba(0,0,0,0)',
        'bordercolor': '#ffffff',
        'borderwidth': 1
    }
}
