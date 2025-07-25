"""Main dashboard application for fairness metrics visualization."""

import pandas as pd
from pathlib import Path
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, ALL, State
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dash import dash_table

from src.utils.data_preprocessing import load_data, preprocess_data, prepare_model_data
from src.models.model_factory import ModelFactory
from src.metrics.fairness_metrics import FairnessMetrics
from src.mitigation.reweighting import Reweighter
from src.utils.logger import logger, log_step, log_progress, log_error_trace, log_model_metrics, log_fairness_metrics
from config.config import (
    PROTECTED_ATTRIBUTES,
    METRICS_OUTPUT_PATH,
    DASH_HOST,
    DASH_PORT,
    DASH_DEBUG
)

def create_dataset_summary(model_data):
    """Create a summary of the dataset and model metrics."""
    try:
        if model_data.empty:
            return pd.DataFrame({'Metric': [], 'Value': []})
        
        # Get overall metrics - fix the filtering
        overall_metrics = model_data[
            (model_data['Protected Attribute'] == 'Overall') &
            (model_data['Metric Type'].isin([
                'Accuracy',
                'Precision (Macro Avg)',
                'Recall (Macro Avg)',
                'F1-Score (Macro Avg)'
            ]))
        ].copy()
        
        # Debug log
        logger.debug(f"Overall metrics data: {overall_metrics.to_dict()}")
        
        def get_metric_value(metric_type):
            try:
                metric_rows = overall_metrics[overall_metrics['Metric Type'] == metric_type]
                if not metric_rows.empty:
                    value = metric_rows['Value'].iloc[0]
                    if pd.notnull(value) and isinstance(value, (int, float)):
                        return f"{value:.2%}"
                return 'N/A'
            except Exception as e:
                logger.error(f"Error getting metric {metric_type}: {str(e)}")
                return 'N/A'
        
        # Calculate metrics
        summary_data = [
            {'Metric': 'Model Type', 'Value': model_data['Model'].iloc[0]},
            {'Metric': 'Accuracy', 'Value': get_metric_value('Accuracy')},
            {'Metric': 'Precision (Macro)', 'Value': get_metric_value('Precision (Macro Avg)')},
            {'Metric': 'Recall (Macro)', 'Value': get_metric_value('Recall (Macro Avg)')},
            {'Metric': 'F1-Score (Macro)', 'Value': get_metric_value('F1-Score (Macro Avg)')},
            {'Metric': 'Number of Features', 'Value': str(len(PROTECTED_ATTRIBUTES))},
            {'Metric': 'Training Samples', 'Value': "20000"},
            {'Metric': 'Positive Class Ratio', 'Value': "23.90% positive"}
        ]
        
        return pd.DataFrame(summary_data)
    except Exception as e:
        logger.error(f"Error in create_dataset_summary: {str(e)}")
        return pd.DataFrame([
            {'Metric': 'Error', 'Value': 'Unable to load metrics'}
        ])

def load_or_generate_metrics():
    """Load existing metrics or generate new ones if needed."""
    try:
        if Path(METRICS_OUTPUT_PATH).exists():
            logger.info(f"Loading existing metrics from {METRICS_OUTPUT_PATH}")
            return pd.read_parquet(METRICS_OUTPUT_PATH)
        
        logger.info("Generating new metrics...")
        
        # Load and preprocess data
        log_step("Loading data", total_steps=5)
        raw_data = load_data()
        
        log_step("Preprocessing data", total_steps=5)
        X, y = preprocess_data(raw_data)
        
        log_step("Preparing model data", total_steps=5)
        train_data, test_data = prepare_model_data(X, y)
        
        # Train and evaluate standard models
        log_step("Training and evaluating standard models", total_steps=5)
        standard_results = ModelFactory.train_and_evaluate_all(
            train_data['X_processed'],
            train_data['y'],
            test_data['X_processed'],
            test_data['y']
        )
        
        # Calculate standard fairness metrics
        standard_metrics = []
        for model_name, results in standard_results.items():
            logger.info(f"Calculating standard fairness metrics for {model_name}")
            
            # Add overall metrics first
            overall_metrics = pd.DataFrame([
                {
                    'Model': model_name,
                    'Strategy': 'Standard',
                    'Protected Attribute': 'Overall',
                    'Group': 'Overall',
                    'Metric Type': 'Accuracy',
                    'Value': results.accuracy
                },
                {
                    'Model': model_name,
                    'Strategy': 'Standard',
                    'Protected Attribute': 'Overall',
                    'Group': 'Overall',
                    'Metric Type': 'Precision (Macro Avg)',
                    'Value': results.classification_report['macro avg']['precision']
                },
                {
                    'Model': model_name,
                    'Strategy': 'Standard',
                    'Protected Attribute': 'Overall',
                    'Group': 'Overall',
                    'Metric Type': 'Recall (Macro Avg)',
                    'Value': results.classification_report['macro avg']['recall']
                },
                {
                    'Model': model_name,
                    'Strategy': 'Standard',
                    'Protected Attribute': 'Overall',
                    'Group': 'Overall',
                    'Metric Type': 'F1-Score (Macro Avg)',
                    'Value': results.classification_report['macro avg']['f1-score']
                }
            ])
            
            # Calculate fairness metrics
            metrics = FairnessMetrics(
                test_data['y'],
                results.y_pred,
                test_data['X_fairness'][PROTECTED_ATTRIBUTES],
                model_name,
                'Standard'
            )
            fairness_metrics = metrics.calculate_all_metrics()
            
            # Combine overall and fairness metrics
            standard_metrics.append(pd.concat([overall_metrics, fairness_metrics], ignore_index=True))
            
            # Log model performance
            log_model_metrics(model_name, {
                'accuracy': results.accuracy,
                'f1_score': results.classification_report['macro avg']['f1-score']
            })
        
        # Apply reweighting and evaluate
        log_step("Applying reweighting and evaluating models", total_steps=5)
        reweighter = Reweighter(PROTECTED_ATTRIBUTES)
        sample_weights = reweighter.fit_transform(
            train_data['X_raw'],
            train_data['y']
        )
        
        reweighted_results = ModelFactory.train_and_evaluate_all(
            train_data['X_processed'],
            train_data['y'],
            test_data['X_processed'],
            test_data['y'],
            sample_weights
        )
        
        # Calculate reweighted fairness metrics
        reweighted_metrics = []
        for model_name, results in reweighted_results.items():
            logger.info(f"Calculating reweighted fairness metrics for {model_name}")
            
            # Add overall metrics first
            overall_metrics = pd.DataFrame([
                {
                    'Model': model_name,
                    'Strategy': 'Reweighted',
                    'Protected Attribute': 'Overall',
                    'Group': 'Overall',
                    'Metric Type': 'Accuracy',
                    'Value': results.accuracy
                },
                {
                    'Model': model_name,
                    'Strategy': 'Reweighted',
                    'Protected Attribute': 'Overall',
                    'Group': 'Overall',
                    'Metric Type': 'Precision (Macro Avg)',
                    'Value': results.classification_report['macro avg']['precision']
                },
                {
                    'Model': model_name,
                    'Strategy': 'Reweighted',
                    'Protected Attribute': 'Overall',
                    'Group': 'Overall',
                    'Metric Type': 'Recall (Macro Avg)',
                    'Value': results.classification_report['macro avg']['recall']
                },
                {
                    'Model': model_name,
                    'Strategy': 'Reweighted',
                    'Protected Attribute': 'Overall',
                    'Group': 'Overall',
                    'Metric Type': 'F1-Score (Macro Avg)',
                    'Value': results.classification_report['macro avg']['f1-score']
                }
            ])
            
            # Calculate fairness metrics
            metrics = FairnessMetrics(
                test_data['y'],
                results.y_pred,
                test_data['X_fairness'][PROTECTED_ATTRIBUTES],
                model_name,
                'Reweighted'
            )
            fairness_metrics = metrics.calculate_all_metrics()
            
            # Combine overall and fairness metrics
            reweighted_metrics.append(pd.concat([overall_metrics, fairness_metrics], ignore_index=True))
            
            # Log model performance
            log_model_metrics(f"{model_name} (Reweighted)", {
                'accuracy': results.accuracy,
                'f1_score': results.classification_report['macro avg']['f1-score']
            })
        
        # Combine all metrics
        combined_metrics = pd.concat(
            standard_metrics + reweighted_metrics,
            ignore_index=True
        )
        
        # Save metrics
        logger.info(f"Saving metrics to {METRICS_OUTPUT_PATH}")
        combined_metrics.to_parquet(METRICS_OUTPUT_PATH, index=False)
        logger.info("Metrics generation complete!")
        
        return combined_metrics
    except Exception as e:
        log_error_trace(e)
        raise

# Load metrics data
metrics_df = load_or_generate_metrics()
available_models = sorted(metrics_df['Model'].unique())

# Read HTML template
with open('src/templates/dashboard.html', 'r') as file:
    html_template = file.read()

# Initialize Dash app with index_string
app = dash.Dash(
    __name__,
    index_string=html_template
)

# Define layout with CSS classes from template
app.layout = html.Div([
    html.H1("Fairness Auditing Dashboard", className="dashboard-title"),
    
    html.Div([
        html.Label("Select Model:", style={'font-weight': 'bold'}),
        html.Div([
            html.Button(
                model,
                id={'type': 'model-btn', 'index': model},
                n_clicks=0,
                className="model-btn" + (" selected" if model == available_models[0] else "")
            ) for model in available_models
        ], className="model-buttons"),
        dcc.Store(id='selected-model', data=available_models[0] if available_models else None)
    ], className="model-selector"),
    
    # Replace the overall performance chart with dataset summary
    html.Div([
        html.Div([
            html.H3("Model & Dataset Summary", className="chart-title"),
            dash_table.DataTable(
                id='dataset-summary-table',
                columns=[
                    {'name': 'Metric', 'id': 'Metric'},
                    {'name': 'Value', 'id': 'Value'}
                ],
                style_cell={
                    'textAlign': 'left',
                    'padding': '15px',
                    'font-family': 'Inter, sans-serif',
                    'color': '#1e293b',
                    'fontSize': '14px'
                },
                style_header={
                    'backgroundColor': '#2563eb',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'border': '1px solid #e2e8f0',
                    'fontSize': '14px'
                },
                style_data={
                    'backgroundColor': 'white',
                    'border': '1px solid #e2e8f0',
                    'color': '#1e293b',
                    'fontWeight': '400'
                },
                style_table={
                    'height': '350px',
                    'overflowY': 'auto'
                }
            )
        ], className="chart-box"),
        
        html.Div([
            html.H3("Disparate Impact Ratio (DIR)", className="chart-title"),
            dcc.Graph(
                id='disparate-impact-chart', 
                className="chart",
                style={'height': '450px', 'width': '100%'},
                config={'responsive': False, 'displayModeBar': False}
            )
        ], className="chart-box"),
        
        html.Div([
            html.H3("Equal Opportunity Difference (EOD)", className="chart-title"),
            dcc.Graph(
                id='equal-opportunity-chart', 
                className="chart",
                style={'height': '450px', 'width': '100%'},
                config={'responsive': False, 'displayModeBar': False}
            )
        ], className="chart-box"),
        
        html.Div([
            html.H3("PPR and TPR by Group", className="chart-title"),
            dcc.Graph(
                id='ppr-tpr-chart', 
                className="chart",
                style={'height': '450px', 'width': '100%'},
                config={'responsive': False, 'displayModeBar': False}
            )
        ], className="chart-box")
    ], className="chart-container")
])

def apply_consistent_layout(fig, height=450):
    """Apply consistent layout settings to all charts."""
    try:
        fig.update_layout(
            height=height,
            autosize=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font={'size': 11},
            margin=dict(t=40, l=60, r=40, b=60),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        return fig
    except Exception as e:
        print(f"Error in apply_consistent_layout: {str(e)}")
        return go.Figure()

# Callback to update selected model in dcc.Store
@app.callback(
    Output('selected-model', 'data'),
    [Input({'type': 'model-btn', 'index': ALL}, 'n_clicks')],
    [State('selected-model', 'data')]
)
def update_selected_model(n_clicks_list, current_model):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_model
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id:
        import json
        btn = json.loads(button_id)
        return btn['index']
    return current_model

# Update all charts based on selected model
@app.callback(
    [
        Output('dataset-summary-table', 'data'),
        Output('disparate-impact-chart', 'figure'),
        Output('equal-opportunity-chart', 'figure'),
        Output('ppr-tpr-chart', 'figure')
    ],
    Input('selected-model', 'data')
)
def update_charts(selected_model):
    """Update all charts based on selected model."""
    try:
        logger.debug(f"Updating charts for model: {selected_model}")
        
        # Initialize empty figures
        empty_fig = go.Figure()
        empty_fig = apply_consistent_layout(empty_fig)
        empty_data = pd.DataFrame({'Metric': [], 'Value': []}).to_dict('records')
        
        if not selected_model:
            logger.warning("No model selected")
            return empty_data, empty_fig, empty_fig, empty_fig
        
        # Filter data for selected model
        model_data = metrics_df[metrics_df['Model'] == selected_model].copy()
        if model_data.empty:
            logger.warning(f"No data found for model: {selected_model}")
            return empty_data, empty_fig, empty_fig, empty_fig
        
        logger.info(f"Generating visualizations for {selected_model}")
        logger.debug(f"Model data shape: {model_data.shape}")
        logger.debug(f"Model data columns: {model_data.columns}")
        logger.debug(f"Model data head: {model_data.head().to_dict()}")
        
        # Create dataset summary
        summary_data = create_dataset_summary(model_data).to_dict('records')
        
        # Fixed categories
        protected_attributes = PROTECTED_ATTRIBUTES
        ppr_tpr_types = ['PPR', 'TPR']
        
        # Disparate Impact Chart
        dir_data = model_data[
            (model_data['Metric Type'] == 'Disparate Impact Ratio') &
            (model_data['Protected Attribute'] != 'Overall')
        ].copy()
        
        logger.debug(f"DIR data shape: {dir_data.shape}")
        logger.debug(f"DIR data: {dir_data.to_dict()}")
        
        if not dir_data.empty:
            dir_fig = px.bar(
                dir_data,
                x='Protected Attribute',
                y='Value',
                color='Strategy',
                barmode='group',
                title=f'Disparate Impact Ratio - {selected_model}',
                labels={'Value': 'DIR', 'Protected Attribute': 'Attribute'},
                color_discrete_map={'Standard': '#2ca02c', 'Reweighted': '#d62728'}
            )
            dir_fig.update_layout(
                xaxis={
                    'categoryorder': 'array', 
                    'categoryarray': protected_attributes,
                    'tickangle': -45
                },
                bargap=0.2,
                bargroupgap=0.1,
                title_x=0.5,
                title_font_size=14
            )
            dir_fig.add_hline(
                y=1.0,
                line_dash="dot",
                line_color="grey",
                annotation_text="Ideal",
                annotation_position="top right"
            )
            dir_fig.update_traces(
                texttemplate='%{y:.2f}',
                textposition='outside',
                textfont_size=10
            )
            dir_fig = apply_consistent_layout(dir_fig)
        else:
            dir_fig = empty_fig
        
        # Equal Opportunity Chart
        eod_data = model_data[
            (model_data['Metric Type'] == 'Equal Opportunity Difference') &
            (model_data['Protected Attribute'] != 'Overall')
        ].copy()
        
        logger.debug(f"EOD data shape: {eod_data.shape}")
        logger.debug(f"EOD data: {eod_data.to_dict()}")
        
        if not eod_data.empty:
            eod_fig = px.bar(
                eod_data,
                x='Protected Attribute',
                y='Value',
                color='Strategy',
                barmode='group',
                title=f'Equal Opportunity Difference - {selected_model}',
                labels={'Value': 'EOD', 'Protected Attribute': 'Attribute'},
                color_discrete_map={'Standard': '#2ca02c', 'Reweighted': '#d62728'}
            )
            eod_fig.update_layout(
                xaxis={
                    'categoryorder': 'array', 
                    'categoryarray': protected_attributes,
                    'tickangle': -45
                },
                bargap=0.2,
                bargroupgap=0.1,
                title_x=0.5,
                title_font_size=14
            )
            eod_fig.add_hline(
                y=0.0,
                line_dash="dot",
                line_color="grey",
                annotation_text="Ideal",
                annotation_position="top right"
            )
            eod_fig.update_traces(
                texttemplate='%{y:.2f}',
                textposition='outside',
                textfont_size=10
            )
            eod_fig = apply_consistent_layout(eod_fig)
        else:
            eod_fig = empty_fig
        
        # PPR and TPR Chart
        ppr_tpr_data = model_data[
            (model_data['Protected Attribute'] != 'Overall') &
            (model_data['Metric Type'].isin(ppr_tpr_types))
        ].copy()
        
        logger.debug(f"PPR/TPR data shape: {ppr_tpr_data.shape}")
        logger.debug(f"PPR/TPR data: {ppr_tpr_data.head().to_dict()}")
        
        if not ppr_tpr_data.empty:
            ppr_tpr_fig = px.bar(
                ppr_tpr_data,
                x='Group',
                y='Value',
                color='Metric Type',
                facet_col='Protected Attribute',
                facet_row='Strategy',
                title=f'PPR and TPR by Group - {selected_model}',
                labels={'Value': 'Rate', 'Group': 'Group'},
                color_discrete_map={'PPR': '#6a0572', 'TPR': '#00a8cc'}
            )
            
            ppr_tpr_fig.update_layout(
                height=400,
                autosize=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font={'size': 10},
                margin=dict(t=80, l=60, r=40, b=60),
                title_x=0.5,
                title_font_size=14,
                bargap=0.2,
                bargroupgap=0.1,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.15,
                    xanchor="center",
                    x=0.5
                )
            )
            ppr_tpr_fig.update_traces(
                texttemplate='%{y:.2f}',
                textposition='outside',
                textfont_size=9
            )
            ppr_tpr_fig.update_xaxes(tickangle=-45)
            ppr_tpr_fig = apply_consistent_layout(ppr_tpr_fig, height=400)
        else:
            ppr_tpr_fig = apply_consistent_layout(go.Figure(), height=400)
        
        return summary_data, dir_fig, eod_fig, ppr_tpr_fig
    
    except Exception as e:
        logger.error(f"Error updating charts: {str(e)}")
        return empty_data, empty_fig, empty_fig, empty_fig

if __name__ == '__main__':
    logger.info("Starting Fairness Dashboard")
    logger.info(f"Server running at http://{DASH_HOST}:{DASH_PORT}")
    
    # Load initial metrics
    logger.info("Loading metrics data...")
    metrics_df = load_or_generate_metrics()
    available_models = sorted(metrics_df['Model'].unique())
    logger.info(f"Available models: {available_models}")
    
    # Run the server
    app.run(
        host=DASH_HOST,
        port=DASH_PORT,
        debug=DASH_DEBUG
    )