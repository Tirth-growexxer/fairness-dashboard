"""Integration tests for dashboard functionality."""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.data_preprocessing import load_data, preprocess_data, prepare_model_data
from src.models.model_factory import ModelFactory
from src.metrics.fairness_metrics import FairnessMetrics
from config.config import PROTECTED_ATTRIBUTES, RANDOM_SEED, METRICS_OUTPUT_PATH

class TestDashboardFunctionality(unittest.TestCase):
    """Integration tests for dashboard functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment for dashboard tests."""
        print("Setting up dashboard integration test environment...")
        
        # Load and prepare test data
        np.random.seed(RANDOM_SEED)
        cls.raw_data = load_data()
        cls.X, cls.y = preprocess_data(cls.raw_data)
        cls.train_data, cls.test_data = prepare_model_data(cls.X, cls.y)
        
        # Train models and generate metrics for testing
        cls.models_results = ModelFactory.train_and_evaluate_all(
            cls.train_data['X_processed'],
            cls.train_data['y'],
            cls.test_data['X_processed'],
            cls.test_data['y']
        )
        
        # Generate sample metrics data
        cls.sample_metrics = cls._generate_sample_metrics()
        
        print("Dashboard test environment ready")
    
    @classmethod
    def _generate_sample_metrics(cls):
        """Generate sample metrics data for dashboard testing."""
        all_metrics = []
        
        for model_name, results in cls.models_results.items():
            # Add overall metrics
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
                    'Metric Type': 'F1-Score (Macro Avg)',
                    'Value': results.classification_report['macro avg']['f1-score']
                }
            ])
            
            # Add fairness metrics
            fairness_calculator = FairnessMetrics(
                cls.test_data['y'],
                results.y_pred,
                cls.test_data['X_fairness'][PROTECTED_ATTRIBUTES],
                model_name,
                'Standard'
            )
            fairness_metrics = fairness_calculator.calculate_all_metrics()
            
            all_metrics.extend([overall_metrics, fairness_metrics])
        
        return pd.concat(all_metrics, ignore_index=True)
    
    def test_metrics_data_loading(self):
        """Test metrics data loading functionality."""
        print("\n=== Testing Metrics Data Loading ===")
        
        # Test metrics data structure
        self.assertIsInstance(self.sample_metrics, pd.DataFrame)
        self.assertGreater(len(self.sample_metrics), 0)
        
        # Check required columns
        required_columns = {'Model', 'Strategy', 'Protected Attribute', 'Group', 'Metric Type', 'Value'}
        self.assertTrue(required_columns.issubset(set(self.sample_metrics.columns)))
        
        # Check available models
        available_models = sorted(self.sample_metrics['Model'].unique())
        expected_models = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'XGBoost']
        self.assertEqual(available_models, expected_models)
        
        print(f"Available models: {available_models}")
        print(f"Total metrics records: {len(self.sample_metrics)}")
        
        # Test model selection functionality
        for model in available_models:
            model_data = self.sample_metrics[self.sample_metrics['Model'] == model]
            self.assertGreater(len(model_data), 0)
            print(f"  {model}: {len(model_data)} metrics")
    
    def test_dataset_summary_generation(self):
        """Test dataset summary generation for dashboard."""
        print("\n=== Testing Dataset Summary Generation ===")
        
        def create_dataset_summary(model_data):
            """Simulate the dashboard's create_dataset_summary function."""
            if model_data.empty:
                return pd.DataFrame({'Metric': [], 'Value': []})
            
            # Get overall metrics
            overall_metrics = model_data[
                (model_data['Protected Attribute'] == 'Overall') &
                (model_data['Metric Type'].isin([
                    'Accuracy',
                    'Precision (Macro Avg)',
                    'Recall (Macro Avg)',
                    'F1-Score (Macro Avg)'
                ]))
            ].copy()
            
            def get_metric_value(metric_type):
                metric_rows = overall_metrics[overall_metrics['Metric Type'] == metric_type]
                if not metric_rows.empty:
                    value = metric_rows['Value'].iloc[0]
                    if pd.notnull(value) and isinstance(value, (int, float)):
                        return f"{value:.2%}"
                return 'N/A'
            
            # Create summary data
            summary_data = [
                {'Metric': 'Model Type', 'Value': model_data['Model'].iloc[0]},
                {'Metric': 'Accuracy', 'Value': get_metric_value('Accuracy')},
                {'Metric': 'F1-Score (Macro)', 'Value': get_metric_value('F1-Score (Macro Avg)')},
                {'Metric': 'Number of Features', 'Value': str(len(PROTECTED_ATTRIBUTES))},
                {'Metric': 'Training Samples', 'Value': str(len(self.train_data['y']))},
                {'Metric': 'Test Samples', 'Value': str(len(self.test_data['y']))}
            ]
            
            return pd.DataFrame(summary_data)
        
        # Test summary generation for each model
        for model_name in ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']:
            model_data = self.sample_metrics[self.sample_metrics['Model'] == model_name]
            summary = create_dataset_summary(model_data)
            
            # Validate summary structure
            self.assertIsInstance(summary, pd.DataFrame)
            self.assertEqual(list(summary.columns), ['Metric', 'Value'])
            self.assertGreater(len(summary), 0)
            
            print(f"\n{model_name} Summary:")
            for _, row in summary.iterrows():
                print(f"  {row['Metric']}: {row['Value']}")
    
    def test_chart_data_preparation(self):
        """Test data preparation for dashboard charts."""
        print("\n=== Testing Chart Data Preparation ===")
        
        def prepare_disparate_impact_chart_data(model_data):
            """Simulate DIR chart data preparation."""
            dir_data = model_data[model_data['Metric Type'] == 'Disparate Impact Ratio']
            
            chart_data = {
                'attributes': [],
                'standard_values': [],
                'reweighted_values': []
            }
            
            for attribute in PROTECTED_ATTRIBUTES:
                attr_data = dir_data[dir_data['Protected Attribute'] == attribute]
                
                standard_value = None
                reweighted_value = None
                
                for _, row in attr_data.iterrows():
                    if row['Strategy'] == 'Standard':
                        standard_value = row['Value']
                    elif row['Strategy'] == 'Reweighted':
                        reweighted_value = row['Value']
                
                if standard_value is not None:
                    chart_data['attributes'].append(attribute)
                    chart_data['standard_values'].append(standard_value)
                    chart_data['reweighted_values'].append(reweighted_value or 0)
            
            return chart_data
        
        def prepare_equal_opportunity_chart_data(model_data):
            """Simulate EOD chart data preparation."""
            eod_data = model_data[model_data['Metric Type'] == 'Equal Opportunity Difference']
            
            chart_data = {
                'attributes': [],
                'standard_values': [],
                'reweighted_values': []
            }
            
            for attribute in PROTECTED_ATTRIBUTES:
                attr_data = eod_data[eod_data['Protected Attribute'] == attribute]
                
                standard_value = None
                reweighted_value = None
                
                for _, row in attr_data.iterrows():
                    if row['Strategy'] == 'Standard':
                        standard_value = row['Value']
                    elif row['Strategy'] == 'Reweighted':
                        reweighted_value = row['Value']
                
                if standard_value is not None:
                    chart_data['attributes'].append(attribute)
                    chart_data['standard_values'].append(standard_value)
                    chart_data['reweighted_values'].append(reweighted_value or 0)
            
            return chart_data
        
        def prepare_ppr_tpr_chart_data(model_data):
            """Simulate PPR/TPR chart data preparation."""
            ppr_data = model_data[model_data['Metric Type'] == 'PPR']
            tpr_data = model_data[model_data['Metric Type'] == 'TPR']
            
            chart_data = {
                'groups': [],
                'attributes': [],
                'ppr_values': [],
                'tpr_values': []
            }
            
            # Combine PPR and TPR data
            for _, ppr_row in ppr_data.iterrows():
                matching_tpr = tpr_data[
                    (tpr_data['Protected Attribute'] == ppr_row['Protected Attribute']) &
                    (tpr_data['Group'] == ppr_row['Group']) &
                    (tpr_data['Strategy'] == ppr_row['Strategy'])
                ]
                
                if not matching_tpr.empty:
                    chart_data['groups'].append(ppr_row['Group'])
                    chart_data['attributes'].append(ppr_row['Protected Attribute'])
                    chart_data['ppr_values'].append(ppr_row['Value'])
                    chart_data['tpr_values'].append(matching_tpr.iloc[0]['Value'])
            
            return chart_data
        
        # Test chart data preparation for each model
        for model_name in ['Logistic Regression', 'Decision Tree']:
            print(f"\nTesting chart data for {model_name}:")
            model_data = self.sample_metrics[self.sample_metrics['Model'] == model_name]
            
            # Test DIR chart data
            dir_chart_data = prepare_disparate_impact_chart_data(model_data)
            self.assertIsInstance(dir_chart_data, dict)
            self.assertIn('attributes', dir_chart_data)
            self.assertIn('standard_values', dir_chart_data)
            print(f"  DIR chart: {len(dir_chart_data['attributes'])} attributes")
            
            # Test EOD chart data
            eod_chart_data = prepare_equal_opportunity_chart_data(model_data)
            self.assertIsInstance(eod_chart_data, dict)
            self.assertIn('attributes', eod_chart_data)
            print(f"  EOD chart: {len(eod_chart_data['attributes'])} attributes")
            
            # Test PPR/TPR chart data
            ppr_tpr_chart_data = prepare_ppr_tpr_chart_data(model_data)
            self.assertIsInstance(ppr_tpr_chart_data, dict)
            self.assertIn('groups', ppr_tpr_chart_data)
            print(f"  PPR/TPR chart: {len(ppr_tpr_chart_data['groups'])} groups")
    
    def test_model_selection_functionality(self):
        """Test model selection functionality."""
        print("\n=== Testing Model Selection Functionality ===")
        
        def simulate_model_selection(selected_model, metrics_data):
            """Simulate model selection callback."""
            if not selected_model or metrics_data.empty:
                return {
                    'summary': pd.DataFrame(),
                    'dir_data': {},
                    'eod_data': {},
                    'ppr_tpr_data': {}
                }
            
            # Filter data for selected model
            model_data = metrics_data[metrics_data['Model'] == selected_model]
            
            # Generate summary
            summary_data = [
                {'Metric': 'Model Type', 'Value': selected_model},
                {'Metric': 'Total Metrics', 'Value': str(len(model_data))}
            ]
            summary = pd.DataFrame(summary_data)
            
            # Generate chart data (simplified)
            dir_data = {'attributes': [], 'values': []}
            eod_data = {'attributes': [], 'values': []}
            ppr_tpr_data = {'groups': [], 'ppr': [], 'tpr': []}
            
            return {
                'summary': summary,
                'dir_data': dir_data,
                'eod_data': eod_data,
                'ppr_tpr_data': ppr_tpr_data
            }
        
        # Test model selection for each available model
        available_models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
        
        for model in available_models:
            result = simulate_model_selection(model, self.sample_metrics)
            
            # Validate result structure
            self.assertIn('summary', result)
            self.assertIn('dir_data', result)
            self.assertIn('eod_data', result)
            self.assertIn('ppr_tpr_data', result)
            
            # Validate summary
            self.assertIsInstance(result['summary'], pd.DataFrame)
            if not result['summary'].empty:
                self.assertEqual(result['summary'].iloc[0]['Value'], model)
            
            print(f"  {model}: Model selection test passed")
        
        # Test invalid model selection
        invalid_result = simulate_model_selection('Invalid Model', self.sample_metrics)
        self.assertTrue(invalid_result['summary'].empty)
        print("  Invalid model selection: Handled correctly")
    
    def test_data_filtering_and_aggregation(self):
        """Test data filtering and aggregation for dashboard."""
        print("\n=== Testing Data Filtering and Aggregation ===")
        
        # Test filtering by strategy
        standard_data = self.sample_metrics[self.sample_metrics['Strategy'] == 'Standard']
        reweighted_data = self.sample_metrics[self.sample_metrics['Strategy'] == 'Reweighted']
        
        print(f"Standard strategy records: {len(standard_data)}")
        print(f"Reweighted strategy records: {len(reweighted_data)}")
        
        # Test filtering by protected attribute
        for attribute in PROTECTED_ATTRIBUTES:
            attr_data = self.sample_metrics[self.sample_metrics['Protected Attribute'] == attribute]
            print(f"  {attribute}: {len(attr_data)} records")
            
            if len(attr_data) > 0:
                # Test metric type distribution
                metric_types = attr_data['Metric Type'].value_counts()
                self.assertGreater(len(metric_types), 0)
                print(f"    Metric types: {list(metric_types.index)}")
        
        # Test aggregation by model and strategy
        aggregated = self.sample_metrics.groupby(['Model', 'Strategy']).size().reset_index(name='count')
        self.assertGreater(len(aggregated), 0)
        print(f"Model-Strategy combinations: {len(aggregated)}")
        
        # Test value statistics
        numeric_metrics = self.sample_metrics[pd.notnull(self.sample_metrics['Value'])]
        if len(numeric_metrics) > 0:
            value_stats = numeric_metrics['Value'].describe()
            print(f"Value statistics:")
            print(f"  Count: {value_stats['count']}")
            print(f"  Mean: {value_stats['mean']:.4f}")
            print(f"  Min: {value_stats['min']:.4f}")
            print(f"  Max: {value_stats['max']:.4f}")
    
    def test_dashboard_error_handling(self):
        """Test error handling in dashboard functionality."""
        print("\n=== Testing Dashboard Error Handling ===")
        
        # Test with empty metrics data
        empty_df = pd.DataFrame()
        
        def safe_model_selection(selected_model, metrics_data):
            """Safe model selection with error handling."""
            try:
                if metrics_data.empty:
                    return {'error': 'No metrics data available'}
                
                if selected_model not in metrics_data['Model'].unique():
                    return {'error': f'Model {selected_model} not found'}
                
                return {'success': True, 'model': selected_model}
            except Exception as e:
                return {'error': str(e)}
        
        # Test empty data handling
        result = safe_model_selection('Logistic Regression', empty_df)
        self.assertIn('error', result)
        print(f"Empty data handling: {result['error']}")
        
        # Test invalid model handling
        result = safe_model_selection('Invalid Model', self.sample_metrics)
        self.assertIn('error', result)
        print(f"Invalid model handling: {result['error']}")
        
        # Test valid model handling
        result = safe_model_selection('Logistic Regression', self.sample_metrics)
        self.assertIn('success', result)
        print(f"Valid model handling: Success")
        
        # Test malformed data handling
        malformed_df = pd.DataFrame({'wrong_column': [1, 2, 3]})
        result = safe_model_selection('Logistic Regression', malformed_df)
        self.assertIn('error', result)
        print(f"Malformed data handling: {result['error']}")
    
    def test_chart_configuration_validation(self):
        """Test chart configuration and validation."""
        print("\n=== Testing Chart Configuration Validation ===")
        
        def validate_chart_config(chart_data, chart_type):
            """Validate chart configuration."""
            if chart_type == 'bar':
                required_keys = ['x', 'y']
            elif chart_type == 'scatter':
                required_keys = ['x', 'y']
            elif chart_type == 'grouped_bar':
                required_keys = ['x', 'y', 'color']
            else:
                return {'valid': False, 'error': f'Unknown chart type: {chart_type}'}
            
            for key in required_keys:
                if key not in chart_data:
                    return {'valid': False, 'error': f'Missing required key: {key}'}
                
                if not chart_data[key]:
                    return {'valid': False, 'error': f'Empty data for key: {key}'}
            
            return {'valid': True}
        
        # Test valid chart configurations
        valid_bar_config = {'x': ['A', 'B', 'C'], 'y': [1, 2, 3]}
        result = validate_chart_config(valid_bar_config, 'bar')
        self.assertTrue(result['valid'])
        print("Valid bar chart config: PASSED")
        
        valid_grouped_bar_config = {
            'x': ['A', 'B'], 
            'y': [1, 2], 
            'color': ['red', 'blue']
        }
        result = validate_chart_config(valid_grouped_bar_config, 'grouped_bar')
        self.assertTrue(result['valid'])
        print("Valid grouped bar chart config: PASSED")
        
        # Test invalid chart configurations
        invalid_config = {'x': ['A', 'B']}  # Missing 'y'
        result = validate_chart_config(invalid_config, 'bar')
        self.assertFalse(result['valid'])
        print(f"Invalid chart config: {result['error']}")
        
        empty_config = {'x': [], 'y': []}  # Empty data
        result = validate_chart_config(empty_config, 'bar')
        self.assertFalse(result['valid'])
        print(f"Empty chart config: {result['error']}")

if __name__ == '__main__':
    # Set random seed for reproducible tests
    np.random.seed(RANDOM_SEED)
    
    # Run tests with high verbosity
    unittest.main(verbosity=2) 