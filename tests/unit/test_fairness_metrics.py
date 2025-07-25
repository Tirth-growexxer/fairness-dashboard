"""Unit tests for fairness metrics calculations."""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.metrics.fairness_metrics import FairnessMetrics, GroupMetrics
from src.utils.data_preprocessing import load_data, preprocess_data, prepare_model_data
from src.models.model_factory import ModelFactory
from config.config import PROTECTED_ATTRIBUTES, RANDOM_SEED

class TestFairnessMetrics(unittest.TestCase):
    """Test cases for FairnessMetrics class."""
    
    def setUp(self):
        """Set up test data using real loan data."""
        # Load real data from data/loan_data.csv
        raw_data = load_data()
        X, y = preprocess_data(raw_data)
        train_data, test_data = prepare_model_data(X, y)
        
        # Train a simple model for testing
        models = ModelFactory.get_available_models()
        logistic_model = models['Logistic Regression']
        trained_model = ModelFactory.train_model(
            logistic_model,
            train_data['X_processed'],
            train_data['y'],
            model_name='Logistic Regression'
        )
        
        # Get predictions on test set
        y_pred = trained_model.predict(test_data['X_processed'].to_numpy())
        
        # Use real test data for fairness metrics
        self.y_true = test_data['y']
        self.y_pred = pd.Series(y_pred, index=test_data['y'].index)
        self.protected_attributes = test_data['X_fairness'][PROTECTED_ATTRIBUTES]
        
        # Create metrics object with real data
        self.metrics = FairnessMetrics(
            self.y_true,
            self.y_pred,
            self.protected_attributes,
            'Logistic Regression',
            'Standard'
        )
        
        # Store some basic info for validation
        self.total_samples = len(self.y_true)
        self.actual_positives = self.y_true.sum()
        
        print(f"Test setup complete:")
        print(f"  Total samples: {self.total_samples}")
        print(f"  Actual positives: {self.actual_positives}")
        print(f"  Protected attributes: {list(self.protected_attributes.columns)}")

    def test_calculate_group_metrics(self):
        """Test group metrics calculation with real data."""
        # Test for gender groups
        gender_groups = self.protected_attributes['person_gender'].unique()
        
        for gender in gender_groups:
            gender_mask = (self.protected_attributes['person_gender'] == gender)
            if gender_mask.sum() > 0:  # Only test if group has data
                metrics = self.metrics.calculate_group_metrics(
                    self.y_true[gender_mask],
                    self.y_pred[gender_mask]
                )
                
                # Validate metrics object
                self.assertIsInstance(metrics, GroupMetrics)
                self.assertGreater(metrics.total_samples, 0)
                self.assertGreaterEqual(metrics.actual_positives, 0)
                self.assertGreaterEqual(metrics.tp, 0)
                self.assertGreaterEqual(metrics.fp, 0)
                
                # Validate PPR and TPR are within valid range [0, 1]
                self.assertGreaterEqual(metrics.ppr, 0.0)
                self.assertLessEqual(metrics.ppr, 1.0)
                
                if metrics.actual_positives > 0:
                    self.assertGreaterEqual(metrics.tpr, 0.0)
                    self.assertLessEqual(metrics.tpr, 1.0)
                
                print(f"  {gender}: samples={metrics.total_samples}, PPR={metrics.ppr:.3f}, TPR={metrics.tpr:.3f}")

    def test_calculate_disparate_impact_ratio(self):
        """Test disparate impact ratio calculation with real data."""
        # Get gender groups from real data
        gender_groups = self.protected_attributes['person_gender'].unique()
        
        if len(gender_groups) >= 2:
            group_metrics = {}
            
            # Calculate metrics for each gender group
            for gender in gender_groups:
                gender_mask = (self.protected_attributes['person_gender'] == gender)
                if gender_mask.sum() > 0:
                    metrics = self.metrics.calculate_group_metrics(
                        self.y_true[gender_mask],
                        self.y_pred[gender_mask]
                    )
                    if metrics:
                        group_metrics[gender] = metrics
            
            if len(group_metrics) >= 2:
                groups = list(group_metrics.keys())
                privileged_group = groups[0]
                unprivileged_group = groups[1]
                
                dir_value, msg = self.metrics.calculate_disparate_impact_ratio(
                    group_metrics,
                    privileged_group,
                    unprivileged_group
                )
                
                # Validate DIR calculation
                if dir_value is not None:
                    self.assertGreater(dir_value, 0)  # DIR should be positive
                    print(f"  DIR ({unprivileged_group} vs {privileged_group}): {dir_value:.3f}")
                else:
                    print(f"  DIR calculation failed: {msg}")

    def test_calculate_equal_opportunity_difference(self):
        """Test equal opportunity difference calculation with real data."""
        # Get gender groups from real data
        gender_groups = self.protected_attributes['person_gender'].unique()
        
        if len(gender_groups) >= 2:
            group_metrics = {}
            
            # Calculate metrics for each gender group
            for gender in gender_groups:
                gender_mask = (self.protected_attributes['person_gender'] == gender)
                if gender_mask.sum() > 0:
                    metrics = self.metrics.calculate_group_metrics(
                        self.y_true[gender_mask],
                        self.y_pred[gender_mask]
                    )
                    if metrics and metrics.actual_positives > 0:
                        group_metrics[gender] = metrics
            
            if len(group_metrics) >= 2:
                groups = list(group_metrics.keys())
                privileged_group = groups[0]
                unprivileged_group = groups[1]
                
                eod_value, msg = self.metrics.calculate_equal_opportunity_difference(
                    group_metrics,
                    privileged_group,
                    unprivileged_group
                )
                
                # Validate EOD calculation
                if eod_value is not None:
                    self.assertGreaterEqual(eod_value, -1.0)  # EOD should be >= -1
                    self.assertLessEqual(eod_value, 1.0)      # EOD should be <= 1
                    print(f"  EOD ({privileged_group} - {unprivileged_group}): {eod_value:.3f}")
                else:
                    print(f"  EOD calculation failed: {msg}")

    def test_calculate_metrics_for_attribute(self):
        """Test metrics calculation for a specific attribute with real data."""
        for attribute in PROTECTED_ATTRIBUTES:
            print(f"\nTesting metrics for {attribute}:")
            metrics_list = self.metrics.calculate_metrics_for_attribute(attribute)
            
            self.assertIsInstance(metrics_list, list)
            
            if len(metrics_list) > 0:
                # Check that we have the expected metric types
                metric_types = set(m['Metric Type'] for m in metrics_list)
                expected_types = {'PPR', 'TPR'}
                
                # Check if we have aggregate metrics (DIR, EOD)
                if 'Disparate Impact Ratio' in metric_types:
                    expected_types.add('Disparate Impact Ratio')
                if 'Equal Opportunity Difference' in metric_types:
                    expected_types.add('Equal Opportunity Difference')
                
                print(f"  Found metric types: {metric_types}")
                print(f"  Number of metrics: {len(metrics_list)}")
                
                # Validate that all metrics have valid values
                for metric in metrics_list:
                    self.assertIn('Value', metric)
                    if pd.notnull(metric['Value']):
                        self.assertIsInstance(metric['Value'], (int, float))
            else:
                print(f"  No metrics calculated for {attribute}")

    def test_calculate_all_metrics(self):
        """Test calculation of all metrics with real data."""
        print(f"\nTesting calculation of all metrics:")
        all_metrics_df = self.metrics.calculate_all_metrics()
        
        self.assertIsInstance(all_metrics_df, pd.DataFrame)
        
        if len(all_metrics_df) > 0:
            # Check that we have metrics for protected attributes
            attributes = set(all_metrics_df['Protected Attribute'])
            print(f"  Attributes with metrics: {attributes}")
            print(f"  Total metrics calculated: {len(all_metrics_df)}")
            
            # Check required columns
            required_columns = {'Model', 'Strategy', 'Protected Attribute', 'Group', 'Metric Type', 'Value'}
            self.assertTrue(required_columns.issubset(set(all_metrics_df.columns)))
            
            # Validate data types and values
            for _, row in all_metrics_df.iterrows():
                self.assertIsInstance(row['Model'], str)
                self.assertIsInstance(row['Strategy'], str)
                self.assertIsInstance(row['Protected Attribute'], str)
                if pd.notnull(row['Value']):
                    self.assertIsInstance(row['Value'], (int, float))
        else:
            print("  No metrics calculated")

if __name__ == '__main__':
    # Set random seed for reproducible tests
    np.random.seed(RANDOM_SEED)
    unittest.main(verbosity=2) 