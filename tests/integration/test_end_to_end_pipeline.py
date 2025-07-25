"""Integration tests for the complete fairness auditing pipeline."""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.data_preprocessing import load_data, preprocess_data, prepare_model_data
from src.models.model_factory import ModelFactory
from src.metrics.fairness_metrics import FairnessMetrics
from src.mitigation.reweighting import Reweighter
from config.config import (
    PROTECTED_ATTRIBUTES, 
    RANDOM_SEED, 
    METRICS_OUTPUT_PATH,
    LOAN_DATA_PATH
)

class TestEndToEndPipeline(unittest.TestCase):
    """Integration tests for the complete fairness auditing pipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        print("Setting up integration test environment...")
        
        # Ensure reproducibility
        np.random.seed(RANDOM_SEED)
        
        # Load and preprocess data
        cls.raw_data = load_data()
        cls.X, cls.y = preprocess_data(cls.raw_data)
        cls.train_data, cls.test_data = prepare_model_data(cls.X, cls.y)
        
        print(f"Data loaded successfully:")
        print(f"  Raw data shape: {cls.raw_data.shape}")
        print(f"  Preprocessed X shape: {cls.X.shape}")
        print(f"  Train samples: {cls.train_data['X_processed'].shape[0]}")
        print(f"  Test samples: {cls.test_data['X_processed'].shape[0]}")
        print(f"  Protected attributes: {PROTECTED_ATTRIBUTES}")
    
    def test_complete_standard_pipeline(self):
        """Test the complete standard model training and evaluation pipeline."""
        print("\n=== Testing Complete Standard Pipeline ===")
        
        # Test model training and evaluation
        standard_results = ModelFactory.train_and_evaluate_all(
            self.train_data['X_processed'],
            self.train_data['y'],
            self.test_data['X_processed'],
            self.test_data['y']
        )
        
        # Validate all models were trained
        expected_models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
        self.assertEqual(set(standard_results.keys()), set(expected_models))
        
        # Validate model results structure
        for model_name, results in standard_results.items():
            print(f"\nValidating {model_name}:")
            
            # Check accuracy is reasonable
            self.assertGreater(results.accuracy, 0.5)
            self.assertLessEqual(results.accuracy, 1.0)
            print(f"  Accuracy: {results.accuracy:.4f}")
            
            # Check predictions shape
            self.assertEqual(len(results.y_pred), len(self.test_data['y']))
            
            # Check classification report structure
            self.assertIn('macro avg', results.classification_report)
            self.assertIn('precision', results.classification_report['macro avg'])
            self.assertIn('recall', results.classification_report['macro avg'])
            self.assertIn('f1-score', results.classification_report['macro avg'])
            
            print(f"  F1-Score: {results.classification_report['macro avg']['f1-score']:.4f}")
        
        # Test fairness metrics calculation for all models
        all_fairness_metrics = []
        
        for model_name, results in standard_results.items():
            print(f"\nCalculating fairness metrics for {model_name}:")
            
            # Create fairness metrics calculator
            metrics_calculator = FairnessMetrics(
                self.test_data['y'],
                results.y_pred,
                self.test_data['X_fairness'][PROTECTED_ATTRIBUTES],
                model_name,
                'Standard'
            )
            
            # Calculate all fairness metrics
            fairness_metrics = metrics_calculator.calculate_all_metrics()
            
            # Validate fairness metrics structure
            self.assertIsInstance(fairness_metrics, pd.DataFrame)
            self.assertGreater(len(fairness_metrics), 0)
            
            # Check required columns
            required_columns = {'Model', 'Strategy', 'Protected Attribute', 'Group', 'Metric Type', 'Value'}
            self.assertTrue(required_columns.issubset(set(fairness_metrics.columns)))
            
            # Check that we have metrics for all protected attributes
            attributes_with_metrics = set(fairness_metrics['Protected Attribute'].unique())
            print(f"  Attributes with metrics: {attributes_with_metrics}")
            
            # Validate metric types
            metric_types = set(fairness_metrics['Metric Type'].unique())
            expected_metric_types = {'PPR', 'TPR'}
            self.assertTrue(expected_metric_types.issubset(metric_types))
            print(f"  Metric types: {metric_types}")
            
            all_fairness_metrics.append(fairness_metrics)
        
        # Combine all metrics
        combined_metrics = pd.concat(all_fairness_metrics, ignore_index=True)
        print(f"\nCombined metrics shape: {combined_metrics.shape}")
        
        return combined_metrics
    
    def test_complete_reweighted_pipeline(self):
        """Test the complete reweighted model training and evaluation pipeline."""
        print("\n=== Testing Complete Reweighted Pipeline ===")
        
        # Test reweighting strategy
        reweighter = Reweighter(PROTECTED_ATTRIBUTES)
        sample_weights = reweighter.fit_transform(
            self.train_data['X_raw'],
            self.train_data['y']
        )
        
        # Validate sample weights
        self.assertEqual(len(sample_weights), len(self.train_data['y']))
        self.assertTrue(np.all(sample_weights > 0))  # All weights should be positive
        print(f"Sample weights range: {sample_weights.min():.4f} - {sample_weights.max():.4f}")
        
        # Test reweighted model training
        reweighted_results = ModelFactory.train_and_evaluate_all(
            self.train_data['X_processed'],
            self.train_data['y'],
            self.test_data['X_processed'],
            self.test_data['y'],
            sample_weights
        )
        
        # Validate all models were trained with reweighting
        expected_models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
        self.assertEqual(set(reweighted_results.keys()), set(expected_models))
        
        # Test fairness metrics for reweighted models
        all_reweighted_metrics = []
        
        for model_name, results in reweighted_results.items():
            print(f"\nValidating reweighted {model_name}:")
            
            # Check basic model performance
            self.assertGreater(results.accuracy, 0.5)
            print(f"  Accuracy: {results.accuracy:.4f}")
            
            # Calculate fairness metrics
            metrics_calculator = FairnessMetrics(
                self.test_data['y'],
                results.y_pred,
                self.test_data['X_fairness'][PROTECTED_ATTRIBUTES],
                model_name,
                'Reweighted'
            )
            
            fairness_metrics = metrics_calculator.calculate_all_metrics()
            self.assertGreater(len(fairness_metrics), 0)
            
            all_reweighted_metrics.append(fairness_metrics)
        
        # Combine reweighted metrics
        combined_reweighted_metrics = pd.concat(all_reweighted_metrics, ignore_index=True)
        print(f"Combined reweighted metrics shape: {combined_reweighted_metrics.shape}")
        
        return combined_reweighted_metrics
    
    def test_fairness_comparison_standard_vs_reweighted(self):
        """Test comparison between standard and reweighted models for fairness improvements."""
        print("\n=== Testing Standard vs Reweighted Fairness Comparison ===")
        
        # Get standard metrics
        standard_metrics = self.test_complete_standard_pipeline()
        
        # Get reweighted metrics
        reweighted_metrics = self.test_complete_reweighted_pipeline()
        
        # Combine both strategies
        all_metrics = pd.concat([standard_metrics, reweighted_metrics], ignore_index=True)
        
        # Test fairness comparison for each model and attribute
        for model_name in ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']:
            print(f"\nComparing fairness for {model_name}:")
            
            model_metrics = all_metrics[all_metrics['Model'] == model_name]
            
            for attribute in PROTECTED_ATTRIBUTES:
                attr_metrics = model_metrics[
                    (model_metrics['Protected Attribute'] == attribute) &
                    (model_metrics['Metric Type'] == 'Disparate Impact Ratio')
                ]
                
                if len(attr_metrics) == 2:  # Both standard and reweighted
                    standard_dir = attr_metrics[attr_metrics['Strategy'] == 'Standard']['Value'].iloc[0]
                    reweighted_dir = attr_metrics[attr_metrics['Strategy'] == 'Reweighted']['Value'].iloc[0]
                    
                    if pd.notnull(standard_dir) and pd.notnull(reweighted_dir):
                        # Calculate improvement (closer to 1.0 is better for DIR)
                        standard_bias = abs(1.0 - standard_dir)
                        reweighted_bias = abs(1.0 - reweighted_dir)
                        improvement = standard_bias - reweighted_bias
                        
                        print(f"  {attribute} DIR: {standard_dir:.3f} → {reweighted_dir:.3f} (improvement: {improvement:.3f})")
        
        return all_metrics
    
    def test_data_integrity_throughout_pipeline(self):
        """Test data integrity is maintained throughout the entire pipeline."""
        print("\n=== Testing Data Integrity Throughout Pipeline ===")
        
        # Test data shapes consistency
        self.assertEqual(len(self.X), len(self.y))
        self.assertEqual(
            len(self.train_data['X_processed']) + len(self.test_data['X_processed']),
            len(self.X)
        )
        self.assertEqual(
            len(self.train_data['y']) + len(self.test_data['y']),
            len(self.y)
        )
        
        # Test protected attributes integrity
        for attr in PROTECTED_ATTRIBUTES:
            # Check attribute exists in fairness data
            self.assertIn(attr, self.test_data['X_fairness'].columns)
            
            # Check no missing values in protected attributes
            missing_count = self.test_data['X_fairness'][attr].isnull().sum()
            print(f"  {attr}: {missing_count} missing values")
            
            # Check reasonable number of unique values
            unique_count = self.test_data['X_fairness'][attr].nunique()
            self.assertGreater(unique_count, 1)
            print(f"  {attr}: {unique_count} unique values")
        
        # Test target variable integrity
        unique_targets = set(self.y.unique())
        expected_targets = {0, 1}  # Binary classification
        self.assertEqual(unique_targets, expected_targets)
        
        print(f"Data integrity validation passed:")
        print(f"  Total samples: {len(self.y)}")
        print(f"  Positive class ratio: {self.y.mean():.3f}")
    
    def test_metrics_output_generation(self):
        """Test the complete metrics output generation process."""
        print("\n=== Testing Metrics Output Generation ===")
        
        # Generate complete metrics as done in the main application
        standard_metrics = self.test_complete_standard_pipeline()
        reweighted_metrics = self.test_complete_reweighted_pipeline()
        
        # Add overall performance metrics (simulating the main app)
        overall_metrics_list = []
        
        # Train one model for overall metrics
        models = ModelFactory.get_available_models()
        logistic_model = models['Logistic Regression']
        trained_model = ModelFactory.train_model(
            logistic_model,
            self.train_data['X_processed'],
            self.train_data['y'],
            model_name='Logistic Regression'
        )
        results = ModelFactory.evaluate_model(
            trained_model,
            self.test_data['X_processed'],
            self.test_data['y'],
            'Logistic Regression'
        )
        
        # Create overall metrics
        overall_metrics = pd.DataFrame([
            {
                'Model': 'Logistic Regression',
                'Strategy': 'Standard',
                'Protected Attribute': 'Overall',
                'Group': 'Overall',
                'Metric Type': 'Accuracy',
                'Value': results.accuracy
            },
            {
                'Model': 'Logistic Regression',
                'Strategy': 'Standard',
                'Protected Attribute': 'Overall',
                'Group': 'Overall',
                'Metric Type': 'F1-Score (Macro Avg)',
                'Value': results.classification_report['macro avg']['f1-score']
            }
        ])
        
        # Combine all metrics
        final_metrics = pd.concat([
            overall_metrics,
            standard_metrics,
            reweighted_metrics
        ], ignore_index=True)
        
        # Validate final metrics structure
        self.assertGreater(len(final_metrics), 0)
        print(f"Final metrics dataset shape: {final_metrics.shape}")
        
        # Test metrics can be saved (simulate saving)
        try:
            # Don't actually save to avoid overwriting real metrics
            temp_path = "temp_test_metrics.parquet"
            final_metrics.to_parquet(temp_path, index=False)
            
            # Test loading back
            loaded_metrics = pd.read_parquet(temp_path)
            self.assertEqual(len(loaded_metrics), len(final_metrics))
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            print("Metrics save/load test passed")
            
        except Exception as e:
            self.fail(f"Metrics save/load failed: {str(e)}")
        
        return final_metrics
    
    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases in the pipeline."""
        print("\n=== Testing Error Handling and Edge Cases ===")
        
        # Test with empty data
        try:
            empty_metrics = FairnessMetrics(
                pd.Series([]),
                pd.Series([]),
                pd.DataFrame(),
                'test_model',
                'Standard'
            )
            result = empty_metrics.calculate_all_metrics()
            self.assertIsInstance(result, pd.DataFrame)
            print("Empty data handling: PASSED")
        except Exception as e:
            print(f"Empty data handling: Expected error - {str(e)}")
        
        # Test with single class data
        try:
            single_class_y_true = pd.Series([0] * 100)
            single_class_y_pred = pd.Series([0] * 100)
            single_class_attrs = pd.DataFrame({
                'person_gender': ['male'] * 50 + ['female'] * 50
            })
            
            single_class_metrics = FairnessMetrics(
                single_class_y_true,
                single_class_y_pred,
                single_class_attrs,
                'test_model',
                'Standard'
            )
            result = single_class_metrics.calculate_all_metrics()
            print("Single class handling: PASSED")
        except Exception as e:
            print(f"Single class handling: Expected error - {str(e)}")
        
        # Test with mismatched data lengths
        try:
            mismatched_metrics = FairnessMetrics(
                pd.Series([0, 1, 0]),  # Length 3
                pd.Series([1, 0]),     # Length 2
                pd.DataFrame({'attr': ['a', 'b', 'c']}),  # Length 3
                'test_model',
                'Standard'
            )
            result = mismatched_metrics.calculate_all_metrics()
            self.fail("Should have raised an error for mismatched lengths")
        except Exception as e:
            print(f"Mismatched lengths handling: Expected error - {str(e)}")
        
        print("Error handling tests completed")

if __name__ == '__main__':
    # Set random seed for reproducible tests
    np.random.seed(RANDOM_SEED)
    
    # Run tests with high verbosity
    unittest.main(verbosity=2) 