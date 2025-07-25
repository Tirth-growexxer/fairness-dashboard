"""Integration tests for the data processing pipeline."""

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
    LOAN_DATA_PATH,
    AGE_BINS,
    AGE_LABELS
)

class TestDataPipeline(unittest.TestCase):
    """Integration tests for the complete data processing pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        np.random.seed(RANDOM_SEED)
        print(f"\nSetting up test environment with seed: {RANDOM_SEED}")
    
    def test_data_loading_and_validation(self):
        """Test data loading from CSV and initial validation."""
        print("\n=== Testing Data Loading and Validation ===")
        
        # Test data loading
        raw_data = load_data()
        
        # Validate data structure
        self.assertIsInstance(raw_data, pd.DataFrame)
        self.assertGreater(len(raw_data), 0)
        print(f"Raw data loaded: {raw_data.shape}")
        
        # Check expected columns
        expected_columns = [
            'person_age', 'person_gender', 'person_education', 'person_income',
            'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
            'credit_score', 'previous_loan_defaults_on_file', 'loan_status'
        ]
        
        for col in expected_columns:
            self.assertIn(col, raw_data.columns, f"Missing column: {col}")
        
        print(f"All expected columns present: {len(expected_columns)}")
        
        # Validate data types and ranges
        self.assertTrue(raw_data['person_age'].min() >= 18)
        self.assertTrue(raw_data['person_age'].max() <= 100)
        self.assertTrue(raw_data['loan_status'].isin([0, 1]).all())
        self.assertTrue(raw_data['person_income'].min() > 0)
        
        print("Data validation passed")
        
        # Check for missing values
        missing_values = raw_data.isnull().sum()
        total_missing = missing_values.sum()
        print(f"Total missing values: {total_missing}")
        
        if total_missing > 0:
            print("Missing values by column:")
            for col, count in missing_values[missing_values > 0].items():
                print(f"  {col}: {count}")
        
        return raw_data
    
    def test_data_preprocessing_pipeline(self):
        """Test the complete data preprocessing pipeline."""
        print("\n=== Testing Data Preprocessing Pipeline ===")
        
        # Load raw data
        raw_data = self.test_data_loading_and_validation()
        
        # Test preprocessing
        X, y = preprocess_data(raw_data)
        
        # Validate preprocessing results
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        
        print(f"Preprocessed data: X shape {X.shape}, y shape {y.shape}")
        
        # Check that age_group was created correctly
        self.assertIn('age_group', X.columns)
        self.assertNotIn('person_age', X.columns)  # Should be dropped
        
        # Validate age groups
        age_groups = X['age_group'].value_counts()
        print(f"Age group distribution: {age_groups.to_dict()}")
        
        expected_age_labels = set(AGE_LABELS)
        actual_age_labels = set(age_groups.index)
        self.assertTrue(actual_age_labels.issubset(expected_age_labels))
        
        # Check protected attributes are present
        for attr in PROTECTED_ATTRIBUTES:
            self.assertIn(attr, X.columns, f"Protected attribute {attr} missing")
        
        print(f"Protected attributes validated: {PROTECTED_ATTRIBUTES}")
        
        # Validate target variable
        self.assertTrue(y.isin([0, 1]).all())
        target_distribution = y.value_counts()
        print(f"Target distribution: {target_distribution.to_dict()}")
        
        return X, y
    
    def test_model_data_preparation(self):
        """Test model data preparation including train/test split and encoding."""
        print("\n=== Testing Model Data Preparation ===")
        
        # Get preprocessed data
        X, y = self.test_data_preprocessing_pipeline()
        
        # Test model data preparation
        train_data, test_data = prepare_model_data(X, y)
        
        # Validate data structure
        self.assertIsInstance(train_data, dict)
        self.assertIsInstance(test_data, dict)
        
        required_train_keys = ['X_raw', 'X_processed', 'y', 'scaler', 'encoder']
        required_test_keys = ['X_raw', 'X_processed', 'y', 'X_fairness']
        
        for key in required_train_keys:
            self.assertIn(key, train_data, f"Missing train key: {key}")
        
        for key in required_test_keys:
            self.assertIn(key, test_data, f"Missing test key: {key}")
        
        print("Data structure validation passed")
        
        # Validate data shapes
        total_samples = len(X)
        train_samples = len(train_data['X_processed'])
        test_samples = len(test_data['X_processed'])
        
        self.assertEqual(train_samples + test_samples, total_samples)
        self.assertEqual(len(train_data['y']), train_samples)
        self.assertEqual(len(test_data['y']), test_samples)
        
        print(f"Data split: {train_samples} train, {test_samples} test")
        
        # Validate feature encoding
        original_features = len(X.columns)
        processed_features = train_data['X_processed'].shape[1]
        
        print(f"Feature expansion: {original_features} → {processed_features}")
        self.assertGreaterEqual(processed_features, original_features)
        
        # Validate fairness data preservation
        for attr in PROTECTED_ATTRIBUTES:
            self.assertIn(attr, test_data['X_fairness'].columns)
        
        print("Fairness data preservation validated")
        
        # Check data types
        self.assertTrue(np.issubdtype(train_data['X_processed'].dtypes.iloc[0], np.number))
        self.assertTrue(np.issubdtype(test_data['X_processed'].dtypes.iloc[0], np.number))
        
        print("Data type validation passed")
        
        return train_data, test_data
    
    def test_model_training_integration(self):
        """Test integration of model training with prepared data."""
        print("\n=== Testing Model Training Integration ===")
        
        # Get prepared data
        train_data, test_data = self.test_model_data_preparation()
        
        # Test model training
        results = ModelFactory.train_and_evaluate_all(
            train_data['X_processed'],
            train_data['y'],
            test_data['X_processed'],
            test_data['y']
        )
        
        # Validate results structure
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        expected_models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
        self.assertEqual(set(results.keys()), set(expected_models))
        
        print(f"Models trained: {list(results.keys())}")
        
        # Validate each model result
        for model_name, result in results.items():
            print(f"\nValidating {model_name}:")
            
            # Check accuracy
            self.assertGreater(result.accuracy, 0.0)
            self.assertLessEqual(result.accuracy, 1.0)
            print(f"  Accuracy: {result.accuracy:.4f}")
            
            # Check predictions
            self.assertEqual(len(result.y_pred), len(test_data['y']))
            self.assertTrue(np.all(np.isin(result.y_pred, [0, 1])))
            
            # Check classification report
            self.assertIn('macro avg', result.classification_report)
            macro_f1 = result.classification_report['macro avg']['f1-score']
            print(f"  Macro F1: {macro_f1:.4f}")
            
            # Validate prediction distribution
            pred_dist = pd.Series(result.y_pred).value_counts()
            print(f"  Prediction distribution: {pred_dist.to_dict()}")
        
        return results, test_data
    
    def test_fairness_metrics_integration(self):
        """Test integration of fairness metrics calculation with model results."""
        print("\n=== Testing Fairness Metrics Integration ===")
        
        # Get model results
        model_results, test_data = self.test_model_training_integration()
        
        # Test fairness metrics for each model
        all_fairness_metrics = []
        
        for model_name, results in model_results.items():
            print(f"\nTesting fairness metrics for {model_name}:")
            
            # Create fairness metrics calculator
            fairness_calculator = FairnessMetrics(
                test_data['y'],
                results.y_pred,
                test_data['X_fairness'][PROTECTED_ATTRIBUTES],
                model_name,
                'Standard'
            )
            
            # Calculate all metrics
            metrics_df = fairness_calculator.calculate_all_metrics()
            
            # Validate metrics structure
            self.assertIsInstance(metrics_df, pd.DataFrame)
            self.assertGreater(len(metrics_df), 0)
            
            # Check required columns
            required_columns = {'Model', 'Strategy', 'Protected Attribute', 'Group', 'Metric Type', 'Value'}
            self.assertTrue(required_columns.issubset(set(metrics_df.columns)))
            
            # Check metrics for each protected attribute
            attributes_with_metrics = set(metrics_df['Protected Attribute'].unique())
            print(f"  Attributes with metrics: {attributes_with_metrics}")
            
            for attr in PROTECTED_ATTRIBUTES:
                attr_metrics = metrics_df[metrics_df['Protected Attribute'] == attr]
                if len(attr_metrics) > 0:
                    metric_types = set(attr_metrics['Metric Type'].unique())
                    print(f"    {attr}: {metric_types}")
                    
                    # Validate metric values
                    for _, row in attr_metrics.iterrows():
                        if pd.notnull(row['Value']):
                            self.assertIsInstance(row['Value'], (int, float))
            
            all_fairness_metrics.append(metrics_df)
        
        # Combine all metrics
        combined_metrics = pd.concat(all_fairness_metrics, ignore_index=True)
        print(f"\nCombined fairness metrics: {combined_metrics.shape}")
        
        return combined_metrics
    
    def test_reweighting_integration(self):
        """Test integration of reweighting strategy with the pipeline."""
        print("\n=== Testing Reweighting Integration ===")
        
        # Get prepared data
        train_data, test_data = self.test_model_data_preparation()
        
        # Test reweighting
        reweighter = Reweighter(PROTECTED_ATTRIBUTES)
        sample_weights = reweighter.fit_transform(
            train_data['X_raw'],
            train_data['y']
        )
        
        # Validate sample weights
        self.assertEqual(len(sample_weights), len(train_data['y']))
        self.assertTrue(np.all(sample_weights > 0))
        self.assertTrue(np.all(np.isfinite(sample_weights)))
        
        print(f"Sample weights generated: {len(sample_weights)}")
        print(f"Weight range: {sample_weights.min():.4f} - {sample_weights.max():.4f}")
        print(f"Weight mean: {sample_weights.mean():.4f}")
        
        # Test reweighted model training
        reweighted_results = ModelFactory.train_and_evaluate_all(
            train_data['X_processed'],
            train_data['y'],
            test_data['X_processed'],
            test_data['y'],
            sample_weights
        )
        
        # Validate reweighted results
        self.assertIsInstance(reweighted_results, dict)
        self.assertGreater(len(reweighted_results), 0)
        
        expected_models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
        self.assertEqual(set(reweighted_results.keys()), set(expected_models))
        
        print(f"Reweighted models trained: {list(reweighted_results.keys())}")
        
        # Compare with standard results (get standard results first)
        standard_results = ModelFactory.train_and_evaluate_all(
            train_data['X_processed'],
            train_data['y'],
            test_data['X_processed'],
            test_data['y']
        )
        
        # Compare accuracies
        print("\nStandard vs Reweighted Accuracy Comparison:")
        for model_name in expected_models:
            standard_acc = standard_results[model_name].accuracy
            reweighted_acc = reweighted_results[model_name].accuracy
            diff = reweighted_acc - standard_acc
            
            print(f"  {model_name}: {standard_acc:.4f} → {reweighted_acc:.4f} (Δ{diff:+.4f})")
        
        return reweighted_results, test_data
    
    def test_complete_pipeline_integration(self):
        """Test the complete pipeline from data loading to final metrics."""
        print("\n=== Testing Complete Pipeline Integration ===")
        
        # Execute complete pipeline
        print("Step 1: Loading and preprocessing data...")
        raw_data = load_data()
        X, y = preprocess_data(raw_data)
        train_data, test_data = prepare_model_data(X, y)
        
        print("Step 2: Training standard models...")
        standard_results = ModelFactory.train_and_evaluate_all(
            train_data['X_processed'],
            train_data['y'],
            test_data['X_processed'],
            test_data['y']
        )
        
        print("Step 3: Calculating standard fairness metrics...")
        standard_fairness_metrics = []
        for model_name, results in standard_results.items():
            fairness_calculator = FairnessMetrics(
                test_data['y'],
                results.y_pred,
                test_data['X_fairness'][PROTECTED_ATTRIBUTES],
                model_name,
                'Standard'
            )
            metrics_df = fairness_calculator.calculate_all_metrics()
            standard_fairness_metrics.append(metrics_df)
        
        print("Step 4: Applying reweighting...")
        reweighter = Reweighter(PROTECTED_ATTRIBUTES)
        sample_weights = reweighter.fit_transform(train_data['X_raw'], train_data['y'])
        
        print("Step 5: Training reweighted models...")
        reweighted_results = ModelFactory.train_and_evaluate_all(
            train_data['X_processed'],
            train_data['y'],
            test_data['X_processed'],
            test_data['y'],
            sample_weights
        )
        
        print("Step 6: Calculating reweighted fairness metrics...")
        reweighted_fairness_metrics = []
        for model_name, results in reweighted_results.items():
            fairness_calculator = FairnessMetrics(
                test_data['y'],
                results.y_pred,
                test_data['X_fairness'][PROTECTED_ATTRIBUTES],
                model_name,
                'Reweighted'
            )
            metrics_df = fairness_calculator.calculate_all_metrics()
            reweighted_fairness_metrics.append(metrics_df)
        
        print("Step 7: Combining all metrics...")
        all_metrics = pd.concat(
            standard_fairness_metrics + reweighted_fairness_metrics,
            ignore_index=True
        )
        
        # Validate final results
        self.assertIsInstance(all_metrics, pd.DataFrame)
        self.assertGreater(len(all_metrics), 0)
        
        # Check we have both strategies
        strategies = set(all_metrics['Strategy'].unique())
        self.assertIn('Standard', strategies)
        self.assertIn('Reweighted', strategies)
        
        # Check we have all models
        models = set(all_metrics['Model'].unique())
        expected_models = {'Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'}
        self.assertEqual(models, expected_models)
        
        # Check we have protected attributes
        attributes = set(all_metrics['Protected Attribute'].unique())
        expected_attributes = set(PROTECTED_ATTRIBUTES)
        self.assertTrue(expected_attributes.issubset(attributes))
        
        print(f"\nPipeline completed successfully:")
        print(f"  Final metrics dataset: {all_metrics.shape}")
        print(f"  Models: {len(models)}")
        print(f"  Strategies: {len(strategies)}")
        print(f"  Protected attributes: {len(attributes & expected_attributes)}")
        
        # Summary statistics
        metric_types = all_metrics['Metric Type'].value_counts()
        print(f"  Metric types: {dict(metric_types)}")
        
        return all_metrics
    
    def test_pipeline_reproducibility(self):
        """Test that the pipeline produces reproducible results."""
        print("\n=== Testing Pipeline Reproducibility ===")
        
        # Run pipeline twice with same seed
        results1 = self.test_complete_pipeline_integration()
        
        # Reset seed and run again
        np.random.seed(RANDOM_SEED)
        results2 = self.test_complete_pipeline_integration()
        
        # Compare results
        self.assertEqual(len(results1), len(results2))
        print(f"Both runs produced {len(results1)} metrics")
        
        # Check that the same metrics were calculated
        metrics1_summary = results1.groupby(['Model', 'Strategy', 'Protected Attribute', 'Metric Type']).size()
        metrics2_summary = results2.groupby(['Model', 'Strategy', 'Protected Attribute', 'Metric Type']).size()
        
        self.assertTrue(metrics1_summary.equals(metrics2_summary))
        print("Pipeline reproducibility validated")

if __name__ == '__main__':
    # Set random seed for reproducible tests
    np.random.seed(RANDOM_SEED)
    
    # Run tests with high verbosity
    unittest.main(verbosity=2) 