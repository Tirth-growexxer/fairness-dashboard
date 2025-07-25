"""Model factory for creating and managing classifiers."""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.utils.logger import logger  

@dataclass
class ModelResults:
    """Container for model evaluation results."""
    accuracy: float
    classification_report: Dict[str, Any]
    y_pred: np.ndarray
    model: BaseEstimator

class ModelFactory:
    """Factory class for creating and managing different classifiers."""
    
    @staticmethod
    def get_available_models() -> Dict[str, BaseEstimator]:
        """Return dictionary of available types of models"""
        logger.debug("Creating model instances...")
        
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                solver='liblinear',
                max_iter=1000
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=42,
                max_depth=10
            ),
            'Random Forest': RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=10
            ),
            'XGBoost': XGBClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=6
            )
        }
        
        logger.info(f"Created {len(models)} model instances: {list(models.keys())}")
        return models

    @staticmethod
    def train_model(
        model: BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weights: Optional[np.ndarray] = None,
        model_name: str = "Unknown"
    ) -> BaseEstimator:
        """
        Train a model with optional sample weights.
        """
        try:
            logger.info(f"Training {model_name}...")
            logger.debug(f"Training data shape: {X_train.shape}")
            logger.debug(f"Training target distribution: {y_train.value_counts().to_dict()}")
            
            if sample_weights is not None:
                logger.debug(f"Using sample weights - min: {sample_weights.min():.4f}, max: {sample_weights.max():.4f}, mean: {sample_weights.mean():.4f}")
            
            # Convert data to appropriate format
            X_train_np = X_train.to_numpy()
            y_train_np = y_train.to_numpy()
            
            # Train with weights if provided and supported
            try:
                if sample_weights is not None:
                    model.fit(X_train_np, y_train_np, sample_weight=sample_weights)
                    logger.info(f"{model_name} trained with sample weights")
                else:
                    model.fit(X_train_np, y_train_np)
                    logger.info(f"{model_name} trained without sample weights")
            except TypeError as e:
                logger.warning(f"{model_name} does not support sample weights. Training without them. Error: {e}")
                model.fit(X_train_np, y_train_np)
            
            # Log model parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
                logger.debug(f"{model_name} parameters: {params}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            raise

    @staticmethod
    def evaluate_model(
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "Unknown"
    ) -> ModelResults:
        """
        Evaluate a trained model.
        """
        try:
            logger.info(f"Evaluating {model_name}...")
            logger.debug(f"Test data shape: {X_test.shape}")
            logger.debug(f"Test target distribution: {y_test.value_counts().to_dict()}")
            
            # Convert data to appropriate format
            X_test_np = X_test.to_numpy()
            y_test_np = y_test.to_numpy()
            
            # Generate predictions
            logger.debug(f"Generating predictions for {model_name}...")
            y_pred = model.predict(X_test_np)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_np, y_pred)
            report = classification_report(y_test_np, y_pred, output_dict=True)
            
            # Check prediction distribution
            unique, counts = np.unique(y_pred, return_counts=True)
            pred_dist = dict(zip(unique, counts))
            
            # Log evaluation results
            logger.info(f"{model_name} Evaluation Results:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {report['macro avg']['precision']:.4f}")
            logger.info(f"  Recall: {report['macro avg']['recall']:.4f}")
            logger.info(f"  Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
            logger.info(f"  Prediction distribution: {pred_dist}")
            
            # Check for potential issues
            if len(pred_dist) == 1:
                logger.warning(f"{model_name} is predicting only one class: {list(pred_dist.keys())}")
            
            if accuracy < 0.5:
                logger.warning(f"{model_name} has very low accuracy: {accuracy:.4f}")
            
            return ModelResults(
                accuracy=accuracy,
                classification_report=report,
                y_pred=y_pred,
                model=model
            )
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            raise

    @staticmethod
    def train_and_evaluate_all(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, ModelResults]:
        """
        Train and evaluate all available models.
        """
        try:
            logger.info("Starting batch training and evaluation of all models...")
            logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
            
            if sample_weights is not None:
                logger.info(f"Using sample weights for bias mitigation")
            else:
                logger.info("Training without sample weights (standard approach)")
            
            results = {}
            models = ModelFactory.get_available_models()
            
            for name, model in models.items():
                logger.info(f"\nProcessing {name}...")
                
                try:
                    # Train model
                    trained_model = ModelFactory.train_model(
                        model,
                        X_train,
                        y_train,
                        sample_weights,
                        name
                    )
                    
                    # Evaluate model
                    results[name] = ModelFactory.evaluate_model(
                        trained_model,
                        X_test,
                        y_test,
                        name
                    )
                    
                    logger.info(f"{name} processing completed successfully")
                    
                except Exception as model_error:
                    logger.error(f"Error processing {name}: {str(model_error)}")
                    continue
            
            logger.info(f"\nBatch processing completed. Successfully trained {len(results)} models.")
            
            # Log summary comparison
            if results:
                logger.info("Model Performance Summary:")
                for name, result in results.items():
                    logger.info(f"  {name}: Accuracy={result.accuracy:.4f}, F1={result.classification_report['macro avg']['f1-score']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in train_and_evaluate_all: {str(e)}")
            raise 