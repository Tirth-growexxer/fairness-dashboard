"""Fairness metrics calculation module."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from dataclasses import dataclass
from src.utils.logger import logger

@dataclass
class GroupMetrics:
    """Container for group-specific metrics."""
    ppr: float  # Proportion of Positive Predictions
    tpr: float  # True Positive Rate
    total_samples: int
    actual_positives: int
    tp: int  # True Positives
    fp: int  # False Positives
    fn: int  # False Negatives
    tn: int  # True Negatives

class FairnessMetrics:
    """Calculate and store fairness metrics for protected attributes."""
    
    def __init__(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        protected_attributes: pd.DataFrame,
        model_name: str,
        strategy: str = 'Standard'
    ):
        """
        Initialize FairnessMetrics calculator.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attributes: DataFrame containing protected attribute columns
            model_name: Name of the model being evaluated
            strategy: Strategy used (e.g., 'Standard' or 'Reweighted')
        """
        self.y_true = pd.Series(y_true)
        self.y_pred = pd.Series(y_pred)
        self.protected_attributes = protected_attributes.copy()
        self.model_name = model_name
        self.strategy = strategy
        self.metrics_data = []
        
        # Validate inputs
        if len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        if len(self.y_true) != len(self.protected_attributes):
            raise ValueError("y_true and protected_attributes must have the same length")

    def calculate_overall_metrics(self) -> List[Dict]:
        """Calculate overall model performance metrics."""
        try:
            # Calculate overall metrics
            accuracy = accuracy_score(self.y_true, self.y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_true,
                self.y_pred,
                average='macro'
            )
            
            # Create metrics list
            overall_metrics = [
                {
                    'Model': self.model_name,
                    'Strategy': self.strategy,
                    'Protected Attribute': 'Overall',
                    'Group': 'Overall',
                    'Metric Type': metric_type,
                    'Value': value
                }
                for metric_type, value in [
                    ('Accuracy', accuracy),
                    ('Precision (Macro Avg)', precision),
                    ('Recall (Macro Avg)', recall),
                    ('F1-Score (Macro Avg)', f1)
                ]
            ]
            
            logger.debug(f"Overall metrics for {self.model_name} ({self.strategy}): "
                        f"Accuracy={accuracy:.4f}, F1={f1:.4f}")
            
            return overall_metrics
        except Exception as e:
            logger.error(f"Error calculating overall metrics: {str(e)}")
            return []

    def calculate_group_metrics(
        self,
        y_true_group: pd.Series,
        y_pred_group: pd.Series
    ) -> Optional[GroupMetrics]:
        """Calculate metrics for a specific group."""
        try:
            if len(y_true_group) == 0:
                return None
            
            # Convert to numpy arrays and handle NaN values
            y_true_np = pd.to_numeric(y_true_group, errors='coerce').fillna(0).astype(int).values
            y_pred_np = pd.to_numeric(y_pred_group, errors='coerce').fillna(0).astype(int).values
            
            # Ensure arrays are the same length
            if len(y_true_np) != len(y_pred_np):
                logger.warning("Prediction and truth arrays have different lengths")
                return None
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(
                y_true_np,
                y_pred_np,
                labels=[0, 1]
            ).ravel()
            
            total_predictions = len(y_pred_np)
            ppr = (tp + fp) / total_predictions if total_predictions > 0 else 0
            
            actual_positives = tp + fn
            tpr = tp / actual_positives if actual_positives > 0 else 0
            
            metrics = GroupMetrics(
                ppr=ppr,
                tpr=tpr,
                total_samples=total_predictions,
                actual_positives=actual_positives,
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn
            )
            
            logger.debug(f"Group metrics - Size: {total_predictions}, "
                        f"PPR: {ppr:.4f}, TPR: {tpr:.4f}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating group metrics: {str(e)}")
            return None

    def calculate_disparate_impact_ratio(
        self,
        group_metrics: Dict[str, GroupMetrics],
        privileged_group: str,
        unprivileged_group: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """Calculate Disparate Impact Ratio between two groups."""
        try:
            if (privileged_group not in group_metrics or
                unprivileged_group not in group_metrics):
                missing_groups = []
                if privileged_group not in group_metrics:
                    missing_groups.append(f"privileged({privileged_group})")
                if unprivileged_group not in group_metrics:
                    missing_groups.append(f"unprivileged({unprivileged_group})")
                msg = f"Missing group metrics: {', '.join(missing_groups)}"
                logger.warning(f"DIR calculation failed: {msg}")
                return None, msg
            
            ppr_privileged = group_metrics[privileged_group].ppr
            ppr_unprivileged = group_metrics[unprivileged_group].ppr
            
            logger.debug(f"DIR calculation - Privileged({privileged_group}): PPR={ppr_privileged:.4f}, "
                        f"Unprivileged({unprivileged_group}): PPR={ppr_unprivileged:.4f}")
            
            if ppr_privileged > 0:
                dir_value = ppr_unprivileged / ppr_privileged
                logger.info(f"DIR: {dir_value:.4f} ({unprivileged_group} vs {privileged_group})")
                return dir_value, None
            else:
                msg = f"Privileged group PPR is zero (PPR={ppr_privileged})"
                logger.warning(f"DIR calculation failed: {msg}")
                return None, msg
        except Exception as e:
            logger.error(f"Error calculating DIR: {str(e)}")
            return None, str(e)

    def calculate_equal_opportunity_difference(
        self,
        group_metrics: Dict[str, GroupMetrics],
        privileged_group: str,
        unprivileged_group: str
    ) -> Tuple[Optional[float], Optional[str]]:
        """Calculate Equal Opportunity Difference between two groups."""
        try:
            if (privileged_group not in group_metrics or
                unprivileged_group not in group_metrics):
                missing_groups = []
                if privileged_group not in group_metrics:
                    missing_groups.append(f"privileged({privileged_group})")
                if unprivileged_group not in group_metrics:
                    missing_groups.append(f"unprivileged({unprivileged_group})")
                msg = f"Missing group metrics: {', '.join(missing_groups)}"
                logger.warning(f"EOD calculation failed: {msg}")
                return None, msg
            
            metrics_p = group_metrics[privileged_group]
            metrics_u = group_metrics[unprivileged_group]
            
            logger.debug(f"EOD calculation - Privileged({privileged_group}): TPR={metrics_p.tpr:.4f}, "
                        f"actual_positives={metrics_p.actual_positives}, "
                        f"Unprivileged({unprivileged_group}): TPR={metrics_u.tpr:.4f}, "
                        f"actual_positives={metrics_u.actual_positives}")
            
            if metrics_p.actual_positives > 0 and metrics_u.actual_positives > 0:
                eod_value = metrics_p.tpr - metrics_u.tpr
                logger.info(f"EOD: {eod_value:.4f} ({privileged_group} - {unprivileged_group})")
                return eod_value, None
            else:
                msg = f"Insufficient actual positives - Privileged: {metrics_p.actual_positives}, Unprivileged: {metrics_u.actual_positives}"
                logger.warning(f"EOD calculation failed: {msg}")
                return None, msg
        except Exception as e:
            logger.error(f"Error calculating EOD: {str(e)}")
            return None, str(e)

    def calculate_metrics_for_attribute(
        self,
        attribute: str
    ) -> List[Dict]:
        """Calculate all fairness metrics for a specific protected attribute."""
        try:
            attr_metrics = []
            
            # Get unique groups and their populations
            groups = self.protected_attributes[attribute].value_counts()
            if len(groups) < 2:
                logger.warning(f"Insufficient groups for attribute {attribute}: only {len(groups)} groups found")
                return []
            
            # Filter out groups with no data after alignment
            data = pd.DataFrame({
                'y_true': self.y_true,
                'y_pred': self.y_pred,
                'group': self.protected_attributes[attribute]
            }).dropna()
            
            # Check which groups actually have data
            actual_groups = data['group'].value_counts()
            if len(actual_groups) < 2:
                logger.warning(f"After data alignment, insufficient groups for {attribute}: only {len(actual_groups)} groups found")
                return []
            
            # Calculate metrics for each group first to check actual positives
            group_metrics = {}
            for group in actual_groups.index:
                group_data = data[data['group'] == group]
                logger.debug(f"Group {group}: {len(group_data)} samples")
                
                if not group_data.empty:
                    metrics = self.calculate_group_metrics(
                        group_data['y_true'],
                        group_data['y_pred']
                    )
                    if metrics:
                        group_metrics[group] = metrics
                        logger.debug(f"Group {group} metrics calculated: PPR={metrics.ppr:.4f}, TPR={metrics.tpr:.4f}, "
                                   f"samples={metrics.total_samples}, actual_positives={metrics.actual_positives}")
                        
                        # Add individual group metrics
                        attr_metrics.extend([
                            {
                                'Model': self.model_name,
                                'Strategy': self.strategy,
                                'Protected Attribute': attribute,
                                'Group': str(group),
                                'Metric Type': 'PPR',
                                'Value': metrics.ppr
                            },
                            {
                                'Model': self.model_name,
                                'Strategy': self.strategy,
                                'Protected Attribute': attribute,
                                'Group': str(group),
                                'Metric Type': 'TPR',
                                'Value': metrics.tpr
                            }
                        ])
                    else:
                        logger.warning(f"Failed to calculate metrics for group {group}")
                else:
                    logger.warning(f"No data for group {group}")
            
            logger.info(f"Successfully calculated metrics for groups: {list(group_metrics.keys())}")
            
            # Smart group selection based on attribute type and data availability
            if attribute == 'age_group':
                # For age_group, find two largest groups with actual positives for EOD
                groups_with_positives = [(g, m) for g, m in group_metrics.items() if m.actual_positives > 0]
                if len(groups_with_positives) >= 2:
                    # Sort by sample size and take two largest with positives
                    groups_with_positives.sort(key=lambda x: x[1].total_samples, reverse=True)
                    privileged_group = groups_with_positives[0][0]
                    unprivileged_group = groups_with_positives[1][0]
                    logger.info(f"Age group special selection - using groups with actual positives")
                else:
                    # Fallback to largest groups
                    privileged_group = actual_groups.index[0]
                    unprivileged_group = actual_groups.index[1] if len(actual_groups) > 1 else actual_groups.index[0]
            else:
                # Use the two largest groups for comparison (more robust)
                privileged_group = actual_groups.index[0]  # largest group
                unprivileged_group = actual_groups.index[1]  # second largest group
            
            logger.info(f"Calculating metrics for {attribute}")
            logger.info(f"  Groups found: {dict(groups)}")
            logger.info(f"  Groups with data: {dict(actual_groups)}")
            logger.info(f"  Privileged group: {privileged_group} (n={actual_groups[privileged_group]})")
            logger.info(f"  Unprivileged group: {unprivileged_group} (n={actual_groups[unprivileged_group]})")
            
            # Calculate and add aggregate metrics
            if len(group_metrics) >= 2:
                dir_value, dir_msg = self.calculate_disparate_impact_ratio(
                    group_metrics,
                    privileged_group,
                    unprivileged_group
                )
                if dir_value is not None:
                    attr_metrics.append({
                        'Model': self.model_name,
                        'Strategy': self.strategy,
                        'Protected Attribute': attribute,
                        'Group': f'DI ({unprivileged_group} vs {privileged_group})',
                        'Metric Type': 'Disparate Impact Ratio',
                        'Value': dir_value
                    })
                else:
                    logger.warning(f"Could not calculate DIR for {attribute}: {dir_msg}")
                
                eod_value, eod_msg = self.calculate_equal_opportunity_difference(
                    group_metrics,
                    privileged_group,
                    unprivileged_group
                )
                if eod_value is not None:
                    attr_metrics.append({
                        'Model': self.model_name,
                        'Strategy': self.strategy,
                        'Protected Attribute': attribute,
                        'Group': f'EO ({privileged_group} - {unprivileged_group})',
                        'Metric Type': 'Equal Opportunity Difference',
                        'Value': eod_value
                    })
                else:
                    logger.warning(f"Could not calculate EOD for {attribute}: {eod_msg}")
            else:
                logger.warning(f"Not enough groups with valid metrics for {attribute}: only {len(group_metrics)} groups")
            
            return attr_metrics
        except Exception as e:
            logger.error(f"Error calculating metrics for {attribute}: {str(e)}")
            return []

    def calculate_all_metrics(self) -> pd.DataFrame:
        """Calculate all fairness metrics for all protected attributes."""
        try:
            # Start with overall metrics
            all_metrics = self.calculate_overall_metrics()
            
            # Calculate fairness metrics for each protected attribute
            for attribute in self.protected_attributes.columns:
                logger.info(f"Calculating metrics for attribute: {attribute}")
                attribute_metrics = self.calculate_metrics_for_attribute(attribute)
                all_metrics.extend(attribute_metrics)
            
            # Convert to DataFrame
            metrics_df = pd.DataFrame(all_metrics)
            
            # Log summary
            logger.info(f"Calculated {len(metrics_df)} metrics for "
                       f"{self.model_name} ({self.strategy})")
            
            return metrics_df
        except Exception as e:
            logger.error(f"Error calculating all metrics: {str(e)}")
            return pd.DataFrame() 