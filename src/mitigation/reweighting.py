"""Reweighting bias mitigation strategy implementation."""

import pandas as pd
import numpy as np
from typing import Union, List, Optional
from dataclasses import dataclass

@dataclass
class ReweightingResults:
    """Container for reweighting results."""
    sample_weights: np.ndarray
    group_weights: pd.Series
    group_counts: pd.Series
    protected_groups: pd.Series

class Reweighter:
    """Implements the reweighting bias mitigation strategy."""
    
    def __init__(
        self,
        protected_attributes: Union[str, List[str]],
        normalize_weights: bool = True
    ):
        """
        Initialize the reweighter.
        
        Args:
            protected_attributes: Column name(s) to use for reweighting
            normalize_weights: Whether to normalize weights to sum to n_samples
        """
        self.protected_attributes = (
            [protected_attributes]
            if isinstance(protected_attributes, str)
            else protected_attributes
        )
        self.normalize_weights = normalize_weights

    def _create_group_identifier(
        self,
        data: pd.DataFrame,
        protected_columns: List[str]
    ) -> pd.Series:
        """
        Create an intersectional group identifier from multiple protected attributes.
        
        Args:
            data: DataFrame containing protected attribute columns
            protected_columns: List of protected attribute column names
            
        Returns:
            Series containing group identifiers
        """
        # Handle numerical attributes by binning them
        temp_data = data[protected_columns].copy()
        for col in protected_columns:
            if temp_data[col].dtype in ['int64', 'float64']:
                temp_data[col] = pd.qcut(
                    temp_data[col],
                    q=5,
                    labels=[f'{col}_bin_{i}' for i in range(5)],
                    duplicates='drop'
                )
        
        # Combine all attributes into a single identifier
        return temp_data.astype(str).agg('-'.join, axis=1)

    def calculate_weights(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> ReweightingResults:
        """
        Calculate sample weights for bias mitigation.
        
        The weight for each sample (g,y) is calculated as:
        w(g,y) = P(Y=y) / (P(A=g) * P(Y=y|A=g))
        
        where:
        - P(Y=y) is the overall proportion of instances with label y
        - P(A=g) is the overall proportion of instances in protected group g
        - P(Y=y|A=g) is the proportion of instances with label y within group g
        
        Args:
            X: Features DataFrame containing protected attributes
            y: Target labels
            
        Returns:
            ReweightingResults containing calculated weights and metadata
        """
        # Verify all protected attributes exist in X
        missing_attrs = [attr for attr in self.protected_attributes if attr not in X.columns]
        if missing_attrs:
            raise ValueError(f"Protected attributes not found in data: {missing_attrs}")
        
        # Create intersectional group identifier if multiple attributes
        if len(self.protected_attributes) > 1:
            protected_groups = self._create_group_identifier(X, self.protected_attributes)
        else:
            protected_groups = X[self.protected_attributes[0]]
        
        # Calculate overall proportions
        p_y = y.value_counts(normalize=True)
        
        # Calculate group proportions
        group_counts = protected_groups.value_counts()
        p_g = group_counts / len(protected_groups)
        
        # Calculate conditional proportions P(Y=y|A=g)
        p_y_given_g = pd.DataFrame({
            'group': protected_groups,
            'label': y
        }).groupby('group')['label'].value_counts(normalize=True)
        
        # Initialize weights array
        sample_weights = np.zeros(len(X))
        
        # Calculate weights for each sample
        for i, (group, label) in enumerate(zip(protected_groups, y)):
            p_y_val = p_y.get(label, 0.0)
            p_g_val = p_g.get(group, 0.0)
            p_y_given_g_val = p_y_given_g.get((group, label), 0.0)
            
            if p_g_val > 0 and p_y_given_g_val > 0:
                weight = p_y_val / (p_g_val * p_y_given_g_val)
            else:
                weight = 1.0  # Default weight if calculation not possible
            
            sample_weights[i] = weight
        
        # Normalize weights if requested
        if self.normalize_weights:
            sample_weights = (
                sample_weights / np.sum(sample_weights) * len(sample_weights)
            )
        
        # Create group weights Series
        group_weights = pd.Series(
            sample_weights,
            index=protected_groups.index
        ).groupby(protected_groups).mean()
        
        return ReweightingResults(
            sample_weights=sample_weights,
            group_weights=group_weights,
            group_counts=group_counts,
            protected_groups=protected_groups
        )

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> np.ndarray:
        """
        Calculate and return sample weights.
        
        Args:
            X: Features DataFrame containing protected attributes
            y: Target labels
            
        Returns:
            Array of sample weights
        """
        results = self.calculate_weights(X, y)
        return results.sample_weights 