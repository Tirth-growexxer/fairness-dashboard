"""Data preprocessing utilities."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from config.config import (
    LOAN_DATA_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    AGE_BINS,
    AGE_LABELS
)

import logging
logger = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
    """Load the loan data from CSV file."""
    try:
        logger.info("Starting data loading process...")
        df = pd.read_csv(LOAN_DATA_PATH)
        logger.info(f"Dataset loaded successfully from {LOAN_DATA_PATH}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Dataset columns: {list(df.columns)}")
        
        # Validate data types
        expected_columns = {
            'person_age': 'float64',
            'person_gender': 'object',
            'person_education': 'object',
            'person_income': 'float64',
            'person_emp_exp': 'int64',
            'person_home_ownership': 'object',
            'loan_amnt': 'float64',
            'loan_intent': 'object',
            'loan_int_rate': 'float64',
            'loan_percent_income': 'float64',
            'cb_person_cred_hist_length': 'float64',
            'credit_score': 'int64',
            'previous_loan_defaults_on_file': 'object',
            'loan_status': 'int64'
        }
        
        # Convert columns to expected types
        logger.info("Converting columns to expected data types...")
        for col, dtype in expected_columns.items():
            if col not in df.columns:
                raise ValueError(f"Required column {col} not found in dataset")
            try:
                if dtype == 'object':
                    df[col] = df[col].astype(str)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if dtype == 'int64':
                        df[col] = df[col].fillna(df[col].median()).astype(int)
                    else:
                        df[col] = df[col].fillna(df[col].median())
            except Exception as e:
                logger.error(f"Error converting column {col} to {dtype}: {str(e)}")
                raise
        
        # Validate no missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning("Missing values found after processing:")
            for col, count in missing[missing > 0].items():
                logger.warning(f"  {col}: {count} missing values")
        else:
            logger.info("No missing values found after processing")
        
        # Log basic statistics
        logger.info(f"Target variable distribution: {df['loan_status'].value_counts().to_dict()}")
        logger.info(f"Age range: {df['person_age'].min():.1f} - {df['person_age'].max():.1f}")
        logger.info(f"Income range: ${df['person_income'].min():,.0f} - ${df['person_income'].max():,.0f}")
        
        return df
        
    except FileNotFoundError:
        logger.warning(f"Data file not found at {LOAN_DATA_PATH}. Creating dummy data.")
        return create_dummy_data()
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def create_dummy_data(n_samples: int = 50000) -> pd.DataFrame:
    """Create dummy data for testing purposes with realistic distributions."""
    try:
        logger.info(f"Creating dummy dataset with {n_samples} samples")
        
        np.random.seed(RANDOM_SEED)
        
        # Create more realistic age distribution
        age_weights = [0.4, 0.35, 0.2, 0.05]  # 18-30, 30-45, 45-60, 60+
        ages = []
        for i, (min_age, max_age, weight) in enumerate(zip([18, 30, 45, 60], [30, 45, 60, 80], age_weights)):
            count = int(n_samples * weight)
            ages.extend(np.random.randint(min_age, max_age, count))
        
        # Fill remaining samples
        while len(ages) < n_samples:
            ages.append(np.random.randint(18, 80))
        ages = ages[:n_samples]
        
        # Create loan approval rates that vary by age (more realistic)
        loan_status = []
        for age in ages:
            if age < 30:
                approval_rate = 0.25  # Young people have lower approval
            elif age < 45:
                approval_rate = 0.35  # Middle age has higher approval
            elif age < 60:
                approval_rate = 0.30  # Pre-retirement moderate approval
            else:
                approval_rate = 0.20  # Seniors have lower approval
            
            loan_status.append(np.random.choice([0, 1], p=[1-approval_rate, approval_rate]))
        
        data = {
            'person_age': ages,
            'person_gender': np.random.choice(['male', 'female'], n_samples),
            'person_education': np.random.choice(
                ['Bachelor', 'Master', 'High School', 'Associate', 'Doctorate'],
                n_samples,
                p=[0.3, 0.2, 0.3, 0.15, 0.05]  # Realistic education distribution
            ),
            'person_income': np.random.randint(20000, 150000, n_samples),
            'person_emp_exp': np.random.randint(0, 20, n_samples),
            'person_home_ownership': np.random.choice(
                ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
                n_samples,
                p=[0.45, 0.15, 0.35, 0.05]  # Realistic ownership distribution
            ),
            'loan_amnt': np.random.randint(1000, 35000, n_samples),
            'loan_intent': np.random.choice(
                ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
                n_samples
            ),
            'loan_int_rate': np.random.uniform(5.0, 20.0, n_samples),
            'loan_percent_income': np.random.uniform(0.05, 0.5, n_samples),
            'cb_person_cred_hist_length': np.random.randint(2, 15, n_samples),
            'credit_score': np.random.randint(500, 800, n_samples),
            'previous_loan_defaults_on_file': np.random.choice(['No', 'Yes'], n_samples),
            'loan_status': loan_status
        }
        
        df = pd.DataFrame(data)
        
        # Convert to correct types
        df['person_age'] = df['person_age'].astype(float)
        df['person_income'] = df['person_income'].astype(float)
        df['loan_amnt'] = df['loan_amnt'].astype(float)
        df['loan_int_rate'] = df['loan_int_rate'].astype(float)
        df['loan_percent_income'] = df['loan_percent_income'].astype(float)
        df['cb_person_cred_hist_length'] = df['cb_person_cred_hist_length'].astype(float)
        df['credit_score'] = df['credit_score'].astype(int)
        df['loan_status'] = df['loan_status'].astype(int)
        
        logger.info("Dummy dataset created successfully")
        logger.info(f"Dummy dataset shape: {df.shape}")
        logger.info(f"Dummy target distribution: {df['loan_status'].value_counts().to_dict()}")
        
        # Log age distribution
        age_groups_temp = pd.cut(df['person_age'], bins=AGE_BINS, labels=AGE_LABELS, right=False, include_lowest=True)
        logger.info(f"Age group distribution: {age_groups_temp.value_counts().to_dict()}")
        
        # Log approval rates by age group
        for age_label in AGE_LABELS:
            mask = age_groups_temp == age_label
            if mask.sum() > 0:
                approval_rate = df.loc[mask, 'loan_status'].mean()
                logger.info(f"Age group {age_label}: {mask.sum()} samples, {approval_rate:.3f} approval rate")
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating dummy data: {str(e)}")
        raise

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the raw data by handling missing values, converting types,
    and creating derived features.
    """
    try:
        logger.info("Starting data preprocessing...")
        
        # Create a copy to avoid modifying the original
        df = df.copy()
        original_shape = df.shape
        
        # Convert previous_loan_defaults to numeric
        logger.debug("Converting categorical variables to numeric...")
        df['previous_loan_defaults_on_file'] = (
            df['previous_loan_defaults_on_file']
            .map({'No': 0, 'Yes': 1})
            .astype(int)
        )
        
        # Create age groups
        logger.debug(f"Creating age groups using bins: {AGE_BINS} with labels: {AGE_LABELS}")
        df['age_group'] = pd.cut(
            df['person_age'],
            bins=AGE_BINS,
            labels=AGE_LABELS,
            right=False,
            include_lowest=True
        )
        
        # Log age group distribution
        age_dist = df['age_group'].value_counts()
        logger.info(f"Age group distribution: {age_dist.to_dict()}")
        
        df.drop(columns=['person_age'], inplace=True)
        
        # Split features and target
        X = df.drop('loan_status', axis=1)
        y = df['loan_status'].astype(int)
        
        logger.info(f"Data preprocessing completed successfully")
        logger.info(f"Original shape: {original_shape}, Final shape: {X.shape}")
        logger.info(f"Features: {list(X.columns)}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def prepare_model_data(
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Prepare data for model training by splitting into train/test sets
    and applying necessary transformations.
    """
    try:
        logger.info("Starting model data preparation...")
        
        # Split data
        logger.info(f"Splitting data with test_size={TEST_SIZE}, random_state={RANDOM_SEED}")
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y
        )
        
        logger.info(f"Train set shape: {X_train_raw.shape}, Test set shape: {X_test_raw.shape}")
        logger.info(f"Train target distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")
        
        # Keep a copy for fairness metrics
        X_test_for_fairness = X_test_raw.copy()
        
        # Identify feature types
        numerical_features = X_train_raw.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_train_raw.select_dtypes(include=['object', 'category']).columns
        
        logger.info(f"Numerical features ({len(numerical_features)}): {list(numerical_features)}")
        logger.info(f"Categorical features ({len(categorical_features)}): {list(categorical_features)}")
        
        # Initialize transformers
        logger.debug("Initializing data transformers...")
        scaler = MinMaxScaler()
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        # Transform numerical features
        logger.debug("Scaling numerical features...")
        X_train_num = scaler.fit_transform(X_train_raw[numerical_features])
        X_test_num = scaler.transform(X_test_raw[numerical_features])
        
        # Transform categorical features
        logger.debug("Encoding categorical features...")
        X_train_cat = encoder.fit_transform(X_train_raw[categorical_features])
        X_test_cat = encoder.transform(X_test_raw[categorical_features])
        
        # Create DataFrames with transformed features
        encoded_feature_names = encoder.get_feature_names_out(categorical_features)
        logger.info(f"Created {len(encoded_feature_names)} encoded categorical features")
        
        X_train_processed = pd.DataFrame(
            np.hstack([X_train_num, X_train_cat]),
            columns=list(numerical_features) + list(encoded_feature_names),
            index=X_train_raw.index
        )
        
        X_test_processed = pd.DataFrame(
            np.hstack([X_test_num, X_test_cat]),
            columns=list(numerical_features) + list(encoded_feature_names),
            index=X_test_raw.index
        )
        
        # Package the results
        train_data = {
            'X_raw': X_train_raw,
            'X_processed': X_train_processed,
            'y': y_train,
            'scaler': scaler,
            'encoder': encoder
        }
        
        test_data = {
            'X_raw': X_test_raw,
            'X_processed': X_test_processed,
            'y': y_test,
            'X_fairness': X_test_for_fairness
        }
        
        logger.info("Model data preparation completed successfully")
        logger.info(f"Final processed feature count: {X_train_processed.shape[1]}")
        
        return train_data, test_data
        
    except Exception as e:
        logger.error(f"Error preparing model data: {str(e)}")
        raise 