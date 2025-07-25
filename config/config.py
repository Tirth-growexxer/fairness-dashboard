"""Configuration settings for the fairness dashboard project."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

# Data settings
LOAN_DATA_PATH = "data/loan_data.csv"
METRICS_OUTPUT_PATH = "data/combined_fairness_metrics.parquet"

# Model settings
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Protected attributes
PROTECTED_ATTRIBUTES = [
    'person_gender',
    'age_group',
    'person_education',
    'person_home_ownership'
]

# Age binning
AGE_BINS = [18, 30, 45, 60, float('inf')]
AGE_LABELS = ['18-30', '30-45', '45-60', '60+']

# Dashboard settings
DASH_HOST = "127.0.0.1"
DASH_PORT = 5004
DASH_DEBUG = True

# Fairness thresholds
DISPARATE_IMPACT_THRESHOLD = 0.8  # Below this is considered unfair
EQUAL_OPPORTUNITY_THRESHOLD = 0.1  # Absolute difference above this is considered unfair

# Ensure directories exist
for directory in [DATA_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 