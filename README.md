# 🏦 AI Fairness Auditing Dashboard

A comprehensive machine learning fairness auditing system that evaluates bias in loan approval models across different demographic groups and implements bias mitigation strategies.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Dash](https://img.shields.io/badge/Dash-2.14+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Machine Learning Models](#machine-learning-models)
- [Fairness Metrics](#fairness-metrics)
- [Bias Mitigation](#bias-mitigation)
- [Dashboard Features](#dashboard-features)
- [Testing](#testing)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project addresses one of the most critical challenges in modern AI: **algorithmic fairness**. It provides a complete pipeline for:

1. **Training multiple ML models** for loan approval prediction
2. **Evaluating fairness** across protected demographic attributes
3. **Implementing bias mitigation** strategies (reweighting)
4. **Visualizing results** through an interactive dashboard
5. **Comprehensive testing** with unit and integration tests

The system helps financial institutions and ML practitioners identify, measure, and mitigate bias in their loan approval algorithms, ensuring fair treatment across different demographic groups.

## ✨ Features

### 🤖 Machine Learning Pipeline
- **4 Different Models**: Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Automated Training & Evaluation**: Complete pipeline from data preprocessing to model evaluation
- **Cross-Model Comparison**: Compare performance and fairness across different algorithms

### ⚖️ Fairness Assessment
- **Multiple Fairness Metrics**: Disparate Impact Ratio (DIR), Equal Opportunity Difference (EOD)
- **Protected Attributes**: Gender, Age Group, Education Level, Home Ownership
- **Group-wise Analysis**: Detailed breakdown by demographic subgroups
- **Bias Detection**: Automated flagging of potential bias issues

### 🛠️ Bias Mitigation
- **Reweighting Strategy**: Pre-processing bias mitigation technique
- **Before/After Comparison**: Visualize the impact of bias mitigation
- **Configurable Parameters**: Customizable fairness thresholds

### 📊 Interactive Dashboard
- **Real-time Visualization**: Dynamic charts and tables
- **Model Selection**: Switch between different ML models
- **Comparative Analysis**: Side-by-side comparison of standard vs. reweighted models
- **Export Capabilities**: Download results and metrics

### 🧪 Comprehensive Testing
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Real Data Testing**: Tests using actual loan dataset

## 📁 Project Structure

```
fairness-dashboard/
├── 📱 app.py                          # Main dashboard application
├── 📋 requirements.txt                # Python dependencies
├── 📖 README.md                       # Project documentation
├── 🚫 .gitignore                      # Git ignore rules
│
├── 📊 src/                            # Source code
│   ├── 🧠 models/                     # ML model implementations
│   │   └── model_factory.py          # Model creation and training
│   ├── ⚖️ metrics/                    # Fairness metrics
│   │   └── fairness_metrics.py       # Fairness calculations
│   ├── 🛠️ mitigation/                 # Bias mitigation strategies
│   │   └── reweighting.py            # Reweighting implementation
│   ├── 🔧 utils/                      # Utility functions
│   │   ├── data_preprocessing.py     # Data loading and preprocessing
│   │   └── logger.py                 # Logging configuration
│   ├── 📊 visualization/              # (Future: additional visualizations)
│   └── 🎨 templates/                  # HTML templates
│       └── dashboard.html            # Dashboard UI template
│
├── ⚙️ config/                         # Configuration files
│   └── config.py                     # Project settings and parameters
│
├── 📁 data/                           # Data directory
│   ├── loan_data.csv                 # Main dataset (45,000+ records)
│   └── combined_fairness_metrics.parquet  # Generated metrics
│
├── 🧪 tests/                          # Test suite
│   ├── unit/                         # Unit tests
│   │   └── test_fairness_metrics.py  # Fairness metrics tests
│   └── integration/                  # Integration tests
│       ├── test_end_to_end_pipeline.py    # Complete pipeline tests
│       ├── test_dashboard_functionality.py # Dashboard tests
│       └── test_data_pipeline.py          # Data processing tests
│
├── 📓 notebooks/                      # Jupyter notebooks
│   └── main.ipynb                    # Original research notebook
│
├── 📈 reports/                        # Generated reports and outputs
├── 📝 logs/                          # Application logs
└── 🐍 venv/                          # Virtual environment (excluded from git)
```

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tirth-growexxer/fairness-dashboard.git
   cd fairness-dashboard
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python3 -c "import dash, pandas, sklearn; print('✅ All dependencies installed successfully!')"
   ```

## 🏃‍♂️ Quick Start

### 1. Run the Dashboard
```bash
python3 app.py
```
The dashboard will be available at `http://localhost:5004`

### 2. Run Tests
```bash
# Unit tests
python3 -m unittest tests.unit.test_fairness_metrics -v

# Integration tests
python3 -m unittest tests.integration.test_end_to_end_pipeline -v
python3 -m unittest tests.integration.test_dashboard_functionality -v
python3 -m unittest tests.integration.test_data_pipeline -v
```

### 3. Explore the Notebook
```bash
jupyter notebook notebooks/main.ipynb
```

## 📊 Dataset

### Loan Approval Dataset
- **Size**: 45,000+ records
- **Target**: Binary classification (loan approval/rejection)
- **Features**: 13 input features including demographic and financial information

### Key Features:
| Feature | Description | Type |
|---------|-------------|------|
| `person_age` | Applicant's age | Numeric |
| `person_gender` | Gender (male/female) | Categorical |
| `person_education` | Education level | Categorical |
| `person_income` | Annual income | Numeric |
| `person_emp_exp` | Employment experience (years) | Numeric |
| `person_home_ownership` | Home ownership status | Categorical |
| `loan_amnt` | Loan amount requested | Numeric |
| `loan_intent` | Purpose of the loan | Categorical |
| `loan_int_rate` | Interest rate | Numeric |
| `loan_percent_income` | Loan as % of income | Numeric |
| `cb_person_cred_hist_length` | Credit history length | Numeric |
| `credit_score` | Credit score | Numeric |
| `previous_loan_defaults_on_file` | Previous defaults | Categorical |

### Protected Attributes:
- **Gender**: male, female
- **Age Group**: 18-30, 30-45, 45-60, 60+
- **Education**: High School, Bachelor, Master, Doctorate
- **Home Ownership**: RENT, OWN, MORTGAGE, OTHER

## 🤖 Machine Learning Models

### Implemented Models:
1. **Logistic Regression**
   - Linear baseline model
   - Interpretable coefficients
   - Fast training and prediction

2. **Decision Tree**
   - Non-linear decision boundaries
   - Feature importance insights
   - Easy to interpret rules

3. **Random Forest**
   - Ensemble of decision trees
   - Reduced overfitting
   - Feature importance ranking

4. **XGBoost**
   - Gradient boosting framework
   - High performance on tabular data
   - Advanced regularization

### Model Training Process:
1. **Data Preprocessing**: Scaling, encoding, feature engineering
2. **Train/Test Split**: 80/20 stratified split
3. **Model Training**: With and without sample weights
4. **Evaluation**: Accuracy, precision, recall, F1-score
5. **Fairness Assessment**: Bias metrics across protected groups

## ⚖️ Fairness Metrics

### Disparate Impact Ratio (DIR)
```
DIR = (Positive Rate for Unprivileged Group) / (Positive Rate for Privileged Group)
```
- **Ideal Value**: 1.0 (perfect parity)
- **Threshold**: < 0.8 indicates potential bias
- **Interpretation**: Measures selection rate differences

### Equal Opportunity Difference (EOD)
```
EOD = TPR(Privileged) - TPR(Unprivileged)
```
- **Ideal Value**: 0.0 (no difference)
- **Threshold**: |EOD| > 0.1 indicates potential bias
- **Interpretation**: Measures true positive rate differences

### Group Metrics:
- **PPR (Positive Prediction Rate)**: Proportion of positive predictions
- **TPR (True Positive Rate)**: Sensitivity/Recall for each group
- **Group Sample Sizes**: Number of samples per demographic group

## 🛠️ Bias Mitigation

### Reweighting Strategy
The system implements a **pre-processing bias mitigation technique**:

1. **Group Identification**: Create intersectional groups from protected attributes
2. **Weight Calculation**: Compute inverse probability weights
3. **Sample Reweighting**: Apply weights during model training
4. **Fairness Improvement**: Reduce bias while maintaining performance

### Implementation:
```python
from src.mitigation.reweighting import Reweighter

# Initialize reweighter
reweighter = Reweighter(protected_attributes=['person_gender', 'age_group'])

# Calculate sample weights
sample_weights = reweighter.fit_transform(X_train, y_train)

# Train model with weights
model.fit(X_train, y_train, sample_weight=sample_weights)
```

## 📊 Dashboard Features

### 🎛️ Interactive Controls
- **Model Selection**: Choose from 4 different ML models
- **Strategy Comparison**: Standard vs. Reweighted models
- **Real-time Updates**: Dynamic chart updates

### 📈 Visualizations

#### 1. Model Performance Summary
- Accuracy, Precision, Recall, F1-Score
- Training and test sample sizes
- Model-specific performance metrics

#### 2. Disparate Impact Analysis
- Bar charts showing DIR across protected attributes
- Color-coded fairness indicators (green = fair, red = biased)
- Comparison between standard and reweighted models

#### 3. Equal Opportunity Analysis
- EOD visualization across demographic groups
- Threshold-based bias detection
- Before/after mitigation comparison

#### 4. Group-wise Metrics
- PPR and TPR for each demographic subgroup
- Sample size information
- Statistical significance indicators

### 🎨 UI Features
- **Responsive Design**: Works on desktop and mobile
- **Modern Interface**: Clean, professional styling
- **Intuitive Navigation**: Easy model switching
- **Export Options**: Download charts and data

## 🧪 Testing

### Test Coverage
- **Unit Tests**: Individual component functionality
- **Integration Tests**: End-to-end pipeline validation
- **Real Data Tests**: Validation with actual loan dataset

### Running Tests
```bash
# Run all tests
python3 -m unittest discover tests/ -v

# Run specific test categories
python3 -m unittest tests.unit.test_fairness_metrics -v
python3 -m unittest tests.integration.test_end_to_end_pipeline -v

# Run with coverage (if pytest-cov installed)
pytest tests/ --cov=src --cov-report=html
```

### Test Features
- **Real Data Integration**: Tests use actual loan dataset
- **Reproducible Results**: Fixed random seeds
- **Comprehensive Validation**: Model training, fairness calculation, dashboard functionality
- **Error Handling**: Edge case and error condition testing

## ⚙️ Configuration

### Key Configuration Files

#### `config/config.py`
```python
# Protected attributes for fairness analysis
PROTECTED_ATTRIBUTES = [
    'person_gender',
    'age_group', 
    'person_education',
    'person_home_ownership'
]

# Fairness thresholds
DISPARATE_IMPACT_THRESHOLD = 0.8
EQUAL_OPPORTUNITY_THRESHOLD = 0.1

# Dashboard settings
DASH_HOST = "127.0.0.1"
DASH_PORT = 5004
DASH_DEBUG = True
```

### Customization Options
- **Add New Models**: Extend `ModelFactory` class
- **New Fairness Metrics**: Add to `FairnessMetrics` class
- **Custom Protected Attributes**: Modify configuration
- **Visualization Themes**: Update dashboard templates

## 🔧 Advanced Usage

### Adding New Models
```python
# In src/models/model_factory.py
def get_available_models():
    models = {
        # Existing models...
        'Your Model': YourModelClass(
            # your parameters
        )
    }
    return models
```

### Custom Fairness Metrics
```python
# In src/metrics/fairness_metrics.py
def calculate_your_metric(self, group_metrics, privileged_group, unprivileged_group):
    # Your fairness metric implementation
    return metric_value
```

### Environment Variables
```bash
# Optional environment variables
export FAIRNESS_DASHBOARD_PORT=8080
export FAIRNESS_DASHBOARD_DEBUG=False
export FAIRNESS_LOG_LEVEL=INFO
```

## 📈 Performance Benchmarks

### Typical Performance (45K records):
- **Data Loading**: ~2 seconds
- **Preprocessing**: ~3 seconds  
- **Model Training**: ~10-30 seconds (all 4 models)
- **Fairness Calculation**: ~5 seconds
- **Dashboard Loading**: ~2 seconds

### Memory Usage:
- **Base Application**: ~200MB
- **With Data Loaded**: ~400MB
- **During Training**: ~600MB

## 🚨 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Kill process using the port
sudo lsof -t -i tcp:5004 | xargs kill -9

# Or change port in config/config.py
DASH_PORT = 8080
```

#### Missing Dependencies
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

#### Data Loading Issues
```bash
# Verify data file exists
ls -la data/loan_data.csv

# Check file permissions
chmod 644 data/loan_data.csv
```

### Logging
Logs are stored in `logs/` directory:
- `dashboard.log`: Application logs
- `fairness_metrics.log`: Fairness calculation logs
- `model_training.log`: ML training logs

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run test suite: `python3 -m unittest discover tests/ -v`
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include type hints where appropriate
- Add logging for important operations

### Adding Features
1. **New Models**: Extend `ModelFactory`
2. **New Metrics**: Add to `FairnessMetrics`
3. **UI Components**: Update dashboard templates
4. **Tests**: Add corresponding test cases

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Scikit-learn**: Machine learning framework
- **Plotly/Dash**: Interactive visualization
- **XGBoost**: Gradient boosting implementation
- **Pandas**: Data manipulation and analysis

## 📞 Support

For questions, issues, or contributions:
- 📧 Create an issue on GitHub
- 📚 Check the documentation
- 🔍 Search existing issues

---

**Made with ❤️ for AI Fairness**

*This project aims to make machine learning more fair and equitable for everyone.* 