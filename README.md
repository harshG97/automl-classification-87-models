# AutoML with AutoGluon: Comparing 87 Models for Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![AutoGluon](https://img.shields.io/badge/AutoGluon-1.1.1-orange.svg)](https://auto.gluon.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project demonstrates the power of **AutoML (Automated Machine Learning)** using AutoGluon to predict depression from health and lifestyle factors. The notebook automatically trains, tunes, and evaluates **87 different machine learning models** with minimal code, showcasing how AutoGluon simplifies the ML workflow while achieving state-of-the-art performance.

### Key Highlights

- 🤖 **87 models trained automatically** including Random Forests, XGBoost, LightGBM, CatBoost, Neural Networks, and ensemble methods
- 🎯 **Binary classification** task for depression prediction
- 📊 **Comprehensive EDA** with visualizations and statistical analysis
- 🔄 **Advanced ensembling** strategies including weighted averaging and top-k model selection
- ⚡ **GPU acceleration** supported for faster training
- 📈 **Multiple evaluation metrics** including ROC-AUC, accuracy, and prediction time

## Project Structure

```
.
├── automl-with-autogluon-87-models-enhanced.ipynb  # Enhanced notebook with detailed markdown
├── requirements.txt                                 # Python dependencies (pip)
├── environment.yml                                  # Conda environment file (optional)
├── README.md                                        # This file
└── data/                                           # Dataset directory (create manually)
│    ├── train.csv                                   # Competition training data
│    ├── test.csv                                    # Competition test data
│    └── final_depression_dataset_1.csv              # Original dataset
│
└── outputs/                                # Model outputs
    ├── submission.csv
    └── sample_submission.csv
```

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for acceleration

### Installation

1. **Clone or download this repository**

```bash
git clone <repository-url>
cd automl-autogluon-depression
```

2. **Installation**

#### Using Anaconda/Conda

```bash
# Create a new conda environment
conda create -n autogluon-env python=3.10 -y

# Activate the environment
conda activate autogluon-env

# Install dependencies
# Note: Install AutoGluon via pip even in conda environment (recommended by AutoGluon team)
pip install -r requirements.txt
```

#### Using Anaconda with environment.yml

```bash
conda env create -f environment.yml
conda activate autogluon-env
```

## 📓 Notebook Sections

### 1. Environment Setup
- Installing AutoGluon and dependencies
- Configuring Ray for distributed computing

### 2. Data Loading & Preprocessing
- Loading competition and original datasets
- Harmonizing data formats
- Combining datasets for improved performance
- Feature engineering and cleaning

### 3. Exploratory Data Analysis (EDA)
- Target variable distribution
- Feature correlation analysis
- Missing value assessment
- Statistical summaries

### 4. AutoGluon Model Training
- Configuration of TabularPredictor
- Automatic model selection and training
- Hyperparameter tuning
- Cross-validation

### 5. Model Evaluation
- Leaderboard of all 87 models
- Performance metrics (ROC-AUC, accuracy)
- Model comparison and analysis

### 6. Predictions & Ensembling
- Individual model predictions
- Weighted ensemble predictions
- Top-k model averaging
- Custom threshold optimization

### 7. Submission Generation
- Creating submission files
- Multiple prediction strategies
- Final model selection

## Model Performance

The notebook trains 87 models including:

- **Tree-based models**: Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost
- **Neural Networks**: TabularNeuralNet with various architectures
- **Ensemble methods**: Weighted ensemble, stacked models
- **K-Nearest Neighbors**: KNN variants
- **Linear models**: Logistic Regression (when applicable)

Best models are automatically identified and ranked by validation performance.

## Configuration Options

You can customize the AutoGluon training by modifying these parameters:

```python
# In the notebook
RETUNE_AUTOGLUON = False  # Set to True to retrain models
RETUNE_AUTOGLUON_PREDICTIONS = False  # Set to True to regenerate predictions

# AutoGluon predictor settings
predictor = TabularPredictor(
    label='Depression',
    eval_metric='roc_auc',  # Change evaluation metric
    problem_type='binary'   # Classification type
)

predictor.fit(
    train_data=train_df,
    presets='best_quality',  # Options: 'best_quality', 'high_quality', 'good_quality', 'medium_quality'
    time_limit=900,          # Training time in seconds (15 minutes)
    verbosity=2              # 0-4, higher = more verbose
)
```

## Key Features

### AutoGluon Advantages
- **Automated ML Pipeline**: No need to manually select algorithms or tune hyperparameters
- **Smart Ensembling**: Automatically creates and optimizes ensemble models
- **Efficient Search**: Uses state-of-the-art optimization techniques
- **Production Ready**: Models can be easily deployed and served

### Advanced Techniques Used
- **Multiple prediction strategies**: Testing various ensemble approaches
- **Threshold optimization**: Fine-tuning decision boundaries
- **Model stacking**: Creating meta-models from base predictions


## Performance Optimization

### Training Speed
- Use GPU acceleration (requires CUDA)
- Reduce `time_limit` for faster experiments
- Use `presets='medium_quality'` for quicker results

### Model Quality
- Increase `time_limit` for better hyperparameter tuning
- Use `presets='best_quality'` for maximum accuracy
- Combine multiple datasets for more training data

## Contributing

Contributions are welcome!

## Resources

- [AutoGluon Documentation](https://auto.gluon.ai/)
- [AutoGluon Tutorials](https://auto.gluon.ai/stable/tutorials/index.html)
- [Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s4e11)
- [AutoML Best Practices](https://auto.gluon.ai/stable/tutorials/tabular/tabular-essentials.html)

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

*Built with ❤️ for financial ML and ensemble learning*