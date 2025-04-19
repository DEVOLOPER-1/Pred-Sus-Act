# Pred-Sus-Act


## Project Overview
Pred-Sus-Act focuses on detecting network anomalies using machine learning techniques. It consists of three main parts:
1. **Exploratory Data Analysis (EDA)**: Loading and preparing the dataset for analysis
2. **Feature Selection**: Using LASSO regression to identify the most important features for anomaly detection
3. **Model Training**: Implementing and evaluating various machine learning models for classification

## Project Structure

### 1. LASSO Feature Selection (`LASSO_feature_selection.ipynb`)
- Performs feature selection using LASSO regression
- Identifies the most significant features for anomaly detection
- Saves selected features to `data/original/LASSO_selected_features.csv`

Key steps:
- Data loading and preprocessing
- Feature engineering (numeric and categorical features)
- LASSO regression with alpha parameter tuning
- Feature importance visualization
- Saving selected features

### 2. Exploratory Data Analysis (`EDA.ipynb`)
- Loads the dataset and performs initial data exploration
- Uses ANOVA for feature selection
- Visualizes the distribution of features and target variable
- Handles unbalanced classes
- Applies dimensionality reduction techniques (PCA, t-SNE, UMAP) for visualization
- Applies `SMOTEN` as best technique for oversampling the minority class
- Visualizes the impact of different oversampling techniques on the dataset
- Saves the processed dataset to `data/resampled/ANOVA_selected_features.csv`

### 3. Model Training (`Models.ipynb`)
- Implements and evaluates four machine learning models:
  1. Support Vector Machine (SVM) Classifier
  2. Decision Tree Classifier
  3. Random Forest Classifier
  4. Logistic Regression
- Includes an ensemble stacking classifier
- Generates performance metrics and saves reports to SQLite database

Key components:
- Data loading and train-test split
- Model fine-tuning with hyperparameter optimization
- Performance metric generation and visualization
- Model comparison and ensemble learning

## Key Features

### Feature Selection
- Uses ANOVA for feature selection
- Uses LASSO regression for feature selection
- Evaluates feature importance based on coefficients
- Handles both numeric and categorical features
- Visualizes the impact of different alpha values on model performance

### Machine Learning Models
1. **SVM Classifier**
   - Linear kernel implementation
   - C parameter tuning for optimal recall
   - Achieves high accuracy in anomaly detection

2. **Decision Tree Classifier**
   - Uses entropy as splitting criterion
   - Visualizes the decision tree structure
   - Provides interpretable classification rules

3. **Random Forest Classifier**
   - Ensemble of decision trees
   - Uses entropy for node splitting
   - Handles high-dimensional feature space effectively

4. **Logistic Regression**
   - Linear classification model
   - Regularization parameter (C) tuning
   - Efficient for binary classification tasks

5. **Stacking Ensemble**
   - Combines predictions from all base models
   - Uses Random Forest as final estimator
   - Potentially improves overall performance

## Performance Evaluation
- Implements comprehensive metric reporting:
  - Precision, recall, and F1-score for each class
  - Accuracy, macro and weighted averages
  - Visualizations of model performance

## Data Management
- Uses SQLite database (`models_reports.db`) to store:
  - Test metadata (model names, dataset versions)
  - Model parameters
  - Performance metrics

## How to Use
1. Clone the repository and navigate to the project directory
2. Activate poetry environment
3. Install dependencies using `poetry install`
4. Run the Jupyter notebooks in order:
   - `LASSO_feature_selection.ipynb`
   - `EDA.ipynb`
   - `Models.ipynb`

## Requirements
- Navigate to `pyproject.toml` for project dependencies

## Future Improvements
- Complete hyperparameter tuning for Random Forest
- Experiment with additional models (e.g., Neural Networks)
- Implement more sophisticated feature engineering
- Add cross-validation for more robust evaluation

This project provides a comprehensive framework for network anomaly detection, from feature selection to model evaluation, with a focus on interpretability and performance.
