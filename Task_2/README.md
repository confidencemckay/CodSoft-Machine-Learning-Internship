# Credit Card Fraud Detection Model
### Overview

This project implements a machine learning–based credit card fraud detection system designed to classify transactions as either fraudulent or legitimate. The objective is to explore transaction data, preprocess features, visualize key patterns, and compare multiple classification models to evaluate their effectiveness in fraud detection.

## Dataset
- File: `fraudTrain.csv`
- Target Variable: `is_fraud`
  - `0` → Legitimate transaction
  - `1` → Fraudulent transaction

The dataset contains transaction details, merchant information, customer attributes, and transaction outcomes. 

## Project Workflow
### 1. Data Preprocessing
- Loaded the dataset using Pandas
- Removed personally identifiable and irrelevant columns such as:
  - Customer names, addresses, card numbers, transaction IDs, and timestamps
- Converted categorical features into numerical values using **Label Encoding**
- Prepared a clean dataset suitable for machine learning

### 2. Exploratory Data Analysis (EDA)
Visualizations were created to better understand the dataset:
- **Fraud Distribution Pie Chart**
  - Shows the imbalance between fraudulent and non-fraudulent transactions
- **Correlation Heatmap**
  - Displays relationships between numerical features
  - Weak correlations were filtered out for clarity
These visualizations help identify patterns and relationships within the data.

### 3. Feature Scaling & Data Splitting
- Split the data into **training (70%)** and **testing (30%)** sets
- Standardized numerical features using StandardScaler to improve model performance

## Machine Learning Models
Three supervised classification models were trained and evaluated:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
Each model was trained on the same preprocessed data for fair comparison.

## Model Evaluation
Models were evaluated using:
- **Classification Report**
  - Precision
  - Recall
  - F1-score
- Predictions on unseen test data
This allows for a clear comparison of model performance in identifying fraudulent transactions.

## Technologies & Libraries Used
- **Python**
- **Pandas & NumPy** – Data handling and numerical operations
- **Scikit-learn** – Machine learning models and evaluation
- **Matplotlib & Seaborn** – Data visualization

## Results
- Successfully trained and compared multiple models for fraud detection
- Random Forest demonstrated strong performance due to its ability to capture complex patterns
- Logistic Regression provided an interpretable baseline model
- Decision Tree offered simplicity and explainability

## Conclusion
This project demonstrates a complete machine learning pipeline, including data preprocessing, exploratory analysis, feature engineering, model training, and evaluation. It highlights practical skills in applying supervised learning techniques to real-world financial fraud detection problems.
