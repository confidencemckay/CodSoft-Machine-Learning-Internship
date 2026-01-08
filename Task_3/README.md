# Customer Churn Prediction Model
### Overview

This project focuses on predicting customer churn using machine learning techniques. Customer churn refers to customers who stop using a company’s service. By identifying patterns associated with churn, businesses can take proactive steps to improve customer retention.
The model uses historical customer data to classify whether a customer is likely to exit (churn) or remain with the company.

## Dataset
- Source: `Churn_Modelling.csv`
- Target Variable: `Exited`
  - `0` → Customer stayed
  - `1` → Customer exited

The dataset includes customer demographic information, account details, and financial attributes such as:
- Geography
- Gender
- Age
- Credit score
- Balance
- Estimated salary
- Number of products
- Tenure

## Key Steps in the Project
### 1. Data Exploration & Cleaning
- Inspected dataset structure and summary statistics
- Checked for null values and duplicates
- Removed unnecessary columns (`RowNumber`, `CustomerId`, `Surname`)
- Converted numerical fields to appropriate data types

### 2. Data Visualization
Exploratory Data Analysis (EDA) was performed using **Seaborn** and **Matplotlib**, including:
- Gender distribution vs churn status
- Overall churn distribution
- Geographic distribution of customers
- Class imbalance visualization (before and after resampling)
- These visualizations help understand customer behavior and churn trends.

### 3. Data Preprocessing
- Encoded categorical variables (`Gender`, `Geography`) using Label Encoding
- Addressed class imbalance using downsampling
- Standardized numerical features using StandardScaler
- Split data into training and testing sets (80/20 split)

## Machine Learning Models Used
The following supervised learning models were trained and evaluated:
- **Decision Tree Classifier** 
- **Logistic Regression** 
- **Random Forest Classifier** 
- Gradient Boosting Classifier** 
Each model was evaluated using:
- Accuracy Score
- Classification Report (Precision, Recall, F1-score)

## Model Evaluation & Comparison
Model performance was compared based on accuracy, and results were summarized in a comparison table for easy analysis.
The comparison helps determine which algorithm performs best for churn prediction on this dataset.

## Technologies & Libraries
- **Python**
- **Pandas & NumPy** – Data handling
- **Matplotlib & Seaborn** – Data visualization
- **Scikit-learn** – Machine learning models and evaluation

## Results
- Successfully built and evaluated multiple classification models
- Addressed class imbalance to improve predictive performance
- Identified Gradient Boosting and Random Forest as strong performers for churn prediction

## Possible Improvements
- Hyperparameter tuning for better performance
- Feature importance analysis
- Use of SMOTE for advanced imbalance handling
- Deployment as a web app or API
- Integration with real-time customer data

## Conclusion
This project demonstrates a complete machine learning workflow, from data preprocessing and visualization to model training, evaluation, and comparison. It provides practical insights into customer churn prediction and serves as a strong foundation for real-world business analytics and predictive modeling.
