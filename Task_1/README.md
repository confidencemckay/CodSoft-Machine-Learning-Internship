# Genre Prediction Model (Text Classification)
## Overview

This project implements a **text-based genre prediction system** using machine learning and Natural Language Processing (NLP) techniques. The model predicts the **genre of a movie or media item** based solely on its textual description.

By transforming text data into numerical features using **TF-IDF vectorization** and applying **Logistic Regression**, the project demonstrates an effective and interpretable approach to multi-class text classification.

## Dataset

The project uses two text-based datasets:
- **Training dataset**: 'train_data.txt'
  Contains:
  - ID
  - TITLE
  - GENRE (target label)

  ## DESCRIPTION

- **Test dataset**: 'test_data.txt'
  Contains:
  - ID
  - TITLE

  ## DESCRIPTION

  The files use a custom delimiter (:::) and are loaded using Pandas.

## Objective
- Train a machine learning model to classify genres based on textual descriptions
- Evaluate model performance using validation data
- Predict genres for unseen test data
- Export predictions for further analysis or submission

## Project Workflow
### 1. Data Loading
- Reads training and test datasets from '.txt' files
- Assigns appropriate column names
- Handles custom delimiters using Python’s parsing engine

### 2. Data Preprocessing
- Extracts the **DESCRIPTION** field as the input feature
- Uses the **GENRE** column as the target label
- Splits training data into training and validation sets (80/20)

### 3. Feature Extraction (TF-IDF)
- Converts text descriptions into numerical vectors using **TF-IDF**
- Removes English stop words
- Limits vocabulary size to the top 5,000 features for efficiency

### 4. Model Training
- Trains a **Logistic Regression** classifier
- Uses a higher iteration limit to ensure convergence for multi-class data

### 5. Model Evaluation
The model is evaluated on validation data using:
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-score)

This provides insight into how well the model performs across different genres.

### 6. Test Data Prediction
- Applies the trained TF-IDF vectorizer to test descriptions
- Predicts genres for unseen data
- Saves results to a CSV file:
  'predicted_test_data.csv'

### Technologies Used
- **Python**
- **Pandas** – Data handling
- **Scikit-learn** – Machine learning & NLP
- **TF-IDF Vectorization** – Text feature extraction

### Output
- Console output showing validation accuracy and classification metrics
- CSV file containing predicted genres for test data

## Possible Improvements
- Experiment with advanced models (Naive Bayes, SVM, Transformers)
- Perform hyperparameter tuning
- Use n-grams for richer text representation
- Apply class balancing techniques
- Deploy as an API or web application

## Conclusion

This project demonstrates a complete **NLP text classification pipeline**, from raw text processing to model training, evaluation, and prediction. It is suitable for applications such as movie genre classification, document categorization, and content tagging.
