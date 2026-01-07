# Genre Prediction Model

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load train and test datasets
train_data_file = "train_data.txt"
test_data_file = "test_data.txt"

train_df = pd.read_csv(train_data_file, sep=" ::: ", engine='python', header=None, names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
test_df = pd.read_csv(test_data_file, sep=" ::: ", engine='python', header=None, names=['ID', 'TITLE', 'DESCRIPTION'])

# Preprocessing
X = train_df['DESCRIPTION']
y = train_df['GENRE']

# Split the train data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# Train the model using Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# Predict on validation data
y_pred = model.predict(X_val_tfidf)

# Evaluation
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred, zero_division=0))

# Test data preprocessing and prediction
test_descriptions = test_df['DESCRIPTION']
test_tfidf = tfidf_vectorizer.transform(test_descriptions)
test_predictions = model.predict(test_tfidf)

# Save test predictions
test_df['PREDICTED_GENRE'] = test_predictions
test_df.to_csv("predicted_test_data.csv", index=False)
print("Predictions saved to predicted_test_data.csv")
