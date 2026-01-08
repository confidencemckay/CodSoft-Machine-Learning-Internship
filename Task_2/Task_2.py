# Credit Card Fraud Detection Model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from matplotlib import pyplot as plt

# Deployment
import joblib

# Load the dataset
df = pd.read_csv("fraudTrain.csv")

# Data Processing
# Drop columns not necessary
df.drop(["Unnamed: 0", 'cc_num', 'first', 'last', 'street', 'city', 'state', 'zip', 'dob', 'trans_num',
         'trans_date_trans_time'], axis=1, inplace=True)

# Handle categorical columns into numerical data
df['merchant'] = LabelEncoder().fit_transform(df['merchant'])
df['category'] = LabelEncoder().fit_transform(df['category'])
df['gender'] = LabelEncoder().fit_transform(df['gender'])
df['job'] = LabelEncoder().fit_transform(df['job'])

# Visualize data
# Pie chart
exit_counts = df["is_fraud"].value_counts()
plt.figure(figsize=(7, 7))
plt.subplot(1, 2, 1)  # Subplot for the pie chart
plt.pie(exit_counts, labels=["NO", "YES"], autopct="%0.0f%%")
plt.title("Pie Chart of Fraud Counts")
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

# Heatmap
pd.options.display.float_format = "{:,.2f}".format

corr_matrix = df.corr(method='pearson')

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
corr_matrix[(corr_matrix < 0.3) & (corr_matrix > -0.3)] = 0

cmap = "mako"

# the heatmap
sns.heatmap(corr_matrix, mask=mask, vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot_kws={"size": 9, "color": "black"}, square=True, cmap=cmap, annot=True)
plt.show()

# Split the data into training and testing sets
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Evaluation
lr_pred = lr_model.predict(X_test)
print("Logistic Regression Results:")
print(classification_report(y_test, lr_pred))

# Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Evaluation
dt_pred = dt_model.predict(X_test)
print("\nDecision Tree Results:")
print(classification_report(y_test, dt_pred))

# Random Forest model
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)

# Evaluation
rf_pred = rf_model.predict(X_test)
print("\nRandom Forest Results:")
print(classification_report(y_test, rf_pred))

# Deployment
joblib.dump(rf_model, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")
