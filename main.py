# Customer Churn Prediction Model

# Import libraries
import pandas as pd
# Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
# Models Libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# Metrics Libraries
from sklearn.metrics import accuracy_score, classification_report
# Data Processing Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("Churn_Modelling.csv")

# Dataset Information and Samples
df.info()
df.head(5)

# Exited Count
is_Exited = df["Exited"].value_counts()
print(f"\nCount value of 'Exited' customers: \nYes: {is_Exited[1]} \nNo: {is_Exited[0]}")

# Search for Null values and Duplicates
print(f"\nNull value count: {df.isna().sum().sum()}")
print(f"Duplicate value count: {df.duplicated().sum()}\n")

# Dataset Visualization
sns.set_style('whitegrid')
sns.set_palette('pastel')

fig, axb = plt.subplots(ncols=2, nrows=1, figsize=(15, 8))

# Exited Gender Distribution
explode = [0.1, 0.1]
df.groupby('Gender')['Exited'].count().plot.pie(explode=explode, autopct="%1.1f%%", ax=axb[0])

ax = sns.countplot(x="Gender", hue="Exited", data=df, ax=axb[1])

# Exited Bar Graph
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center',
                xytext=(0, 10), textcoords='offset points')

plt.title("Distribution of Gender with Exited Status")
plt.xlabel("Gender")
plt.ylabel("Count")

plt.show()

# Exited Counts Pie Chart
is_Exited = df["Exited"].value_counts()
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.pie(is_Exited, labels=["No", "Yes"], autopct="%0.0f%%")
plt.title("Exited Counts")

# Distribution of Geography Pie Chart
plt.subplot(1, 2, 2)
geography_counts = df['Geography'].value_counts()
plt.pie(geography_counts, labels=geography_counts.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Distribution of Geography')

plt.tight_layout()
plt.show()

# Data Processing
# Delete unnecessary Columns
df = df.drop(['RowNumber', 'Surname', 'CustomerId'], axis=1)

df['Balance'] = df['Balance'].astype(int)
df['EstimatedSalary'] = df['EstimatedSalary'].astype(int)

# Label Encoding and Fitting and Transformation of data
le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])
df['Geography'] = le.fit_transform(df['Geography'])

# Sampling and Standard Scaler
No_class = df[df["Exited"] == 0]
yes_class = df[df["Exited"] == 1]

No_class = resample(No_class, replace=False, n_samples=len(yes_class))
down_samples = pd.concat([yes_class, No_class], axis=0)

X = down_samples.drop("Exited", axis=1)
y = down_samples["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

original_class_counts = df["Exited"].value_counts()
downsampled_class_counts = down_samples["Exited"].value_counts()
original_percentages = original_class_counts / len(df) * 100
downsampled_percentages = downsampled_class_counts / len(down_samples) * 100

plt.figure(figsize=(12, 6))

# Bar chart for original class distribution
plt.subplot(1, 2, 1)
bars_1 = plt.bar(original_class_counts.index, original_class_counts.values, color=['orange', 'green'])
for bar, label in zip(bars_1, original_percentages):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f'{label:.2f}%', ha='center', va='bottom')
plt.title('Original Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(original_class_counts.index, ['Not Exited', 'Exited'])

# Bar chart for downsampled class distribution
plt.subplot(1, 2, 2)
bars_2 = plt.bar(downsampled_class_counts.index, downsampled_class_counts.values, color=['orange', 'green'])
for bar, label in zip(bars_2, downsampled_percentages):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f'{label:.2f}%', ha='center', va='bottom')
plt.title('Downsampled Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(downsampled_class_counts.index, ['Not Exited', 'Exited'])

plt.tight_layout()
plt.show()

# Model Training and Evaluation
## Decision Tree
DT = DecisionTreeClassifier(max_depth=5, random_state=0)
DT.fit(X_train, y_train)

# Generate Classification Report
predict_DT = DT.predict(X_test)
print(f"Decision Tree Classification Report: \n{classification_report(y_test, predict_DT)}")

# Calculate Accuracy
DT_accuracy = accuracy_score(predict_DT, y_test)
print('Decision Tree Model accuracy is: {:.3f}%'.format(DT_accuracy * 100))

## Logistic Regression
LR_model = LogisticRegression()
LR_model.fit(X_train, y_train)

# Generate Classification Report
predict_LR = LR_model.predict(X_test)
print(f"\nLogistic Regression Classification Report: \n{classification_report(y_test, predict_LR)}")

# Calculate Accuracy
LR_accuracy = accuracy_score(predict_LR, y_test)
print('Logistic Regression accuracy is: {:.3f}%'.format(LR_accuracy * 100))

## Random Forest
RF = RandomForestClassifier(n_estimators=60, random_state=0)
RF.fit(X_train, y_train)

# Generate Classification Report
predict_RF = RF.predict(X_test)
print(f"\nRandom Forest Classification Report: \n{classification_report(y_test, predict_RF)}")

# Calculate Accuracy
RF_accuracy = accuracy_score(predict_RF, y_test)
print('Random Forest model accuracy is: {:.3f}%'.format(RF_accuracy * 100))

## Gradient Boosting
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

gb_classifier.fit(X_train, y_train)
y_pred = gb_classifier.predict(X_test)

# Generate Classification Report
report = classification_report(y_test, y_pred)
print("\nGradient Boosting Classification Report:\n", report)

# Calculate Accuracy
gb_accuracy = accuracy_score(y_test, y_pred)
print('Gradient Boost model accuracy is: {:.3f}%'.format(gb_accuracy * 100))


# Model Comparison
Algorithms = ['Random Forest', 'Gradient Boosting', 'Decision Tree', 'Logistic Regression']
accuracy = [RF_accuracy, gb_accuracy, DT_accuracy, LR_accuracy]

FinalResult = pd.DataFrame({'Algorithm': Algorithms, 'Accuracy': accuracy})
print(FinalResult)

