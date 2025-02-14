import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Load student data
training_data = pd.read_csv('student_performance.csv')

# Drop the target label from the dataframe
X_train = training_data.drop(columns=['Passed Final Exam'])
# The target label
y_train = training_data['Passed Final Exam']

# Create a dummy for the invested effort column
X_train = pd.get_dummies(X_train)

# Create a Gaussian Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)

# Prepare the test data
test_data = pd.read_csv('test_student_data.csv')
X_test = test_data.drop(columns=['Passed Final Exam'])
X_test = pd.get_dummies(X_test)
y_test = test_data['Passed Final Exam']

# Make predictions
y_predict = model.predict(X_test)
print("Predictions: ")
print(y_predict)
print("Actual: ")
print(y_test.values)

# Confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion matrix: ")
print(confusion_matrix(y_test, y_predict))

# Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy: ")
print(accuracy_score(y_test, y_predict))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("student_performance.csv")

# Set plot style
sns.set(style="whitegrid")

# Countplot for Invested Effort
plt.figure(figsize=(8, 5))
sns.countplot(x='Invested Effort', data=df, order=['low', 'medium', 'high'], palette='viridis')
plt.title("Distribution of Invested Effort")
plt.xlabel("Invested Effort")
plt.ylabel("Count")
plt.show()

# Bar plot for pass rates by effort level
plt.figure(figsize=(8, 5))
sns.barplot(x='Invested Effort', y='Passed Final Exam', data=df, order=['low', 'medium', 'high'], palette='coolwarm')
plt.title("Final Exam Pass Rate by Effort Level")
plt.xlabel("Invested Effort")
plt.ylabel("Pass Rate")
plt.show()

# Stacked bar plot for Passed Last Exam vs Passed Final Exam
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Passed Last Exam", hue="Passed Final Exam", multiple="stack", discrete=True, shrink=0.8, palette="magma")
plt.title("Passed Last Exam vs Passed Final Exam")
plt.xlabel("Passed Last Exam")
plt.ylabel("Count")
plt.xticks([0, 1], ["Failed", "Passed"])
plt.legend(title="Passed Final Exam", labels=["Failed", "Passed"])
plt.show()