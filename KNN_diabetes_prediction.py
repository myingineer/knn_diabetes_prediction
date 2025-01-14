import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Load the dataset
df = pd.read_csv('diabetes_data.csv')

# Replace the missing values with 0
replace_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for columns in replace_zero:
    df[columns] = df[columns].replace(0, np.nan) # Replace 0 with NaN
    mean = int(df[columns].mean(skipna=True)) # Calculate the mean and skip NaN
    df[columns] = df[columns].replace(np.nan, mean) # Replace NaN with the mean. This way we replace with the average value of the column

# Split the dataset into features and target
X = df.iloc[:, 0:8] # Features
y = df.iloc[:, 8] # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc = StandardScaler() # Standardize features by removing the mean and scaling to unit variance
X_train = sc.fit_transform(X_train) # Fit to data, then transform it
X_test = sc.transform(X_test) # Perform standardization by centering and scaling

# Train the KNN model 
knn = KNeighborsClassifier(n_neighbors=11, p=2) # p=2 is for Euclidean distance
knn.fit(X_train, y_train) # Fit the model using X as training

# Predict the test set results
y_pred = knn.predict(X_test) # Predict the class labels for the provided data

# Evaluate the model
cm = confusion_matrix(y_test, y_pred) # Compute confusion matrix to evaluate the accuracy of a classification
print("Confusion Matrix: ")
print(cm)

accuracy = accuracy_score(y_test, y_pred) # Compute the accuracy
print("Accuracy: ", accuracy)

classification_report = classification_report(y_test, y_pred) # Build a text report showing the main classification metrics
print("Classification Report: ")
print(classification_report)

# Confusion matrix heatmap
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
