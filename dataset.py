import pandas as pd # used for data manipulation
import numpy as np # used for mathematical operations
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve


# Load dataset
file_path = "German Credit Card Data Small Set- Assignment 2-15-20 (1) (9)(1).xlsx" # relative path of the credit card xlsx file
df = pd.read_excel(file_path) # read the dataset
df.columns = df.columns.astype(str)  # Convert all column names to strings

# Split dataset
train_data = df.iloc[:750]  # First 750 rows for training
test_data = df.iloc[750:]  # Remaining rows for testing

# Define independent (X) and dependent (y) variables for training and test data
X_train = train_data.iloc[:, :-1]  # Columns A-G
Y_train = train_data.iloc[:, -1]   # Column H (dependent variable)

X_test = test_data.iloc[:, :-1]
Y_test = test_data.iloc[:, -1]

# Train a model (Logistic Regression)
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, Y_train)

# get probability predictions
y_probs = model.predict(X_test)

# Make predictions
Y_pred = model.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(Y_test, Y_pred, labels=[1, 2])

# Penalty scoring
penalty_score = cm[0, 1] * 1 + cm[1, 0] * 5  # Misclassification penalties

# Print results and classification report to the console for testing
print("Confusion Matrix:\n", cm)
print("Penalty Score:", penalty_score)
print("\nClassification Report")
print(classification_report(Y_test, Y_pred))

# Using the Seaborn library, visualize the confusion matrix as a heatmap
sns.heatmap(cm, annot=True, fmt="d", cbar=True, cmap=sns.cubehelix_palette(as_cmap=True), xticklabels=["Good", "Bad"], yticklabels=["Good", "Bad"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()