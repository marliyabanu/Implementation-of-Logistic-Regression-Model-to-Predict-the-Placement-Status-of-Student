# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program and import required libraries (pandas, sklearn).

2.Create or load dataset and separate it into

     * Features (CGPA, IQ, Projects) → X
     
     * Target (Placement status) → y
     
3.Split the dataset into training and testing sets.

4.Train the Logistic Regression model using the training data.

5.Predict and evaluate the model using test data and display the placement result.


## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: B MARLIYA BANU B
RegisterNumber:  212225040229
*/
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Create dataset (no CSV needed)
data = pd.DataFrame({
    'cgpa': [7.5, 6.2, 8.0, 5.5, 7.8, 6.5, 7.0, 6.8, 8.2, 5.9],
    'iq': [120, 100, 130, 90, 125, 105, 115, 110, 135, 95],
    'projects': [2, 1, 3, 0, 2, 1, 2, 1, 3, 0],
    'placed': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
})

# Features (X) and Target (y)
X = data[['cgpa', 'iq', 'projects']]
y = data['placed']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Predict for new student
new_student = pd.DataFrame([[7.2, 115, 2]], columns=['cgpa', 'iq', 'projects'])
result = model.predict(new_student)

# Output result
if result[0] == 1:
    print("Student is likely to be Placed")
else:
    print("Student is NOT likely to be Placed")
```

## Output:
<img width="401" height="127" alt="WhatsApp Image 2026-05-11 at 9 12 53 AM" src="https://github.com/user-attachments/assets/8380df17-f8b4-4f67-a970-5c19f56b36da" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
