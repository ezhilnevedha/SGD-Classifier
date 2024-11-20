# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Dataset  
2. Inspect and Explore the Dataset  
3. Drop Irrelevant Columns  
4. Convert Categorical Variables to Numeric  
5. Encode Categorical Variables with Integer Codes  
6. Prepare Features (`X`) and Target (`Y`) Variables  
7. Define Logistic Regression Functions: Sigmoid, Loss, Gradient Descent  
8. Train the Logistic Regression Model Using Gradient Descent  
9. Predict and Evaluate Model Accuracy  
10. Test Model on New Data Points  

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Ezhil Nevedha.K
RegisterNumber:  212223230055
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_clf = SGDClassifier(max_iter = 1000, tol = 1e-3)
sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

## Output:
![image](https://github.com/user-attachments/assets/6627ec3c-7911-4cc6-a3f8-5a8e5fbcc48f)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
