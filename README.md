# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the process
2. Data Preparation
3. Hypothesis Definition
4. Cost Function
5. Parameter Update Rule
6. Iterative Training
7. Model Evaluation
8. End the process
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: V. YOGESH
RegisterNumber:  212223230250
*/
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
```
![image](https://github.com/user-attachments/assets/f3e2e149-6cd1-42e7-84e9-27eb0b2dd2b2)
```
df.info()
```
![image](https://github.com/user-attachments/assets/b2ee5be0-0edf-4d7d-86b1-ecdb956fcb97)
```
X=df.drop(columns=['AveOccup','target'])
X.info()
```
![image](https://github.com/user-attachments/assets/064ec120-2a08-460e-a712-d0226d9b8f5b)
```
Y=df[['AveOccup','target']]
Y.info()
```
![Screenshot 2024-09-11 093027](https://github.com/user-attachments/assets/e7bddeb2-af38-4368-99e8-48dca4339e93)
```
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
X.head()
```
![Screenshot 2024-09-11 093126](https://github.com/user-attachments/assets/10fc1bdb-1044-4c6b-8d62-9b8f291b4f48)
```
scaler_X=StandardScaler()
scaler_Y=StandardScaler()

X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
print(X_train)
```
![image](https://github.com/user-attachments/assets/0f022b00-4b20-46ec-b790-793d86872fed)
```
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])
```
![image](https://github.com/user-attachments/assets/ca4fc763-e2b9-4df4-823b-1cb85b48dcc8)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
