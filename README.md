# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
Developed by: THARUN SRIDHAR
RegisterNumber: 212223230230
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```

## Output:

**HEAD**

![image](https://github.com/user-attachments/assets/171aeaf0-4ea5-4264-8c3c-76ac347abbd4)

**TAIL**

![image](https://github.com/user-attachments/assets/c73dc8e8-7a93-4d90-9c94-46acbd70a75f)

**ARRAY VALUE OF X**

![image](https://github.com/user-attachments/assets/ae076d3f-377d-4c02-9c7c-d971d914ff39)

**ARRAY VALUE OF Y**

![image](https://github.com/user-attachments/assets/6e9fb846-6781-425c-81d0-0ec7ed2d0a5f)

**VALUES OF Y PREDICTION**

![image](https://github.com/user-attachments/assets/b6cf8083-823e-4c99-9b0e-f76798dde8a1)

**ARRAY VALUE OF Y TEST**

![image](https://github.com/user-attachments/assets/0281f60c-0393-482c-a930-547e89ccf3ae)

**TRAINING SET GRAPH**

![image](https://github.com/user-attachments/assets/7c5a2d63-7cc4-4b65-9ade-08b92036ab63)

**TEST SET GRAPH**

![image](https://github.com/user-attachments/assets/315f454d-e08c-4ee8-87f7-4f9cc1146768)

**VALUES OF MSE, MAE AND RSME**
![image](https://github.com/user-attachments/assets/546780ad-5b2c-4908-ac4f-a886d2cdb2a4)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
