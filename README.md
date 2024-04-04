# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import all the packages that helps to implement Decision Tree.
2. Download and upload required csv file or dataset for predecting Employee Churn
3. Initialize variables with required features.
4. And implement Decision tree classifier to predict Employee Churn.

## Program:

```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Gokul J
RegisterNumber: 212222230038  
```

```python
import pandas as pd
data=pd.read_csv('/content/Employee_EX6.csv')
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

# Output:

## Dataset:

![318748631-5080869e-abbe-4fee-890d-e1084b8eb802](https://github.com/Gokul0117/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121165938/d1fb7bdb-8d28-4995-8584-8db63caa6344)

## Accuracy :

![318748407-64074082-f9ef-459f-8051-de6953ed0cdc](https://github.com/Gokul0117/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121165938/80a6848a-0031-4882-a2c5-b8dc51ff6aeb)

## dt.predict:
![318742279-c9e7f9e5-0153-4941-b72f-4138123858a3](https://github.com/Gokul0117/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121165938/c1ff186c-02dd-479f-90b0-d8669ccf308a)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
