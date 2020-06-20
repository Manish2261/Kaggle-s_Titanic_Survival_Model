# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 19:02:05 2020

@author: Lenovo
"""
#Importing the Libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb

#Importing the Dataset:
train = pd.read_csv('train.csv')
test  =pd.read_csv('test.csv')
y = train.loc[:,'Survived'].values

#Data Exploratory Analysis:

#Check Missing Values:
#print(train.isna().sum())
#print(test.isna().sum())

#Dropping the Column:
#Feature: PassengerId, Survived, Ticket :
train = train.drop(labels = ['PassengerId', 'Survived', 'Ticket'], axis = 1)
test  = test.drop(labels = ['PassengerId', 'Ticket'], axis = 1)

#Handling the Missing Values:

#Training data
#Featurre: Age :>> Mean Values
train['Age'] = train['Age'].fillna(train['Age'].mean())
#Feature: Cabin :>> 0 or 1
train['Cabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1 )
#Feature: Embarked :>> 0,1,2
#sdf = train.groupby('Embarked').size()
train['Embarked'] = train['Embarked'].fillna('S')

#Testing Values:
#Feature:Age :>>Mean Values
test['Age'] = test['Age'].fillna(test['Age'].mean())
#Feature: Cabin :>> 0,1:
test['Cabin'] = test['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
#Feature: Fare :>> Mean value:
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

#C#heck Missing Values:
#print(train.isna().sum())
#print(test.isna().sum())

#Encoding the Categorical Features:
#Feature: Sex : LabelEncoding
#Training Data:
le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
#Testing Data:
test['Sex'] = le.fit_transform(test['Sex'])

#Mapping the Embarked column with Key Values:
#Feature:>> Embarked >>> 0,1,2
#Training Data
map_emb = {'S' : 0, 'Q': 1, 'C':2}
train['Embarked'] = train['Embarked'].map(lambda d: map_emb[d])
#Testing Data:
test['Embarked'] = test['Embarked'].map(lambda d: map_emb[d])

#Abstracting the title from the 'Name' feature:

#Training Data:
#Feature: Name =>> Titles with values 0,1,2
##Split Names to Extract Titles:
train['title'] = train.Name.str.split(", ", n=1, expand = True)[1].str.split(". ", n=1, expand = True).iloc[:,0]
tr = train.title.unique()
##Replace title with the respective codes:
map_tit = {'Master':0,'Miss':0, 'Miss':0, 'Don':0, 'Dr':0, 'Mr':1, 'Rev':1, 'Major':1,'Lady': 2}
train['title'] = train['title'].map(lambda x: map_tit[x] if x in map_tit else 3)
##Dropping the name Column:
train = train.drop(labels = ['Name'], axis = 1)

#Testing Data:
#Feature:Name=>> Titles with 0,1,2>>>
test['title'] = test.Name.str.split(", ", n=1, expand = True)[1].str.split(". ", n=1, expand = True).iloc[:,0]
test['title'] = test['title'].map(lambda x: map_tit[x] if x in map_tit else 3)
test = test.drop(labels = ['Name'], axis = 1)

#Defining Variable for Training and Testing Features:
x_train = train.iloc[:,:].values
x_test  = test.iloc[:,:].values

#Feature Scaling of data:
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
y       = sc.fit_transform(y.reshape(-1,1))
x_test  = sc.transform(x_test)


#Part - 3rd : Training of the Model:

"""
#Using SVC Classifier:
regressor = SVC(kernel = 'rbf' )
regressor.fit(x_train,y)
y_pred = regressor.predict(x_test)
# >>> With Plain SVM got an accuracy of 66.507% on test data

#using RFC Classifier:
regressor_1 = RandomForestClassifier(n_estimators = 700, criterion = 'entropy')
regressor_1.fit(x_train,y)
y_pred = regressor_1.predict(x_test)
# >>>Using RFC, accuracy hits 76.076%

#Using AdaBoost Classifier:
regressor = AdaBoostClassifier(n_estimators = 5000, learning_rate = 0.30, random_state = 5)
regressor.fit(x_train,y)
y_pred = regressor.predict(x_test)
# >>> Using ABC, accurcay is 76.55%

#Using AdaBoost Classifier:
regressor = ExtraTreesClassifier(n_estimators = 5000,  random_state = 5)
regressor.fit(x_train,y)
y_pred = regressor.predict(x_test)
# >>> Using ETC, accurcay is 74.162%

#Using GradientBoosting Classsifier:
regressor = GradientBoostingClassifier(n_estimators = 5000,  random_state = 5)
regressor.fit(x_train,y)
y_pred = regressor.predict(x_test)
# >>> Using GBC, accurcay is 70.334%

#Using XGBoosting Claasifier:
regressor = xgb.XGBClassifier()
regressor.fit(x_train,y)
y_pred = regressor.predict(x_test)
# >>> Using XGB, accurcay is 72.727%

#Ensembling RFC and XGB Classifiers:
regressor = RandomForestClassifier(n_estimators = 700, criterion = 'entropy')
regressor = xgb.XGBClassifier()
regressor.fit(x_train,y)
y_pred = sc.inverse_transform(regressor.predict(x_test))
# >>> Using GBC, accurcay is 59.330%
"""



