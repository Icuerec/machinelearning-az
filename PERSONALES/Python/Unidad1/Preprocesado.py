# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 18:50:13 2021

@author: iwanc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Unidad1/Data.csv')

#Dividir dataset
x = df.iloc[:,:-1]

y = np.array(df.iloc[:,-1])

#Limpieza dataset
x['Age'].fillna(np.mean(x['Age']),inplace = True)
x['Salary'].fillna(np.mean(x['Salary']),inplace = True)

#Variables categóricas
df2 = pd.get_dummies(x['Country'])

x = x.join(df2)
del(x['Country'])

y = list(map(lambda x: 1 if x == 'Yes' else '0',y))
y = np.array(y,dtype=(int))

#Conjunto de entrenamiento y conjunto de testing

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size= 0.2, random_state=0)

#Estandarización del conjunto de datos X
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)




             