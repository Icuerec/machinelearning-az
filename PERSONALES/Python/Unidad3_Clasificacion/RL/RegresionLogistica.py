# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 19:03:04 2021

@author: Icuerec
"""
#Regresión Logística

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

#importamos, estudiamos, limpiamos y dividimos en variables ind/dep el df
df = pd.read_csv('Unidad3_Clasificacion/RL/Social_Network_Ads.csv')

X = df.iloc[:,1:-1]
y = np.array(df.iloc[:,-1])

X['Gender'] = pd.get_dummies(X['Gender'])

#Escalamos los datos
sc_X = StandardScaler()
X.iloc[:,1:] = sc_X.fit_transform(X.iloc[:,1:])

#Dividimos en test y train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#Ajustar el modelo en el X_train
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

#Evaluamos el modelo con el X_test
y_pred = classifier.predict(X_test)

#Matriz de confusión y puntuación total
cm = confusion_matrix(y_test, y_pred)
score = classifier.score(X, y)

print('La puntuación es: ' + str(score))
print('')
print('Y la matriz de confusión...')
print(cm)
