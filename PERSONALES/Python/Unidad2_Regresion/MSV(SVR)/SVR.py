# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 18:43:46 2021

@author: Icuerec
"""
#Máquina soporte vectorial
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR


df = pd.read_csv('Unidad2_Regresion/MSV(SVR)/Position_Salaries.csv')

#Transformo los datos tanto de X e y en float para que a la hora de regularizar, el SC se entrene bien (:S)
X = np.array(df.Level).reshape(-1,1).astype(float)
y = np.array(df.Salary).reshape(-1,1).astype(float)


#Indicamos que queremos crear un modelo de regresión con máquinas de soporte vectorial
#Le indicamos que el nucle en este caso lo queremos Gausiano (rbf)
regression = SVR(kernel='rbf')
regression.fit(X,y)

#Predecimos y vemos que algo falla
y_pred = regression.predict(np.array([6.5]).reshape(-1,1))

#Comprobamos el modelo visualmente
plt.scatter(X,y,color = 'red')
plt.plot(X,regression.predict(X).reshape(-1,1),color = 'blue')
plt.show()

#EScalamos a ver si es por eso

from sklearn.preprocessing import StandardScaler

Sc_X = StandardScaler()
Sc_Y = StandardScaler()
X = Sc_X.fit_transform(X)
y = Sc_Y.fit_transform(y)


#Creamos el nuevo modelo a ver si se ha solucionado
regression2 = SVR(kernel='rbf')
regression2.fit(X,y)

nivelEsc = Sc_X.transform([[7.5]]) 

#Predecimos y vemos que algo falla
y_pred2 = regression2.predict(nivelEsc)
#Visualizamos
plt.scatter(X,y,color = 'red')
plt.plot(X,regression2.predict(X).reshape(-1,1),color = 'blue')