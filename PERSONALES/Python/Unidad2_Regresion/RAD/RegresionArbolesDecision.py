# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 19:20:48 2021

@author: Icuerec
"""
#Regresión con arboles de decisión

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('Unidad2_Regresion/RAD/Position_Salaries.csv')

X = np.array(df.Level).reshape(-1,1).astype(float)
y = np.array(df.Salary).reshape(-1,1).astype(float)

regression = DecisionTreeRegressor(random_state=10)
regression.fit(X,y)

y_pred = regression.predict([[6.5]])

#Visualizamos el modelo
plt.scatter(X,y,color = 'red')
plt.plot(X,regression.predict(X).reshape(-1,1),color = 'blue')
plt.show()

#En este caso hay un overfitting obvio. Realmente crea escalones con la media entre los dos puntos. Veamoslo
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(-1,1)
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regression.predict(X_grid).reshape(-1,1),color = 'blue')