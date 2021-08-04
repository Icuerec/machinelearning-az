# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 18:55:22 2021

@author: Icuerec
"""

#REGRESION POLINÓMICA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Unidad2_Regresion/RP/Position_Salaries.csv')

X = np.array(df.Level).reshape(-1,1)

y = np.array(df.Salary).reshape(-1,1)

#No se puede generar conjunto de testing puesto que hay pocos datos.

#Ajustar una regresión lineal simple
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Crear un modelo de regresión polinómica.

#Importar el método para ajustar la regresion poliómica 
from sklearn.preprocessing import PolynomialFeatures

#Transformamos nuestro dataset según hasta que grado queremos ajustar
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)

#Y ahora entrenamos el modelo de regresión lineal
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#Visualización de la comparativa de los resultados de los modelos
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.plot(X,lin_reg_2.predict(X_poly),color = 'green')
plt.legend(['Referencias','Linear Regression','Polynomial Regression'])
plt.title('Linear Regression VS. Polynomial Regressión')
plt.xlabel('Salary')
plt.ylabel('Level')
plt.show()

#Probamos ahora con uno de grado 3
poly_reg3 = PolynomialFeatures(degree = 3)
X_poly3 = poly_reg3.fit_transform(X)

lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly3,y)

#Comparamos resultados con el de grado 2
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg_2.predict(X_poly),color = 'green')
plt.plot(X,lin_reg_3.predict(X_poly3),color = 'Orange')
plt.legend(['Referencias','PR Grade2', 'PR Grade3'])
plt.title('Polynomial Regressión: Grade 2 vs Grade 3')
plt.xlabel('Salary')
plt.ylabel('Level')

prediccion6ymediolinear = lin_reg.predict(np.array([6.5]).reshape(-1,1))
prediccion6ymediogrado3 = lin_reg_3.predict(poly_reg3.fit_transform(np.array(6.5).reshape(-1,1)))
prediccion6ymediogrado2 = lin_reg_2.predict(poly_reg.fit_transform(np.array(6.5).reshape(-1,1)))
