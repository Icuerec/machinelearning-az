# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 18:26:24 2021

@author: Icuerec
"""
#Regresión lineal múltiple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import statsmodels.api as sm

df = pd.read_csv('Unidad2_Regresion/RLM/50_Startups.csv')

#Tratamos las categóricas y evitamos la multicorrelacionalidad eliminando una de las variables
dummiesStates = pd.get_dummies(df.State).iloc[:,:-1]
for column in dummiesStates.columns:
    dummiesStates[column] = dummiesStates[column].apply(lambda x: int(x))
        
X = df.drop(['State'],axis = 1)
X = X.join(dummiesStates)
X = X.drop('Profit',axis = 1)

y = df['Profit']

#Separamos el dataset en test y en entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state=0)

#Entrenamos el modelo y lo probamos
regression = LinearRegression()
regression.fit(X_train, y_train)

#Guardamos la predicción y la puntuación para comparar los resultados mas adelante.
predictionFull = regression.predict(X_test)
scoreFull = regression.score(X,y)

#------------------------Preparación del modelo usando la eliminación hacia atrás---------------------------------

#Creamos una columna para saber el coeficiente del término independiente y creamos el mínimo valor p que aceptaremos
X['Independiente'] = 1
SL = 0.05

#Creamos una función donde iremos eliminando las columnas con un pvalor superior al límite que hemos decidido dar. (Podemos sacar los pvalue mas fácilmente con pvalues... :S)
def eliminacionAtras (X,y,SL):
    
    maxCol = len(X.columns)
    X_act = X

    for n in range(maxCol):
        
        regression_OLS = sm.OLS(endog = y, exog = X_act).fit()
        mdl = regression_OLS.summary()
        '''
        print(mdl)
        print('******************************************')
        '''
        table = pd.DataFrame(mdl.tables[1])
        table.columns = table.iloc[0]
        table = table.drop(0,axis = 0)
        pvals = table.iloc[:,4].astype(str).astype(float)
        pvals = np.array(pvals[pvals != 1])
        pmax = pvals.max()
        
        if pmax >= SL:
            pmaxpos = int(np.where( pvals == pmax)[0])
            X_act = X_act.drop(X_act.columns[pmaxpos],axis = 1)
        else:
            break
     
    return X_act
        
X_opt = eliminacionAtras(X,y,SL)

#Comprobamos las predicciones y las puntuaciones de nuestro modelo reducido.

X_trainO, X_testO, y_trainO, y_testO = train_test_split(X_opt,y,test_size= 0.2, random_state=0)

#Entrenamos el modelo y lo probamos
regression = LinearRegression()
regression.fit(X_trainO, y_trainO)

predictionClean = regression.predict(X_testO)
scoreClean = regression.score(X_opt,y)
