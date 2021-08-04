# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 20:51:19 2021

@author: Icuerec
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importa el dataset, limpialo, separa en variables dependientes e independientes y crea el conjunto de entrenamiento y el de test.

df = pd.read_csv('')

#Limpieza

#Separación

#Conjuntos test y train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=int, test_size=float)


#Ajusta el modelo de regresión
regression = 
regression.fit

#Predecir nuestros datos
y_pred = regression.predict()

#Visualizar los datos 