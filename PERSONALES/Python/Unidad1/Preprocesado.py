# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 18:50:13 2021

@author: iwanc
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Unidad1/Data.csv')

x = df.iloc[:,:-1]

y = np.array(df.iloc[:,-1])

x['Age'].fillna(np.mean(x['Age']),inplace = True)
x['Salary'].fillna(np.mean(x['Salary']),inplace = True)

df2 = pd.get_dummies(x['Country'])

x = x.join(df2)

del(x['Country'])
