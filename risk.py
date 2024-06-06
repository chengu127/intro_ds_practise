#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 22:54:57 2024

@author: chengu
"""


# provides the relative frequencies of distinct values in a sample. It is a non-parametric estimator of the probability mass function (PMF).
import csv
import matplotlib.pyplot as plt
data =[]
header =[]
with open('data/portland.csv', mode='r') as f:
    reader =csv.reader(f)
    header =tuple(next(reader))
    for row in reader:
        try:
            data.append((int(row[0]),int(row[1]),int(row[2])))
        except Exception as e:
            print(e)
import numpy as np
D = np.array(data)
x=D[:,0]
y=D[:,2]



    
#linear regression
r =lambda a: np.mean(np.power((a[0]*x+a[1]-y),2))
import scipy.optimize as so
result= so.minimize(r,(0,0),method ='Nelder-Mead')
x_pred = np.linspace(np.min(x),np.max(x),2)
y_pred = x_pred*result['x'][0]+result['x'][1]
plt.scatter(x,y)
plt.plot(x_pred,y_pred,color='green')