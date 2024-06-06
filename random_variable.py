#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:02:42 2024

@author: chengu
"""

from Utils import load_sms
sms_data = load_sms()
print(sms_data[:2])
interesting_words = set(['free','prize'])
TF10={True:1, False:0}
Z_obs=[TF10[not interesting_words.isdisjoint(word.lower() for word in line[0].split(' '))] for line in sms_data]
print(Z_obs)
Y_obs =[y for x,y in sms_data]
import numpy as np
def F_X_12(x):
    TF10 ={True:1, False :0}
    return np.mean([TF10[(x1 <= x[0]) and (x2 <= x[1])] for x1,x2 in zip (Y_obs,Z_obs)])
for x1 in range(0,2):
    for x2 in range(0,2):
        print(F_X_12((x1,x2)),end=',\t')
    print('\n')
        