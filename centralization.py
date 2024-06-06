#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 19:54:43 2024

@author: chengu
"""
from Utils import load_sms
import numpy as np
sms_data = load_sms()
interest_word =set(['free','prize'])
TF10 = {True:1, False:0}
z_obs = [TF10[not interest_word.isdisjoint([word.lower() for word in line[0].split(' ')])]for line in sms_data]
y_obs = [y for x,y in sms_data]

def epsilon_bernoulli(n,alpha):
    return np.sqrt(-1/(2*n)*np.log((alpha)/2))

#epsilon = epsilon_bernoulli(len(y_obs), 0.05)
#mean_y_obs = np.mean(y_obs)
#print("[%.3f,%.3f]" %(mean_y_obs-epsilon, mean_y_obs+epsilon))
#true probability of getting a spam emil
#This subset represents the conditional distribution of the true labels given that the condition is met, p(y=1|z=1)
y_mid_z = [y for z,y in zip(z_obs,y_obs) if z==1]
epsilon = epsilon_bernoulli(len(y_mid_z), 0.05)
mean_y_obs = np.mean(y_mid_z)
print("[%.3f,%.3f]" %(mean_y_obs-epsilon, mean_y_obs+epsilon))


