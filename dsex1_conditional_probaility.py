#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 23:49:08 2024
calculate the conditional probability P(X=1| Y=1)
x is if the text is spam or not
y is contain the word "free"

@author: chengu
"""

import csv
def load_sms():
    lines =[]
    hamspam ={'ham':0, 'spam':1}
    with open ('data/spam.csv', mode ='r', encoding ='latin-1') as f:
        reader = csv.reader(f)
        header = next(reader)
        lines = [(line[1],hamspam[line[0]]) for line in reader]
    return lines

sms_data = load_sms()


#p(y=1) contains the word "free"
#check if the word free is in the test message
#count the probability sum(numbers of word free occurs /len(probability))
def check_free(txt):
    if "free" in txt:
        return 1
    return 0

sms_free = [check_free(txt) for txt, label in sms_data]
total_sms_free=sum(sms_free)
                  
proby = total_sms_free/len(sms_free)
print(proby)

#calculate probability x to check is the text is spam or not
X = [label for txt,label in sms_data] 
Y = sms_free
x_y = [x*y for x,y in zip(X,Y)]   
final_prob = sum(x_y)/len(x_y)
print(final_prob)
print("conditional prob", final_prob/proby)