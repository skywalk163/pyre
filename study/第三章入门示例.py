# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 14:11:41 2017

@author: Administrator
"""

import math
from numpy import *
import numpy as np
import pandas as pd
import pandas.io.data as web

from time import time 
random.seed(20000)
t0=time()

S0=100.;K=105.;T=1.0;r=0.05;sigma=0.2
M=50;dt=T/M;I=25000

S=S0*exp(cumsum((r-0.5*sigma**2)*dt+sigma*math.sqrt(dt)*random.standard_normal((M+1,I)),axis=0))

S[0]=S0

C0=math.exp(-r*T)*sum(maximum(S[-1]-K,0))/I

tnp2=time()-t0 
print ('European Option Value %7.3f' %C0)
print("Duration in Secconds %7.3f" %tnp2)

import matplotlib.pyplot as plt
fig=plt.subplots()
plt.plot(S[:,:10])
plt.grid(True)
plt.xlabel('time stemp')
plt.ylabel('index level')
fig=plt.subplots()
plt.hist(S[-1],bins=50)
plt.grid(True)
plt.xlabel('Index level')
plt.ylabel('frequency')

fig=plt.subplots()
plt.hist(np.maximum(S[-1] -K,0),bins=50)
plt.grid(True)
plt.xlabel('option inner value')
plt.ylabel('frequency')
plt.ylim(0,50000)

print('到期时没有价值的为',sum(S[-1]<K))

'''
技术分析
'''
sp500=web.DataReader('^GSPC',data_source='yahoo',start='1/1/1990',end='3/14/2017')
print(sp500.info())
fig=plt.subplots()
sp500['Close'].plot(grid=True ,figsize=(8,5))

