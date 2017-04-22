# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:12:43 2017

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
import random
position=0
walk=[position]
steps=1000
for i in range(steps):
    step=1 if random.randint(0,1) else -1
    position += step
    walk.append(position)
    
nsteps=1000
nwalks=5000
draws=np.random.randint(0,2,size=(nwalks,nsteps))

steps=np.where(draws >0,1,-1)
walks=steps.cumsum(1)
        
plt.plot(walks)    