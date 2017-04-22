# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:21:14 2017

@author: Administrator
"""

'''
积分
第一次的时候，赋值出错，a=0.5 b=0.5,结果运行报错：ValueError: math domain error
经网络搜索，是说运算错误，比如把0放在除数那里了。仔细检查，才发现是b赋值错误，这样a=b，区间是0啊！后来改成b=9.5就ok了

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.integrate as sci
from matplotlib.patches import Polygon

def f(x):
    return np.sin(x)+0.5*x

a=0.5
b=9.5
x=np.linspace(0,10)
y=f(x)


fig,ax=plt.subplots(figsize=(7,5))
plt.plot(x,y,'b',linewidth=2)
plt.ylim(ymin=0)

#最低、最高区域
Ix=np.linspace(a,b)
Iy=f(Ix)
verts=[(a,0)]+list (zip(Ix,Iy))+[(b,0)]
poly=Polygon(verts,facecolor='0.7',edgecolor='0.5')
ax.add_patch(poly)

#lbaels 表
plt.text(0.75*(a+b),1.5,r'$\int_a^b f(x)dx$',horizontalalignment='center',fontsize=20)
plt.figtext(0.9,0.075,'$x$' )
plt.figtext(0.075,0.9,'$f(x)$')

ax.set_xticks((a,b))
ax.set_xticklabels('ab')
ax.set_yticks([f(a),f(b)])
   
#数值积分 古董高斯求积 自适应求积 与龙贝格积分
   
sci.fixed_quad(f,a,b)[0]
sci.quad(f,a,b)[0]
sci.romberg(f,a,b)

#通过模拟求取积分
for i in range(1,200):
    np.random.seed(1000)
    x=np.random.random(i*10)*(b-a)+a
    print (np.sum(f(x))/len(x)*(b-a))
    
