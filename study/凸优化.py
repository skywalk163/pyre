# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 22:00:35 2017

@author: Administrator
凸优化
我py3 版本里，定义函数时不能用2个括号，即def fm((x,y)):
所以后面都用的一个括号。
fo函数直接用单数据传入。

在后期调用fm的时候，可以用指针把tuple、list、array等传入，如：fm(*opt1)
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spo
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


def fm(x,y):
    return (np.sin(x))+0.05*x**2+np.sin(y)+0.05*y**2
    
x=np.linspace(-10,10,50)
y=np.linspace(-10,10,50)
X,Y=np.meshgrid(x,y)
Z=fm(X,Y)

fig=plt.figure(figsize=(9,6))
ax=fig.gca(projection='3d')
surf=ax.plot_surface(X,Y,Z,rstride=2,cstride=2,cmap=mpl.cm.coolwarm,linewidth=0.5,antialiased=True)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
fig.colorbar(surf,shrink=0.5,aspect=5)
#按照书上例题，过不了，修改成fo（xy），x,y=xy 才ok
def fo(xy):
    x,y=xy
    z=np.sin(x)+0.05*x**2+np.sin(y)+0.05*y**2
    if output==True:
        print('%8.4f %8.4f %8.4f' %(x,y,z))
    return z

output=True
rranges = slice(-10, 10.1, 5),slice(-10, 10.1, 5)
spo.brute(fo, rranges, finish=None)
rranges1 = ((-10, 10.1, 0.1),(-10, 10.1, 0.1))
output=False
opt1=spo.brute(fo,((-10, 10.1, 0.1),(-10, 10.1, 0.1)),finish=None)
print(opt1,fm(*(opt1)))