# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 10:51:52 2017

@author: Administrator
"""
'''
#逼近法
14的时候，拟合True 
'''
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)+0.5*x
    
x=np.linspace(-2*np.pi,2*np.pi,50)
#plt.plot(x,f(x),'b')
#plt.grid(True)
#plt.xlabel('x')
#plt.ylabel('f(x)')

while(0):
    dcn=input('欢迎使用逼近法,请输入多项式拟合度"0"退出，reg=:')

    if dcn=='0' :
        break
    reg=np.polyfit(x,f(x),deg=int(dcn))
    ry=np.polyval(reg,x)
    
    plt.plot(x,f(x),'b',label='f(x)')
    plt.plot(x,ry,'r.',label='regression')
    
    plt.legend(loc=0)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    print('拟合度reg=%s 是否所有都拟合：%s  均方差MSE=%s '%(dcn, np.allclose(f(x),ry) ,np.sum((f(x)-ry)**2)/len(x)))
#    print('拟合度reg=%s 是否所有都拟合：%s  均方差MSE=%s '%(dcn, dcn,dcn))

'''
matrix=np.zeros((3+1,len(x)))
matrix[3,:]=np.sin(x)
#matrix[3,:]=x**3
matrix[2,:]=x**2
matrix[1,:]=x
matrix[0,:]=1
#reg=np.linalg.lstsq(matrix.T,f(x))[0]
reg=np.linalg.lstsq(matrix.T,f(x))[0]

reg
ry=np.dot(reg,matrix)
plt.plot(x,f(x),'b',label='f(x)')
plt.plot(x,ry,'r.',label='regression')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
print ('回归拟合度%s,均方差%s' %(np.allclose(f(x),ry),np.sum(f(x)-ry)**2/len(x)))

print(reg)
'''

'''
生成有噪声的数据

xn=np.linspace(-2*np.pi,2*np.pi,50)
xn=xn+0.15*np.random.standard_normal(len(xn))
yn=f(xn)+0.25*np.random.standard_normal(len(xn))

reg=np.polyfit(xn,yn,7)
ry=np.polyval(reg,xn)
plt.plot(xn,yn,'b^',label='f(x)')
plt.plot(xn,ry,'ro',label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')
'''
"""
未排序数据
"""

xu=np.random.rand(50)*4*np.pi -2*np.pi
yu=f(xu)
print(xu[:10].round(2),yu[:10].round(2))
reg=np.polyfit(xu,yu,5)
ry=np.polyval(reg,xu)

plt.plot (xu,yu,'b^',label='f(x)')
plt.plot(xu,ry,'ro',label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

