# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:18:46 2017

@author: Administrator
"""

'''
符号计算
SymPy为数学表达式提供了三个基本的渲染器：
1 基于LaTex的渲染器
2 基于unicode的渲染器
3 基于ASCII的渲染器

'''
import time
import sympy as sy
x=sy.Symbol('x')
y=sy.Symbol('y')

3+sy.sqrt(x)-4**2

f=x**2+3+0.5*x**2+3/2

sy.simplify(f)

sy.init_printing(pretty_print=False,use_unicode=False)
print(sy.pretty(f))
print(sy.pretty(sy.sqrt(x)+0.5))
#如果按照例子中，pi取40万位的话，我的电脑大约要等10秒钟
dcnow=time.time()
pi_str=str(sy.N(sy.pi,40000))
print(pi_str[:255],time.time()-dcnow,"770508 在pi的第51067位！150715在316576位！",pi_str.find('770508'))


#解方程
print(sy.solve(x**2-1),sy.solve(x**2-1-3),sy.solve(x**3+0.5*x**2-1))
print('求x*x+y*y=0的解：',sy.solve(x**2+y**2))
'''
SymPy的另一个长处是积分和微分
'''
a,b=sy.symbols('a b')
#打印符号积分
print( sy.pretty(sy.Integral(sy.sin(x)+0.5*x,(x,a,b))))
#积分函数的反导数
int_func=sy.integrate(sy.sin(x)+0.5*x,x)
print('积分函数的反导数:\r\n',sy.pretty(int_func))

#求积分
Fb=int_func.subs(x,9.5).evalf()
Fa=int_func.subs(x,0.5).evalf()
print('Fb-Fa的差就是积分的准确值:',Fb-Fa)

int_func_limits=sy.integrate(sy.sin(x)+0.5*x,(x,a,b))
print("符号积分上下限得到符号解：\r\n",sy.pretty(int_func_limits))

dcj=int_func_limits.subs({a:0.5,b:9.5}).evalf()
print('代入数值，使用字典对象代表多个替代值，并求值可以得到积分值：\r\n',dcj)

dcj1=sy.integrate(sy.sin(x)+0.5*x,(x,0.5,9.5))
print('提供量化的积分上下限，在一步中得出准确的值：\r\n',dcj1)


'''
微分 p232
对不定积分求导通常应该得出原函数，我们对前面的符号反导数应用diff函数
'''
print(int_func.diff())

f=(sy.sin(x)+0.05*x**2+sy.sin(y)+0.05*y**2)
del_x=sy.diff(f,x)
del_y=sy.diff(f,y)

print('两个变量x y的偏微分',del_x,' '  , del_y)

xo=sy.nsolve(del_x,-1.5)
yo=sy.nsolve(del_y,-1.5)
dcf=f.subs({x:xo,y:yo}).evalf()
print('从数值上求出两个方程式的解，xo=%s yo=%s 全局最小值dcf=%s' %(xo,yo,dcf))

xo=sy.nsolve(del_x,1.5)
yo=sy.nsolve(del_y,1.5)
dcf1=f.subs({x:xo,y:yo}).evalf()
print('没有根据/随机的猜测，同样可能使算法陷入某个局部最小值而非全局最小值：xo=%s yo=%s 局部最小值dcf1=%s' %(xo,yo,dcf1))
