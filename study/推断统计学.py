# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:02:11 2017

@author: Administrator
第十章 随机数
"""
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

npr.rand(10)
npr.rand(5,5)

#生成[5,10)内的随机数
a=5.
b=10.
npr.rand(10)*(b-a)+a

npr.rand(5,5)*(b-a)+a
'''
生成简单随机数函数
rand randn 来自标准正态分布的一个或多个样本
randint（low,high ,size) 从low到high之间的随机整数
random_sample 半开区间[0.0 ,1.0)内的随机浮点数
random
ranf
sample
choice 给定一维数组中的随机样本
bytes 随机字节
'''
sample_size=500
rn1=npr.rand(sample_size,3)
rn2=npr.randint(0,10,sample_size)
rn3=npr.sample(size=sample_size)
a=[0,25,50,75,100]
rn4=npr.choice(a,size=sample_size)

fig,((ax1,ax2),(ax3,ax4))=plt.subplots(nrows=2,ncols=2,figsize=(7,7))
ax1.hist(rn1,bins=25,stacked=True)
ax1.set_title('rand')
ax1.set_ylabel('frequency')
ax1.grid(True)
ax2.hist(rn2,bins=25)
ax2.set_title('randint')
ax2.grid(True)
ax3.hist(rn3,bins=25)
ax3.set_title('sample')
ax3.set_ylabel('frequency')
ax3.grid(True)
ax4.hist(rn4,bins=25)
ax4.set_title('choice')
ax4.grid(True)

'''
标准正态分布
正态分布
卡方分布
泊松分布

'''

sample_size=500
rn1=npr.standard_normal(sample_size)
rn2=npr.normal(100,20,sample_size)
rn3=npr.chisquare(df=0.5,size=sample_size)
rn4=npr.poisson(lam=1.0,size=sample_size)

fig,((ax1,ax2),(ax3,ax4))=plt.subplots(nrows=2,ncols=2,figsize=(7,7))
ax1.hist(rn1,bins=25)
ax1.set_title('standard normal')
ax1.set_ylabel('frequency')
ax1.grid(True)
ax2.hist(rn2,bins=25)
ax2.set_title('normal(100,20)')
ax2.grid(True)
ax3.hist(rn3,bins=25)
ax3.set_title('chi square')
ax3.set_ylabel('frequency')
ax3.grid(True)
ax4.hist(rn4,bins=25)
ax4.set_title('poisson')
ax4.grid(True)

'''
模拟 p241
蒙特卡罗MCS模拟是金融学中最重要的数值技术之一 
期权定价模型bs
'''
S0=100
r=0.05
sigma=0.25
T=2.0
I=10000
ST1=S0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*npr.standard_normal(I))

fig=plt.subplots(nrows=1,ncols=1,figsize=(7,7))
plt.hist(ST1,bins=50)
plt.xlabel('index level of bs ')
plt.ylabel('frequency')
plt.grid(True)

'''
尝试使用lognormal函数直接得出随机变量值
'''
ST2=S0* npr.lognormal((r-0.5*sigma**2)*T,sigma*np.sqrt(T),size=I)
fig=plt.subplots(nrows=1,ncols=1,figsize=(7,7))

plt.hist(ST2,bins=50)
plt.xlabel('index level of lognormal')
plt.ylabel('frequency')
plt.grid(True)

'''
使用scipy.stats子库和下面定义的助手函数 print_statisstics比较模拟结果的分布特性：

'''

import scipy.stats as scs
def print_staistics(a1,a2):
    '''Prints selected statistics
    Parameters
    ==========
    a1,a2:ndarray objects
    results object from simulation
    '''
    
    sta1=scs.describe(a1)
    sta2=scs.describe(a2)
    print("%14s %14s %14s" %('statistic','data set 1','data set 2'))
    print (45*'-')
    print ("%14s %14.3f %14.3f" %('size',sta1[0],sta2[0]))
    print ("%14s %14.3f %14.3f" %('min',sta1[1][0],sta2[1][0]))
    print ("%14s %14.3f %14.3f" %('max',sta1[1][1],sta2[1][1]))
    print ("%14s %14.3f %14.3f" %('mean',sta1[2],sta2[2]))
    print ("%14s %14.3f %14.3f" %('std',np.sqrt(sta1[3]),np.sqrt(sta2[3])))
    print ("%14s %14.3f %14.3f" %('skew',sta1[4],sta2[4]))
    print ("%14s %14.3f %14.3f" %('kurtosis',sta1[5],sta2[5]))
    
print_staistics(ST1,ST2)
    
'''
随机过程
金融学中的随机过程通常表现为马尔科夫特性：明天的过程值只依赖于今天的过程状态，而不依赖其它任何历史状态，甚至不依赖整个路径历史，这种过程也被称为无记忆过程！
几何布朗运动
'''
I=10000
M=50
dt=T/M
S=np.zeros((M+1,I))
S[0]=S0
for t in range (1,M+1):
    S[t] =S[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*npr.standard_normal(I))

fig=plt.subplots()
plt.hist(S[-1],bins=S0)
plt.xlabel('index level of SDE')
plt.ylabel('frequency')
plt.grid(True)

print('布朗运动正态分布')
print_staistics(S[-1],ST2)
'''
展示前10调模拟路径
'''
fig=plt.subplots()
plt.plot(S[:, :10],lw=1.5)
plt.xlabel('time')
plt.ylabel('index level')
plt.grid(True)

"""
平方根扩散
另一类重要的金融过程是均值回归过程，用于建立短期利率或者波动过程的模型。平方根扩散由Cox、Ingersoll和Ross1985年提出，公式提供了对应的SDE
大部分随机过程会产生偏差，使用欧拉格式可能更合适，这种特殊的格式在文献中通常称为完全截断。

"""
x0=0.05
kappa=3.0
theta=0.02
sigma=0.1

I=10000
M=50
dt=T/M
def srd_euler():
    xh=np.zeros((M+1,I))
    x1=np.zeros_like(xh)
    xh[0]=x0
    x1[0]=x0
    for t in range(1,M+1):
        xh[t]=(xh[t-1] + kappa*(theta-np.maximum(xh[t-1],0))*dt + sigma* np.sqrt(np.maximum(xh[t-1],0))*np.sqrt(dt)*npr.standard_normal(I))
    x1=np.maximum(xh,0)
    return x1
x1=srd_euler()
#直方图展示模拟结果
fig=plt.subplots()
plt.hist(x1[-1],bins=50)
plt.xlabel('value of oula')
plt.ylabel('frequency')
plt.grid(True)
#展示前10条模拟路径
fig=plt.subplots()
plt.plot(x1[:,:10],lw=1.5)
plt.xlabel('time of oula')
plt.ylabel('index level')
plt.grid(True)
'''
寻求更精确的结果，基于自由度、非中心参数的卡方分布平方根扩散的精确离散化格式
不知道为什么，exact收敛于0.002 ，明显有10倍量级问题。另外扩散路径明显不对 
'''
x0=0.05
kappa=3.0
theta=0.02
sigma=0.1

I=10000
M=50

def srd_exact():
    x2=np.zeros((M+1,I))
    x2[0]=x0
    for i in range(1,M+1):
        df=4*theta*kappa/sigma**2
        c=(sigma**2*(1-np.exp(-kappa*dt)))/(4*kappa)
        nc=np.exp(-kappa*dt)/c*x2[t-1]
        x2[t]=c*npr.noncentral_chisquare(df,nc,size=I)
    return x2
    
x2=srd_exact()

fig=plt.subplots()
plt.hist(x2[-1],bins=50)
plt.xlabel('value of jingque')
plt.ylabel('frequency')
plt.grid(True)

fig=plt.subplots()
plt.plot(x2[:,:10],lw=1.5)
plt.xlabel('time of jingque' )
plt.ylabel('index level')
plt.grid(True)

print_staistics(x1[-1],x2[-1])
'''I=250000
%time x1=srd_euler()
%time x2=srd_exact()
'''

'''
随机波动率，Heston模型
'''
S0=100.
r=0.05
v0=0.1
kappa=3.0
theta=0.25
sigma=0.1
rho=0.6
T=1.0

#克列斯基分解
corr_mat=np.zeros((2,2))
corr_mat[0,:]=[1.0,rho]
corr_mat[1,:]=[rho,1.0]
cho_mat=np.linalg.cholesky(corr_mat)

print(cho_mat)
M=50
I=10000
ran_num=npr.standard_normal((2,M+1,I))

#平方根扩散，欧拉格式
dt=T/M
v=np.zeros_like(ran_num[0])
vh=np.zeros_like(v)
v[0]=v0
vh[0]=v0
for t in range(1,M+1):
    ran =np.dot(cho_mat,ran_num[:,t,:])
    vh[t]=(vh[t-1]+kappa*(theta-np.maximum(vh[t-1],0))*dt+sigma*np.sqrt(np.maximum(vh[t-1],0))*np.sqrt(dt)+ran[1])
v=np.maximum(vh,0)

#使用几何布朗运动的精确欧拉格式
S=np.zeros_like(ran_num[0])
S[0]=S0
for t in range(1,M+1):
    ran=np.dot(cho_mat,ran_num[:,t,:])
    S[t]=S[t-1]*np.exp((r-0.5*v[t])*dt+np.sqrt(v[t])*ran[0]*np.sqrt(dt))
    
#用直方图展示了指数水平过程和波动性过程的模拟结果
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,5))
ax1.hist(S[-1],bins=50)
ax1.set_xlabel('index level')
ax1.set_ylabel('frequency')
ax1.grid(True)
ax2.hist(v[-1],bins=50)
ax2.set_xlabel('volatility')
ax2.grid(True)


fig,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(7,6))
ax1.plot(S[:,:10],lw=1.5)
ax1.set_ylabel('index level')
ax1.grid(True)
ax2.plot(v[:,:10],lw=1.5)
ax2.set_xlabel('time')
ax2.set_ylabel('volatility')

ax2.grid(True)
print_staistics(S[-1],v[-1])

'''
跳跃扩散
p255
'''
S0=100.
r=0.05
sigma=0.2
lamb=0.75
mu=-0.6
delta=0.25
T=1.0

#为了模拟跳跃扩散，我们需要生成3组独立随机数
M=50
I=10000
dt=T/M
rj=lamb*(np.exp(mu+0.5*delta**2)-1)
S=np.zeros((M+1,I))
S[0]=S0
sn1=npr.standard_normal((M+1,I))
sn2=npr.standard_normal((M+1,I))
poi=npr.poisson(lamb*dt,(M+1,I))
for t in range(1,M+1,1):
    S[t]=S[t-1]*(np.exp((r-rj-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*sn1[t])+(np.exp(mu+delta*sn2[t])-1)*poi[t])
    S[t]=np.maximum(S[t],0)

fig=plt.subplots()    
plt.hist(S[-1],bins=50)
plt.xlabel('value of tiaoyue')
plt.ylabel('frequency')
plt.grid(True)

fig=plt.subplots()
plt.plot(S[:,:10],lw=1.5)
plt.xlabel('time of tiaoyue')
plt.ylabel('index level')
plt.grid(True)

'''
方差缩减
修改随机生成器种子值：

'''
print("%15s %15s" %('Mean','Std.DEviation'))
print (31*'-')
for i in range(1,31,2):
    npr.seed(1000)
    sn=npr.standard_normal(i**2*10000)
    print ("%15.12f %15.12f" %(sn.mean(),sn.std()))
    
print('i',i**2*10000)
#用numpy的函数concatenate可以简洁的实现上述方法：
sn=npr.standard_normal(10000/2)
sn=np.concatenate((sn,-sn))
print(np.shape(sn))
print("%15s %15s" %('Mean','Std. Deviation'))
print (31*'-')
for i in range(1,31,2):
    npr.seed(1000)
    sn=npr.standard_normal(i**2*10000/2)
    sn=np.concatenate((sn,-sn))
    
    print ("%15.12f %15.12f" %(sn.mean(),sn.std()))
    
'''
使用方差缩减技术-矩匹配，有助于在一个步骤中更正第一个和第二个统计矩：
'''
sn=npr.standard_normal(10000)
print ('sn.mean=',sn.mean(),'sn.std=',sn.std())

sn_new=(sn-sn.mean())/sn.std()
print('sn_new.mean()=',sn_new.mean(),'sn_new.std()=',sn_new.std())

'''生成用于过程模拟的标准正态随机数'''
def gen_sn(M,I,anti_paths=True,no_match=True):
    '''
    
    '''
    if anti_paths is True:
        sn=npr.standard_normal((M+1,I/2))
        sn=np.concatenate((sn,-sn),axis=1)
    else:
        sn-npr.standard_normal((M+1,I))
    if no_match is True:
        sn=(sn-sn.mean())/sn.std()
        
    return sn
'''估值p259 欧式期权 美式期权
风险世界中，未定权益的价值是风险中立（鞅）测度下的折现后预期收益
先考虑欧式期权'''


S0=100.
r=0.05
sigma=0.25
T=1.0
I=50000
def gbm_mcs_stat(K):
    '''
    K:float
    returns:C0 :float
    '''
    sn=gen_sn(1,I)
    ST=S0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*sn[1])
    hT=np.maximum(ST-K,0)
    C0=np.exp(-r*T)*1/I*np.sum(hT)
    return C0
print('行权价K=105时，风险中立预期定价C0=',gbm_mcs_stat(K=105.))

'''动态模拟方法'''
M=50
def gbm_mcs_dyna(K,option='call'):
    '''
    K:float
    option:string
    returns C0:float
    '''
    dt=T/M
    S=np.zeros((M+1,I))
    S[0]=S0
    sn=gen_sn(M,I)
    for t in range(1,M+1):
        S[t]=S[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*sn[t])
    if option =='call':
        hT=np.maximum(S[-1]-K,0)
    else:
        hT=np.maximum(K-S[-1],0)
    C0=np.exp(-r*T)*1/I*np.sum(hT)
    return C0
    
dc1=gbm_mcs_dyna(K=110. ,option='call')
dc2=gbm_mcs_dyna(K=110. ,option='put')
print ('K=110时，看涨期权价格估算%f 看跌期权价格估算%f' %(dc1,dc2))

'''
这些基于模拟的估值方法与布莱克估值公式得出的基准值相比表现如何？
'''

from bsm_functions import bsm_call_value
stat_res=[]
dyna_res=[]
anal_res=[]
k_list=np.arange(80.,120.1,5.)
np.random.seed(20000)
for K in k_list:
    stat_res.append(gbm_mcs_stat(K))
    dyna_res.append(gbm_mcs_dyna(K))
    anal_res.append(bsm_call_value(S0,K,T,r,sigma))
stat_res=np.array(stat_res)
dyna_res=np.array(dyna_res)
anal_res=np.array(anal_res)

"""
将今年柜台模拟方法的结果与精确的分析值相比
"""
fig,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(8,6))
ax1.plot(k_list,anal_res,'b',label='analytical')
ax1.plot(k_list,stat_res,'ro',label='static')
ax1.set_ylabel('European call option value')
ax1.grid(True)
ax1.legend(loc=0)
ax1.set_ylim(ymin=0)
wi=1.0
ax2.bar(k_list-wi/2,(anal_res-stat_res)/anal_res*100,wi)
ax2.set_xlabel('strike')
ax2.set_ylabel('difference in %')
ax2.set_xlim(left=75,right=125)
ax2.grid(True)

'''
上面所有估值的差异都小于1%,下面动态的差异略大
'''
fig,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(8,6))
ax1.plot(k_list,anal_res,'b',label='analytical')
ax1.plot(k_list,dyna_res,'ro',label='dynamic')
ax1.set_ylabel('European call option value')
ax1.grid(True)
ax1.legend(loc=0)
ax1.set_ylim(ymin=0)
wi=1.0
ax2.bar(k_list-wi/2,(anal_res-dyna_res)/anal_res*100,wi)
ax2.set_xlabel('strike')
ax2.set_ylabel('difference in %')
ax2.set_xlim(left=75,right=125)
ax2.grid(True)


'''
美式期权
以最优截止问题形式出现的美式期权价格

'''
def gbm_mcs_amer(K,option='call'):
    dt=T/M
    df=np.exp(-r*dt)
    
    S=np.zeros((M+1,I))
    S[0]=S0
    sn=gen_sn(M,I)
    for t in range(1,M+1):
        S[t]=S[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*sn[t])
        
    if  option=='call':
        h=np.maximum(S-K,0)
    else:
        h=np.maximum(K-S,0)
    V=np.copy(h)
    for t in range(M-1,0,-1):
        reg=np.polyfit(S[t],V[t+1]*df,7)
        C=np.polyval(reg,S[t])
        V[t]=np.where(C>h[t],V[t+1]*df,h[t])
    C0=df*1/I*np.sum(V[1])
    return C0
    
dc1=gbm_mcs_amer(110.,option='call')
dc2=gbm_mcs_amer(110.,option='put')
print('看涨看跌期权值：',dc1,dc2)
            
'''
估算期权溢价
'''
euro_res=[]
amer_res=[]
k_list=np.arange(80.,120.1,5.)
for K in k_list:
    euro_res.append(gbm_mcs_dyna(K,'put'))
    amer_res.append(gbm_mcs_amer(K,'put'))
euro_res=np.array(euro_res)
amer_res=np.array(amer_res)
"""对于所选择的行权价范围，溢价可能最高达到10%"""
fig,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(8,6))
ax1.plot(k_list,euro_res,'b',label='European put')
ax1.plot(k_list,amer_res,'ro',label='American put')
ax1.set_ylabel('call option value')
ax1.grid(True)
ax1.legend(loc=0)
wi=1.0

ax2.bar(k_list-wi/2,(amer_res-euro_res)/euro_res*100,wi)
ax2.set_xlabel('strike')
ax2.set_ylabel('early exercise preminu in %')
ax2.set_xlim(left=75,right=125)
ax2.grid(True)

"""
风险测度
风险价值VaR
"""
S0=100
r=0.05
sigma=0.25
T=30/365.
I=10000
ST=S0*np.exp((r-0.5*sigma**2)*T +sigma*np.sqrt(T)*npr.standard_normal(I))

R_gbm=np.sort(ST-S0)
fig=plt.subplots()
plt.hist(R_gbm,bins=50)
plt.xlabel('absolute return')
plt.ylabel('frequency')
plt.grid(True)

'''
定义干兴趣的百分比，在列表兑现peres中，0.1转换为置信度100%-0.1%=99.9% 本例中，置信度位99.9%的30日VaR为20.2货币单位，而89%置信度下为8.9个货币单位

'''
percs=[0.01,0.1,1.,2.5,5.0,10.0]
var=scs.scoreatpercentile(R_gbm,percs)
print("%16s %16s" %('Confidence Level','Value-at-Risk'))
print('-'*33)
for pair in zip(percs,var):
    print ('%16.2f %16.3f ' %(100-pair[0],-pair[1]))
    
dt=30./365/M
rj=lamb*(np.exp(mu+0.5*delta**2)-1)
S=np.zeros((M+1,I))
S[0]=S0
sn1=npr.standard_normal((M+1,I))
sn2=npr.standard_normal((M+1,I))
poi=npr.poisson(lamb*dt,(M+1,I))
for t in range(1,M+1,1):
    S[t]=S[t-1]*(np.exp((r-rj-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*sn1[t])+(np.exp(mu+delta*sn2[t])-1)*poi[t])
    S[t]=np.maximum(S[t],0)
R_jd=np.sort(S[-1]-S0)
'''
利用均值为负数的跳跃成分，从正态分布的角度看，在左侧有明显的“大尾巴”
'''
fig=plt.subplots()
plt.hist(R_jd,bins=50)
plt.xlabel('absolute return of 30day')
plt.ylabel('frequency')
plt.grid(True)

'''置信度90%的VaR'''

percs=[0.01,0.1,1.,2.5,5.0,10.0]
var=scs.scoreatpercentile(R_jd,percs)
print("%16s %16s" %('Confidence Level','Value-at-Risk'))
print('-'*33)
for pair in zip(percs,var):
    print ('%16.2f %16.3f ' %(100-pair[0],-pair[1]))
'''这说明标准VaR测度在捕捉金融市场经常遇到的尾部风险方面的问题
下面用图形方式展示两种情况的VaR测度以便比较'''
fig=plt.subplots()
percs=list(np.arange(0.0,10.1,0.1))
gbm_var=scs.scoreatpercentile(R_gbm,percs)
jd_var=scs.scoreatpercentile(R_jd,percs)
plt.plot(percs,gbm_var,'b',lw=1.5,label='GBM')
plt.plot(percs,jd_var,'r',lw=1.5,label='JD')

plt.legend(loc=4)
plt.xlabel('100-confidence level [%]')
plt.ylabel('value-at-risk')
plt.grid(True)
plt.ylim(ymax=0.0)

'''
信用价值调整
违约概率和（平均）损失水平
'''
S0=100.
r=0.05
sigma=0.2
T=1.
I=100000
ST=S0*np.exp((r-0.5*sigma**2)*T +sigma*np.sqrt(T)*npr.standard_normal(I))

L=0.5
p=0.01

#泊松分布
D=npr.poisson(p*T,I)
D=np.where(D>1,1,D)

dct0=np.exp(-r*T)*1/I*np.sum(ST)
CVaR=np.exp(-r*T)*1/I*np.sum(L*D*ST)
S0_CVA=np.exp(-r*T)*1/I*np.sum((1-L*D)*ST)
S0_adj=S0-CVaR
#违约概率1%,10万次
dct1=np.count_nonzero(L*D*ST)
print('如没有违约，资产当日现值%f CVaR=%f 风险调整后的资产现值S0_CVA=%f 大约为S0_adj=%f 违约概率1%%，10万次模拟下预期结果违约次数%f' %(dct0,CVaR,S0_CVA,S0_adj,dct1))
fig=plt.subplots()
plt.hist(L*D*ST,bins=S0)
plt.xlabel('loss')
plt.ylabel('frequency')
plt.grid(True)
plt.ylim(ymax=175)

#考虑欧式看涨期权，行权价100时的价值大约为10.4个货币单位
K=100.
hT=np.maximum(ST-K,0)
C0=np.exp(-r*T)*1/I*np.sum(hT)

CVaR=np.exp(-r*T)*1/I*np.sum(L*D*hT)
#调整后的期权价值大约低了5分
C0_CVA=np.exp(-r*T)*1/I*np.sum((1-L*D)*hT)

dc1=np.count_nonzero(L*D*hT)
dc2=np.count_nonzero(D)
dc3=I-np.count_nonzero(hT)
print('欧式看涨期权，行权价100时价值大约为%f个货币单位，在相同违约概率和水平假设下，CVaR=%f 调整后期权价值大约降低5分，C0_CVA=%f 因违约引起的亏损次数%i 违约次数%i 亏损次数%i' %(C0,CVaR,C0_CVA,dc1,dc2,dc3))
fig=plt.subplots()
plt.hist(L*D*hT,bins=S0)
plt.xlabel('loss')
plt.ylabel('frequency')
plt.grid(True)
plt.ylim(ymax=350)

