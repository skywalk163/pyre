#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 21:26:14 2017

@author: skywalk
"""

#判断月份

import time
import tushare as ts
import pandas as pd
import numpy as np
dct2=pd.date_range(2010,2017)

dcmon=time.ctime()[4:7]
dc={"Jan":1,'Feb':1,'Mar':1,'Apr':0.5,'May':0.4,"Jun":0.2,'jul':1,'Aug':1,'Sep':0.8,'Oct':0.7,'Nov':0.5,"Dec":0.8}
print(dc[dcmon])

'''
 一月     January      （Jan）2.  二月      February   （Feb）3.  三月      March        （Mar）
 4.  四月      April           （Apr）5.  五月      May           （May）
 6.  六月      June           （Jun）7.  七月      July             （Jul）
 8.  八月      August        （Aug）9.  九月      September  （Sep）
 10.  十月     October      （Oct） 11.  十一月   November （Nov）12.  十二月   December （Dec）
 '''
dctime=time.ctime()
dctime=time.strftime('%Y%m%d')
print(dctime) 

dcdata =ts.get_k_data('000001',index=True,start='2010-01-01')
dct=dcdata.date

dct1=pd.date_range('2010','2011')
rng=dct1
dcts= pd.Series(np.random.randn(len(rng)), index=rng)
dcts[0:10].to_period('M')
dcts[0:10].to_period('W')
'''
Series 才有to_period 
'''
#dcdata.asfreq('M')
#dcdata.index=dcdata['date']
#dct3=dcdata.asfreq(freq='M')
#print(dct3)
#dctemp=dct
#for i in dct :
#    print (i,)
    
print(dcdata.date[5:7])
#拿到月初和月末的数据    
#dcdata['yuemo']=dcdata.date[5:7] !=dcdata.date.shift(-1)[5:7]
#dctemp=dcdata[dcdata.yuemo==True]

#for i in range(1,len(dcdata)):
#    print (type(i),i,dcdata[i])
    
    
df = pd.DataFrame({'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]})
print(df)

df.loc[df.AAA >= 5,'BBB'] = -1;
df.loc[df.AAA >= 5,['BBB','CCC']] = 555
df.loc[df.AAA < 5,['BBB','CCC']] = 2000
#蒙板的这个地方出错，还没有看明白
#找到原因，是网络断的缘故。


df_mask = pd.DataFrame({'AAA' : [True] * 4, 'BBB' : [False] * 4,'CCC' : [True, False] * 2})
df.where(df_mask,-1000)

#再来一次where的使用
df = pd.DataFrame({'AAA' : [4,5,6,7], 'BBB' : [10,20,30,40],'CCC' : [100,50,-30,-50]})
df['logic'] = np.where(df['AAA'] > 5,'high','low')
dcdata['logic'] = np.where(str(dcdata['date'])[5:7]!=str(dcdata['date'].shift(-1))[5:7],True,False)
df.loc[(df['BBB'] > 25) | (df['CCC'] >= 75), 'AAA'] = 0.1; df

print(df,"===============================")

#把数据写入一个df中
#dct1=dct1.append(pd.DataFrame({'date':'2002','close':12.5},index=[2]))
dfom=pd.DataFrame()
dfomi=0
dctemp=(dcdata.iloc[0].date)[5:7]
for i in dcdata.index :
    #print (type(i),i,)
    if (dcdata.iloc[i].date)[5:7]!= dctemp :
        #print (i,dcdata.iloc[i])
        dct5=dcdata.iloc[i]
        dfom=dfom.append(pd.DataFrame({'date':dct5.date,'open':dct5.open,'close':dcdata.iloc[i-1].close,'m':dct5.date[5:7]},index=[dfomi]))
        dfomi=dfomi+1
    dctemp=(dcdata.iloc[i].date)[5:7]
#算出每月的收益率    
dfom['mr']=(dfom.shift(-1).close-dfom.open)/dfom.open
    
#聚合月数据
dct6=dfom.groupby('m')
print(dct6.mean())

def mmean(dfin):
    dcdata=dfin
    dfom=pd.DataFrame()
    dfomi=0
    dctemp=(dcdata.iloc[0].date)[5:7]
    for i in dcdata.index :
        #print (type(i),i,)
        if (dcdata.iloc[i].date)[5:7]!= dctemp :
            #print (i,dcdata.iloc[i])
            dct5=dcdata.iloc[i]
            dfom=dfom.append(pd.DataFrame({'date':dct5.date,'open':dct5.open,'close':dcdata.iloc[i-1].close,'m':dct5.date[5:7]},index=[dfomi]))
            dfomi=dfomi+1
        dctemp=(dcdata.iloc[i].date)[5:7]
    #算出每月的收益率    
    dfom['mr']=(dfom.shift(-1).close-dfom.open)/dfom.open
        
    #聚合月数据
    dct6=dfom.groupby('m')
    return (dct6)