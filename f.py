#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:14:05 2017

@author: skywalk
"""

import tushare as ts
import pandas as pd
import numpy as np
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