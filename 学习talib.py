#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 09:54:00 2017

@author: skywalk
"""
import talib as ta
import tushare as ts
import matplotlib.pyplot as plt



#获取收盘价数据，拿到的是Dataframe，转换成ndarray 
dcclose=ts.get_k_data('000001',index=True)['close'].values
#调用talib ma均线指标
ma30 = ta.MA(dcclose, timeperiod=30, matype=0)
ma120 = ta.MA(dcclose, timeperiod=120, matype=0)

#画图均线
plt.plot(ma30)
plt.plot(ma120)

macd, macdsignal, macdhist = ta.MACD(dcclose, fastperiod=12, slowperiod=26, signalperiod=9)
plt.plot(macd)

dcdata =ts.get_k_data('000001',index=True)
open,high,low,close = dcdata['open'].values,dcdata['high'].values,dcdata['low'].values,dcdata['close'].values
#Engulfing Pattern 吞噬模式
integer = ta.CDLENGULFING(open, high, low, close)
print(integer)
                    

import numpy as np
# note that all ndarrays must be the same length!
inputs = {
    'open': np.random.random(100),
    'high': np.random.random(100),
    'low': np.random.random(100),
    'close': np.random.random(100),
    'volume': np.random.random(100)
}

from talib import abstract


sma = abstract.SMA
sma = abstract.Function('sma')

print(type(sma),sma)
dctemp=sma(inputs)

output=ta.abstract.MA(inputs)

plt.plot(output)
