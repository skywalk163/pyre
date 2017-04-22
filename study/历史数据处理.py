# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 22:21:53 2017

@author: Administrator
"""

import tushare as ts
import pandas as pd

#拿到所有数据
aa=ts.get_k_data('000001', index=True,start='1990-12-17', end='2017-1-3')

aa['yc']=aa.shift(1)['close']

yuandan=(
'1991-1-2',
'1992-1-2',
'1993-1-4',
'1994-1-3',
'1991-1-2',
'1991-1-2',
'1991-1-2',
'1991-1-2',
'1991-1-2',
'1991-1-2',
'1991-1-2',
'1991-1-2',
'1991-1-2',

)

bb=aa[aa['date']>"2010-12-31"]
cc=bb.head(1)
d=1990
while d < 2018 :
    e=str(d)+'-12-31'
    d=d+1
    #print(type(e),e)
    bb=aa[aa['date']>e]
    cc=bb.head(1)
    print(cc)
    
    