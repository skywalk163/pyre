from datetime import date
import datetime
date1 = datetime.date( 2013, 4,28 )
date2 = datetime.date( 2014,5,4 )
#print(date1)
#rint(type(date1))
testd=datetime.date(1990,12,19)
testd=datetime.date(testd.year,testd.month,testd.day+1)
#print(testd)
lista=[]

# for line in open("D:/快盘/work/python/SH999999.csv"):
#     year1,so1,sh1,sl1,sc1,aa1,bbbb = line.split(",")
#     test=(year1,so1,sh1,sl1,sc1)
#     #print(test)
#     if year1==str(testd) or year1==str(date1):
#         pass
#         #print(year1,test)
da512=[]
dcm='04'
dcd='01'
for line in open("D:/快盘/work/python/SH999999.csv"):
    year1,so1,sh1,sl1,sc1,aa1,bbbb = line.split(",")
    test=(year1,so1,sh1,sl1,sc1,aa1,bbbb)
    #print(test)
    ii=0
    if year1.split('-')[1]==dcm and year1.split('-')[2]==dcd:
        ii=ii+1
        #print(test)
        da512.append(test)
        
zs=0.0
zf=0.0
ysum=0.0
yyang=0
yall=0
print(type(zs),type(zf))
for ld in da512:
#     if zs<10:
#         zs=float(ld[4])
#         print(zs)
#     else:
#         zf=(float(ld[4])-zs)/zs*100
#         zs=float(ld[4])
#     print(zs)
# 
    yy=(float(ld[4])-float(ld[1]))/float(ld[1])*100
    yall=yall+1
    if yy>0:
        yyang =yyang+1
    ysum=ysum+yy
    print ('%s  当日涨幅%.2f%%' % (ld[0],yy))    
print('历史上%s.%s日大盘走势' %(dcm,dcd))
print('总计%d天 其中收阳%d天 收阳几率%.2f%% 当日总涨跌幅为%.2f%%' % (yall,yyang,yyang/yall*100,ysum))


