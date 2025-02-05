##一、python数据结构和读取
import pandas as pd
import numpy as np
#创建字典
dic = {"a":1, "b":2,"c":3}
dic["a"]
dic.keys()
dic.values()
list(dic.values())
#创建列表
lst=[1,2,3]
lst.append('a')
#创建数据框
dic = {"a": {"一": 1, "二": 2},
       "b": {"一": 10, "二": 20},
       "c": {"一": 100, "二": 200}}
data = pd.DataFrame(dic)  # 创建Dataframe
print(data)
#自定义函数
def myfun (a,b):
  c=a**b
  return c
myfun(8,2)

#循环语句
for i in range(4):
       print(i)

#读取数据
df=pd.read_csv("starbucks_store_worldwide.csv")
#data=pd.read_excel("文件名.xlsx",ndex_col=0)#第一列为index
print(df.head(1))
print(df.info())


#二、绘图操作
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib
from matplotlib import font_manager#导入包
font={'family':"Microsoft Yahei",
     'size':'10'}
matplotlib.rc("font",**font)#设置中文显示

#绘制折线图
fig=plt.figure(figsize=(20,8),dpi=80)
x=range(2,26,2)
y1=range(3,15,1)
y2=np.random.normal(0,2,12)#生成12个N（0，2）分布随机数
plt.plot(x,y1,label="北京",color='g',linewidth=1,linestyle=':')
plt.plot(x,y2,label="上海",color='b',linewidth=5,linestyle='--')#label用于图例,color为颜色,linewidth设置线条大小,linestyle线条形状，slpha为透明度0-1
plt.xlabel("时间")
plt.ylabel("温度")
plt.title("某段时间内温度")
plt.legend()#图例
#绘制条形图
fig=plt.figure(figsize=(20,8),dpi=80)#设置图片大小
x=['1','2','3']
y1=[40,59,29]
y2=np.random.normal(3,1,50)
plt.bar(x,y1,width=0.3)#width调整宽度
#绘制横着的条形图
fig=plt.figure(figsize=(20,8),dpi=80)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.barh(x,y1,height=0.1)#height调整宽度
#绘制多组数据条形图
#fig=plt.figure(figsize=(20,8),dpi=80)#设置图片大小，需整体运行
fig=plt.figure(figsize=(20,8),dpi=80)
bar_width=0.2#不能太大
x1=list(range(1,5))
x2=[i+bar_width for i in x1]
x3=[i+bar_width for i in x2] #设置横坐标（关键）
y1=[40,59,29,33]
y2=[30,59,21,33]
y3=[40,29,59,33]
plt.bar(x1,y1,width=0.2,label='day14')#width调整宽度
plt.bar(x2,y2,width=0.2,label='day15')
plt.bar(x3,y3,width=0.2,label='day16')
plt.xticks(x2)#调整横坐标值为x2
plt.xticks(x2,['A1','B1','C1','D1'])#更换横坐标为A1,B1,C1,D1
plt.xticks(fontsize=19)#调整横坐标字体
plt.yticks(fontsize=19)
plt.legend()

#散点图
fig=plt.figure(figsize=(20,8),dpi=80)#设置图片大小
x=np.random.normal(0,1,50)
y1=np.random.normal(3,1,50)
y2=np.random.normal(3,1,50)
plt.xticks(fontsize=19)
plt.yticks(fontsize=19)
plt.scatter(x,y1)
plt.scatter(x,y2)

#直方图
fig=plt.figure(figsize=(20,8),dpi=80)
x=np.random.normal(0,1,100)
bins=20#组数,100个以内的数据分为5-12组
bin_width=3#组距，以多少间隔分组
plt.hist(x,bins,density=True)#density=True为频率图，默认为频数图
plt.grid(alpha=0.3)#添加网格

#画箱线图
fig=plt.figure(figsize=(20,8),dpi=80)
GDP1=np.array([90371,123607,57191,107624,94172,70653,157279])
GDP2=np.array([58496,45724,36183,53164,56388,42964,43475])
GDP3=np.array([55774,75828,47944,48981,46433,48902,54217])
plt.boxplot([GDP1,GDP2,GDP3],patch_artist=True,boxprops={'color':'black','facecolor':'steelblue'},medianprops={'linestyle':'-','color':'orange'},labels=['east','middle','west'])
plt.ylabel('GDP',fontsize=10)
plt.xlabel('region',fontsize=10)
plt.show()


#三、描述性统计与数理统计
from scipy import stats
import pandas as pd
import numpy as np
from statsmodels.compat import lzip

a=pd.read_excel("aa.xls",index_col=0)#第一列为index
data=pd.DataFrame(a)
n=len(data)
sz=data.iloc[0:n,[0]]#取第一列
hs=data.iloc[0:n,[1]]#取第二列
print(sz)
# 数据描述
print(data.describe())
# 常用数字特征
print(sz.mean()) # 均值
print(sz.mode()) # 众数
print(sz.mad()) # 平均离差
print(sz.std()) # 标准差
print(sz.skew()) # 偏度
print(sz.kurt()) # 峰度
# 检验上证的均值是否为2195
stats.ttest_1samp(sz, 2195)
# 检验两只股票的均值是否相等
stats.ttest_ind(sz,hs)
# 检验两只股票波动（方差）是否相等
szaa=data["shangzheng"]
hsaa=data["hengsheng"]
stats.levene(szaa,hsaa) # stats.levene函数只支持一维数据

#四、回归
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

# 导入数据
data = pd.read_csv("data2.csv", index_col=[])
data = pd.DataFrame(data)
print(data.info())#查看各变量情况
print(data.head())
x = data.iloc[:, 6:10]  # 设置自变量
x1 = data.iloc[:, [6]] # 设置自变量
y = data.iloc[:, [5]]  # 设置因变量
# 一元线性回归
lm = ols('y~x1', data=data).fit()
print(lm.summary())
# 多元线性回归
lm1 = ols('y~x', data=data).fit()
print(lm1.summary())  # 显示回归结果
# 数据的预测值
y_pred = lm.predict(x1)
#画图
#数据散点图
plt.scatter(x1, y, color='red', label="data")
# 绘制线性拟合线
plt.plot(x1, y_pred, color='black', linewidth=3, label="line")
# 添加图标标签
plt.legend(loc=2)
plt.xlabel("lnp")
plt.ylabel("lntc")
# 显示图像
# plt.savefig("lines.jpg") #保存图像
plt.show()

