#!/usr/bin/env python
# coding: utf-8


# In[3]:

#/padasをインポート。ローカルに保存したcsvファイルを読み込み,変数「allData」に格納し、出力。
import pandas as pd
allData = pd.read_csv(r'C:\Users\ooooooo\WeatherYokohama.csv',encoding="SHIFT-JIS")
allData


# In[5]:

#機械学習ライブラリscikit-learn(sklearn)のなかから、線形回帰のLinearRegressionを呼び出す。
#その後、説明変数として'neutron(counts/10min)'、目的変数として'muon(counts/10min)'を定義した後、表作成し、プロット。
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(allData['neutron(counts/10min)'],allData['muon(counts/10min)'])
plt.xlabel('neutron(counts/10min)')
plt.ylabel('muon(counts/10min)')


# In[7]:
#機械学習ライブラリscikit-learn(sklearn)のなかから、LinearRegression、つまり線形回帰を呼び出す。
#線形回帰型の変数『model1』を用意。   説明変数を'neutron(counts/10min)'、目的変数を'muon(counts/10min)'と定義
#「X1」と「Y1」のそれぞれに定義を行い、『model1』に「X1」,「Y1」をセットして線形回帰を実行。
#その後、した後、線形回帰を実行しています。
from sklearn.linear_model import LinearRegression
X1 = allData[['neutron(counts/10min)']]
Y1 = allData['muon(counts/10min)']
model1 = LinearRegression()
model1.fit(X1, Y1)


# In[8]:

#『中性子』と『ミューオン』を検知した回数(10秒あたり)の散布図と回帰直線を可視化。
plt.scatter(allData['neutron(counts/10min)'],allData['muon(counts/10min)'])
plt.xlabel('neutron(counts/10min)')
plt.ylabel('muon(counts/10min)')
plt.plot(X1, model1.predict(X1))




# In[11]:

#機械学習ライブラリscikit-learn(sklearn)のなかから、LinearRegression、つまり線形回帰を呼び出す。
#線形回帰型の変数『model1』を用意。
#「X1」と「Y1」のそれぞれに定義を行い、『model1』に「X1」,「Y1」をセットして線形回帰を実行。
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
X1 = allData[['muon(counts/10min)']]
Y1 = allData['tempreture']
model1.fit(X1, Y1)


# In[12]:

#『ミューオン』を検知した回数(10秒あたり)と横浜市の『日ごとの平均気温』の散布図と回帰直線を可視化。
plt.scatter(allData['muon(counts/10min)'],allData['tempreture'])
plt.xlabel('muon(counts/10min)')
plt.ylabel('tempreture')
plt.plot(X1, model1.predict(X1))


# In[14]:

#相関係数をcorr関数を使って導出する。
allData[["muon(counts/10min)","tempreture"]].corr()

