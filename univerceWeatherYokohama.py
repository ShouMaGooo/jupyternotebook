#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
allData = pd.read_csv(r'C:\Users\sho\WeatherYokohama.csv',encoding="SHIFT-JIS")
allData


# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(allData['neutron(counts/10min)'],allData['muon(counts/10min)'])
plt.xlabel('neutron(counts/10min)')
plt.ylabel('muon(counts/10min)')


# In[7]:


from sklearn.linear_model import LinearRegression
X1 = allData[['neutron(counts/10min)']]
Y1 = allData['muon(counts/10min)']
model1 = LinearRegression()
model1.fit(X1, Y1)


# In[8]:


plt.scatter(allData['neutron(counts/10min)'],allData['muon(counts/10min)'])
plt.xlabel('neutron(counts/10min)')
plt.ylabel('muon(counts/10min)')
plt.plot(X1, model1.predict(X1))


# In[9]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(allData['neutron(counts/10min)'],allData['tempreture'])
plt.xlabel('neutron(counts/10min)')
plt.ylabel('tempreture')


# In[10]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(allData['muon(counts/10min)'],allData['tempreture'])
plt.xlabel('muon(counts/10min)')
plt.ylabel('tempreture')


# In[11]:


from sklearn.linear_model import LinearRegression
X1 = allData[['muon(counts/10min)']]
Y1 = allData['tempreture']
model1 = LinearRegression()
model1.fit(X1, Y1)


# In[12]:


plt.scatter(allData['muon(counts/10min)'],allData['tempreture'])
plt.xlabel('muon(counts/10min)')
plt.ylabel('tempreture')
plt.plot(X1, model1.predict(X1))


# In[14]:


allData[["muon(counts/10min)","tempreture"]].corr()

