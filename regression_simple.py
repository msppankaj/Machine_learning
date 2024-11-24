#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
boston=load_boston()
import seaborn as sns


# In[2]:


boston.data.shape


# In[3]:


df=pd.DataFrame(boston.data,columns=boston.feature_names)


# In[4]:


df['target']=pd.Series(boston.target)


# In[5]:


df


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


cor_matrix=df.corr()


# In[9]:


sns.heatmap(data=cor_matrix,annot=True)
plt.show()


# In[10]:


#regression with single variable:
x=df['RM']
y=df['target']


# In[11]:


x=np.array(x).reshape(-1,1)
y=np.array(y).reshape(-1,1)


# In[12]:


y.shape
x.shape


# In[13]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=5)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[14]:


ytrain.shape


# In[15]:


linr_model=LinearRegression()


# In[16]:


linr_model.fit(xtrain,ytrain)


# In[27]:


linr_model.coef_


# In[29]:


linr_model.intercept_


# In[31]:


## evaluation matrxi for regression:
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# In[22]:


linpredict=linr_model.predict(xtest)


# In[33]:


linpredict


# In[38]:


mse=mean_squared_error(linpredict,ytest)
mae=mean_absolute_error(linpredict,ytest)
r2=r2_score(linpredict,ytest)
rmse=np.sqrt(mse)


# In[39]:


rmse


# In[30]:


##cross validation score


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


mse=mean_squared_error(xtrain,linpredict)


# In[21]:


mse

