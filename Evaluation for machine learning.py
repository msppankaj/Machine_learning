#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston,load_iris
boston=load_boston()
iris=load_iris()
import seaborn as sns


# In[2]:


dir(boston)


# In[5]:


dfreg=pd.DataFrame(boston.data,columns=boston.feature_names)
dfcla=pd.DataFrame(iris.data,columns=iris.feature_names)


# In[6]:


dfreg['target']=pd.Series(boston.target)
dfcla['target']=iris.target


# In[9]:


dfreg


# In[37]:


xreg=dfreg.iloc[:,0:13]
yreg=dfreg.iloc[:,13]
xcla=dfcla.iloc[:,0:4]
ycla=dfcla.iloc[:,4]


# In[41]:


ycla


# In[43]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(xcla,ycla,test_size=0.3)


# In[23]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
lr=LinearRegression()
lg=LogisticRegression()


# In[25]:


lr.fit(xreg,yreg)


# In[44]:


lg.fit(xtrain,ytrain)


# In[48]:


yc_pred=lg.predict(xtest)


# In[ ]:


#Evaluation for classification machine learning


# In[71]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_metrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve,auc


# In[ ]:


#https://medium.com/analytics-vidhya/evaluation-metrics-for-classification-models-e2f0d8009d69


# In[72]:


a=accuracy_score(ytest,yc_pred)
b=precision_score(ytest,yc_pred)
c=recall_score(ytest,yc_pred)
d=f1_score(ytest,yc_pred)
e=fbeta_score(ytest,yc_pred)
fpr,tpr,threshold=roc_curve(y,score,pos_label=2)
fpr,tpr,threshold=roc_curve(y,score,pos_label=2)
auc(fpr,tpr)


# In[55]:





# In[51]:





# In[32]:


#evaluation metrics for regression:
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import r2_score ##one more is adjusted


# In[33]:


mean_absolute_error(xreg,yreg)


# In[ ]:





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


# In[17]:





# In[18]:





# In[19]:


## evaluation matrxi for regression:
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# In[20]:


linpredict=linr_model.predict(xtest)


# In[21]:


linpredict


# In[22]:





# In[23]:





# In[24]:





# In[ ]:





# In[26]:





# In[27]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




