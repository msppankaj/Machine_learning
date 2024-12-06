#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
iris=load_iris()


# In[2]:


df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['Target']=iris.target
#iris.target_names
x=df.iloc[:,0:4]
y=df.iloc[:,4]


# In[3]:


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(x,y)


# In[ ]:


##cross validation:
""""K-fold cross-validation
Hold-out cross-validation
Stratified k-fold cross-validation
Leave-p-out cross-validation
Leave-one-out cross-validation
Monte Carlo (shuffle-split)
Time series (rolling cross-validation)"""


# In[ ]:


#K-fold cross-validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv1s=cross_val_score(dtc,x,y,cv=kf)


# In[ ]:


cv1s


# In[ ]:


#Leave-p-out cross-validation
from sklearn.model_selection import cross_val_score,LeavePOut
lpo=LeavePOut(p=10)
cvs3=cross_val_score(dtc,x,y,cv=lpo)


# In[ ]:


cvs3


# In[ ]:


#Hold-out cross-validation
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)


# In[ ]:


#stratified k fold
from sklearn.model_selection import cross_val_score,StratifiedKFold
stkfold=StratifiedKFold(n_splits=5)


# In[ ]:


cvs2=cross_val_score(dtc,x,y,cv=stkfold)


# In[ ]:


cvs2


# In[ ]:


##class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0, monotonic_cst=None)


# In[ ]:


#Leaveoneout
from sklearn.model_selection import cross_val_score,LeaveOneOut()
LOO=LeaveOneOut()
cvs4=cross_val_score(dtc,x,y,cv=loo)


# In[ ]:


cvs4


# In[ ]:


ypred=dtc.predict(xtest)


# In[ ]:


#ypred=np.array(ypred).reshape(-1,1)
#ytest=np.array(ytest).reshape(-1,1)


# In[ ]:


ytest.shape


# In[ ]:


dtc.score(ypred,ytest)


# In[ ]:





# In[ ]:




