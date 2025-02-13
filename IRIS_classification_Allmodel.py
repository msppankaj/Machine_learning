
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snb
from sklearn.datasets import load_iris

iris=load_iris()

dir(iris)

df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target

#DIVIDING target and independednt variable
x=df.iloc[:,0:4]
y=df.iloc[:,4]
#Train_test_split
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=5)

#importing ML model:
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#Allign ML model
lr=LogisticRegression()
svc1=SVC()
knn=KNeighborsClassifier()
dtc=DecisionTreeClassifier()
nbc=GaussianNB()

#ML FIT in model:
lr.fit(xtrain,ytrain)
svc1.fit(xtrain,ytrain)
knn.fit(xtrain,ytrain)
dtc.fit(xtrain,ytrain)
nbc.fit(xtrain,ytrain)

#prediction
ypred_lr=lr.predict(xtest)
ypred_svc1=lr.predict(xtest)
ypred_knn=lr.predict(xtest)
ypred_dtc=lr.predict(xtest)
ypred_nbc=lr.predict(xtest)

#Score
slr=lr.score(xtest,ytest)
sscv1=svc1.score(xtest,ytest)
sknn=knn.score(xtest,ytest)
sdtc=dtc.score(xtest,ytest)
snbc=nbc.score(xtest,ytest)

#accuracy_score
from sklearn.metrics import accuracy_score,recall_score,precision_score,r2_score
acclr=accuracy_score(ypred_lr,ytest)
accsvc1=accuracy_score(ypred_svc1,ytest)
accknn=accuracy_score(ypred_knn,ytest)
accdtc=accuracy_score(ypred_dtc,ytest)
accnbc=accuracy_score(ypred_nbc,ytest)

#confusion_metrix
from sklearn.metrics import confusion_matrix
cmlr=confusion_matrix(ypred_lr,ytest,labels=[0,1,2])
cmsvc1=confusion_matrix(ypred_svc1,ytest,labels=[0,1,2])
cmknn=confusion_matrix(ypred_knn,ytest,labels=[0,1,2])
cmdtc=confusion_matrix(ypred_dtc,ytest,labels=[0,1,2])
cmnbc=confusion_matrix(ypred_nbc,ytest,labels=[0,1,2])

#precision_score
prelr=precision_score(ypred_lr,ytest,average=None)
presvc1=precision_score(ypred_svc1,ytest,average=None)
preknn=precision_score(ypred_knn,ytest,average=None)
predtc=precision_score(ypred_dtc,ytest,average=None)
prenbc=precision_score(ypred_nbc,ytest,average=None)

#recall_score
relr=recall_score(ypred_lr,ytest,average=None)
resvc1=recall_score(ypred_svc1,ytest,average=None)
reknn=recall_score(ypred_knn,ytest,average=None)
redtc=recall_score(ypred_dtc,ytest,average=None)
renbc=recall_score(ypred_nbc,ytest,average=None)

#model_compare:
model=pd.DataFrame({'Model':['lr','svm1','knn','dtc','nbc'],'Score':[slr,sscv1,sknn,sdtc,snbc],'Accuracy_score':[acclr,accsvc1,accknn,accdtc,accnbc],'Recall':[relr,resvc1,reknn,redtc,renbc],'precision':[prelr,presvc1,preknn,predtc,prenbc]})

model

from sklearn.model_selection import cross_val_score,KFold
cross_val_score(lr,x,y,cv=5,ran)

cc=KFold(n_splits=5,shuffle=True,random_state=43)

cross_val_score(lr,x,y,cv=cc)



