# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:19:41 2017

@author: Arpit
"""

import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import random

#suppressing warnings caused by deprications
def warn(*args, **kwargs):
    pass

df=pd.read_csv('original_data.csv')

#df =df.drop(['Unnamed: 25'], axis=1)

#Preprocessing the data
df = df.replace({'\t':''}, regex=True)

df=df.replace(r'\?+', np.nan, regex=True)
df[['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']] = df[['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']].apply(pd.to_numeric)
    
#print(df.dm.unique())

#print(df.dtypes)

df.replace('yes', 1, inplace = True)
df.replace('no', 0, inplace = True)

df.replace('notpresent', 1, inplace = True)
df.replace('present', 0, inplace = True)
df.replace('abnormal', 1, inplace = True)
df.replace('normal', 0, inplace = True)
df.replace('poor', 1, inplace = True)
df.replace('good', 0, inplace = True)
df.replace('ckd', 1, inplace = True)
df.replace('notckd', 0, inplace = True)
#Preprocessing of data done

#with pd.option_context('display.max_rows', None, 'display.max_columns', 25):
    #print(df)
    
cols = df.columns
df= df.astype(float)

#Shuffling the data
df = df.sample(frac=1)

#finding the means of all the columns to replace the missing features
means = []
for i in range(0,25):
    temp = df[df.columns[i]].mean()
    means.append(temp)
    df[cols[i]].replace(np.nan, temp, inplace=True)
    
df.to_csv('updated_clean.csv', sep='\t')

df = pd.read_table('updated_clean.csv')
df = df.astype(float)

#Normalizing the data
df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

#df.to_csv('updated_clean_normalized.csv', sep='\t')

df =df.drop(['Unnamed: 0'], axis=1)

df1 = df.iloc[:,24]
df2 =df.drop(['class'], axis=1)

#Function to find the sigmoid (value between 0 and 1)
def sigmoid(z):
	return 1/(1+np.e**(-z))

#Computing the cost function
def cost(x,y,lamda, theta):
	t=theta
	t[0]=0
	m=320
	h=sigmoid(np.dot(x,theta))
	return (( -1.0/m ) * np.sum( y * (np.log (h)) + ( 1.0 -y ) * (np.log(1.0 -h)) )) + ((lamda/(2*m))*np.sum(t*t)) #regularization

#Performing gradient descent on the cost function
def gradientDescent(x, y, lamda, theta):
	t1= theta
	t1[0] = 0
	converge=[]
	m=320
	for i in range (45000):
		converge.append(cost(x, y, lamda, theta))
		if len(converge)>1:
			if converge[i-1]<converge[i]:
				break
		h=sigmoid(np.dot(x, theta))
		grad = ( 1.0/m ) * np.dot(x.T, ( h - y ) ) + (float(lamda)/m) * t1
		theta=theta-0.01*grad
	return theta

#Finding f-measure values
def findF(x,y,theta):
	predictions=np.dot(x,theta)
	for i in range(len(predictions)):
		if predictions[i]>=0: predictions[i]=1
		else: predictions[i]=0
	#print(predictions)
	#quit()
	tp=0
	fp=0
	fn=0
	for j in range(len(y)):
		if predictions[j]==1 and y[j]==1:tp+=1
		if predictions[j]==1 and y[j]==0:fp+=1
		if predictions[j]==0 and y[j]==1:fn+=1
	p=float(tp)/(tp+fp)
	r=float(tp)/(tp+fn)
	f_measure = 2*p*r/(p+r)
	return f_measure

x_i = np.array(df2)
labels = np.array(df1)
ids = np.array(range(400))
id_t = np.array(random.sample(range(400), 320))
idtest = np.setdiff1d(ids,id_t)
var_t = x_i[id_t,:]
classTrain = labels[id_t]
varTest = x_i[idtest,:]
classTest = labels[idtest]
theta=np.array([0]*24)
alpha = 0.01
lamda=list(np.linspace(-2,4,31))
fs1=[]
fs2=[]
for lam in lamda:
    theta=gradientDescent(var_t, classTrain, lam, theta)
    f1=findF(varTest,classTest,theta)
    f2=findF(var_t,classTrain,theta)
    fs1.append(f1)
    fs2.append(f2)
    
plt.plot(lamda,fs2,'r-',label='Training Data')
plt.plot(lamda,fs1,'g-',label='Test Data')
plt.xlabel('Regularization Parameter- Lambda')
plt.ylabel('f-measure')
plt.title('f-measure  vs Regularization Parameter')
plt.legend()
plt.show()