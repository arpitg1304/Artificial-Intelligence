
# coding: utf-8

# In[1]:


"""
Created on Sat Nov 11 05:17:26 2017

@author: Arpit
"""

"""
Instruction to use the code: run this python file through terminal or in any IDE, it will ask for first
user input for selecting the classifier(SVM or RandomForest). If the user chooses SVM, it will
ask for one more user input for selecting the karnel to be used with the SVC. On selecting the 
kernel, the result will be printed on the console
"""
import warnings
import pandas as pd
import numpy as np

#suppressing warnings caused by deprications
def warn(*args, **kwargs):
    pass
warnings.warn = warn

import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

#Reading original data file
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

#Taking user inputs for the type of classifier to use and the kernel to use for SVM
classifier = input('Choose the classifier to use(Give 1 for SVM, 2 for RandomForest)')
if classifier == '1':
    kernel = input('Choose the kernel to use(Give 1 for linear, 2 for rbf:) ')
    if kernel == '1':
        kernel = 'linear'
    elif kernel == '2':
        kernel = 'rbf'
    clf = svm.SVC(kernel = kernel)
elif classifier =='2':
    clf = RandomForestClassifier(max_depth=2, random_state=0)

#Preparing data for classifiers
Y = np.array(df1.values)
Y[30] = 1
X = np.array(df2.values)
X =X.reshape(-1,24)
Y.reshape(-1,1)

#Training data
X1 = X[:320,:]
Y1 = Y[:320]

#Testing data
X2 = X[320:401]
Y2 = Y[320:401]

#Fitting the data onto the classifier
clf.fit(X1,Y1)

z2 = df2.iloc[3,:]

zz = []
zz1 = []

#Performing predictions and accumulating the results in zz list
for i in range(0,80):
    z_temp = X2[i]
    zz.append(clf.predict(z_temp).tolist())
    
flat_list = [item for sublist in zz for item in sublist]

for i in range(0,320):
    z_temp = X1[i]
    zz1.append(clf.predict(z_temp).tolist())
    
flat_list1 = [item for sublist in zz1 for item in sublist]

#finding the number of 
matches = 0
TP = 0
FP = 0
FN = 0
for i in range(0,80):
    if flat_list[i] == Y2.tolist()[i]:
        matches +=1
    if flat_list[i] == Y2.tolist()[i] ==1:
        TP += 1
    if flat_list[i] == 1 and Y2.tolist()[i] ==0:
        FP += 1
    if flat_list[i] == 0 and Y2.tolist()[i] ==1:
        FN +=1

matches1 = 0
TP1 = 0
FP1 = 0
FN1 = 0
for i in range(0,320):
    if flat_list1[i] == Y1.tolist()[i]:
        matches1 +=1
    if flat_list1[i] == Y1.tolist()[i] ==1:
        TP1 += 1
    if flat_list1[i] == 1 and Y1.tolist()[i] ==0:
        FP1 += 1
    if flat_list1[i] == 0 and Y1.tolist()[i] ==1:
        FN1 +=1
    
#f_measure = 2 x Pre x Rec / Pre + Rec
#Pre = TP/TP+FP ; Rec = TP/Tp+FN
#and TP is the number of true positives (class 1 members predicted as class 1),
#TN is the number of true negatives (class 2 members predicted as class 2),
#FP is the number of false positives (class 2 members predicted as class 1),
#and FN is the number of false negatives (class 1 members predicted as class 2).
Pre = TP/(TP+FP)
Rec = TP/(TP+FN)

Pre1 = TP1/(TP1+FP1)
Rec1 = TP1/(TP1+FN1)

#calculating and printing the f_measure of the algo
f = 2 * Pre * Rec/ (Pre+Rec)
f1 = 2 * Pre1 * Rec1/ (Pre1+Rec1)

#Printing the accuracy of the algorithm    
print('Number of matches when testing test data '+str(matches))
print('f_measure of the algorithm with test data is:'+str(f))

print('Number of matches when testing training data '+str(matches1))
print('f_measure of the algorithm with training data is:'+str(f1))