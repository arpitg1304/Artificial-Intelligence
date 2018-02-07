# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 05:40:15 2017

@author: Arpit
"""
print('Applied Affinity Propagation of scikit learn digit dataset\n')
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import AffinityPropagation
from sklearn import metrics

#Loading digits dataset into the digits variable
digits = load_digits(n_class=10)

#Getting the data from the dataset
data = digits.data

#Getting the labels
truth_labels = digits.target

#Converting labels array to python list
labels = truth_labels.tolist()

#Count is an array of size 10 having the counts of all the digits in the dataset, initially np.zeros(10)
counts = np.zeros(10)

#Assigning the count of digits for every digit to the counts array
for i in range(len(labels)):
    counts[labels[i]] += 1

print('Original count of digits for every number(0-9)\n')
for i in range(len(counts)):
    print(str(i) +': '+ str(counts[i]))

print('\n')

affinity = AffinityPropagation(damping=0.9, preference = -70000, affinity='euclidean')

predictions = affinity.fit_predict(data)

#Creating a list of lists of 10 x 10 size
cluster_wise = [[0 for i in range(10)] for j in range(10)]

for i in range(len(predictions)):
	clusterNumber = predictions[i]
	digit = truth_labels[i]
	cluster_wise[clusterNumber][digit] += 1
    
final_labels = []

for counts in cluster_wise:
    final_labels.append(counts.index(max(counts)))

print('Cluster wise digit counts \n')  
for i in range(len(cluster_wise)):
    print(str(cluster_wise[i]) + ':     ' + str(final_labels[i]))

combined_list = []

for i in range(len(cluster_wise)):
    combined_list.append((final_labels[i] , cluster_wise[i]))

combined_list.sort()

print('\n')
print('Confusion Matrix\n')

#Printing the confusion matrix
for i in range(len(combined_list)):
    print(combined_list[i][1])
    
a1 = truth_labels.tolist()
a2 = predictions.tolist()
fowlkes_mallows_score = metrics.fowlkes_mallows_score(a1,a2)

print("\nThe Fowlkes-Mallow score is :",fowlkes_mallows_score)