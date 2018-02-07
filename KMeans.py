
# coding: utf-8

# In[48]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_table('realdata.txt')
#print(df.info)
data = np.array(df)
i = 0
X = np.array(df)
colors = 10 * ["y", "g", "b", "r"]
new_data = np.array(df)
# Plotting Original Data
for x in X:
    new_data[i][0] = x[1]
    new_data[i][1] = x[2]
    i = i+1
    plt.scatter(x[1], x[2], marker = "o", color = colors,  s = 15, linewidth = 1)

plt.title('Original Data')
plt.xlabel('Length')
plt.ylabel('Width')
plt.show()

class KMeansAlgo:
    
    def __init__(self, cl = 2, tol = .001, iters = 300):
        self.cl = cl
        self.tol = tol
        self.iters = iters
        
    def fit(self, data, cl):
        self.means = {}
        
        #Selecting first cl datapoints as intitial means
        for i in range(cl):
            self.means[i] = data[i]
        
        #iterating the k-means for 300 times(generally it does not take more than 300 iterations for the algorithm to converge)
        for i in range(300):
            self.classes = {}
            
            #Number of classes equals number of number of clusters
            for i in range(cl):
                self.classes[i] = []
            
            #Finding eucledian distances between the data point and all the centroids, select the mean with which the data
            #point has minimum distance
            for dataset in data:
                eucledians = [np.linalg.norm(dataset - self.means[mean]) for mean in self.means]
                min_dist = eucledians.index(min(eucledians))
                self.classes[min_dist].append(dataset)

            #saving old centroids in a variable
            means_old = dict(self.means)

            #Updating the new means as the average of the data_points in all the classes
            for class_i in self.classes:
                self.means[class_i] = np.average(self.classes[class_i], axis =0)

            converged = True
            
            #checking the shift between old and new entroids against a threshold value, if it is greater than the threshold, then the 
            #algorithm will continue checking
            for m in self.means:
                mean_original = means_old[m]
                mean_current = self.means[m]
                if np.sum((mean_current - mean_original)/ mean_original * 100.0) > .0001:
                    converged = False

                if converged:
                    break
    
#Using the classifier and fitting the data in it
clf = KMeansAlgo()
clf.fit(new_data,2)

#Plotting the means on the scatter plot

fig, ax = plt.subplots()
t = 0
for m in clf.means:
    clust = 'Cluster' + str(t+1)
    ax.scatter(clf.means[m][0], clf.means[m][1], color = "k", s = 15, linewidth = 10)
    ax.annotate(clust, (clf.means[m][0], clf.means[m][1]), fontsize = 15, color = "Black")
    t = t+1

#Plotting all the data points
for class_i in clf.classes:
    color = colors[class_i]
    for dataset in clf.classes[class_i]:
        ax.scatter(dataset[0], dataset[1], marker = "o", color = color, s = 15, linewidth = 1)

plt.title('Clustered Data')
plt.xlabel('Length')
plt.ylabel('Width')
plt.show()


# In[ ]:



