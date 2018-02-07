import numpy as np
from sklearn import datasets
from sklearn import metrics

digits = datasets.load_digits(n_class=10)
data = digits.data
labels = digits.target

class KMeansAlgo:
    
    def __init__(self, cl = 10, tol = .001, iters = 100):
        self.cl = cl
        self.tol = tol
        self.iters = iters
        
        
    def fit(self, data, cl):
        self.means = {}
        
        #Selecting first cl datapoints as intitial means
        for i in range(cl):
            self.means[i] = data[i]
        
        #iterating the k-means for 500 times(generally it does not take more than 500 iterations for the algorithm to converge)
        for i in range(500):
            self.classes = {}
            self.labels_class = {}
            self.real_labels = []
            self.indexes = []
        
            #Number of classes equals number of number of clusters
            for j in range(cl):
                self.classes[j] = []
                self.labels_class[j] = []
                #self.indexes[j] = []
            #self.real_labels[j] = []
            #Finding eucledian distances between the data point and all the centroids, select the mean with which the data
            #point has minimum distance
            for i in range(len(data)):
                eucledians = [np.linalg.norm(data[i] - self.means[mean]) for mean in self.means]
                min_dist = eucledians.index(min(eucledians))
                #self.real_labels.append(labels[labels.tolist().index(data.tolist().index(data[i]))])
                self.classes[min_dist].append(data[i])
                self.indexes.append(min_dist)
                self.labels_class[min_dist].append(labels[i])

            #Updating the new means as the average of the data_points in all the classes
            for class_i in self.classes:
                self.means[class_i] = np.average(self.classes[class_i], axis =0)

clf = KMeansAlgo()
clf.fit(data,10)
#for i in range(len(clf.classes)):
#    print(str(i)+ ' ' + str(len(clf.classes[i])))

print('\n')
l_class = []
confusion_matrix = []
for i in range(len(clf.labels_class)):
    temp_zeros = np.zeros(10)
    temp_set = set(clf.labels_class[i])
    temp_list = list(temp_set)
    counts = []
    
    for j in range(len(temp_list)):
        counts.append(clf.labels_class[i].count(temp_list[j]))
        #temp_zeros[j] = clf.labels_class[i].count(temp_list[j])
    
    l_class.append(temp_list[counts.index(max(counts))])
    #confusion_matrix.append(temp_zeros)
    
    for k in range(10):
        temp_zeros[k] = clf.labels_class[i].count(k)
    confusion_matrix.append(temp_zeros)

print('Cluster and its label')
for i in range(len(l_class)):
    print(str(i+1) + ': ' + str(l_class[i]))

print('\n')

print('Confusion Matrix')

for i in range(10):
    print(confusion_matrix[i])
    
predictions = []
list_indexes = []

for i in range(10):
    predictions.append(clf.labels_class[i])
    list_indexes.append(clf.indexes[i])


flat_list = [item for sublist in predictions for item in sublist]
#flat_indexes = [item for sublist in list_indexes for item in sublist] 
l1 = []

for i in range(1797):
    l1.append(labels[clf.indexes[i]])

fowlkes_mallows_score = metrics.fowlkes_mallows_score(l1, flat_list)

#Creating a list of lists of 10 x 10 size
print('\n Clusters and their labels')
for i in range(10):
    print(confusion_matrix[i] , ': ' ,(l_class[i]))

print("\nThe Fowlkes-Mallow score is :",fowlkes_mallows_score)