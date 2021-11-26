# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import sklearn.cluster
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd
import time

# Imports the dataset as a pandas dataframe and converts the dataframe into a
# numpy array
data = np.array(pd.read_csv("WeatherData.csv", header = None))

# Standardizes data as a numpy array where the columns represent different 
# features
def standardize(data):
  standardized_data = []
 
  for j in range(len(data[0])):
    # Calculates the mean for each specific feature
    mean = np.mean(data[:,j])
    # Calculates the standard deviation of each feature
    std = np.std(data[:,j])
    standardized_feature = []
    
    for i in range(len(data[:,0])):
      # Applies the transformation to each measurment of a particular feature
      standardized_feature.append((data[i,j]-mean)/std)
     
    # Appends the a standardized feature to the data as a row
    standardized_data.append(standardized_feature)
  
  stan_data = np.array(standardized_data)  
  # The transpose of the list of standardized features is returned so that the
  # output is in the same form as the input data 
  return np.transpose(stan_data)

standardizedData = standardize(data)


# Initiates a PCA object with 18 principal components from the standardized data
pca = PCA(n_components = 18)
pca.fit(standardizedData)


# Plots the variance explained by all 18 principal components
plt.bar(list(range(1, 19)) , pca.explained_variance_ratio_, tick_label = 
        list(range(1, 19)),  )
plt.ylabel ("Explained variance ratio")
plt.xlabel ("Principal components")
plt.show()

# Calculates the explained variance of the first 9 principal components
np.sum(pca.explained_variance_ratio_[0:9])


# Transforms the standardized data based on the first 9 principal components
reduced_pca = PCA(n_components = 9 )
reduced_pca.fit(standardizedData)
PCA_data = reduced_pca.transform(standardizedData)

# Creates a list of the inertia for different k valuesin the k-means algorithm
inertia = []
for i in range(1,12):
    kmeans = KMeans(n_clusters = i ).fit(PCA_data)
    inertia.append(kmeans.inertia_)

# Plots the inertia against the k value
plt.plot(range(1,12), inertia)
plt.ylabel( "Inertia" )
plt.xlabel( "Number of clusters" )
plt.show()

# Fits the k-means algorithm to both the standardized data and the PCA data
kmeans = KMeans(n_clusters = 4 ).fit(standardizedData)
PCA_kmeans = KMeans(n_clusters = 4 ).fit(PCA_data)
PCA_kmeans_labels = PCA_kmeans.labels_

# Plots the first two principal components coloured according to the k-means 
# clustering
plt.scatter(PCA_data[:,0], PCA_data[:,1] , c = PCA_kmeans_labels)
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.show()

# Fits a DBSCN algorithm to the PCA data
PCA_dbscan = sklearn.cluster.DBSCAN(eps = 4, min_samples = 18).fit(PCA_data) 
PCA_dbscan_labels = PCA_dbscan.labels_
# Calculates the number of measurments allocated as anomalies
sum(PCA_dbscan_labels == -1)

# 3D plot of the result of DBSCAN
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(PCA_data[:,0] , PCA_data[:,1] , PCA_data[:,2] , c = PCA_dbscan_labels)

# Fits the BIRCH algorithm to the standardized data and PCA data
birch = sklearn.cluster.Birch(n_clusters = 4).fit(standardizedData)
PCA_birch = sklearn.cluster.Birch(n_clusters = 4).fit(PCA_data)
PCA_birch_labels = PCA_birch.labels_

# Plots the results of BIRCH clustering
plt.scatter(PCA_data[:,0], PCA_data[:,1] , c = PCA_birch_labels)
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.show()

# Calculates the silhouette score for k-means and BIRCH
silhouette_score(standardizedData, kmeans.labels_)
silhouette_score(PCA_data, PCA_kmeans.labels_)
silhouette_score(standardizedData, birch.labels_)
silhouette_score(PCA_data, PCA_birch.labels_)

# Calculates the Davies-Bouldin index for k-means and BIRCH
davies_bouldin_score(standardizedData, kmeans.labels_)
davies_bouldin_score(PCA_data, PCA_kmeans.labels_)
davies_bouldin_score(standardizedData, birch.labels_)
davies_bouldin_score(PCA_data, PCA_birch.labels_)

# Randomly splits the dataset into training and testing data with the results 
# of the k-means algorithm used as labels
train_data, test_data, train_labels, test_labels = train_test_split(PCA_data,
    PCA_kmeans_labels, train_size = 0.75, random_state = 10)

# Plots the training data with the different classes labelled
plt.scatter(train_data[train_labels == 0 ,0], train_data[train_labels == 0 ,1] ,
            c = 'red', label = 0)
plt.scatter(train_data[train_labels == 1 ,0], train_data[train_labels == 1 ,1] ,
            c = 'blue', label = 1)
plt.scatter(train_data[train_labels == 2 ,0], train_data[train_labels == 2 ,1] ,
            c = 'green', label = 2)
plt.scatter(train_data[train_labels == 3 ,0], train_data[train_labels == 3 ,1] ,
            c = 'yellow', label = 3)
plt.xlabel("1st Principal Component")
plt.ylabel("2nd Principal Component")
plt.legend()
plt.show()

# Creates a list of average F1 scores for different k values
kNN_f1_scores = []
for i in range(1, 31):
    knNeighbours = KNeighborsClassifier(n_neighbors = i, metric = 'euclidean' )
    knNeighbours.fit(train_data, train_labels)
    kNN_f1_scores.append(f1_score(y_true = test_labels,
                              y_pred = knNeighbours.predict(test_data), 
                              average = 'macro'))

plt.plot(range(1, 31), kNN_f1_scores)
plt.ylabel("F1 score")
plt.xlabel("K value")
plt.show()

# Initiates the K nearest neighbours classifier with a k value of 15
knNeighbours = KNeighborsClassifier(n_neighbors = 15, metric = 'euclidean' )

# Trains the k-nearest neighbours classifier 30 times and records the times taken 
kNN_times = []
for i in range(30):
    start_time = time.time()
    knNeighbours.fit(train_data, train_labels)
    end_time = time.time()
    kNN_times.append(end_time-start_time)

# Initiates a naive bayes classifier with the assumption of normal data
naive_bayes = GaussianNB()

# Trains the naive bayes classifier 30 times and records the times taken 
NB_times = []
for i in range(30):
  start_time = time.time()
  naive_bayes.fit(train_data, train_labels) 
  end_time = time.time()
  NB_times.append(end_time-start_time)

# Creates a list of average F1 scores for different tree depths
dt_f1_scores = []
for i in range(1,16):
  decision_tree = tree.DecisionTreeClassifier(max_depth = i)
  decision_tree.fit(train_data, train_labels)
  dt_f1_scores.append(f1_score(y_true = test_labels,
                               y_pred = decision_tree.predict(test_data),
                               average = 'macro'))

plt.plot(range(1,16), dt_f1_scores)
plt.ylabel("F1 score")
plt.xlabel("Tree Depth")
plt.show()

# Initiates a decision tree classifier with a depth of 8 and gini criterion
decision_tree = tree.DecisionTreeClassifier(max_depth = 8, criterion = "gini")

# Trains the decision tree classifier 30 times and records the times taken 
dt_times = []
for i in range(30):
  start_time = time.time()
  decision_tree.fit(train_data, train_labels)
  end_time = time.time()
  dt_times.append(end_time-start_time)

# Initiates an AdaBoost classifier with a 2 depth decision tree as the weak classifier
adaboost = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 2))

# Trains the AdaBoost classifier 30 times and records the times taken 
adaboost_times = []
for i in range(30):
 start_time = time.time()
 adaboost.fit(train_data, train_labels)
 end_time = time.time()
 adaboost_times.append(end_time-start_time)
 
# Calculates accuracy scores for the different classifiers
accuracy_score(y_true = test_labels, y_pred = knNeighbours.predict(test_data))
accuracy_score(y_true = test_labels, y_pred = naive_bayes.predict(test_data))
accuracy_score(y_true = test_labels, y_pred = decision_tree.predict(test_data))
accuracy_score(y_true = test_labels, y_pred = adaboost.predict(test_data))

# Calculates a list of F1 scores for each of the classifiers
f1_score(y_true = test_labels, y_pred = knNeighbours.predict(test_data),
         average = None)
f1_score(y_true = test_labels, y_pred = naive_bayes.predict(test_data),
         average = None)
f1_score(y_true = test_labels, y_pred = decision_tree.predict(test_data),
         average = None)
f1_score(y_true = test_labels, y_pred = adaboost.predict(test_data),
         average = None)

# Returns the mean of the times taken to train the classifiers
np.mean(kNN_times)
np.mean(NB_times)
np.mean(dt_times)
np.mean(adaboost_times) 
