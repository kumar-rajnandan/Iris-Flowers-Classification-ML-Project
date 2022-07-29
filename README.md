# Name - Rajnandan Kumar

# Data Science Intern @ LetsGrowMore
# Task 1
# Beginner Level
# Name of Project : Iris Flowers Classification ML Project
# Dataset http://archive.ics.uci.edu/ml/datasets/Iris
# About Dataset
# • The data set contains 3 classes with 50 instances each, and 150 instances in total, where

# each class refers to a type of iris plant.

# • Class: Iris Setosa,Iris Versicolour, Iris Virginica

# • Format for the data: (sepal_length, sepalwidth, petal_length, petal width)

# ML Algorithm used in the Project
# Machine Learning
# 1)Supervised Machine Learning
# • Decision Tree
#•. Classification
# -KNN
# 2)UnSupervised Machine Learning
# • Clustering
# -Kmean

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv("iris.csv")

df

df.head()

df.tail()

df.isnull().sum()

# Visualization

#catplot

sns.catplot(x = 'Class labels', hue = 'Class labels', kind = 'count', data = df)

# Bar plot foe Class labels vs Petal width

plt.bar(df['Class labels'],df['Petal width'])

# Paired Plot

sns.set()
sns.pairplot(df[['Sepal length','Sepal width','Petal length','Petal width','Class labels']], hue = "Class labels", diag_kind="kde")

df.describe()

df.columns

df.info()

df

# Dropping the Class labels column

x = df.drop(['Class labels'], axis=1)

x

# Encoding the Categorial feature as a one-hot numeric feature

Label_Encode = LabelEncoder()
Y = df['Class labels']
Y = Label_Encode.fit_transform(Y)

Y

x = np.array(x)

x

Y

df['Class labels']. nunique()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size= 0.3, random_state=0)

X_train

X_train.shape

X_test.shape

Y_train.shape

Y_test.shape

# Model Preparation KNN algorithm

# Trainning the model

from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler().fit(X_train)
X_train_std = standard_scaler.transform(X_train)
X_test_std = standard_scaler.transform(X_train)

X_train_std

Y_train

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_std,Y_train)

predict_knn = knn.predict(X_test_std)
accuracy_knn = accuracy_score(Y_train,predict_knn)*100

accuracy_knn

# K mean clustering

df

color_map = np.array(['Red','Green','Blue'])
figure = plt.scatter(df['Petal length'], df['Petal width'], c=color_map[Y], s=30)

x

from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3, random_state=2)
k_means.fit(x)

y_k_means = k_means.fit_predict(x)

centers = k_means.cluster_centers_

centers

color_map=np.array(['Red','Green','Blue'])
labels=np.array(['Iris-setosa', 'Iris-virginica','Iris-versicolour'])
figure=plt.scatter(df['Petal length'], df['Petal width'],c=color_map[k_means.labels_],s=20)

X_train.size

Y_train.size

from sklearn import tree
D_tree = tree.DecisionTreeClassifier()
D_tree.fit(X_train, Y_train)

pred_tree=D_tree.predict(X_test)
accuracy=accuracy_score(Y_test,pred_tree)*100

accuracy

# Thank You
