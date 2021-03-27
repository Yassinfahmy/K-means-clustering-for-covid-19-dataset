# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 20:57:19 2021

@author: Yassin Fahmy and Gianna Jordan
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

# case           =pd.read_csv('Case.csv')
patient_info   =pd.read_csv('PatientInfo.csv')
# time           =pd.read_csv('Time.csv')
# time_age       =pd.read_csv('TimeAge.csv')
# time_gender    =pd.read_csv('TimeGender.csv')
# time_provinance=pd.read_csv('TimeProvince.csv')

#preprocessing
patient_info_mod= patient_info[patient_info['age'].notna()]
patient_info_mod= patient_info_mod[['sex','age','country','province','city','infection_case','state']]
patient_info_mod= patient_info_mod[patient_info_mod['infection_case'].notna()]
patient_info_mod= patient_info_mod[patient_info_mod['city'].notna()]
patient_info_mod= patient_info_mod[patient_info_mod['sex'].notna()]
#one hot encoding
df              =pd.get_dummies(patient_info_mod[['sex','age','country','province','city','infection_case','state']])

#define some variables
clusters                =[]
clusters_centers        =[]
inertia                 =[]
silhouette_coefficients =[]

#try different number of clusters
for i in np.arange(2,16,1):
    km              = KMeans(n_clusters=i, random_state=0).fit(df)
    inertia.append(km.inertia_)
    silhouette_coefficients.append(silhouette_score(df, km.labels_))
    
#plot Sum of squared distances of samples to their closest cluster center vs # of clusters
plt.plot(np.arange(2,16,1),inertia,color='r',marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()
plt.plot(np.arange(2,16,1),silhouette_coefficients,color='g',marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score')
plt.show()

#chosen model 2 or 4 clusters, what do you think Gia?
km              = KMeans(n_clusters=2, random_state=0).fit(df)
clusters        =km.labels_
clusters_centers=km.cluster_centers_

#how do you want to validate the model?
#show the characteristics of each cluster
#any other ideas?