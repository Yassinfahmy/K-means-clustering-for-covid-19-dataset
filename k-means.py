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
col=list(df.columns)

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
idealClusters=silhouette_coefficients.index(max(silhouette_coefficients))+2

km              = KMeans(n_clusters=idealClusters, random_state=0).fit(df)
clusters        =km.labels_
clusters_centers=km.cluster_centers_

#how do you want to validate the model?



#show the characteristics of each cluster
cluster_1=np.count_nonzero(clusters==0)
cluster_2=np.count_nonzero(clusters==1)
c_data=np.concatenate((np.array(patient_info_mod),np.reshape(clusters,(len(clusters),1))),axis=1)

width=0.4
labels=['Released','Isolated','Female','Male','Inf-patient Contact']
c=np.empty([idealClusters,len(labels)])
for i in range(idealClusters-1):
    c[i]=[\
        sum(df.loc[clusters==i,'state_released']),\
        sum(df.loc[clusters==i,'state_isolated']),\
        sum(df.loc[clusters==i,'sex_female']),\
        sum(df.loc[clusters==i,'sex_male']),\
        sum(df.loc[clusters==i,'infection_case_contact with patient'])\
            ]

x=np.arange(len(labels))
f, ax =plt.subplots()
r1=ax.bar(x - width/2,c[0],width,label='Cluster 1')
r2=ax.bar(x + width/2,c[1],width,label='Cluster 2')

ax.set_ylabel('Counts')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()

#any other ideas?