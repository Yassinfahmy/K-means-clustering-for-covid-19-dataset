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


#show the characteristics of each cluster
cluster_1_no=np.count_nonzero(clusters==0)
cluster_2_no=np.count_nonzero(clusters==1)
c_data=np.concatenate((np.array(patient_info_mod),np.reshape(clusters,(len(clusters),1))),axis=1)

width=0.4
labels=['Released','Isolated','Female','Male','Inf-patient Contact']
c=np.empty([idealClusters,len(labels)])

clust1=df.loc[clusters==0]
clust2=df.loc[clusters==1]

for i in range(idealClusters):
    c[i]=[
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

del labels
del f 
del ax
labels=['Female','Male','Under 60', 'Over 60', 'Country China', 'Other Country']
x=np.arange(len(labels))
demos=np.empty([idealClusters,len(labels)])

demos1=np.array([\
        len(clust1.loc[clust1.sex_female==1]),\
        len(clust1.loc[clust1.sex_male==1]),\
        len(clust1.loc[((clust1.age_0s==1) | (clust1.age_10s==1) | (clust1.age_20s==1) | (clust1.age_30s==1) | (clust1.age_40s==1) | (clust1.age_50s==1))]),\
        len(clust1.loc[((clust1.age_60s==1) | (clust1.age_70s==1) | (clust1.age_80s==1) | (clust1.age_90s==1) | (clust1.age_100s==1))]),\
        len(clust1.loc[clust1.country_China==1]),\
        len(clust1.loc[clust1.country_China==0])\
        ])

demos2=np.array([\
        len(clust2.loc[clust2.sex_female==1]),\
        len(clust2.loc[clust2.sex_male==1]),\
        len(clust2.loc[((clust2.age_0s==1) | (clust2.age_10s==1) | (clust2.age_20s==1) | (clust2.age_30s==1) | (clust2.age_40s==1) | (clust2.age_50s==1))]),\
        len(clust2.loc[((clust2.age_60s==1) | (clust2.age_70s==1) | (clust2.age_80s==1) | (clust2.age_90s==1) | (clust2.age_100s==1))]),\
        len(clust2.loc[clust2.country_China==1]),\
        len(clust2.loc[clust2.country_China==0])\
        ])
    
    
f, ax =plt.subplots()
ax.bar(x - width/2,demos1,width,label='Cluster 1')
ax.bar(x + width/2,demos2,width,label='Cluster 2')

ax.set_ylabel('Counts')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.title('Cluster Demographics')
plt.show()


###########################################################  
del labels
labels=['Cluster 1' , 'Cluster 2']


counts=[cluster_1_no,cluster_2_no]

f, ax = plt.subplots()
ax.pie(counts,labels=labels,autopct='%1.1f')
ax.axis('equal')
plt.title('Data Split')
plt.show()




    
###########################################################  Males
counts=[len(clust1.loc[(clust1.sex_male==1) & (clust1.state_released==1)]),len(clust2.loc[(clust2.sex_male==1) & (clust2.state_released==1)])    ]

f, (ax1,ax2) = plt.subplots(1,2)
ax1.pie(counts,autopct='%1.1f')
ax1.axis('equal')
ax1.set_title('Males Released')



counts=[len(clust1.loc[(clust1.sex_male==1) & (clust1.state_released==0)]),len(clust2.loc[(clust2.sex_male==1) & (clust2.state_released==0)])    ]

#f, ax = plt.subplots()
ax2.pie(counts,autopct='%1.1f')
ax2.axis('equal')
ax2.set_title('Males Isolated')
plt.legend(labels,loc="upper left")

plt.show()


###########################################################   Females
counts=[len(clust1.loc[(clust1.sex_female==1) & (clust1.state_released==1)]),len(clust2.loc[(clust2.sex_female==1) & (clust2.state_released==1)])    ]

f, (ax1,ax2) = plt.subplots(1,2)
ax1.pie(counts,autopct='%1.1f')
ax1.axis('equal')
ax1.set_title('Females Released')



counts=[len(clust1.loc[(clust1.sex_female==1) & (clust1.state_released==0)]),len(clust2.loc[(clust2.sex_female==1) & (clust2.state_released==0)])    ]

#f, ax = plt.subplots()
ax2.pie(counts,autopct='%1.1f')
ax2.axis('equal')
ax2.set_title('Females Isolated')
plt.legend(labels,loc="upper left")

plt.show()



###########################################################   Young

counts=[len(clust1.loc[((clust1.age_0s==1) | (clust1.age_10s==1) | (clust1.age_20s==1) | (clust1.age_30s==1) | (clust1.age_40s==1) | (clust1.age_50s==1))\
                       & (clust1.state_released==1)]),\
        len(clust2.loc[((clust2.age_0s==1) | (clust2.age_10s==1) | (clust2.age_20s==1) | (clust2.age_30s==1) | (clust2.age_40s==1) | (clust2.age_50s==1))\
                       & (clust2.state_released==1)])    ]

    
    
 

f, (ax1,ax2) = plt.subplots(1,2)
ax1.pie(counts,autopct='%1.1f')
ax1.axis('equal')
ax1.set_title('Under 60s Released')



counts=[len(clust1.loc[((clust1.age_0s==1) | (clust1.age_10s==1) | (clust1.age_20s==1) | (clust1.age_30s==1) | (clust1.age_40s==1) | (clust1.age_50s==1))\
                       & (clust1.state_released==0)]),\
        len(clust2.loc[((clust2.age_0s==1) | (clust2.age_10s==1) | (clust2.age_20s==1) | (clust2.age_30s==1) | (clust2.age_40s==1) | (clust2.age_50s==1))\
                       & (clust2.state_released==0)])    ]


ax2.pie(counts,autopct='%1.1f')
ax2.axis('equal')
ax2.set_title('Under 60s Isolated')
plt.legend(labels,loc="upper left")
plt.show()




    
###########################################################  Old

counts=[len(clust1.loc[((clust1.age_60s==1) | (clust1.age_70s==1) | (clust1.age_80s==1) | (clust1.age_90s==1) | (clust1.age_100s==1))\
                       & (clust1.state_released==1)]),\
        len(clust2.loc[((clust2.age_60s==1) | (clust2.age_70s==1) | (clust2.age_80s==1) | (clust2.age_90s==1) | (clust2.age_100s==1))\
                       & (clust2.state_released==1)])    ]
    
    
f, (ax1,ax2) = plt.subplots(1,2)
ax1.pie(counts,autopct='%1.1f')
ax1.axis('equal')
ax1.set_title('Over 60s Released')



counts=[len(clust1.loc[((clust1.age_60s==1) | (clust1.age_70s==1) | (clust1.age_80s==1) | (clust1.age_90s==1) | (clust1.age_100s==1))\
                       & (clust1.state_released==0)]),\
        len(clust2.loc[((clust2.age_60s==1) | (clust2.age_70s==1) | (clust2.age_80s==1) | (clust2.age_90s==1) | (clust2.age_100s==1))\
                       & (clust2.state_released==0)])    ]


ax2.pie(counts,autopct='%1.1f')
ax2.axis('equal')
ax2.set_title('Over 60s Isolated')
plt.legend(labels)
plt.show()



