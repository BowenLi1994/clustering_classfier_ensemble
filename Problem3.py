# -*- coding: utf-8 -*-

import numpy as np
import random


def load():
    data=np.loadtxt( 'Data for Assignment 4/Seeds Data/seeds.txt',delimiter='    ')
    return data

def distEclud(vecA, vecB):                  
    #    return np.linalg.norm(vecA - vecB) 
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

#def calcDis(data,clu):
#    clalist=[]  #save distance
##    data=data.tolist()
##    clu=clu.tolist()
#    for i in range(data.shape[0]):
#        clalist.append([])
#        for j in range(clu.shape[0]):
#            dist= np.linalg.norm(data[i,:] - clu[j,:])
#            clalist[i].append(dist)
#    clalist=np.array(clalist)
#    return clalist

def randCent(data, k):
    feature_dimension_number = data.shape[1]   
    #create centers matrix (k*n)
    centroids = np.mat(np.zeros((k,feature_dimension_number)))
    for j in range(feature_dimension_number):     
        minJ = min(data[:,j])          
        rangeJ = float(max(data[:,j]) - minJ)   
        centroids[:,j] = np.mat(minJ + rangeJ * np.random.rand(k,1))
    return centroids

def randChosenCent(data,k):
    m=data.shape[0]
    centroidsIndex=[]
    dataIndex=list(range(m))
    for i in range(k):
        randIndex=np.random.randint(0,len(dataIndex))
        centroidsIndex.append(dataIndex[randIndex])
        del dataIndex[randIndex]
    centroids = data[centroidsIndex]
    return np.mat(centroids)


def kMeansSSE(data,k,distMeas=distEclud, createCent=randCent):
    m = data.shape[0]
    clusterAssment=np.mat(np.zeros((m,2)))
    centroids = createCent(data, k)
    print('initial centroids=',centroids)
    sseOld=0
    sseNew=np.inf
    iterTime=0 
    while(abs(sseNew-sseOld)>0.001 and iterTime<100):
        sseOld=sseNew
        for i in range(m):
            minDist=np.inf
            minIndex=-1
            for j in range(k):
                distJI=distMeas(centroids[j,:],data[i,:])
                if distJI<minDist:
                    minDist=distJI
                    minIndex=j
            clusterAssment[i,:]=minIndex,minDist**2 
        iterTime+=1
        sseNew=sum(clusterAssment[:,1])
        print('the SSE of %d'%iterTime + 'th iteration is %f'%sseNew)

        for cent in range(k):
            ptsInClust=data[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust, axis=0)
    return sseNew, centroids, clusterAssment



if 1:
    data = load()

    sseAveResult = {}
    for k in [3,5,7]:
        sse = []
        for i in range(10):
              sseResult,mycentroids,clusterAssment=kMeansSSE(data,k)
              sse.append(sseResult)
        ave_sse = np.mean(sse)
        sseAveResult[k] = ave_sse
    print(sseAveResult)
 
