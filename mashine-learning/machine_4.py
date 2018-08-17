#################################################
# 本案例用来 测试 K-mean聚类算法
#################################################

from numpy import *
import kmeans
import time
import matplotlib.pyplot as plt

## step 1: load data
print("step 1: load data...")
dataSet = []
fileIn = open('mashine_d_4.txt')
for line in fileIn.readlines():
    lineArr = line.strip().split('\t')
    print(lineArr)
    dataSet.append([float(lineArr[0]), float(lineArr[1])])

## step 2: clustering...
print("step 2: clustering...")
dataSet = mat(dataSet)
print(dataSet)
k = 4
centroids, clusterAssment = kmeans.kmeans(dataSet, k)

## step 3: show the result
print("step 3: show the result...")
kmeans.showCluster(dataSet, k, centroids, clusterAssment)