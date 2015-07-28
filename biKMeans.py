#################################################  
# kmeans: k-means cluster  
# Author :wojiushimogui 
# Date   :  2015年7月28日11:14:01
# HomePage :http://write.blog.csdn.net/postlist 
# Email  : 
################################################# 
# 二分k均值算法的伪代码如下：
#***************************************************************
#将所有数据点看成一个簇
#
#当簇数目小于k时
#
#       对每一个簇
#
#              计算总误差
#
#              在给定的簇上面进行k-均值聚类（k=2）
#
#              计算将该簇一分为二后的总误差
#
#       选择使得误差最小的那个簇进行划分操作
#
#*************************************************************** 
  
from numpy import *  
import time  
import matplotlib.pyplot as plt  
  
  
# calculate Euclidean distance  
def euclDistance(vector1, vector2):  
	return sqrt(sum(power(vector2 - vector1, 2)))  
  
# init centroids with random samples  
def initCentroids(dataSet, k):  
	numSamples, dim = dataSet.shape  
	centroids = zeros((k, dim))  
	for i in range(k):  
		index = int(random.uniform(0, numSamples))  
		centroids[i, :] = dataSet[index, :]  
	return centroids  
  
# k-means cluster  
def kmeans(dataSet, k):  
	numSamples = dataSet.shape[0]  
    # first column stores which cluster this sample belongs to,  
    # second column stores the error between this sample and its centroid  
	clusterAssment = mat(zeros((numSamples, 2)))  
	clusterChanged = True  
  
    ## step 1: init centroids  
	centroids = initCentroids(dataSet, k)  
  
	while clusterChanged:  
		clusterChanged = False  
        ## for each sample  
		for i in range(numSamples):  
			minDist  = 100000.0  
			minIndex = 0  
            ## for each centroid  
            ## step 2: find the centroid who is closest  
			for j in range(k):  
				distance = euclDistance(centroids[j, :], dataSet[i, :])  
				if distance < minDist:  
					minDist  = distance  
					minIndex = j  
              
            ## step 3: update its cluster  
			if clusterAssment[i, 0] != minIndex:  
				clusterChanged = True  
				clusterAssment[i, :] = minIndex, minDist**2  
  
        ## step 4: update centroids  
		for j in range(k):  
			pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]  
			centroids[j, :] = mean(pointsInCluster, axis = 0)  
  
	print ('Congratulations, cluster using k-means complete!'  )
	return centroids, clusterAssment  
  
# bisecting k-means cluster  
def biKmeans(dataSet, k):  
	numSamples = dataSet.shape[0]  
    # first column stores which cluster this sample belongs to,  
    # second column stores the error between this sample and its centroid  
	clusterAssment = mat(zeros((numSamples, 2)))  
  
    # step 1: the init cluster is the whole data set 
	#mean(dataSet,axis=0)返回的是：matrix([[XXX,XXX,XXX]])
	#mean(dataSet, axis = 0).tolist()返回的是：[[XXX,XXX,XXX]],
	#mean(dataSet, axis = 0).tolist()[0],返回的是：[XXX,XXX,XXX]
	centroid = mean(dataSet, axis = 0).tolist()[0]  #列表中的第一个列表元素：即全部数据每个属性对应的均值
	centList = [centroid]  #centList是[[xxx,xxx,xxx]]
	for i in range(numSamples):  
		clusterAssment[i, 1] = euclDistance(mat(centroid), dataSet[i, :])**2 #计算每个样本点与质心之间的距离 
  
	while len(centList) < k:  
        # min sum of square error  
		minSSE = 100000.0  
		numCurrCluster = len(centList)  #当前的簇数
        # for each cluster 
		#找出numCurrCluster个簇中哪个簇分解得到的误差平方和最小的两个簇
		for i in range(numCurrCluster):  
            # step 2: get samples in cluster i
			#选取第i个簇的所有数据，然后将其分成两个两个簇
			pointsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]  
  
            # step 3: cluster it to 2 sub-clusters using k-means 
			#centroids的元素为每个簇的质心
			#splitClusterAssment第一列为样本所属的类别号，第二列为样本到其所属簇的质心的距离的平方
			centroids, splitClusterAssment = kmeans(pointsInCurrCluster, 2)  
  
            # step 4: calculate the sum of square error after split this cluster
			#下面的代码是求误差平方和
			#splitSSE=sum(power(splitClusterAssment[:,1],2))
			splitSSE = sum(splitClusterAssment[:, 1]) 
			#不是标号为第i个簇的误差平方和
			notSplitSSE = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])  
			currSplitSSE = splitSSE + notSplitSSE  #当前所有簇的平方和  
  
            # step 5: find the best split cluster which has the min sum of square error  
			if currSplitSSE < minSSE:  #
				minSSE = currSplitSSE  
				bestCentroidToSplit = i  
				bestNewCentroids = centroids.copy()  
				bestClusterAssment = splitClusterAssment.copy()  
  
        # step 6: modify the cluster index for adding new cluster
		#将新分出来的两个簇的标号一个沿用它父亲的标号，一个用簇的总数来标号。
		bestClusterAssment[nonzero(bestClusterAssment[:, 0].A == 1)[0], 0] = numCurrCluster  
		bestClusterAssment[nonzero(bestClusterAssment[:, 0].A == 0)[0], 0] = bestCentroidToSplit  
  
        # step 7: update and append the centroids of the new 2 sub-cluster 
		
		centList[bestCentroidToSplit] = bestNewCentroids[0, :] #将第一个子簇的质心放在父亲质心的原位置 
		centList.append(bestNewCentroids[1, :]) #将第二个子簇的质心添加在末尾 
  
        # step 8: update the index and error of the samples whose cluster have been changed
		#由第i个簇分解为j、k两个簇所得到的数据将分解之前的数据替换掉
		clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentroidToSplit), :] = bestClusterAssment  
  
	print ('Congratulations, cluster using bi-kmeans complete!' ) 
	return mat(centList), clusterAssment  
  
# show your cluster only available with 2-D data  
def showCluster(dataSet, k, centroids, clusterAssment):  
	numSamples, dim = dataSet.shape  
	if dim != 2:  
		print ("Sorry! I can not draw because the dimension of your data is not 2!"  )
		return 1  
  
	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
	if k > len(mark):  
		print ("Sorry! Your k is too large! please contact wojiushimogui")  
		return 1  
  
    # draw all samples  
	for i in range(numSamples):  
		markIndex = int(clusterAssment[i, 0])  
		plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])  
  
	mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
    # draw the centroids  
	for i in range(k):  
		plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)  
          
	plt.show() 