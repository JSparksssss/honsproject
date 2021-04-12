import math
import collections
import random
import copy
import pylab
import pickle
import numpy
import os
import csv

try:
	import psyco
	psyco.full()
except ImportError:
	pass

FLOAT_MAX = 1e100

class Cluster:
	def __init__(self, visits, segments, keyframes):
		self.visits = [visits]
		self.keyframes = list(keyframes)
		self.segments = [str(segments)]
	def aggregation(self, visits, segments, keyframes):
		for element in visits:
			self.visits.append(element)
		for element in segments:
			self.segments.append(element)
		for element in keyframes:			
			self.keyframes.append(element)

def generatePoints(pointsNumber, radius):
	points = [Point() for _ in range(4 * pointsNumber)]
	originX = [-radius, -radius, radius, radius]
	originY = [-radius, radius, -radius, radius]
	count = 0
	countCenter = 0
	for index, point in enumerate(points):
		count += 1
		r = random.random() * radius
		angle = random.random() * 2 * math.pi
		point.x = r * math.cos(angle) + originX[countCenter]
		point.y = r * math.sin(angle) + originY[countCenter]
		point.group = index
		if count >= pointsNumber * (countCenter + 1):
			countCenter += 1	
	return points

def solveDistanceBetweenPoints(clusterA, clusterB):
    dist_sum = 0
    pair = 0
    for elementA in clusterA:
        for elementB in clusterB:
            dist = numpy.sqrt(numpy.sum(numpy.square(elementA - elementB)))
            dist_sum += dist
            pair += 1
    
    dist_avg = dist_sum/pair
    return dist_avg

def getDistanceMap(clusters):
	distanceMap = {}
	for xindex in clusters.keys():
		for yindex in clusters.keys():
			if(yindex != xindex):
				distanceMap[str(xindex) + '#' + str(yindex)] = solveDistanceBetweenPoints(clusters[xindex].keyframes, clusters[yindex].keyframes)
	distanceMap = sorted(distanceMap.items(), key=lambda dist:dist[1], reverse=False) #length 8911
	return distanceMap

def agglomerativeHierarchicalClustering(clusters, clusterCenterNumber):
	threshold = 40 
	while(len(clusters) > clusterCenterNumber):
		distanceMap = getDistanceMap(clusters)
		for key in distanceMap:
			value = key[1]
			if(value <= threshold):
				low_visit_index = int(key[0].split('#')[0])
				high_visit_index = int(key[0].split('#')[1])
				if(low_visit_index in clusters and high_visit_index in clusters):
					clusters[low_visit_index].aggregation(clusters[high_visit_index].visits,clusters[high_visit_index].segments, clusters[high_visit_index].keyframes)
					del clusters[high_visit_index]
							
		if(len(clusters) <= clusterCenterNumber):
			break
		threshold += 3
	return clusters


def showClusterAnalysisResults(points):
	colorStore = ['or', 'og', 'ob', 'oc', 'om', 'oy', 'ok']
	pylab.figure(figsize=(9, 9), dpi = 80)
	for point in points:
		color = ''
		if point.group < 0:
			color = colorStore[-1 * point.group - 1]
		else:
			color = colorStore[-1]
		pylab.plot(point.x, point.y, color)
	pylab.show()

if __name__ == "__main__":
	arr = []
	clusters = {}
	visit_segment = 0
	# labels = {}
	# for i in range(5):
	# 	index  = i + 1
	# 	my_file_local_path = "0"+str(index)+"\\segmentation_gt ("+str(index)+").txt" #segmentation_gt() belongs to Max
	# 	THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
	# 	my_file = os.path.join(THIS_FOLDER, my_file_local_path)
	# 	with open(my_file) as csvDataFile:
	# 		csvReader = csv.reader(csvDataFile, delimiter="\t")
	# 		for row in csvReader:
	# 			labels[str(visit_segment)] = row[7]
	# 			visit_segment += 1

	# outfile = open('labels','wb')
	# pickle.dump(labels,outfile)
	# outfile.close()

	clusterCenterNumber = 9

	infile = open('VGG16visit_keyframes_sil','rb')
	new_dict = pickle.load(infile)
	infile.close()

	infile = open('labels','rb')
	labels = pickle.load(infile)
	infile.close()

	visit_segment = 0
	for index,element in enumerate(new_dict.values()):
		feat = numpy.array(element)
		feat = feat.reshape(-1,4096)
		cluster = Cluster(labels[str(visit_segment)], visit_segment, feat)
		clusters[visit_segment] = cluster
		visit_segment += 1


	# pair = 0
	# for key in distanceMap:
	# 	lowIndex, highIndex = int(key[0].split('#')[0]), int(key[0].split('#')[1])
	# 	if(labels[lowIndex] == labels[highIndex]):
	# 		arr.append(key[1])
	# 		pair += 1
	# arr = numpy.mean(arr)

	points = agglomerativeHierarchicalClustering(clusters, clusterCenterNumber)
	showClusterAnalysisResults(points)
