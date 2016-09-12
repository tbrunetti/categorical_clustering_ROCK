import argparse
import numpy as np
from heapq import heappush, heappop
from sklearn.metrics import jaccard_similarity_score
from itertools import combinations
import pandas as pd

def initiate_matrix(inputMatrixData, colName, rowName):
	
	patientProfile=[]
	with open(inputMatrixData) as inputData:
		for line in inputData:
			line = line.rstrip('\n').split('\t')
			patientProfile.append(line)
	#convert to numpy array/matrix
	patientProfile=np.array(patientProfile)

	# nxn matrix representing patient profile similarity comparisons in pair-wise manner
	jaccardMatrix = np.zeros((len(patientProfile), len(patientProfile)))
	
	# nxn matrix representing patient profiles that are linked together based on JC similarity threshold
	neighborMatrix = np.zeros((len(patientProfile), len(patientProfile)))

	# naming of rows and columns
	def getNames(inputMatrixData, colName, rowName):
		patNames=[]
		attributeNames=[]
		if rowName != 'None':
			with open(rowName) as patIDs:
				for line in patIDs:
					patNames.append(line.rstrip('\n'))
		else:
			patNames=list(range(len(patientProfile)))

		if colName != 'None':
			with open(colName) as attIDs:
				for line in attIDs:
					attributeNames.append(line.rstrip('\n'))
		else:
			attributeNames=list(range(len(patientProfile[0])))

		return patNames, attributeNames

	
	patNames, attributeNames = getNames(inputMatrixData, colName, rowName)
	labeled_patProfile = pd.DataFrame(patientProfile, index=patNames, columns=attributeNames)
	return labeled_patProfile, patientProfile, jaccardMatrix, neighborMatrix, patNames, attributeNames


def calculate_similarity(patientProfile):

	# calculate the Jaccard Coeffient for between each pair of patients and normalizes to value between 0 and 1
	# high JC, closer to 1, means more similar
	for i in range(0, len(patientProfile)):
		for j in range(0, len(patientProfile)):
			jaccardMatrix[i][j] = jaccard_similarity_score(patientProfile[i], patientProfile[j], normalize=True)

	return jaccardMatrix


def calculate_neighbors(neighborMatrix, jaccardMatrix, patNames, threshold):
	# if greater or equal than threshold, recorded as linked (1) or if less than unlinked (0)
	for i in range(0, len(jaccardMatrix)):
		for j in range(0, len(jaccardMatrix)):
			if jaccardMatrix[i][j] >= threshold:
				neighborMatrix[i][j] = 1

	numLinks = np.dot(neighborMatrix, neighborMatrix)
	labeled_numLinks = pd.DataFrame(numLinks, index=patNames, columns=patNames)
	print labeled_numLinks
	return labeled_numLinks


# for determination of cluster merging
def fitness_measure(labeled_numLinks, threshold):
	pairwise_fitness = []
	# function_theta can be changed to fit data set of interest, below is the most commonly used one
	function_theta = (1.0 - threshold)/(1.0 + threshold)
	
	pairs=combinations(labeled_numLinks, 2)
	
	# calculate the fitness between all combinations of cluster pairs
	for elements in pairs:
		total_links_between_clusters = labeled_numLinks[elements[0]][elements[1]]
		bothClusters = (sum(labeled_numLinks[elements[0]]!=0) + sum(labeled_numLinks[elements[1]]!=0))**(1+2*(function_theta))
		firstCluster = sum(labeled_numLinks[elements[0]]!=0)**(1+2*(function_theta))
		secondCluster = sum(labeled_numLinks[elements[1]]!=0)**(1+2*(function_theta))
		totalDenominator = bothClusters - firstCluster - secondCluster
		fitnessMeasure = float(total_links_between_clusters) / float(totalDenominator)

		# (cluster number, cluster number, fitness between clusters)
		pairwise_fitness.append((elements[0], elements[1], fitnessMeasure))

	pairwise_fitness.sort(key=lambda tup: tup[2], reverse=True)
	print pairwise_fitness
	return pairwise_fitness



def merge_and_update(pairwise_fitness, labeled_numLinks):

	# merges clusters	
	for column in labeled_numLinks:
		labeled_numLinks[column] = labeled_numLinks[column]+labeled_numLinks[pairwise_fitness[0][1]]

	# relabels clusters
	labeled_numLinks = labeled_numLinks.drop(pairwise_fitness[0][1], axis=1)
	labeled_numLinks = labeled_numLinks.drop(pairwise_fitness[0][1])
	labeled_numLinks.rename(columns={pairwise_fitness[0][0]:str(pairwise_fitness[0][0])+','+str(pairwise_fitness[0][1])}, inplace=True)
	labeled_numLinks.rename(index={pairwise_fitness[0][0]:str(pairwise_fitness[0][0])+','+str(pairwise_fitness[0][1])}, inplace=True)
	return labeled_numLinks



if __name__=='__main__':
	parser=argparse.ArgumentParser(description='Categorical classification using ROCK (RObust Clustering using linKs)')
	parser.add_argument('-kclusters', default='2', dest='kclusters', type=int, help='[INT] Number of subtypes or clusters to expect default: 2')
	#column=attributes, row=patients
	parser.add_argument('-input', required=True, dest='matrixFile', help='Full path to tab-delimited "matrix" file')
	parser.add_argument('-threshold', default='0.5', dest='minJC', type=float, help='[FLOAT] Number between 0 and 1 for min Jaccard Coeffienct to be considered a neighbor (not same as linked) default: 0.5')
	parser.add_argument('--colNames', default='None', dest='colNames', help='file with order column names, one name per line, should be list of attribute/dims')
	parser.add_argument('--rowNames', default='None', dest='rowNames', help='file with order of row names, one name per line, should be patient IDs')
	args=parser.parse_args()

	# initiate_matrix, calculate_similarity, and calculate_neighbors only need to be called once
	labeled_patProfile, patientProfile, jaccardMatrix, neighborMatrix, patNames, attributeNames = initiate_matrix(inputMatrixData=args.matrixFile, colName=args.colNames, rowName=args.rowNames)
	jaccardMatrix = calculate_similarity(patientProfile)
	labeled_numLinks = calculate_neighbors(neighborMatrix, jaccardMatrix, patNames, threshold=args.minJC)
	
	while len(labeled_numLinks) > args.kclusters:
		pairwise_fitness = fitness_measure(labeled_numLinks, threshold=args.minJC)
		labeled_numLinks = merge_and_update(pairwise_fitness, labeled_numLinks)
	print labeled_numLinks