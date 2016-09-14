import argparse
from time import gmtime, strftime
import numpy as np
from sklearn.metrics import jaccard_similarity_score
from itertools import combinations
import pandas as pd
import os
import math

def initiate_matrix(inputMatrixData, colName, rowName):
	
	patientProfile=[]
	with open(inputMatrixData) as inputData:
		for line in inputData:
			line = line.rstrip('\n').split('\t')
			patientProfile.append(line)
	#convert to numpy array/matrix
	patientProfile=np.array(patientProfile)

	# nxn matrix representing patient profile similarity comparisons in pair-wise manner
	similarityMatrix = np.zeros((len(patientProfile), len(patientProfile)))
	
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
	return labeled_patProfile, patientProfile, similarityMatrix, neighborMatrix, patNames, attributeNames


def calculate_similarity(patientProfile, similarityMatrix, metric):
	
	def calc_SMC(patientProfile, similarityMatrix):
		# calculate the simple matching coefficient
		# equal weights are placed on a matching 0's as matching 1's
		# higher SMC, closer to 1, means more similar
		for i in range(0, len(patientProfile)):
			for j in range(0, len(patientProfile)):
				first = patientProfile[i]
				second = patientProfile[j]
				total_matches = sum([int(1) for a,b in zip(first, second) if a==b])
				percent_match = float(float(total_matches)/len(patientProfile[0]))
				similarityMatrix[i, j] = percent_match
		
		return similarityMatrix

	def calc_JC(patientProfile, similarityMatrix):
		# calculate the Jaccard Coeffient for between each pair of patients and normalizes to value between 0 and 1
		# high JC, closer to 1, means more similar
		for i in range(0, len(patientProfile)):
			for j in range(0, len(patientProfile)):
				similarityMatrix[i, j] = jaccard_similarity_score(patientProfile[i], patientProfile[j], normalize=True)
		
		return similarityMatrix

	if metric == 'JC':
		calc_JC(patientProfile, similarityMatrix);
		return similarityMatrix
	elif metric =='SMC':
		calc_SMC(patientProfile, similarityMatrix);
		return similarityMatrix
	else:
		print "error:  metric specified does not exist"
	

def calculate_neighbors(neighborMatrix, similarityMatrix, patNames, threshold):
	# if greater or equal than threshold, recorded as linked (1) or if less than unlinked (0)
	for i in range(0, len(similarityMatrix)):
		for j in range(0, len(similarityMatrix)):
			if similarityMatrix[i][j] >= threshold:
				neighborMatrix[i][j] = 1

	numLinks = np.dot(neighborMatrix, neighborMatrix)
	labeled_numLinks = pd.DataFrame(numLinks, index=patNames, columns=patNames)

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

	# sorts list so first tuple is most fit cluster pair
	pairwise_fitness.sort(key=lambda tup: tup[2], reverse=True)
	return pairwise_fitness



def merge_and_update(pairwise_fitness, labeled_numLinks):

	# merges clusters	
	for column in labeled_numLinks:
		labeled_numLinks[column] = labeled_numLinks[column]+labeled_numLinks[pairwise_fitness[0][1]]

	# relabels clusters post-merging
	labeled_numLinks = labeled_numLinks.drop(pairwise_fitness[0][1], axis=1)
	labeled_numLinks = labeled_numLinks.drop(pairwise_fitness[0][1])
	labeled_numLinks.rename(columns={pairwise_fitness[0][0]:str(pairwise_fitness[0][0])+','+str(pairwise_fitness[0][1])}, inplace=True)
	labeled_numLinks.rename(index={pairwise_fitness[0][0]:str(pairwise_fitness[0][0])+','+str(pairwise_fitness[0][1])}, inplace=True)
	
	return labeled_numLinks

def cluster_summaries_binary_attributes(labeled_numLinks, labeled_patProfile, outputDirectory, numClusters, metric, thresh, dataName):
	# makes appropriate output directories and files
	os.mkdir(outputDirectory+'/'+str(dataName))
	os.chdir(outputDirectory+'/'+str(dataName))
	parametersFile = open('parameters_'+str(dataName)+'.txt', 'w')
	parametersFile.write('clusters: '+str(numClusters)+'\n'+'similarity_Metric: '+str(metric)+'\n'+'min_threshold_for_similarity:'+str(thresh))
	outFile = open('final_output_'+str(dataName)+'.txt', 'w')
	row_names=list(labeled_patProfile.index.values)
	attribute_names = list(labeled_patProfile.columns.values)
	column_entropy = []
	
	# calculates entropy of the entire unclustered data set
	for v in range(0, len(attribute_names)):
		num_ones = sum([int(labeled_patProfile[attribute_names[v]][row_names[x]]) for x in range(0, len(row_names))])
		# len(row_names) gets total number of patients
		num_zeros = len(row_names) - num_ones
		if num_ones == 1 or num_ones == 0:
			column_entropy.append(0)
		else:
			partial_entropy_ones = float(num_ones)/len(row_names)*(math.log(float(num_ones)/len(row_names), 2))
			partial_entropy_zeros = float(num_zeros)/len(row_names)*(math.log(float(num_zeros)/len(row_names), 2))
			column_entropy.append(-1*(partial_entropy_ones+partial_entropy_zeros))

	entropy_of_dataset = sum(column_entropy)
	outFile.write('Entropy_of_dataset_unclustered: '+str(entropy_of_dataset)+'\n'+'\n')


	# this function is called each time a new cluster in made, i.e. if 2 clusters, this is called two separate independent times
	def calc_within_cluster_entropy(patIDs, attributes, summary, key, attribute_names, row_names):
		entropy_of_each_attribute = []
		# entropy of 0 means all patients share that attribute
		num_samples_in_cluster = len(patIDs)
		numTot_Attributes = len(summary)

		for i in range(0, numTot_Attributes):
			# calcs the number of samples that have an attribute versus the number that do not
			has_attribute = (float(summary[i])/num_samples_in_cluster)
			not_have_attribute = 1 - float(has_attribute)
			
			# log base 2 of 1 is zero and log base 2 of 0 is undefined so entropy is 0 in both cases
			if has_attribute == 1.0 or has_attribute == 0.0:
				entropy_of_each_attribute.append(0)
			
			# assuming entropy is not 0, this calculates entropy
			else:
				sumOfAtts = sum([has_attribute*(math.log(has_attribute, 2)), not_have_attribute*(math.log(not_have_attribute, 2))])
				entropy = -1*sumOfAtts
				entropy_of_each_attribute.append(entropy)

		# writes entropy for every attribute in each cluster to file
		outFile.write('Entropy of each attribute for clusterID_'+str(key)+'\n')
		outFile.write('\t'.join(attribute_names)+'\n')
		outFile.write('\t'.join([str(entropy_of_each_attribute[x]) for x in range(0, len(entropy_of_each_attribute))])+'\n')

		outFile.write('Total_Entropy_of_cluster_'+str(key)+': '+str(sum(entropy_of_each_attribute))+'\n')
		outFile.write('Weighted_Entropy_of_cluster_'+str(key)+': '+str((float(num_samples_in_cluster)/len(row_names))*(sum(entropy_of_each_attribute)))+'\n')




	# extract cluster names, and attribute names
	clusters = labeled_numLinks.columns.values
	attribute_names.insert(0, 'patID')
	cluster_attributes = {}

	for i in range(0, len(clusters)):
		clusters[i] = clusters[i].split(',')
		cluster_attributes[i] = []
		for x in range(0, len(clusters[i])):
			cluster_attributes[i] = cluster_attributes[i] + [[str(clusters[i][x])] + labeled_patProfile.loc[clusters[i][x]].values.tolist()]
	
	for key in cluster_attributes:
		outFile.write('clusterID_'+str(key)+'\n')
	
		tempArray = np.array(cluster_attributes[key])
		outFile.write('\t'.join(attribute_names)+'\n')
		for x in range(0, len(tempArray)):
			outFile.write('\t'.join(tempArray[x])+'\n')
		
		patIDs, attributes = np.split(tempArray, [1], axis=1)
		# counts total number of pats with each attribute per cluster
		summary = list(np.sum(attributes.astype(int), axis=0))
		
		calc_within_cluster_entropy(patIDs, attributes, summary, key, attribute_names, row_names)
		
		summary.insert(0, 'total_atts_in_cluster_'+str(key))
		str_summary = [str(summary[x]) for x in range(0, len(summary))]
		outFile.write('\t'.join(str_summary)+'\n'+'\n')

		

if __name__=='__main__':
	parser=argparse.ArgumentParser(description='Categorical classification using ROCK (RObust Clustering using linKs)')
	parser.add_argument('-kclusters', default='2', dest='kclusters', type=int, help='[INT] Number of subtypes or clusters to expect default: 2')
	#column=attributes, row=patients
	parser.add_argument('-input', required=True, dest='matrixFile', help='Full path to tab-delimited "matrix" file')
	parser.add_argument('-outDir', default=os.getcwd(), dest='outDir', help='Full path to output directory')
	parser.add_argument('-simMetric', default='JC', dest='metric', help='JC or SMC, similarity metrics to use, Jaccard Coeffient or simple matching coeffient')
	parser.add_argument('-threshold', default='0.5', dest='minThresh', type=float, help='[FLOAT] Number between 0 and 1 for min Jaccard Coeffienct or min SMC to be considered a neighbor (not same as linked) default: 0.5')
	parser.add_argument('--dataName', default=strftime("%Y-%m-%d_%H:%M:%S", gmtime()), dest='name', help='Name of dataset.  Will be used to name output files')
	parser.add_argument('--colNames', default='None', dest='colNames', help='file with order column names, one name per line, should be list of attribute/dims')
	parser.add_argument('--rowNames', default='None', dest='rowNames', help='file with order of row names, one name per line, should be patient IDs')
	args=parser.parse_args()

	# initiate_matrix, calculate_similarity, and calculate_neighbors only need to be called once
	labeled_patProfile, patientProfile, similarityMatrix, neighborMatrix, patNames, attributeNames = initiate_matrix(inputMatrixData=args.matrixFile, colName=args.colNames, rowName=args.rowNames)
	similarityMatrix = calculate_similarity(patientProfile, similarityMatrix, metric=args.metric)
	labeled_numLinks = calculate_neighbors(neighborMatrix, similarityMatrix, patNames, threshold=args.minThresh)
	
	while len(labeled_numLinks) > args.kclusters:
		pairwise_fitness = fitness_measure(labeled_numLinks, threshold=args.minThresh)
		labeled_numLinks = merge_and_update(pairwise_fitness, labeled_numLinks)
	
	cluster_summaries_binary_attributes(labeled_numLinks, labeled_patProfile, outputDirectory=args.outDir, numClusters=args.kclusters, metric=args.metric, thresh=args.minThresh, dataName=args.name)
