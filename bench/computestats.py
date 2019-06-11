import os
import os.path
import json

sourceDirectory = '../output/MAINRUN'
objectCount = 5
topResultCount = 10

inputFiles = os.listdir(sourceDirectory)

for file in inputFiles:
	with open(os.path.join(sourceDirectory, file), 'r') as inputFile:
		print('Processing file', file)
		contents = json.loads(inputFile.read())
		referenceObjectVertexCount = contents['vertexCounts'][0]
		QSIHistogramsNormalised = [[0 for x in range(topResultCount)] for y in range(objectCount)] 
		SIHistogramsNormalised  = [[0 for x in range(topResultCount)] for y in range(objectCount)]
		for i in range(0, objectCount):
			for j in range(0, topResultCount):
				if str(j) in contents['QSIhistograms'][i]:
					QSIHistogramsNormalised[i][j] = float(contents['QSIhistograms'][i][str(j)]) / float(referenceObjectVertexCount)
		for item in QSIHistogramsNormalised:
			print(item)
		print('----')