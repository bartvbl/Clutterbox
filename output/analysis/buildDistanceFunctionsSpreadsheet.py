import json
import os
import os.path
import datetime
import xlwt
import pprint
import statistics

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image

inputDirectory = '../HEIDRUNS/run10_distance_functions_experiment/output/'

outfile = 'final_results/distance_functions_results.xls'

resultMap = {}

clutterResistantMax = 4096.0
weightedHammingMax = 

sphereCounts = None

seedList = []
imageCountList = []

filesToRead = os.listdir(inputDirectory)
for fileindex, file in enumerate(filesToRead):
    print(str(fileindex + 1) + '/' + str(len(filesToRead)), file + '        ', end='\r', flush=True)
    with open(os.path.join(inputDirectory, file), 'r') as openFile:
        # Read JSON file
        try:
            fileContents = json.loads(openFile.read())
        except Exception as e:
            print('FAILED TO READ FILE: ' + str(file))
            print(e)
            continue

    if sphereCounts is None:
        sphereCounts = fileContents['sphereCounts']

        resultMap['clutterResistant'] = {}
        resultMap['weightedHamming'] = {}
        resultMap['hamming'] = {}

        for sphereCount in sphereCounts:
            resultMap['clutterResistant'][str(sphereCount)] = []
            resultMap['weightedHamming'][str(sphereCount)] = []
            resultMap['hamming'][str(sphereCount)] = []

    seedList.append(fileContents['seed'])
    imageCountList.append(fileContents['imageCount'])

    for sphereCount in sphereCounts:
        averageClutterResistantScore = statistics.mean(
            fileContents['measuredDistances']['clutterResistant'][str(sphereCount) + ' spheres'])
        averageWeightedHammingScore = statistics.mean(
            fileContents['measuredDistances']['weightedHamming'][str(sphereCount) + ' spheres'])
        averageHammingScore = statistics.mean(
            fileContents['measuredDistances']['hamming'][str(sphereCount) + ' spheres'])

        resultMap['clutterResistant'][str(sphereCount)].append(averageClutterResistantScore)
        resultMap['weightedHamming'][str(sphereCount)].append(averageWeightedHammingScore)
        resultMap['hamming'][str(sphereCount)].append(averageHammingScore)

print()
print('Writing spreadsheet..')

book = xlwt.Workbook(encoding="utf-8")

resultsSheet = book.add_sheet('results')

# Write header
resultsSheet.write(0, 0, 'Index')
resultsSheet.write(0, 1, 'Seed')
resultsSheet.write(0, 2, 'Image count')
for sphereCountIndex, sphereCount in enumerate(sphereCounts):
    resultsSheet.write(0, 3 + sphereCountIndex, str(sphereCount) + ' spheres')
    resultsSheet.write(0, 3 + sphereCountIndex + len(sphereCounts), str(sphereCount) + ' spheres')
    resultsSheet.write(0, 3 + sphereCountIndex + 2 * len(sphereCounts), str(sphereCount) + ' spheres')

for seedIndex, seed in enumerate(seedList):
    resultsSheet.write(seedIndex + 1, 0, seedIndex + 1)
    resultsSheet.write(seedIndex + 1, 1, seed)
    resultsSheet.write(seedIndex + 1, 2, imageCountList[seedIndex])

    for sphereCountIndex, sphereCount in enumerate(sphereCounts):
        resultsSheet.write(seedIndex + 1, 3 + sphereCountIndex, resultMap['clutterResistant'][str(sphereCount)][seedIndex])
        resultsSheet.write(seedIndex + 1, 3 + sphereCountIndex + len(sphereCounts), resultMap['weightedHamming'][str(sphereCount)][seedIndex])
        resultsSheet.write(seedIndex + 1, 3 + sphereCountIndex + 2 * len(sphereCounts), resultMap['hamming'][str(sphereCount)][seedIndex])

book.save(outfile)

print('Complete.')