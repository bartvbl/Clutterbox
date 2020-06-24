import json
import math
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

inputDirectory = '../HEIDRUNS/run12_quicci_distance_functions_rerun/output/'
baselineDirectory = '../HEIDRUNS/run14_quicci_distance_functions_baseline/'

outfile = 'final_results/distance_functions_results.xls'

resultMap = {}

sphereCounts = None

seedList = []
imageCountList = []

imageSize = 4096

numSphereClutterLevels = 50 + 1
maxDistance = 4096

similarSurfaceClutterResistantHistogram = np.zeros(shape=(numSphereClutterLevels, maxDistance), dtype=np.int64)
similarSurfaceWeightedHammingHistogram = np.zeros(shape=(numSphereClutterLevels, maxDistance), dtype=np.int64)
similarSurfaceHammingHistogram = np.zeros(shape=(numSphereClutterLevels, maxDistance), dtype=np.int64)

baselineClutterResistantHistogram = np.zeros(shape=(maxDistance, imageSize), dtype=np.int64)
baselineWeightedHammingHistogram = np.zeros(shape=(2 * maxDistance, imageSize), dtype=np.int64)
baselineHammingHistogram = np.zeros(shape=(maxDistance, imageSize), dtype=np.int64)


baselineFileToRead = os.listdir(baselineDirectory)
for fileindex, file in enumerate(baselineFileToRead):
    print(str(fileindex + 1) + '/' + str(len(baselineFileToRead)), file + '        ', end='\r', flush=True)
    with open(os.path.join(baselineDirectory, file), 'r') as openFile:
        # Read JSON file
        try:
            fileContents = json.loads(openFile.read())
        except Exception as e:
            print('FAILED TO READ FILE: ' + str(file))
            print(e)
            continue

    for imageIndex, imageBitCount in enumerate(fileContents['imageBitCounts']):
        baselineClutterResistantHistogram[
            maxDistance - 1 - fileContents['measuredDistances']['clutterResistant']['0 spheres'][imageIndex],
            imageBitCount] += 1
        weightedHammingDistance = int(fileContents['measuredDistances']['weightedHamming']['0 spheres'][imageIndex])
        if weightedHammingDistance < 2 * maxDistance:
            baselineWeightedHammingHistogram[
                2 * maxDistance - 1 - weightedHammingDistance,
                imageBitCount] += 1
        baselineHammingHistogram[
            maxDistance - 1 - fileContents['measuredDistances']['hamming']['0 spheres'][imageIndex],
            imageBitCount] += 1

baselineClutterResistantHistogram = np.log10(np.maximum(baselineClutterResistantHistogram, 0.1))
baselineWeightedHammingHistogram = np.log10(np.maximum(baselineWeightedHammingHistogram, 0.1))
baselineHammingHistogram = np.log10(np.maximum(baselineHammingHistogram, 0.1))

baselineHistograms = [baselineClutterResistantHistogram,
                      baselineWeightedHammingHistogram,
                      baselineHammingHistogram]

total_minimum_value = min([np.amin(x) for x in baselineHistograms])
total_maximum_value = max([np.amax(x) for x in baselineHistograms])
normalisation = colors.Normalize(vmin=total_minimum_value,vmax=total_maximum_value)

extent = [0, imageSize, 0, maxDistance]
weightedHammingExtent = [0, imageSize, 0, 2 * maxDistance]

plt.clf()

plot = plt.figure(1)
plt.title('Clutter Resistant')
plt.ylabel('Computed distance using distance function')
plt.xlabel('Number of set pixels in query image')
image = plt.imshow(baselineClutterResistantHistogram, extent=extent, cmap='hot', norm=normalisation)
plot.show()

plot = plt.figure(2)
plt.title('Weighted Hamming')
plt.ylabel('Computed distance using distance function')
plt.xlabel('Number of set pixels in query image')
image = plt.imshow(baselineWeightedHammingHistogram, extent=weightedHammingExtent, cmap='hot', norm=normalisation)
plot.show()

plot = plt.figure(3)
plt.title('Hamming')
plt.ylabel('Computed distance using distance function')
plt.xlabel('Number of set pixels in query image')
image = plt.imshow(baselineHammingHistogram, extent=extent, cmap='hot', norm=normalisation)

halfway = math.log10(5)
colorbar_ticks = [-1.0, 0, halfway, 1, 1 + halfway, 2, 2 + halfway, 3, 3 + halfway, 4, 4 + halfway]
cbar = plt.colorbar(image, ticks=colorbar_ticks)
cbar.ax.set_yticklabels([str(int(round(pow(10, x)))) for x in colorbar_ticks])
cbar.set_label('Sample count', rotation=90)

plot.show()

input()


filesToRead = os.listdir(inputDirectory)
for fileindex, file in enumerate(filesToRead):
    if fileindex == 10:
        break
    print(str(fileindex + 1) + '/' + str(len(filesToRead)), file + '        ', end='\r', flush=True)
    with open(os.path.join(inputDirectory, file), 'r') as openFile:
        # Read JSON file
        try:
            fileContents = json.loads(openFile.read())
        except Exception as e:
            print('FAILED TO READ FILE: ' + str(file))
            print(e)
            continue

    # Do map initialisation based on output file contents
    if sphereCounts is None:
        sphereCounts = fileContents['sphereCounts']

        resultMap['clutterResistant'] = {}
        resultMap['weightedHamming'] = {}
        resultMap['hamming'] = {}

        for sphereCount in sphereCounts:
            resultMap['clutterResistant'][str(sphereCount)] = []
            resultMap['weightedHamming'][str(sphereCount)] = []
            resultMap['hamming'][str(sphereCount)] = []

    # Add contents of file to result set
    seedList.append(fileContents['seed'])
    imageCountList.append(fileContents['imageCount'])

    for sphereIndex, sphereCount in enumerate(sphereCounts):
        averageClutterResistantScore = statistics.mean(
            fileContents['measuredDistances']['clutterResistant'][str(sphereCount) + ' spheres'])
        averageWeightedHammingScore = statistics.mean(
            fileContents['measuredDistances']['weightedHamming'][str(sphereCount) + ' spheres'])
        averageHammingScore = statistics.mean(
            fileContents['measuredDistances']['hamming'][str(sphereCount) + ' spheres'])

        for imageIndex in range(0, fileContents['imageCount']):
            clutterResistantDistance = \
                fileContents['measuredDistances']['clutterResistant'][str(sphereCount) + ' spheres'][imageIndex]
            weightedHammingDistance = \
                int(fileContents['measuredDistances']['weightedHamming'][str(sphereCount) + ' spheres'][imageIndex])
            hammingDistance = \
                fileContents['measuredDistances']['hamming'][str(sphereCount) + ' spheres'][imageIndex]

            if clutterResistantDistance < maxDistance:
                similarSurfaceClutterResistantHistogram[sphereIndex, clutterResistantDistance] += 1
            if weightedHammingDistance < maxDistance:
                similarSurfaceWeightedHammingHistogram[sphereIndex, weightedHammingDistance] += 1
            if hammingDistance < maxDistance:
                similarSurfaceHammingHistogram[sphereIndex, hammingDistance] += 1


        resultMap['clutterResistant'][str(sphereCount)].append(averageClutterResistantScore)
        resultMap['weightedHamming'][str(sphereCount)].append(averageWeightedHammingScore)
        resultMap['hamming'][str(sphereCount)].append(averageHammingScore)

print()

similarSurfaceClutterResistantHistogram = np.log10(np.maximum(similarSurfaceClutterResistantHistogram, 0.1))
similarSurfaceWeightedHammingHistogram = np.log10(np.maximum(similarSurfaceWeightedHammingHistogram, 0.1))
similarSurfaceHammingHistogram = np.log10(np.maximum(similarSurfaceHammingHistogram, 0.1))

extent = [0, numSphereClutterLevels, 0, maxDistance]

plot = plt.figure(1)
plt.title('')
plt.ylabel('rank')
plt.xlabel('clutter percentage')
image = plt.imshow(similarSurfaceWeightedHammingHistogram, extent=extent, cmap='hot')#, norm=normalisation)
#plt.xticks(horizontal_ticks_real_coords, horizontal_ticks_labels)

plot.show()

input()

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