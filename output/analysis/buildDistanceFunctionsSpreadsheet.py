import json
import math
import os
import os.path
import datetime
from multiprocessing import Pool, cpu_count

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
baselineWeightedHammingDumpFile = 'final_results/weighted_hamming_distance_function_baseline.txt'
baselineHammingDumpFile = 'final_results/hamming_distance_function_baseline.txt'
baselineClutterResistantDumpFile = 'final_results/clutter_resistant_distance_function_baseline.txt'

resultMap = {}

sphereCounts = range(0, 510, 10)

seedList = []
imageCountList = []

imageSize = 4096

numSphereClutterLevels = 50 + 1
maxDistance = 4096

similarSurfaceClutterResistantHistogram = np.zeros(shape=(maxDistance, numSphereClutterLevels), dtype=np.int64)
similarSurfaceWeightedHammingHistogram = np.zeros(shape=(maxDistance, numSphereClutterLevels), dtype=np.int64)
similarSurfaceHammingHistogram = np.zeros(shape=(maxDistance, numSphereClutterLevels), dtype=np.int64)

baselineClutterResistantHistogram = np.zeros(shape=(maxDistance), dtype=np.int64)
baselineWeightedHammingHistogram = np.zeros(shape=(2 * maxDistance), dtype=np.int64)
baselineHammingHistogram = np.zeros(shape=(maxDistance), dtype=np.int64)

baselineTotalImageCount = 0

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

    baselineTotalImageCount += fileContents['imageCount']
    for imageIndex, imageBitCount in enumerate(fileContents['imageBitCounts']):
        baselineClutterResistantHistogram[
            fileContents['measuredDistances']['clutterResistant']['0 spheres'][imageIndex]] += 1

        baselineWeightedHammingHistogram[
            int(fileContents['measuredDistances']['weightedHamming']['0 spheres'][imageIndex])] += 1

        baselineHammingHistogram[
            fileContents['measuredDistances']['hamming']['0 spheres'][imageIndex]] += 1

print()
print('Total number of baseline images:', baselineTotalImageCount)
# 176,225,136

np.savetxt(baselineWeightedHammingDumpFile, baselineWeightedHammingHistogram, delimiter=',')
np.savetxt(baselineClutterResistantDumpFile, baselineClutterResistantHistogram, delimiter=',')
np.savetxt(baselineHammingDumpFile, baselineHammingHistogram, delimiter=',')

print('Statistics written to disk.')

sphereClutterTotalImageCount = 0

def chunkify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

filesToRead = os.listdir(inputDirectory)

def process(threadinput):
    threadMaxDistance, threadSphereCounts, threadInputDirectory, threadFilesToRead = threadinput
    threadTotalImageCount = 0
    threadClutterResistantHistogram = np.zeros(shape=(maxDistance, numSphereClutterLevels), dtype=np.int64)
    threadWeightedHammingHistogram = np.zeros(shape=(maxDistance, numSphereClutterLevels), dtype=np.int64)
    threadHammingHistogram = np.zeros(shape=(maxDistance, numSphereClutterLevels), dtype=np.int64)
    for fileindex, file in enumerate(threadFilesToRead):
        if fileindex == 5:
            #break
            pass
        print(str(fileindex + 1) + '/' + str(len(threadFilesToRead)), file + '        ', end='\r', flush=True)
        with open(os.path.join(threadInputDirectory, file), 'r') as openFile:
            # Read JSON file
            try:
                fileContents = json.loads(openFile.read())
            except Exception as e:
                print('FAILED TO READ FILE: ' + str(file))
                print(e)
                continue

        '''# Do map initialisation based on output file contents
        if sphereCounts is None:
            sphereCounts = fileContents['sphereCounts']
    
            resultMap['clutterResistant'] = {}
            resultMap['weightedHamming'] = {}
            resultMap['hamming'] = {}
    
            for sphereCount in sphereCounts:
                resultMap['clutterResistant'][str(sphereCount)] = []
                resultMap['weightedHamming'][str(sphereCount)] = []
                resultMap['hamming'][str(sphereCount)] = []'''

        # Add contents of file to result set
        #seedList.append(fileContents['seed'])
        #imageCountList.append(fileContents['imageCount'])
        threadTotalImageCount += fileContents['imageCount']

        for sphereIndex, sphereCount in enumerate(threadSphereCounts):
            #averageClutterResistantScore = statistics.mean(
            #    fileContents['measuredDistances']['clutterResistant'][str(sphereCount) + ' spheres'])
            #averageWeightedHammingScore = statistics.mean(
            #    fileContents['measuredDistances']['weightedHamming'][str(sphereCount) + ' spheres'])
            #averageHammingScore = statistics.mean(
            #    fileContents['measuredDistances']['hamming'][str(sphereCount) + ' spheres'])

            for imageIndex in range(0, fileContents['imageCount']):
                clutterResistantDistance = \
                    fileContents['measuredDistances']['clutterResistant'][str(sphereCount) + ' spheres'][imageIndex]
                weightedHammingDistance = \
                    int(fileContents['measuredDistances']['weightedHamming'][str(sphereCount) + ' spheres'][imageIndex])
                hammingDistance = \
                    fileContents['measuredDistances']['hamming'][str(sphereCount) + ' spheres'][imageIndex]

                if clutterResistantDistance < threadMaxDistance:
                    threadClutterResistantHistogram[
                        clutterResistantDistance,
                        sphereIndex] += 1
                if weightedHammingDistance < threadMaxDistance:
                    threadWeightedHammingHistogram[
                        weightedHammingDistance,
                        sphereIndex] += 1
                if hammingDistance < threadMaxDistance:
                    threadHammingHistogram[
                        hammingDistance,
                        sphereIndex] += 1


            #resultMap['clutterResistant'][str(sphereCount)].append(averageClutterResistantScore)
            #resultMap['weightedHamming'][str(sphereCount)].append(averageWeightedHammingScore)
            #resultMap['hamming'][str(sphereCount)].append(averageHammingScore)
    return (threadTotalImageCount,
            threadClutterResistantHistogram,
            threadWeightedHammingHistogram,
            threadHammingHistogram)

cpuCount = cpu_count()
print('Processing sphere clutter output using', cpuCount, 'threads..')
processPool = Pool(cpuCount)
sphereClutterOutput = processPool.map(process, [(maxDistance, sphereCounts, inputDirectory, x) for x in list(chunkify(filesToRead, int(len(filesToRead) / cpuCount) + 1))])
processPool.close()

for outputEntry in sphereClutterOutput:
    threadTotalImageCount, threadClutterResistantHistogram, threadWeightedHammingHistogram, threadHammingHistogram = outputEntry
    similarSurfaceWeightedHammingHistogram = np.add(similarSurfaceWeightedHammingHistogram,
                                                    threadWeightedHammingHistogram)
    similarSurfaceClutterResistantHistogram = np.add(similarSurfaceClutterResistantHistogram,
                                                    threadClutterResistantHistogram)
    similarSurfaceHammingHistogram = np.add(similarSurfaceHammingHistogram,
                                                    threadHammingHistogram)
    sphereClutterTotalImageCount += threadTotalImageCount

print()
# 26,791,988
print('Sphere clutter total image count:', sphereClutterTotalImageCount)

plt.clf()

SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

similarSurfaceClutterResistantHistogram = np.log10(np.maximum(similarSurfaceClutterResistantHistogram, 0.1))
similarSurfaceWeightedHammingHistogram = np.log10(np.maximum(similarSurfaceWeightedHammingHistogram, 0.1))
similarSurfaceHammingHistogram = np.log10(np.maximum(similarSurfaceHammingHistogram, 0.1))

clutterSphereHistograms = [similarSurfaceHammingHistogram, similarSurfaceClutterResistantHistogram, similarSurfaceWeightedHammingHistogram]
total_minimum_value = min([np.amin(x) for x in clutterSphereHistograms])
total_maximum_value = max([np.amax(x) for x in clutterSphereHistograms])
normalisation = colors.Normalize(vmin=total_minimum_value,vmax=total_maximum_value)

extent = [0, numSphereClutterLevels, 0, maxDistance]

horizontal_ticks_real_coords = np.arange(0, 60, 10)
horizontal_ticks_labels = [("%i" % x) for x in np.arange(0,600,100)]

plot = plt.figure(1)
plt.title('Clutter Resistant')
plt.ylabel('Distance value')
plt.xlabel('Added sphere count')
image = plt.imshow(similarSurfaceClutterResistantHistogram, interpolation='nearest', extent=extent, cmap='hot', aspect='auto', origin='bottom', norm=normalisation)
plt.xticks(horizontal_ticks_real_coords, horizontal_ticks_labels)
plot.show()

plot = plt.figure(2)
plt.title('Weighted Hamming')
plt.ylabel('Distance value')
plt.xlabel('Added sphere count')
image = plt.imshow(similarSurfaceWeightedHammingHistogram, interpolation='nearest', extent=extent, cmap='hot', aspect='auto', origin='bottom', norm=normalisation)
plt.xticks(horizontal_ticks_real_coords, horizontal_ticks_labels)
plot.show()

plot = plt.figure(3)
plt.title('Hamming')
plt.ylabel('Distance value')
plt.xlabel('Added sphere count')
image = plt.imshow(similarSurfaceHammingHistogram, interpolation='nearest', extent=extent, cmap='hot', aspect='auto', origin='bottom', norm=normalisation)

plt.xticks(horizontal_ticks_real_coords, horizontal_ticks_labels)

colorbar_ticks = [-1.0, 0, 1, 2, 3, 4, 5, 6, 7]
colorbar_ticks_strings = ['0', '1E+00', '1E+01', '1E+02', '1E+03', '1E+04', '1E+05', '1E+06', '1E+07']
cbar = plt.colorbar(image, ticks=colorbar_ticks)
cbar.ax.set_yticklabels(colorbar_ticks_strings)
cbar.set_label('Sample count', rotation=90)

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

