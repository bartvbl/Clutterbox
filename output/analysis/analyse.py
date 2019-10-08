import json
import os
import os.path
from math import sqrt


variety = 'earlycutoff'
resultDirectory = '../HEIDRUNS/output_majorfix_v2_' + variety + '/output'
outfile = 'final_results/results_' + variety + '.csv'

def loadOutputFileDirectory(path):
    originalFiles = os.listdir(path)
    results = {}
    for fileindex, file in enumerate(originalFiles):
        print(str(fileindex+1) + '/' + str(len(originalFiles)), file, end='\r')
        if(file == 'raw'):
        	continue
        with open(os.path.join(path, file), 'r') as openFile:
            fileContents = json.loads(openFile.read())
            results[fileContents['seed']] = fileContents
    return results

def objects(count):
    if count > 1:
        return 'Objects'
    else:
        return 'Object'

print('Loading original files..')
loadedResults = loadOutputFileDirectory(resultDirectory)
print()

print('Processing..')
with open(outfile, 'w') as outputFile:
    anyResult = next(iter(loadedResults.values()))
    outputFile.write('Experiment ID, seed, Total Vertex Count, Total Image Count, , ')
    for i in range(0, anyResult['sampleSetSize']):
        outputFile.write('Vertex Count Object ' + str(i) + ', ')
    outputFile.write(', ')
    for i in range(0, len(anyResult['uniqueVertexCounts'])):
        outputFile.write('Image Count Object ' + str(i) + ', ')
    outputFile.write(', ')
    for i in range(0, anyResult['sampleSetSize']):
        outputFile.write('Distance from Object ' + str(i) + ' to Object 0, ')
    outputFile.write(', ')
    for count in anyResult['sampleObjectCounts']:
        outputFile.write('QSI with ' + str(count) + ' ' + objects(count) + ' in Scene, ')
    outputFile.write(', ')
    for count in anyResult['sampleObjectCounts']:
        outputFile.write('SI with ' + str(count) + ' ' + objects(count) + ' in Scene, ')
    outputFile.write(', ')
    for count in anyResult['sampleObjectCounts']:
        outputFile.write('QSI Top 10 with ' + str(count) + ' ' + objects(count) + ' in Scene, ')
    outputFile.write(', ')
    for count in anyResult['sampleObjectCounts']:
        outputFile.write('SI Top 10 with ' + str(count) + ' ' + objects(count) + ' in Scene, ')
    outputFile.write(', ')
    for count in anyResult['sampleObjectCounts']:
        outputFile.write('QSI Reference Generation Time with ' + str(count) + ' ' + objects(count) + ' in Scene, ')
    outputFile.write(', ')
    for count in anyResult['sampleObjectCounts']:
        outputFile.write('SI Reference Generation Time with ' + str(count) + ' ' + objects(count) + ' in Scene, ')
    outputFile.write(', ')
    for count in anyResult['sampleObjectCounts']:
        outputFile.write('QSI Search Time with ' + str(count) + ' ' + objects(count) + ' in Scene, ')
    outputFile.write(', ')
    for count in anyResult['sampleObjectCounts']:
        outputFile.write('SI Search Time with ' + str(count) + ' ' + objects(count) + ' in Scene, ')
    outputFile.write('\n')


    for fileindex, seed in enumerate(loadedResults):
        print(str(fileindex+1) + '/' + str(len(loadedResults)), seed, end='\r')
        result = loadedResults[seed]
        angleFileContents = {}

        totalImageCount = result['imageCounts'][0] #result['vertexCounts'][0]
        referencePosition = (result['translations'][0])
        usedSampleObjectCount = result['sampleSetSize']
        experimentIterationCount = len(result['sampleObjectCounts'])

        distances = [0] * usedSampleObjectCount
        qsiPercentageAtPlace0 = [0] * experimentIterationCount
        siPercentageAtPlace0 = [0] * experimentIterationCount
        qsiPercentageInTop10 = [0] * experimentIterationCount
        siPercentageInTop10 = [0] * experimentIterationCount

        angle_qsiPercentageAtPlace0 = [0] * experimentIterationCount
        angle_siPercentageAtPlace0 = [0] * experimentIterationCount

        for i in range(0, usedSampleObjectCount):
            vertex = tuple(result['translations'][i])
            dx = vertex[0] - referencePosition[0]
            dy = vertex[1] - referencePosition[1]
            dz = vertex[2] - referencePosition[2]
            distances[i] = sqrt(dx*dx + dy*dy + dz*dz)

        for i in range(0, experimentIterationCount):
            index = i
            if result['version'] == 'v8':
                index = str(i)
            if 'QSIhistograms' in result and '0' in result['QSIhistograms'][index]:
                qsiPercentageAtPlace0[i] = float(result['QSIhistograms'][index]['0']) / float(totalImageCount)

            if 'SIhistograms' in result and '0' in result['SIhistograms'][index]:
                siPercentageAtPlace0[i] = float(result['SIhistograms'][index]['0']) / float(totalImageCount)

            QSITop10Sum = 0
            SITop10Sum = 0
            for j in range(0, 10):
                if 'QSIhistograms' in result and str(j) in result['QSIhistograms'][index]:
                    QSITop10Sum += result['QSIhistograms'][index][str(j)]
                if 'SIhistograms' in result and str(j) in result['SIhistograms'][index]:
                    SITop10Sum += result['SIhistograms'][index][str(j)]

            qsiPercentageInTop10[i] = float(QSITop10Sum) / float(totalImageCount)
            siPercentageInTop10[i] = float(SITop10Sum) / float(totalImageCount)

        qsiSampleGenerationTimes = None
        siSampleGenerationTimes = None
        qsiSearchTimes = None
        siSearchTimes = None

        if not 'SIhistograms' in result:
            siPercentageAtPlace0 = [''] * experimentIterationCount
            siPercentageInTop10 = [''] * experimentIterationCount
            siSampleGenerationTimes = [''] * experimentIterationCount
            siSearchTimes = [''] * experimentIterationCount
        else:
            siSampleGenerationTimes = result['runtimes']['SISampleGeneration']['total']
            siSearchTimes = result['runtimes']['SISearch']['total']

        if not 'QSIhistograms' in result:
            qsiPercentageAtPlace0 = [''] * experimentIterationCount
            qsiPercentageInTop10 = [''] * experimentIterationCount
            qsiSampleGenerationTimes = [''] * experimentIterationCount
            qsiSearchTimes = [''] * experimentIterationCount
        else:
            qsiSampleGenerationTimes = result['runtimes']['QSISampleGeneration']['total']
            qsiSearchTimes = result['runtimes']['QSISearch']['total']



        outputFile.write('%i, %i, %i, %i, ,' % (fileindex, seed, sum(result['vertexCounts']), sum(result['imageCounts'])))
        outputFile.write(', '.join([str(f) for f in result['vertexCounts']]) + ', , ')
        outputFile.write(', '.join([str(f) for f in result['imageCounts']]) + ', , ')
        outputFile.write(', '.join([str(f) for f in distances]) + ', , ')
        outputFile.write(', '.join([str(f) for f in qsiPercentageAtPlace0]) + ', , ')
        outputFile.write(', '.join([str(f) for f in siPercentageAtPlace0]) + ', , ')
        outputFile.write(', '.join([str(f) for f in qsiPercentageInTop10]) + ', , ')
        outputFile.write(', '.join([str(f) for f in siPercentageInTop10]) + ', , ')
        outputFile.write(', '.join([str(f) for f in qsiSampleGenerationTimes]) + ', , ')
        outputFile.write(', '.join([str(f) for f in siSampleGenerationTimes]) + ', , ')
        outputFile.write(', '.join([str(f) for f in qsiSearchTimes]) + ', , ')
        outputFile.write(', '.join([str(f) for f in siSearchTimes]) + ', , ')
        outputFile.write('\n')
print()
print('Complete.')

