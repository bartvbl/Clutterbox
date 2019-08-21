import json
import os
import os.path
from math import sqrt

resultDirectory = '../HEIDRUNS/output_majorfix_v1_smallspinwidth/output'
outfile = 'dump.csv'

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

print('Loading original files..')
loadedResults = loadOutputFileDirectory(resultDirectory)
print()

print('Processing..')
with open(outfile, 'w') as outputFile:
    anyResult = next(iter(loadedResults.values()))
    outputFile.write('Experiment ID, Total Vertex Count, , ')
    for i in range(0, anyResult['sampleSetSize']):
        outputFile.write('Vertex Count Object ' + str(i) + ', ')
    outputFile.write(', ')
    for i in range(0, anyResult['sampleSetSize']):
        outputFile.write('Distance from Object ' + str(i) + ' to Object 0, ')
    outputFile.write(', ')
    for count in anyResult['sampleObjectCounts']:
        outputFile.write('QSI with ' + str(count) + ' Object in Scene, ')
    outputFile.write(', ')
    for count in anyResult['sampleObjectCounts']:
        outputFile.write('SI with ' + str(count) + ' Object in Scene, ')
    outputFile.write(', ')
    for count in anyResult['sampleObjectCounts']:
        outputFile.write('QSI Top 10 with ' + str(count) + ' Object in Scene, ')
    outputFile.write(', ')
    for count in anyResult['sampleObjectCounts']:
        outputFile.write('SI Top 10 with ' + str(count) + ' Object in Scene, ')
    outputFile.write(', ')
    for count in anyResult['sampleObjectCounts']:
        outputFile.write('QSI Generation Time with ' + str(count) + ' Object in Scene, ')
    outputFile.write(', ')
    for count in anyResult['sampleObjectCounts']:
        outputFile.write('SI Generation Time with ' + str(count) + ' Object in Scene, ')
    outputFile.write(', ')
    outputFile.write('QSI Search Time, , , , , , SI Search Time, , , , , , QSI (smaller box), , , , , , SI (smaller box + support angle)\n')


    for fileindex, seed in enumerate(loadedResults):
        print(str(fileindex+1) + '/' + str(len(loadedResults)), seed, end='\r')
        result = loadedResults[seed]
        angleFileContents = {}

        referenceVertexCount = result['vertexCounts'][0]
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
            if '0' in result['QSIhistograms'][i]:
                qsiPercentageAtPlace0[i] = float(result['QSIhistograms'][i]['0']) / float(referenceVertexCount)

            if '0' in result['SIhistograms'][i]:
                siPercentageAtPlace0[i] = float(result['SIhistograms'][i]['0']) / float(referenceVertexCount)

            QSITop10Sum = 0
            SITop10Sum = 0
            for j in range(0, 10):
                if str(j) in result['QSIhistograms'][i]:
                    QSITop10Sum += result['QSIhistograms'][i][str(j)]
                if str(j) in result['SIhistograms'][i]:
                    SITop10Sum += result['SIhistograms'][i][str(j)]

            qsiPercentageInTop10[i] = float(QSITop10Sum) / float(referenceVertexCount)
            siPercentageInTop10[i] = float(SITop10Sum) / float(referenceVertexCount)

        outputFile.write('%i, %i, ,' % (fileindex, sum(result['vertexCounts'])))
        outputFile.write(', '.join([str(f) for f in result['vertexCounts']]) + ', , ')
        outputFile.write(', '.join([str(f) for f in distances]) + ', , ')
        outputFile.write(', '.join([str(f) for f in qsiPercentageAtPlace0]) + ', , ')
        outputFile.write(', '.join([str(f) for f in siPercentageAtPlace0]) + ', , ')
        outputFile.write(', '.join([str(f) for f in qsiPercentageInTop10]) + ', , ')
        outputFile.write(', '.join([str(f) for f in siPercentageInTop10]) + ', , ')
        outputFile.write(', '.join([str(f) for f in result['runtimes']['QSISampleGeneration']['total']]) + ', , ')
        outputFile.write(', '.join([str(f) for f in result['runtimes']['SISampleGeneration']['total']]) + ', , ')
        outputFile.write(', '.join([str(f) for f in result['runtimes']['QSISearch']['total']]) + ', , ')
        outputFile.write(', '.join([str(f) for f in result['runtimes']['SISearch']['total']]) + ', , ')
        outputFile.write('\n')
print()
print('Complete.')

