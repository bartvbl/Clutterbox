import json
import os
import os.path
from math import sqrt

resultDirectory = '../HEIDRUNS/output_majorfix_v1/output'
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
    outputFile.write('Experiment ID, Total Vertex Count, , Vertex Count Object 0, Vertex Count Object 1, Vertex Count Object 2, Vertex Count Object 3, Vertex Count Object 4, , Distance from Object 0 to Object 0, Distance from Object 1 to Object 0, Distance from Object 2 to Object 0, Distance from Object 3 to Object 0, Distance from Object 4 to Object 0, , QSI with 1 Object in Scene, QSI with 2 Objects in Scene, QSI with 3 Objects in Scene, QSI with 4 Objects in Scene, QSI with 5 Objects in Scene, , SI with 1 Object in Scene, SI with 2 Objects in Scene, SI with 3 Objects in Scene, SI with 4 Objects in Scene, SI with 5 Objects in Scene, , QSI Top 10 with 1 Object in Scene, QSI Top 10 with 2 Objects in Scene, QSI Top 10 with 3 Objects in Scene, QSI Top 10 with 4 Objects in Scene, QSI Top 10 with 5 Objects in Scene, , SI Top 10 with 1 Object in Scene, SI Top 10 with 2 Objects in Scene, SI Top 10 with 3 Objects in Scene, SI Top 10 with 4 Objects in Scene, SI Top 10 with 5 Objects in Scene, , QSI Generation Time with 1 Object in Scene, QSI Generation Time with 2 Objects in Scene, QSI Generation Time with 3 Objects in Scene, QSI Generation Time with 4 Objects in Scene, QSI Generation Time with 5 Objects in Scene, , SI Generation Time with 1 Object in Scene, SI Generation Time with 2 Objects in Scene, SI Generation Time with 3 Objects in Scene, SI Generation Time with 4 Objects in Scene, SI Generation Time with 5 Objects in Scene, , QSI Search Time, , , , , , SI Search Time, , , , , , QSI (smaller box), , , , , , SI (smaller box + support angle)\n')
    for fileindex, seed in enumerate(loadedResults):
        print(str(fileindex+1) + '/' + str(len(loadedResults)), seed, end='\r')
        result = loadedResults[seed]
        angleFileContents = {}

        referenceVertexCount = result['vertexCounts'][0]
        referencePosition = (result['translations'][0])
        usedSampleObjectCounts = result['sampleObjectCounts']

        distances = [0, 0, 0, 0, 0]
        qsiPercentageAtPlace0 = [0, 0, 0, 0, 0]
        siPercentageAtPlace0 = [0, 0, 0, 0, 0]
        qsiPercentageInTop10 = [0, 0, 0, 0, 0]
        siPercentageInTop10 = [0, 0, 0, 0, 0]

        angle_qsiPercentageAtPlace0 = [0, 0, 0, 0, 0]
        angle_siPercentageAtPlace0 = [0, 0, 0, 0, 0]

        for i in range(0, len(usedSampleObjectCounts)):
            vertex = tuple(result['translations'][i])
            dx = vertex[0] - referencePosition[0]
            dy = vertex[1] - referencePosition[1]
            dz = vertex[2] - referencePosition[2]
            distances[i] = sqrt(dx*dx + dy*dy + dz*dz)

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
        outputFile.write('%i, %i, %i, %i, %i, ,' % tuple(result['vertexCounts']))
        outputFile.write('%f, %f, %f, %f, %f, ,' % tuple(distances))
        outputFile.write('%f, %f, %f, %f, %f, ,' % tuple(qsiPercentageAtPlace0))
        outputFile.write('%f, %f, %f, %f, %f, ,' % tuple(siPercentageAtPlace0))
        outputFile.write('%f, %f, %f, %f, %f, ,' % tuple(qsiPercentageInTop10))
        outputFile.write('%f, %f, %f, %f, %f, ,' % tuple(siPercentageInTop10))
        outputFile.write('%f, %f, ,' % tuple(result['runtimes']['QSISampleGeneration']['total']))
        outputFile.write('%f, %f, ,' % tuple(result['runtimes']['SISampleGeneration']['total']))
        outputFile.write('%f, %f, ,' % tuple(result['runtimes']['QSISearch']['total']))
        outputFile.write('%f, %f, ,' % tuple(result['runtimes']['SISearch']['total']))
        outputFile.write('\n')
print()
print('Complete.')

