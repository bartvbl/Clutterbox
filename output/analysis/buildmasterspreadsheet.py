import json
import os
import os.path
import datetime
import xlwt
from math import sqrt

# -- settings --

inputDirectories = [
    '../HEIDRUNS/output_qsifix_v4_noearlyexit/output',
    '../HEIDRUNS/output_qsifix_v4_withearlyexit/output',
    '../HEIDRUNS/output_majorfix_v2_lotsofobjects_5_v2/output',
    '../HEIDRUNS/output_qsifix_v4_lotsofobjects_idun_failed/output',
    '../IDUNRUNS/output_lotsofobjects_v4',
    '../IDUNRUNS/output_lotsofobjects_5_v3',
    '../IDUNRUNS/output_smallsupportangle_lotsofobjects',
    '../IDUNRUNS/output_smallsupportangle_lotsofobjects_run2',
    '../IDUNRUNS/output_qsifix_smallsupportangle_rerun',
]
inputDirectoryDatasetNames = [
    'QSI, no early exit',
    'QSI, with early exit',
    'SI, 5 objects, large support angle',
    'QSI, '
]
outfile = 'final_results/master_spreadsheet.xls'

# Last known fault in code: unsigned integer subtraction in QSI comparison function
# Date threshold corresponds to this commit
qsiResultsValidAfter = datetime.datetime(year=2019, month=10, day=7, hour=15, minute=14, second=0, microsecond=0)

# Last known fault in code: override object count did not match
# Date threshold corresponds to this commit
siResultsValidAfter = datetime.datetime(year=2019, month=9, day=26, hour=17, minute=28, second=0, microsecond=0)

# -- global initialisation --

# Maps seeds to a list of (dataset, value) tuples
seedmap_top_result_qsi = {}
seedmap_top_result_si = {}
seedmap_top10_results_qsi = {}
seedmap_top10_results_si = {}

# Settings used to construct the experiment settings table


# -- code --

def extractExperimentSettings(loadedJson):
    settings = {}
    settings['boxSize'] = loadedJson['boxSize']
    settings['sampleObjectCounts'] = loadedJson['sampleObjectCounts']
    settings['sampleSetSize'] = loadedJson['sampleSetSize']
    settings['searchResultCount'] = loadedJson['searchResultCount']
    settings['spinImageSupportAngle'] = loadedJson['spinImageSupportAngle']
    settings['spinImageWidth'] = loadedJson['spinImageWidth']
    settings['spinImageWidthPixels'] = loadedJson['spinImageWidthPixels']
    settings['version'] = loadedJson['version']
    return settings

def loadOutputFileDirectory(path):
    originalFiles = os.listdir(path)
    results = {
        'path': path,
        'results': {},
        'settings': {}
    }

    ignored_count_si = 0
    ignored_count_qsi = 0
    for fileindex, file in enumerate(originalFiles):
        if file == 'raw' or file == 'rawless':
            continue
        filename_creation_time_part = file.split('_')[0]
        creation_time = datetime.datetime.strptime(filename_creation_time_part, "%Y-%m-%d %H-%M-%S")
        if creation_time < qsiResultsValidAfter:
            ignored_count_qsi += 1
        if creation_time < siResultsValidAfter:
            ignored_count_si += 1

    if ignored_count_qsi > 0:
        print('\t%i/%i QSI images were created before the threshold deadline. QSI images will be ignored from this dataset.' % (ignored_count_qsi, len(originalFiles)))
    if ignored_count_si > 0:
        print('\t%i/%i SI images were created before the threshold deadline. SI images will be ignored from this dataset.' % (ignored_count_si, len(originalFiles)))


    previousExperimentSettings = None
    for fileindex, file in enumerate(originalFiles):
        print(str(fileindex+1) + '/' + str(len(originalFiles)), file + '        ', end='\r')
        if file == 'raw' or file == 'rawless':
            continue
        with open(os.path.join(path, file), 'r') as openFile:
            fileContents = json.loads(openFile.read())
            currentExperimentSettings = extractExperimentSettings(fileContents)
            if previousExperimentSettings is not None:
                if currentExperimentSettings != previousExperimentSettings:
                    raise Exception("Experiment settings mismatch in the same batch! File: " + file)
            results[fileContents['seed']] = fileContents
            previousExperimentSettings = currentExperimentSettings
    print()

    results['settings'] = previousExperimentSettings
    print('\t' + str(results['settings']))

    return results

def objects(count):
    if count > 1:
        return 'Objects'
    else:
        return 'Object'

print('Loading original files..')
loadedResults = {}
for directory in inputDirectories:
    print('Loading directory:', directory)
    loadedResults[directory] = loadOutputFileDirectory(directory)


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

# -- Dump to spreadsheet --

book = xlwt.Workbook(encoding="utf-8")

# 1. Create data page for dataset settings table
experimentSheet = book.add_sheet("Experiment Overview")

# 2. Create data page for mapping of seed -> % at rank 0
top0sheet = book.add_sheet("Rank 0 results")

# 3. Create data page for mapping of seed -> % at rank 0 - 9
top10sheet = book.add_sheet("Top 10 results")


print('Complete.')