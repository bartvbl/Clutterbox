import json
import os
import os.path
import datetime
import xlwt
import pprint
import copy
from math import sqrt

# -- settings --

inputDirectories = {
    '../HEIDRUNS/output_qsifix_v4_noearlyexit/output': ('QSI, No early exit, 5 objects', 'HEID'),
    '../HEIDRUNS/output_qsifix_v4_withearlyexit/output': ('QSI, Early exit, 5 objects', 'HEID'),
    '../HEIDRUNS/output_qsifix_v4_lotsofobjects_idun_failed/output': ('Failed jobs from IDUN run', 'HEID'),
    '../IDUNRUNS/output_lotsofobjects_v4': ('primary QSI IDUN run', 'IDUN'),

    '../HEIDRUNS/output_qsifix_v4_lotsofobjects_10_objects_only/output': ('180 support angle, 10 objects', 'HEID'),
    '../HEIDRUNS/output_qsifix_v4_lotsofobjects_5_objects_only/output': ('180 support angle, 5 objects', 'HEID'),
    '../HEIDRUNS/output_qsifix_v4_180deg_si_missing/output': ('180 support angle, 10 objects', 'HEID'),
    '../IDUNRUNS/output_smallsupportangle_lotsofobjects': ('60 support angle, primary', 'IDUN'),
    '../IDUNRUNS/output_qsifix_smallsupportangle_rerun': ('60 support angle, secondary', 'IDUN'),
    '../IDUNRUNS/output_mainchart_si_v4_15': ('180 support angle, 1 & 5 objects', 'IDUN'),
    '../IDUNRUNS/output_mainchart_si_v4_10': ('180 support angle, 10 objects', 'IDUN'),
    '../IDUNRUNS/output_mainchart_si_v4_1': ('180 support angle, 1 object', 'IDUN'),
    '../IDUNRUNS/output_supportanglechart60_si_v4_1': ('60 support angle, 1 object', 'IDUN'),
    '../IDUNRUNS/output_supportanglechart60_si_v4_5': ('60 support angle, 5 objects', 'IDUN'),
    '../HEIDRUNS/output_qsifix_v4_60deg_si_missing/output/': ('60 support angle, 10 objects', 'HEID'),
}
outfile = 'final_results/master_spreadsheet.xls'

# Last known fault in code: unsigned integer subtraction in QSI comparison function
# Date threshold corresponds to this commit
def isQsiResultValid(fileCreationDateTime, resultJson):
    qsiResultsValidAfter = datetime.datetime(year=2019, month=10, day=7, hour=15, minute=14, second=0, microsecond=0)
    return fileCreationDateTime > qsiResultsValidAfter

# Last known fault in code: override object count did not match
# Date threshold corresponds to this commit
def isSiResultValid(fileCreationDateTime, resultJson):
    siResultsValidAfter = datetime.datetime(year=2019, month=9, day=26, hour=17, minute=28, second=0, microsecond=0)
    hasCorrectOverrideObjectCount = 'overrideObjectCount' in resultJson and resultJson['overrideObjectCount'] == 10
    hasCorrectSampleSetSize = resultJson['sampleSetSize'] == 10
    hasValidCreationDateTime = siResultsValidAfter < fileCreationDateTime

    return hasCorrectOverrideObjectCount or hasCorrectSampleSetSize

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
    if 'overrideObjectCount' in loadedJson:
        settings['overrideObjectCount'] = loadedJson['overrideObjectCount']
    else:
        settings['overrideObjectCount'] = max(loadedJson['sampleObjectCounts'])
    if 'descriptors' in loadedJson:
        settings['descriptors'] = loadedJson['descriptors']
    else:
        descriptors = []
        if 'QSIhistograms' in loadedJson:
            descriptors.append('qsi')
        if 'SIhistograms' in loadedJson:
            descriptors.append('si')
        settings['descriptors'] = descriptors
    settings['version'] = loadedJson['version']
    return settings

def loadOutputFileDirectory(path):
    originalFiles = os.listdir(path)
    # Filter out the raw output file directories
    originalFiles = [x for x in originalFiles if x != 'raw' and x != 'rawless']

    results = {
        'path': path,
        'results': {'QSI': {}, 'SI': {}},
        'settings': {}
    }

    jsonCache = {}

    ignoredListQSI = []
    ignoredListSI = []

    allQSIResultsInvalid = False
    allSIResultsInvalid = False
    for fileindex, file in enumerate(originalFiles):
        print(str(fileindex+1) + '/' + str(len(originalFiles)), file + '        ', end='\r')
        with open(os.path.join(path, file), 'r') as openFile:
            # Read JSON file
            try:
                fileContents = json.loads(openFile.read())
            except Exception as e:
                print('FAILED TO READ FILE: ' + str(file))
                print(e)
                continue
            jsonCache[file] = fileContents

            # Check validity of code by using knowledge about previous code changes
            filename_creation_time_part = file.split('_')[0]
            creation_time = datetime.datetime.strptime(filename_creation_time_part, "%Y-%m-%d %H-%M-%S")
            if not isQsiResultValid(creation_time, fileContents):
                ignoredListQSI.append(file)
                allQSIResultsInvalid = True
            if not isSiResultValid(creation_time, fileContents):
                ignoredListSI.append(file)
                allSIResultsInvalid = True

    previousExperimentSettings = None
    for fileindex, file in enumerate(originalFiles):
        print(str(fileindex+1) + '/' + str(len(originalFiles)), file + '        ', end='\r')
        with open(os.path.join(path, file), 'r') as openFile:
            # Read JSON file
            fileContents = jsonCache[file]

            # Check if settings are the same as other files in the folder
            currentExperimentSettings = extractExperimentSettings(fileContents)
            if previousExperimentSettings is not None:
                if currentExperimentSettings != previousExperimentSettings:
                    # Any discrepancy here is a fatal exception. It NEEDS attention regardless
                    raise Exception("Experiment settings mismatch in the same batch! File: " + file)
            previousExperimentSettings = currentExperimentSettings
            results['settings'] = currentExperimentSettings

            # Check for other incorrect settings. Ignore files if detected
            if 0 in fileContents['imageCounts']:
                if file not in ignoredListQSI:
                    ignoredListQSI.append(file)
                if file not in ignoredListSI:
                    ignoredListSI.append(file)
                #print('ignored 0',file)
            if fileContents['spinImageWidthPixels'] == 32:
                if file not in ignoredListQSI:
                    ignoredListQSI.append(file)
                if file not in ignoredListSI:
                    ignoredListSI.append(file)

            # Beauty checks
            if file not in ignoredListQSI and allQSIResultsInvalid:
                ignoredListQSI.append(file)
            if file not in ignoredListSI and allSIResultsInvalid:
                ignoredListSI.append(file)

            containsQSIResults = ('descriptors' in fileContents and 'qsi' in fileContents['descriptors']) or 'QSIhistograms' in fileContents
            containsSIResults = ('descriptors' in fileContents and 'si' in fileContents['descriptors']) or 'SIhistograms' in fileContents

            # Sanity checks are done. We can now add any remaining valid entries to the result lists
            if not file in ignoredListQSI and not allQSIResultsInvalid and containsQSIResults:
                results['results']['QSI'][str(fileContents['seed'])] = fileContents
            if not file in ignoredListSI and not allSIResultsInvalid and containsSIResults:
                results['results']['SI'][str(fileContents['seed'])] = fileContents

    print()
    print('%i/%i QSI files had discrepancies and had to be ignored.' % (len(ignoredListQSI), len(originalFiles)))
    print('%i/%i SI files had discrepancies and had to be ignored.' % (len(ignoredListSI), len(originalFiles)))

    results['settings'] = previousExperimentSettings
    pp = pprint.PrettyPrinter(indent=4)
    #print(pp.pformat(results['settings']))

    return results

def objects(count):
    if count > 1:
        return 'Objects'
    else:
        return 'Object'

print('Loading original files..')
loadedResults = {}
for directory in inputDirectories.keys():
    print('Loading directory:', directory)
    loadedResults[directory] = loadOutputFileDirectory(directory)

def filterResultSet(resultSet, index):

    out = copy.deepcopy(resultSet)

    if 'QSISampleGeneration' in out['runtimes']:
        out['runtimes']['QSISampleGeneration']['total'] = [out['runtimes']['QSISampleGeneration']['total'][index]]
        out['runtimes']['QSISampleGeneration']['meshScale'] = [out['runtimes']['QSISampleGeneration']['meshScale'][index]]
        out['runtimes']['QSISampleGeneration']['redistribution'] = [out['runtimes']['QSISampleGeneration']['redistribution'][index]]
        out['runtimes']['QSISampleGeneration']['generation'] = [out['runtimes']['QSISampleGeneration']['generation'][index]]

    if 'QSISearch' in out['runtimes']:
        out['runtimes']['QSISearch']['total'] = [out['runtimes']['QSISearch']['total'][index]]
        out['runtimes']['QSISearch']['search'] = [out['runtimes']['QSISearch']['search'][index]]

    if 'SISampleGeneration' in out['runtimes']:
        out['runtimes']['SISampleGeneration']['total'] = [out['runtimes']['SISampleGeneration']['total'][index]]
        out['runtimes']['SISampleGeneration']['initialisation'] = [out['runtimes']['SISampleGeneration']['initialisation'][index]]
        out['runtimes']['SISampleGeneration']['sampling'] = [out['runtimes']['SISampleGeneration']['sampling'][index]]
        out['runtimes']['SISampleGeneration']['generation'] = [out['runtimes']['SISampleGeneration']['generation'][index]]

    if 'SISearch' in out['runtimes']:
        out['runtimes']['SISearch']['total'] = [out['runtimes']['SISearch']['total'][index]]
        out['runtimes']['SISearch']['averaging'] = [out['runtimes']['SISearch']['averaging'][index]]
        out['runtimes']['SISearch']['search'] = [out['runtimes']['SISearch']['search'][index]]

    if 'QSIhistograms' in out:
        out['QSIhistograms'] = {'0': out['QSIhistograms'][str(index)]}

    if 'SIhistograms' in out:
        out['SIhistograms'] = {'0': out['SIhistograms'][str(index)]}

    return out

def split(directory):
    print('Splitting', directory)
    global loadedResults

    result = loadedResults[directory]
    del loadedResults[directory]

    setMeta = inputDirectories[directory]
    del inputDirectories[directory]

    for itemCountIndex, itemCount in enumerate(result['settings']['sampleObjectCounts']):
        out = {
            'results': {'QSI': {}, 'SI': {}},
            'settings': {}}
        out['settings'] = result['settings'].copy()
        out['settings']['sampleObjectCounts'] = [itemCount]

        for qsiSeed in result['results']['QSI']:
            out['results']['QSI'][qsiSeed] = filterResultSet(result['results']['QSI'][qsiSeed], itemCountIndex)

        for siSeed in result['results']['SI']:
            out['results']['SI'][siSeed] = filterResultSet(result['results']['SI'][siSeed], itemCountIndex)

        newDirectoryName = directory + ' (' + str(itemCount) + ' objects)'

        loadedResults[newDirectoryName] = out
        inputDirectories[newDirectoryName] = (setMeta[0] + ' (' + str(itemCount) + ' objects)', setMeta[1])




def merge(directory1, directory2, newdirectoryName, newDirectoryClusterName):
    global loadedResults
    global inputDirectories

    print('Merging', directory2, 'into', directory1)

    directory1_contents = loadedResults[directory1]
    directory2_contents = loadedResults[directory2]

    del loadedResults[directory1]
    del loadedResults[directory2]
    del inputDirectories[directory1]
    del inputDirectories[directory2]

    if directory1_contents['settings'] != directory2_contents['settings']:
        print('WARNING: Directories %s and %s have different generation settings, and may therefore not be compatible to be merged!' % (directory1, directory2))

    combinedResults = {'results': {'QSI': {}, 'SI': {}}, 'settings': directory1_contents['settings']}

    # Initialising with the original results
    combinedResults['results']['QSI'] = directory1_contents['results']['QSI']
    combinedResults['results']['SI'] = directory1_contents['results']['SI']

    # Now we merge any missing results into it
    for type in directory2_contents['results']:
        for seed in directory2_contents['results'][type].keys():
            if seed not in combinedResults['results'][type]:
                combinedResults['results'][type][seed] = directory2_contents['results'][type][seed]

    loadedResults[newdirectoryName] = combinedResults
    inputDirectories[newdirectoryName] = (newdirectoryName, newDirectoryClusterName)

print('Merging similar datasets..')
split('../IDUNRUNS/output_smallsupportangle_lotsofobjects')
split('../IDUNRUNS/output_qsifix_smallsupportangle_rerun')
split('../IDUNRUNS/output_mainchart_si_v4_15')

# QSI runs
merge('../IDUNRUNS/output_lotsofobjects_v4', '../HEIDRUNS/output_qsifix_v4_lotsofobjects_idun_failed/output', 'QSI, 1, 5, and 10 objects', 'HEID + IDUN')

# SI 180 degrees, 1 object
merge('../IDUNRUNS/output_mainchart_si_v4_1', '../IDUNRUNS/output_mainchart_si_v4_15 (1 objects)', 'SI 180 degrees, 1 object', 'IDUN')

# SI 180 degrees, 5 objects
merge('../HEIDRUNS/output_qsifix_v4_lotsofobjects_5_objects_only/output', '../IDUNRUNS/output_mainchart_si_v4_15 (5 objects)', 'SI 180 degrees, 5 objects', 'HEID + IDUN')

# SI 180 degrees, 10 objects
merge('../HEIDRUNS/output_qsifix_v4_lotsofobjects_10_objects_only/output', '../IDUNRUNS/output_mainchart_si_v4_10', 'SI 180 degrees, 10 objects', 'HEID + IDUN')
merge('SI 180 degrees, 10 objects', '../HEIDRUNS/output_qsifix_v4_180deg_si_missing/output', 'SI 180 degrees, 10 objects', 'HEID + IDUN')

# SI 60 degrees, 1 object
merge('../IDUNRUNS/output_supportanglechart60_si_v4_1', '../IDUNRUNS/output_smallsupportangle_lotsofobjects (1 objects)', 'SI 60 degrees, 1 object intermediate', 'IDUN')
merge('SI 60 degrees, 1 object intermediate', '../IDUNRUNS/output_qsifix_smallsupportangle_rerun (1 objects)', 'SI 60 degrees, 1 object', 'IDUN')

# SI 60 degrees, 5 objects
merge('../IDUNRUNS/output_supportanglechart60_si_v4_5', '../IDUNRUNS/output_smallsupportangle_lotsofobjects (5 objects)', 'SI 60 degrees, 5 objects', 'IDUN')

# SO 60 degrees, 10 objects
merge('../HEIDRUNS/output_qsifix_v4_60deg_si_missing/output/', '../IDUNRUNS/output_smallsupportangle_lotsofobjects (10 objects)', 'SI 60 deg 10 objects intermediate', 'HEID + IDUN')
merge('SI 60 deg 10 objects intermediate', '../IDUNRUNS/output_qsifix_smallsupportangle_rerun (10 objects)', 'SI 60 degrees, 10 objects', 'HEID + IDUN')

print('Processing..')
seedSet = set()
for directory in inputDirectories.keys():
    for seed in loadedResults[directory]['results']['QSI'].keys():
        seedSet.add(seed)
    for seed in loadedResults[directory]['results']['SI'].keys():
        seedSet.add(seed)
seedList = [x for x in seedSet]

print('\tFound', len(seedSet), 'seeds in result sets')




# -- Dump to spreadsheet --

print('Dumping spreadsheet..')

book = xlwt.Workbook(encoding="utf-8")

# Create data page for dataset settings table
experimentSheet = book.add_sheet("Experiment Overview")
allColumns = set()
for directoryIndex, directory in enumerate(inputDirectories.keys()):
    result = loadedResults[directory]
    allColumns = allColumns.union(set(result['settings']))

# Overview table headers
for keyIndex, key in enumerate(allColumns):
    experimentSheet.write(0, keyIndex + 1, str(key))
experimentSheet.write(0, len(allColumns) + 1, 'Cluster')
experimentSheet.write(0, len(allColumns) + 2, 'QSI Count')
experimentSheet.write(0, len(allColumns) + 3, 'SI Count')

# Overview table contents
for directoryIndex, directory in enumerate(inputDirectories.keys()):
    directoryName, cluster = inputDirectories[directory]
    experimentSheet.write(directoryIndex + 1, 0, directoryName)
    result = loadedResults[directory]
    for keyIndex, key in enumerate(allColumns):
        if key in result['settings']:
            experimentSheet.write(directoryIndex + 1, keyIndex + 1, str(result['settings'][key]))
        else:
            experimentSheet.write(directoryIndex + 1, keyIndex + 1, ' ')

    experimentSheet.write(directoryIndex + 1, len(allColumns) + 1, cluster)
    experimentSheet.write(directoryIndex + 1, len(allColumns) + 2, len(result['results']['QSI']))
    experimentSheet.write(directoryIndex + 1, len(allColumns) + 3, len(result['results']['SI']))



# Sheets
top0sheetQSI = book.add_sheet("Rank 0 QSI results")
top0sheetSI = book.add_sheet("Rank 0 SI results")
top10sheetQSI = book.add_sheet("Top 10 QSI results")
top10sheetSI = book.add_sheet("Top 10 SI results")
qsiGenerationSpeedSheet = book.add_sheet("QSI Generation Times")
siGenerationSpeedSheet = book.add_sheet("SI Generation Times")
qsiComparisonSpeedSheet = book.add_sheet("QSI Comparison Times")
siComparisonSpeedSheet = book.add_sheet("SI Comparison Times")
vertexCountSheet = book.add_sheet("Reference Image Count")
totalVertexCountSheet = book.add_sheet("Total Image Count")
totalTriangleCountSheet = book.add_sheet("Total Triangle Count")

# seed column is 0, data starts at column 1
currentColumn = 1

for directoryIndex, directory in enumerate(inputDirectories.keys()):

    # Writing seed column
    if directoryIndex == 0:
        top0sheetQSI.write(0, 0, 'seed')
        top0sheetSI.write(0, 0, 'seed')
        top10sheetQSI.write(0, 0, 'seed')
        top10sheetSI.write(0, 0, 'seed')
        qsiGenerationSpeedSheet.write(0, 0, 'seed')
        siGenerationSpeedSheet.write(0, 0, 'seed')
        qsiComparisonSpeedSheet.write(0, 0, 'seed')
        siComparisonSpeedSheet.write(0, 0, 'seed')
        vertexCountSheet.write(0, 0, 'seed')
        totalVertexCountSheet.write(0, 0, 'seed')
        totalTriangleCountSheet.write(0, 0, 'seed')

        for seedIndex, seed in enumerate(seedList):
            top0sheetQSI.write(seedIndex + 1, 0, seed)
            top0sheetSI.write(seedIndex + 1, 0, seed)
            top10sheetQSI.write(seedIndex + 1, 0, seed)
            top10sheetSI.write(seedIndex + 1, 0, seed)
            qsiGenerationSpeedSheet.write(seedIndex + 1, 0, seed)
            siGenerationSpeedSheet.write(seedIndex + 1, 0, seed)
            qsiComparisonSpeedSheet.write(seedIndex + 1, 0, seed)
            siComparisonSpeedSheet.write(seedIndex + 1, 0, seed)
            vertexCountSheet.write(seedIndex + 1, 0, seed)
            totalVertexCountSheet.write(seedIndex + 1, 0, seed)
            totalTriangleCountSheet.write(seedIndex + 1, 0, seed)

    resultSet = loadedResults[directory]
    directoryName, _ = inputDirectories[directory]

    # Writing column headers
    for sampleCountIndex, sampleObjectCount in enumerate(resultSet['settings']['sampleObjectCounts']):
        columnHeader = directoryName + ' (' + str(sampleObjectCount) + ' ' + objects(
                               len(resultSet['settings']['sampleObjectCounts'])) + ')'
        top0sheetQSI.write(0, currentColumn + sampleCountIndex, columnHeader)
        top0sheetSI.write(0, currentColumn + sampleCountIndex, columnHeader)
        top10sheetQSI.write(0, currentColumn + sampleCountIndex, columnHeader)
        top10sheetSI.write(0, currentColumn + sampleCountIndex, columnHeader)

        qsiGenerationSpeedSheet.write(0, currentColumn + sampleCountIndex, columnHeader)
        siGenerationSpeedSheet.write(0, currentColumn + sampleCountIndex, columnHeader)
        qsiComparisonSpeedSheet.write(0, currentColumn + sampleCountIndex, columnHeader)
        siComparisonSpeedSheet.write(0, currentColumn + sampleCountIndex, columnHeader)

        vertexCountSheet.write(0, currentColumn + sampleCountIndex, columnHeader)
        totalVertexCountSheet.write(0, currentColumn + sampleCountIndex, columnHeader)
        totalTriangleCountSheet.write(0, currentColumn + sampleCountIndex, columnHeader)


    for seedIndex, seed in enumerate(seedList):
        if seed in resultSet['results']['QSI']:
            for sampleCountIndex, sampleObjectCount in enumerate(resultSet['settings']['sampleObjectCounts']):
                # Top 1 performance
                entry = resultSet['results']['QSI'][seed]
                totalImageCount = entry['imageCounts'][0]
                experimentIterationCount = len(resultSet['settings']['sampleObjectCounts'])
                percentageAtPlace0 = float(entry['QSIhistograms'][str(sampleCountIndex)]['0']) / float(totalImageCount)
                top0sheetQSI.write(seedIndex + 1, currentColumn + sampleCountIndex, percentageAtPlace0)

                # Top 10 performance
                totalImageCountInTop10 = sum(
                    [entry['QSIhistograms'][str(sampleCountIndex)][str(x)] for x in range(0, 10) if
                    str(x) in entry['QSIhistograms'][str(sampleCountIndex)]])
                percentInTop10 = float(totalImageCountInTop10) / float(totalImageCount)
                top10sheetQSI.write(seedIndex + 1, currentColumn + sampleCountIndex, percentInTop10)

                # generation execution time
                generationTime = entry['runtimes']['QSISampleGeneration']['total'][sampleCountIndex]
                qsiGenerationSpeedSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, generationTime)

                # search execution time
                comparisonTime = entry['runtimes']['QSISearch']['total'][sampleCountIndex]
                qsiComparisonSpeedSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, comparisonTime)

                # Vertex count sanity check
                vertexCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, entry['imageCounts'][0])
                totalVertexCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, sum(entry['imageCounts'][0:sampleObjectCount]))
                totalTriangleCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, sum(entry['vertexCounts'][0:sampleObjectCount]) / 3)
        else:
            for sampleCountIndex, sampleObjectCount in enumerate(resultSet['settings']['sampleObjectCounts']):
                top0sheetQSI.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                top10sheetQSI.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                qsiGenerationSpeedSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                qsiComparisonSpeedSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')

        if seed in resultSet['results']['SI']:
            for sampleCountIndex, sampleObjectCount in enumerate(resultSet['settings']['sampleObjectCounts']):
                # Top 1 performance
                entry = resultSet['results']['SI'][seed]
                totalImageCount = entry['imageCounts'][0]
                experimentIterationCount = len(resultSet['settings']['sampleObjectCounts'])
                percentageAtPlace0 = float(entry['SIhistograms'][str(sampleCountIndex)]['0']) / float(totalImageCount)
                top0sheetSI.write(seedIndex + 1, currentColumn + sampleCountIndex, percentageAtPlace0)

                # Top 10 performance
                totalImageCountInTop10 = sum(
                    [entry['SIhistograms'][str(sampleCountIndex)][str(x)] for x in range(0, 10) if
                    str(x) in entry['SIhistograms'][str(sampleCountIndex)]])
                percentInTop10 = float(totalImageCountInTop10) / float(totalImageCount)
                top10sheetSI.write(seedIndex + 1, currentColumn + sampleCountIndex, percentInTop10)

                # generation execution time
                generationTime = entry['runtimes']['SISampleGeneration']['total'][sampleCountIndex]
                siGenerationSpeedSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, generationTime)

                # search execution time
                comparisonTime = entry['runtimes']['SISearch']['total'][sampleCountIndex]
                siComparisonSpeedSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, comparisonTime)

                # Vertex count sanity check
                vertexCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, entry['imageCounts'][0])
                totalVertexCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, sum(entry['imageCounts'][0:sampleObjectCount]))
                totalTriangleCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, sum(entry['vertexCounts'][0:sampleObjectCount]) / 3)
        else:
            for sampleCountIndex, sampleObjectCount in enumerate(resultSet['settings']['sampleObjectCounts']):
                top0sheetSI.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                top10sheetSI.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                siGenerationSpeedSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                siComparisonSpeedSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')

    # Moving on to the next column
    currentColumn += len(resultSet['settings']['sampleObjectCounts'])

# beauty addition.. Cuts off final column
for seedIndex, seed in enumerate(seedList + ['dummy entry for final row']):
    top0sheetQSI.write(seedIndex, currentColumn, ' ')
    top10sheetQSI.write(seedIndex, currentColumn, ' ')
    qsiGenerationSpeedSheet.write(seedIndex, currentColumn, ' ')
    qsiComparisonSpeedSheet.write(seedIndex, currentColumn, ' ')

    top0sheetSI.write(seedIndex, currentColumn, ' ')
    top10sheetSI.write(seedIndex, currentColumn, ' ')
    siGenerationSpeedSheet.write(seedIndex, currentColumn, ' ')
    siComparisonSpeedSheet.write(seedIndex, currentColumn, ' ')

    vertexCountSheet.write(seedIndex, currentColumn, ' ')
    totalVertexCountSheet.write(seedIndex, currentColumn, ' ')
    totalTriangleCountSheet.write(seedIndex, currentColumn, ' ')







book.save(outfile)
print('Complete.')