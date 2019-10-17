import json
import os
import os.path
import datetime
import xlwt
import pprint
from math import sqrt

# -- settings --

inputDirectories = [
    '../HEIDRUNS/output_qsifix_v4_noearlyexit/output',
    '../HEIDRUNS/output_qsifix_v4_withearlyexit/output',
    '../IDUNRUNS/output_lotsofobjects_v4',
    '../HEIDRUNS/output_qsifix_v4_lotsofobjects_idun_failed/output',
    '../IDUNRUNS/output_smallsupportangle_lotsofobjects',
    '../IDUNRUNS/output_qsifix_smallsupportangle_rerun',
]
inputDirectoryDatasetNames = [
    'QSI, no early exit',
    'QSI, with early exit',
    'QSI, primary QSI IDUN run',
    'QSI, failed jobs from IDUN run',
    'SI, 60 support angle, primary IDUN run',
    'SI, 60 support angle, secondary IDUN run'
]
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
    if 'descriptors' in loadedJson:
        settings['descriptors'] = loadedJson['descriptors']
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
            fileContents = json.loads(openFile.read())
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
for directory in inputDirectories:
    print('Loading directory:', directory)
    loadedResults[directory] = loadOutputFileDirectory(directory)


print('Processing..')
seedSet = set()
for directory in inputDirectories:
    for seed in loadedResults[directory]['results']['QSI'].keys():
        seedSet.add(seed)
    for seed in loadedResults[directory]['results']['SI'].keys():
        seedSet.add(seed)
seedList = [x for x in seedSet]

print('\tFound', len(seedSet), 'seeds in result sets')




# -- Dump to spreadsheet --

print('Dumping spreadsheet..')

book = xlwt.Workbook(encoding="utf-8")

# 1. Create data page for dataset settings table
experimentSheet = book.add_sheet("Experiment Overview")
for directoryIndex, directory in enumerate(inputDirectories):
    directoryName = inputDirectoryDatasetNames[directoryIndex]
    experimentSheet.write(directoryIndex + 1, 0, directoryName)
    resultSet = loadedResults[directory]
    for keyIndex, key in enumerate(resultSet['settings'].keys()):
        if directoryIndex == 0:
            experimentSheet.write(0, keyIndex + 1, str(key))
        experimentSheet.write(directoryIndex + 1, keyIndex + 1, str(resultSet['settings'][key]))

# 2. Create data page for mapping of seed -> % at rank 0
top0sheetQSI = book.add_sheet("Rank 0 QSI results")
top0sheetSI = book.add_sheet("Rank 0 SI results")
top10sheetQSI = book.add_sheet("Top 10 QSI results")
top10sheetSI = book.add_sheet("Top 10 SI results")
currentColumn = 1
top0sheetQSI.write(0, 0, 'seed')
top0sheetSI.write(0, 0, 'seed')
top10sheetQSI.write(0, 0, 'seed')
top10sheetSI.write(0, 0, 'seed')
vertexCountSheet = book.add_sheet("Vertex Count Sanity Check")

for directoryIndex, directory in enumerate(inputDirectories):
    if directoryIndex == 0:
        for seedIndex, seed in enumerate(seedList):
            top0sheetQSI.write(seedIndex + 1, 0, str(seed))
            top0sheetSI.write(seedIndex + 1, 0, str(seed))
    resultSet = loadedResults[directory]
    directoryName = inputDirectoryDatasetNames[directoryIndex]
    for sampleCountIndex, sampleObjectCount in enumerate(resultSet['settings']['sampleObjectCounts']):
        top0sheetQSI.write(0, currentColumn + sampleCountIndex,
                           directoryName + ' (' + str(sampleObjectCount) + ' ' + objects(
                               len(resultSet['settings']['sampleObjectCounts'])) + ')')
        top0sheetSI.write(0, currentColumn + sampleCountIndex,
                          directoryName + ' (' + str(sampleObjectCount) + ' ' + objects(
                              len(resultSet['settings']['sampleObjectCounts'])) + ')')
        top10sheetQSI.write(0, currentColumn + sampleCountIndex,
                           directoryName + ' (' + str(sampleObjectCount) + ' ' + objects(
                               len(resultSet['settings']['sampleObjectCounts'])) + ')')
        top10sheetSI.write(0, currentColumn + sampleCountIndex,
                          directoryName + ' (' + str(sampleObjectCount) + ' ' + objects(
                              len(resultSet['settings']['sampleObjectCounts'])) + ')')
    for seedIndex, seed in enumerate(seedList):
        if seed in resultSet['results']['QSI']:
            for sampleCountIndex, sampleObjectCount in enumerate(resultSet['settings']['sampleObjectCounts']):
                entry = resultSet['results']['QSI'][seed]
                experimentIterationCount = len(resultSet['settings']['sampleObjectCounts'])
                totalImageCount = entry['imageCounts'][0]
                percentageAtPlace0 = float(entry['QSIhistograms'][str(sampleCountIndex)]['0']) / float(totalImageCount)
                top0sheetQSI.write(seedIndex + 1, currentColumn + sampleCountIndex, str(percentageAtPlace0))

                top10ImageCount = sum(entry['imageCounts'][0:min(10, len(entry['imageCounts']))])
                totalImageCountInTop10 = sum(
                    [entry['QSIhistograms'][str(sampleCountIndex)][str(x)] for x in range(0, 10) if
                    str(x) in entry['QSIhistograms'][str(sampleCountIndex)]])
                percentInTop10 = float(totalImageCountInTop10) / float(top10ImageCount)
                top10sheetQSI.write(seedIndex + 1, currentColumn + sampleCountIndex, str(percentInTop10))

                vertexCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, entry['imageCounts'][0])
        if seed in resultSet['results']['SI']:
            for sampleCountIndex, sampleObjectCount in enumerate(resultSet['settings']['sampleObjectCounts']):
                entry = resultSet['results']['SI'][seed]
                experimentIterationCount = len(resultSet['settings']['sampleObjectCounts'])
                totalImageCount = entry['imageCounts'][0]
                percentageAtPlace0 = float(entry['SIhistograms'][str(sampleCountIndex)]['0']) / float(totalImageCount)
                top0sheetSI.write(seedIndex + 1, currentColumn + sampleCountIndex, str(percentageAtPlace0))

                top10ImageCount = sum(entry['imageCounts'][0:min(10, len(entry['imageCounts']))])
                totalImageCountInTop10 = sum(
                    [entry['SIhistograms'][str(sampleCountIndex)][str(x)] for x in range(0, 10) if
                    str(x) in entry['SIhistograms'][str(sampleCountIndex)]])
                percentInTop10 = float(totalImageCountInTop10) / float(top10ImageCount)
                top10sheetSI.write(seedIndex + 1, currentColumn + sampleCountIndex, str(percentInTop10))

                vertexCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, entry['imageCounts'][0])
    currentColumn += len(resultSet['settings']['sampleObjectCounts'])







book.save(outfile)
print('Complete.')