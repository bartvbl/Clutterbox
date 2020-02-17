import json
import os
import os.path
import datetime
import xlwt
import pprint
import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from PIL import Image

# --- SETTINGS ---

# Master input directories
inputDirectories = {
    '../HEIDRUNS/output_qsifix_v4_noearlyexit/output': ('QSI, No early exit, 5 objects', 'HEID'),
    '../HEIDRUNS/output_qsifix_v4_withearlyexit/output': ('QSI, Early exit, 5 objects', 'HEID'),
    '../HEIDRUNS/output_qsifix_v4_lotsofobjects_idun_failed/output': ('Failed jobs from IDUN run', 'HEID'),
    '../IDUNRUNS/output_lotsofobjects_v4': ('primary QSI IDUN run', 'IDUN'),
    '../HEIDRUNS/output_seeds_qsi_v4_5objects_missing/output': ('re-run of 5 object QSI results that were missing raw files', 'HEID'),

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
    '../HEIDRUNS/output_qsifix_si_v4_60deg_5objects_missing/output/': ('60 support angle, 5 objects', 'HEID'),

    '../HEIDRUNS/run1_3dsc_main/output/': ('3DSC', 'HEID'),
    #'../IDUNRUNS/output_run4_3dsc_main/run1/': ('3DSC, 1, 5, 10 objects', 'IDUN'),
}

# The location where the master spreadsheet should be written to
outfile = 'final_results/master_spreadsheet.xls'

# Settings for clutter heatmaps
heatmapSize = 256

rawInputDirectories = {
    'QSI': ['../HEIDRUNS/output_seeds_qsi_v4_5objects_missing/output/raw', '../IDUNRUNS/output_lotsofobjects_v4/raw'],
    'SI': ['../HEIDRUNS/output_qsifix_v4_lotsofobjects_5_objects_only/output/raw', '../IDUNRUNS/output_mainchart_si_v4_15/raw'],
    '3DSC': ['../HEIDRUNS/run1_3dsc_main/output/raw'],
    #'3DSC': ['../IDUNRUNS/output_run4_3dsc_main/run1/raw'],
}
rawInputObjectCount = 5

clutterFileDirectories = ['../clutter/output_5objects/output']



# Distill the result set down to the entries we have all data for
removeSeedsWithMissingEntries = True

# Cut down the result set to a specific number of entries
resultSetSizeLimit = 1500
enableResultSetSizeLimit = True


# --- start of code ---

print()
print(' === Processing of experiment output files into the master spreadsheet ===')
print('This spreadsheet contains the exact data used to construct the charts in the paper')
print('Having a machine with 24GB of RAM is probably a necessity for running this')
print()

matplotlib.use('Qt5Agg')

# Last known fault in code: unsigned integer subtraction in RICI comparison function
# Date threshold corresponds to this commit
def isQsiResultValid(fileCreationDateTime, resultJson):
    riciResultsValidAfter = datetime.datetime(year=2019, month=10, day=7, hour=15, minute=14, second=0, microsecond=0)
    return fileCreationDateTime > riciResultsValidAfter


# Last known fault in code: override object count did not match
# Date threshold corresponds to this commit
def isSiResultValid(fileCreationDateTime, resultJson):
    siResultsValidAfter = datetime.datetime(year=2019, month=9, day=26, hour=17, minute=28, second=0, microsecond=0)
    hasCorrectOverrideObjectCount = 'overrideObjectCount' in resultJson and resultJson['overrideObjectCount'] == 10
    hasCorrectSampleSetSize = resultJson['sampleSetSize'] == 10
    hasValidCreationDateTime = siResultsValidAfter < fileCreationDateTime

    return hasCorrectOverrideObjectCount or hasCorrectSampleSetSize

def is3dscResultValid(fileCreationTime, resultJson):
    return True


# -- global initialisation --

# Maps seeds to a list of (dataset, value) tuples
seedmap_top_result_rici = {}
seedmap_top_result_si = {}
seedmap_top_result_3dsc = {}
seedmap_top10_results_rici = {}
seedmap_top10_results_si = {}
seedmap_top10_results_3dsc = {}

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
        if '3DSChistograms' in loadedJson:
            descriptors.append('3dsc')
        settings['descriptors'] = descriptors
    settings['version'] = loadedJson['version']
    return settings


def loadOutputFileDirectory(path):
    originalFiles = os.listdir(path)
    # Filter out the raw output file directories
    originalFiles = [x for x in originalFiles if x != 'raw' and x != 'rawless']

    results = {
        'path': path,
        'results': {'QSI': {}, 'SI': {}, '3DSC': {}},
        'settings': {}
    }

    jsonCache = {}

    ignoredListRICI = []
    ignoredListSI = []
    ignoredList3DSC = []

    allRICIResultsInvalid = False
    allSIResultsInvalid = False
    all3DSCResultsInvalid = False

    for fileindex, file in enumerate(originalFiles):
        print(str(fileindex + 1) + '/' + str(len(originalFiles)), file + '        ', end='\r', flush=True)
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
                ignoredListRICI.append(file)
                allRICIResultsInvalid = True
            if not isSiResultValid(creation_time, fileContents):
                ignoredListSI.append(file)
                allSIResultsInvalid = True
            if not is3dscResultValid(creation_time, fileContents):
                ignoredList3DSC.append(file)
                all3DSCResultsInvalid = True

    previousExperimentSettings = None
    for fileindex, file in enumerate(originalFiles):
        print(str(fileindex + 1) + '/' + str(len(originalFiles)), file + '        ', end='\r', flush=True)

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
            if file not in ignoredListRICI:
                ignoredListRICI.append(file)
            if file not in ignoredListSI:
                ignoredListSI.append(file)
            if file not in ignoredList3DSC:
                ignoredList3DSC.append(file)
            # print('ignored 0',file)
        if fileContents['spinImageWidthPixels'] == 32:
            if file not in ignoredListRICI:
                ignoredListRICI.append(file)
            if file not in ignoredListSI:
                ignoredListSI.append(file)
            if file not in ignoredList3DSC:
                ignoredList3DSC.append(file)

        # Beauty checks
        if file not in ignoredListRICI and allRICIResultsInvalid:
            ignoredListRICI.append(file)
        if file not in ignoredListSI and allSIResultsInvalid:
            ignoredListSI.append(file)
        if file not in ignoredList3DSC and all3DSCResultsInvalid:
            ignoredList3DSC.append(file)

        containsRICIResults = ('descriptors' in fileContents and 'qsi' in fileContents[
            'descriptors']) or 'QSIhistograms' in fileContents
        containsSIResults = ('descriptors' in fileContents and 'si' in fileContents[
            'descriptors']) or 'SIhistograms' in fileContents
        contains3DSCResults = ('descriptors' in fileContents and '3dsc' in fileContents[
            'descriptors']) or '3DSChistograms' in fileContents

        # Sanity checks are done. We can now add any remaining valid entries to the result lists
        if not file in ignoredListRICI and not allRICIResultsInvalid and containsRICIResults:
            results['results']['QSI'][str(fileContents['seed'])] = fileContents
        if not file in ignoredListSI and not allSIResultsInvalid and containsSIResults:
            results['results']['SI'][str(fileContents['seed'])] = fileContents
        if not file in ignoredList3DSC and not all3DSCResultsInvalid and contains3DSCResults:
            results['results']['3DSC'][str(fileContents['seed'])] = fileContents

    print()
    # print('%i/%i RICI files had discrepancies and had to be ignored.' % (len(ignoredListRICI), len(originalFiles)))
    # print('%i/%i SI files had discrepancies and had to be ignored.' % (len(ignoredListSI), len(originalFiles)))
    # print('%i/%i 3DSC files had discrepancies and had to be ignored.' % (len(ignoredList3DSC), len(originalFiles)))

    results['settings'] = previousExperimentSettings
    pp = pprint.PrettyPrinter(indent=4)
    # print(pp.pformat(results['settings']))

    return results


def objects(count):
    if count > 1:
        return 'Objects'
    else:
        return 'Object'



print('Loading raw data files..')
loadedRawResults = {'QSI': {}, 'SI': {}, '3DSC': {}}
for algorithm in rawInputDirectories:
    for path in rawInputDirectories[algorithm]:
        print('Loading raw directory:', path)
        rawFiles = os.listdir(path)
        for fileIndex, file in enumerate(rawFiles):
            print(str(fileIndex + 1) + '/' + str(len(rawFiles)), file + '        ', end='\r', flush=True)
            seed = file.split('.')[0].split("_")[2]
            if not seed in loadedRawResults[algorithm]:
                with open(os.path.join(path, file), 'r') as openFile:
                    # Read JSON file
                    try:
                        rawFileContents = json.loads(openFile.read())
                        loadedRawResults[algorithm][seed] = rawFileContents[algorithm][str(rawInputObjectCount)]
                    except Exception as e:
                        print('FAILED TO READ FILE: ' + str(file))
                        print(e)
                        continue
        print()


print()
print('Loading input data files..')
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

    if '3DSCSampleGeneration' in out['runtimes']:
        out['runtimes']['3DSCSampleGeneration']['total'] = [out['runtimes']['3DSCSampleGeneration']['total'][index]]
        out['runtimes']['3DSCSampleGeneration']['initialisation'] = [out['runtimes']['3DSCSampleGeneration']['initialisation'][index]]
        out['runtimes']['3DSCSampleGeneration']['sampling'] = [out['runtimes']['3DSCSampleGeneration']['sampling'][index]]
        out['runtimes']['3DSCSampleGeneration']['pointCounting'] = [out['runtimes']['3DSCSampleGeneration']['pointCounting'][index]]
        out['runtimes']['3DSCSampleGeneration']['generation'] = [out['runtimes']['3DSCSampleGeneration']['generation'][index]]

    if '3DSCSearch' in out['runtimes']:
        out['runtimes']['3DSCSearch']['total'] = [out['runtimes']['3DSCSearch']['total'][index]]
        out['runtimes']['3DSCSearch']['search'] = [out['runtimes']['3DSCSearch']['search'][index]]

    if 'QSIhistograms' in out:
        out['QSIhistograms'] = {'0': out['QSIhistograms'][str(index)]}

    if 'SIhistograms' in out:
        out['SIhistograms'] = {'0': out['SIhistograms'][str(index)]}

    if '3DSChistograms' in out:
        out['3DSChistograms'] = {'0': out['3DSChistograms'][str(index)]}

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
            'results': {'QSI': {}, 'SI': {}, '3DSC': {}},
            'settings': {}}
        out['settings'] = result['settings'].copy()
        out['settings']['sampleObjectCounts'] = [itemCount]

        for qsiSeed in result['results']['QSI']:
            out['results']['QSI'][qsiSeed] = filterResultSet(result['results']['QSI'][qsiSeed], itemCountIndex)

        for siSeed in result['results']['SI']:
            out['results']['SI'][siSeed] = filterResultSet(result['results']['SI'][siSeed], itemCountIndex)

        for scSeed in result['results']['3DSC']:
            out['results']['3DSC'][scSeed] = filterResultSet(result['results']['3DSC'][scSeed], itemCountIndex)

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
        print(
            'WARNING: Directories %s and %s have different generation settings, and may therefore not be compatible to be merged!' % (
            directory1, directory2))
        print('Directory 1:', directory1_contents['settings'])
        print('Directory 2:', directory2_contents['settings'])

    combinedResults = {'results': {'QSI': {}, 'SI': {}, '3DSC': {}}, 'settings': directory1_contents['settings']}

    # Initialising with the original results
    combinedResults['results']['QSI'] = directory1_contents['results']['QSI']
    combinedResults['results']['SI'] = directory1_contents['results']['SI']
    combinedResults['results']['3DSC'] = directory1_contents['results']['3DSC']

    additionCount = 0

    # Now we merge any missing results into it
    for type in directory2_contents['results']:
        for seed in directory2_contents['results'][type].keys():
            if seed not in combinedResults['results'][type]:
                combinedResults['results'][type][seed] = directory2_contents['results'][type][seed]
                additionCount += 1

    loadedResults[newdirectoryName] = combinedResults
    inputDirectories[newdirectoryName] = (newdirectoryName, newDirectoryClusterName)

    print('\tAdded', additionCount, 'new values')
    return additionCount

# Small hack, but silences a warning that does not apply here
loadedResults['../HEIDRUNS/output_qsifix_v4_lotsofobjects_idun_failed/output']['settings']['overrideObjectCount'] = 10

print('\nRestructuring datasets..\n')
split('../IDUNRUNS/output_smallsupportangle_lotsofobjects')
split('../IDUNRUNS/output_qsifix_smallsupportangle_rerun')
split('../IDUNRUNS/output_mainchart_si_v4_15')
split('../IDUNRUNS/output_lotsofobjects_v4')
split('../HEIDRUNS/output_qsifix_v4_lotsofobjects_idun_failed/output')
print()

# QSI 1 object
merge('../IDUNRUNS/output_lotsofobjects_v4 (1 objects)', '../HEIDRUNS/output_qsifix_v4_lotsofobjects_idun_failed/output (1 objects)',
      'QSI, 1 object', 'HEID + IDUN')

# QSI 5 objects
merge('../HEIDRUNS/output_seeds_qsi_v4_5objects_missing/output', '../IDUNRUNS/output_lotsofobjects_v4 (5 objects)', 'QSI primary intermediate', 'HEID + IDUN')
merge('QSI primary intermediate', '../HEIDRUNS/output_qsifix_v4_lotsofobjects_idun_failed/output (5 objects)',
      'QSI, 5 objects', 'HEID + IDUN')

# QSI 10 objects
merge('../IDUNRUNS/output_lotsofobjects_v4 (10 objects)', '../HEIDRUNS/output_qsifix_v4_lotsofobjects_idun_failed/output (10 objects)',
      'QSI, 10 objects', 'HEID + IDUN')

# SI 180 degrees, 1 object
merge('../IDUNRUNS/output_mainchart_si_v4_1', '../IDUNRUNS/output_mainchart_si_v4_15 (1 objects)',
      'SI 180 degrees, 1 object', 'IDUN')

# SI 180 degrees, 5 objects
additionCount = merge('../HEIDRUNS/output_qsifix_v4_lotsofobjects_5_objects_only/output',
                      '../IDUNRUNS/output_mainchart_si_v4_15 (5 objects)', 'SI 180 degrees, 5 objects', 'HEID')
# this merge is mainly to remove the dataset from the input batch. We ultimately want the HEIDRUNS results exclusively because
# we use these results to compare runtimes
assert (additionCount == 0)

# SI 180 degrees, 10 objects
merge('../HEIDRUNS/output_qsifix_v4_lotsofobjects_10_objects_only/output', '../IDUNRUNS/output_mainchart_si_v4_10',
      'SI 180 degrees, 10 objects', 'HEID + IDUN')
merge('SI 180 degrees, 10 objects', '../HEIDRUNS/output_qsifix_v4_180deg_si_missing/output',
      'SI 180 degrees, 10 objects', 'HEID + IDUN')

# SI 60 degrees, 1 object
merge('../IDUNRUNS/output_supportanglechart60_si_v4_1',
      '../IDUNRUNS/output_smallsupportangle_lotsofobjects (1 objects)', 'SI 60 degrees, 1 object intermediate', 'IDUN')
merge('SI 60 degrees, 1 object intermediate', '../IDUNRUNS/output_qsifix_smallsupportangle_rerun (1 objects)',
      'SI 60 degrees, 1 object', 'IDUN')

# SI 60 degrees, 5 objects
merge('../HEIDRUNS/output_qsifix_si_v4_60deg_5objects_missing/output/', '../IDUNRUNS/output_supportanglechart60_si_v4_5', 'SI 60 degrees, 5 objects, intermediate', 'HEID + IDUN')
merge('SI 60 degrees, 5 objects, intermediate',
      '../IDUNRUNS/output_smallsupportangle_lotsofobjects (5 objects)', 'SI 60 degrees, 5 objects', 'HEID + IDUN')

# SO 60 degrees, 10 objects
merge('../HEIDRUNS/output_qsifix_v4_60deg_si_missing/output/',
      '../IDUNRUNS/output_smallsupportangle_lotsofobjects (10 objects)', 'SI 60 deg 10 objects intermediate',
      'HEID + IDUN')
merge('SI 60 deg 10 objects intermediate', '../IDUNRUNS/output_qsifix_smallsupportangle_rerun (10 objects)',
      'SI 60 degrees, 10 objects', 'HEID + IDUN')

print()
split('../HEIDRUNS/run1_3dsc_main/output/')

print('\nProcessing..\n')

seedSet = set()
for directory in inputDirectories.keys():
    for seed in loadedResults[directory]['results']['QSI'].keys():
        seedSet.add(seed)
    for seed in loadedResults[directory]['results']['SI'].keys():
        seedSet.add(seed)
    for seed in loadedResults[directory]['results']['3DSC'].keys():
        seedSet.add(seed)
seedList = [x for x in seedSet]

print('Found', len(seedSet), 'seeds in result sets')



if removeSeedsWithMissingEntries:
    print('\nRemoving missing entries..')
    missingSeeds = []

    for directory in loadedResults:
        cutCount = 0
        for seed in seedList:
            if len(loadedResults[directory]['results']['QSI']) > 0:
                if not seed in loadedResults[directory]['results']['QSI']:
                    missingSeeds.append(seed)
                    cutCount += 1
            if len(loadedResults[directory]['results']['SI']) > 0:
                if not seed in loadedResults[directory]['results']['SI']:
                    missingSeeds.append(seed)
                    cutCount += 1
            if len(loadedResults[directory]['results']['3DSC']) > 0:
                if not seed in loadedResults[directory]['results']['3DSC']:
                    missingSeeds.append(seed)
                    cutCount += 1
        print(directory, '- removed seed count:', cutCount)

    print('Detected', len(set(missingSeeds)), 'unique seeds with missing entries. Removing..')

    for missingSeed in missingSeeds:
        for directory in loadedResults:
            if missingSeed in loadedResults[directory]['results']['QSI']:
                del loadedResults[directory]['results']['QSI'][missingSeed]
            if missingSeed in loadedResults[directory]['results']['SI']:
                del loadedResults[directory]['results']['SI'][missingSeed]
            if missingSeed in loadedResults[directory]['results']['3DSC']:
                del loadedResults[directory]['results']['3DSC'][missingSeed]
        if missingSeed in seedList:
            del seedList[seedList.index(missingSeed)]

print('\nLoading clutter files..')
clutterFileMap = {}
for clutterFileDirectory in clutterFileDirectories:
    print('Reading directory', clutterFileDirectory)
    clutterFiles = os.listdir(clutterFileDirectory)
    for clutterFileIndex, clutterFile in enumerate(clutterFiles):
        print(str(clutterFileIndex + 1) + '/' + str(len(clutterFiles)), clutterFile + '        ', end='\r', flush=True)
        with open(os.path.join(clutterFileDirectory, clutterFile), 'r') as openFile:
            # Read JSON file
            try:
                clutterFileContents = json.loads(openFile.read())
                seed = clutterFileContents['sourceFile'].split('.')[0].split('_')[2]
                clutterFileMap[seed] = clutterFileContents
            except Exception as e:
                print('FAILED TO READ FILE: ' + str(file))
                print(e)
                continue
    print()

# Find the seeds for which all input sets have data
print('\nComputing condensed seed set')
print('Starting raw QSI seed set size:', len(loadedRawResults['QSI'].keys()))
rawSeedList = [x for x in loadedRawResults['QSI'].keys() if x in loadedRawResults['SI'].keys()]
print('Merged with SI:', len(rawSeedList))
rawSeedList = [x for x in rawSeedList if x in loadedRawResults['3DSC'].keys()]
print('Merged with 3DSC:', len(rawSeedList))
rawSeedList = [x for x in rawSeedList if x in seedList]
print('Merged with seedList:', len(rawSeedList))
rawSeedList = [x for x in rawSeedList if x in clutterFileMap.keys()]
print('Seeds remaining:', len(rawSeedList), '(after merging with clutter file map)')

# We want to make sure our dataset contains entries for ALL data points
seedList = rawSeedList

if enableResultSetSizeLimit:
    seedList = [x for index, x in enumerate(seedList) if index < resultSetSizeLimit]

print()
print('Reduced result set to size', len(seedList))
print()



# Create heatmap histograms
hist_rici = np.zeros(shape=(heatmapSize, heatmapSize), dtype=np.int64)
hist_si = np.zeros(shape=(heatmapSize, heatmapSize), dtype=np.int64)
hist_3dsc = np.zeros(shape=(heatmapSize, heatmapSize), dtype=np.int64)

print('Computing histograms..')
histogramEntryCount = 0
for rawSeedIndex, rawSeed in enumerate(rawSeedList):
    print(str(rawSeedIndex + 1) + '/' + str(len(rawSeedList)) + " processed", end='\r', flush=True)

    clutterValues = clutterFileMap[rawSeed]['clutterValues']
    riciRanks = loadedRawResults['QSI'][rawSeed]
    siRanks = loadedRawResults['SI'][rawSeed]
    scRanks = loadedRawResults['3DSC'][rawSeed]

    if not(len(clutterValues) == len(riciRanks) == len(siRanks) == len(scRanks)):
        print('WARNING: batch size mismatch at seed', rawSeed , '!', [len(clutterValues), len(riciRanks), len(siRanks), len(scRanks)])

    histogramEntryCount += len(clutterValues)

    for i in range(0, len(clutterValues)):
        clutterValue = clutterValues[i]
        index_rici = riciRanks[i]
        index_si = siRanks[i]
        index_3dsc = scRanks[i]

        # Apparently some NaN in there
        if clutterValue is None:
            continue

        xBin = int((1.0 - clutterValue) * heatmapSize)
        if xBin >= heatmapSize:
            continue

        yBin_rici = index_rici
        if yBin_rici < heatmapSize:
            hist_rici[heatmapSize - 1 - yBin_rici, xBin] += 1

        yBin_si = index_si
        if yBin_si < heatmapSize:
            hist_si[heatmapSize - 1 - yBin_si, xBin] += 1

        yBin_3dsc = index_3dsc
        if yBin_3dsc < heatmapSize:
            hist_3dsc[heatmapSize - 1 - yBin_3dsc, xBin] += 1

print('Histogram computed over', histogramEntryCount, 'values')



hist_rici = np.log10(np.maximum(hist_rici,0.1))
hist_si = np.log10(np.maximum(hist_si,0.1))
hist_3dsc = np.log10(np.maximum(hist_3dsc,0.1))

extent = [0, heatmapSize, 0, heatmapSize]

# Plot heatmap
plt.clf()

colorbar_ticks = np.arange(0, 8, 1)
total_minimum_value = min(np.amin(hist_rici), np.amin(hist_si), np.amin(hist_3dsc))
total_maximum_value = max(np.amax(hist_rici), np.amax(hist_si), np.amax(hist_3dsc))
print('range:', total_minimum_value, total_maximum_value)
normalisation = colors.Normalize(vmin=total_minimum_value,vmax=total_maximum_value)

horizontal_ticks_real_coords = np.arange(0,256,25.599*2.0)
horizontal_ticks_labels = [("%.1f" % x) for x in np.arange(0,1.1,0.2)]

riciplt = plt.figure(1)
plt.title('')
plt.ylabel('rank')
plt.xlabel('clutter percentage')
rici_im = plt.imshow(hist_rici, extent=extent, cmap='nipy_spectral', norm=normalisation)
plt.xticks(horizontal_ticks_real_coords, horizontal_ticks_labels)

scplot = plt.figure(2)
plt.title('')
plt.ylabel('rank')
plt.xlabel('clutter percentage')
scim_im = plt.imshow(hist_3dsc, extent=extent, cmap='nipy_spectral', norm=normalisation)
plt.xticks(horizontal_ticks_real_coords, horizontal_ticks_labels)

siplt = plt.figure(3)
plt.title('')
plt.ylabel('rank')
plt.xlabel('clutter percentage')
si_im = plt.imshow(hist_si, extent=extent, cmap='nipy_spectral', norm=normalisation)
plt.xticks(horizontal_ticks_real_coords, horizontal_ticks_labels)
si_cbar = plt.colorbar(si_im, ticks=colorbar_ticks)
si_cbar.ax.set_yticklabels(["{:.0E}".format(x) for x in np.power(10, colorbar_ticks)])
si_cbar.set_label('Sample count', rotation=90)

riciplt.show()
scplot.show()
siplt.show()

input()

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
experimentSheet.write(0, len(allColumns) + 2, 'RICI Count')
experimentSheet.write(0, len(allColumns) + 3, 'SI Count')
experimentSheet.write(0, len(allColumns) + 4, '3DSC Count')

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
    experimentSheet.write(directoryIndex + 1, len(allColumns) + 2, len([x for x in result['results']['QSI'] if x in seedList]))
    experimentSheet.write(directoryIndex + 1, len(allColumns) + 3, len([x for x in result['results']['SI'] if x in seedList]))
    experimentSheet.write(directoryIndex + 1, len(allColumns) + 4, len([x for x in result['results']['3DSC'] if x in seedList]))

# Sheets
top0sheetRICI = book.add_sheet("Rank 0 RICI results")
top0sheetSI = book.add_sheet("Rank 0 SI results")
top0sheet3DSC = book.add_sheet("Rank 0 3DSC results")

top10sheetRICI = book.add_sheet("Top 10 RICI results")
top10sheetSI = book.add_sheet("Top 10 SI results")
top10sheet3DSC = book.add_sheet("Top 10 3DSC results")

generationSpeedSheetRICI = book.add_sheet("RICI Generation Times")
generationSpeedSheetSI = book.add_sheet("SI Generation Times")
generationSpeedSheet3DSC = book.add_sheet("3DSC Generation Times")

comparisonSpeedSheetRICI = book.add_sheet("RICI Comparison Times")
comparisonSpeedSheetSI = book.add_sheet("SI Comparison Times")
comparisonSpeedSheet3DSC = book.add_sheet("3DSC Comparison Times")

vertexCountSheet = book.add_sheet("Reference Image Count")
totalVertexCountSheet = book.add_sheet("Total Image Count")
totalTriangleCountSheet = book.add_sheet("Total Triangle Count")

# seed column is 0, data starts at column 1
currentColumn = 1

for directoryIndex, directory in enumerate(inputDirectories.keys()):

    # Writing seed column
    if directoryIndex == 0:
        top0sheetRICI.write(0, 0, 'seed')
        top0sheetSI.write(0, 0, 'seed')
        top0sheet3DSC.write(0, 0, 'seed')

        top10sheetRICI.write(0, 0, 'seed')
        top10sheetSI.write(0, 0, 'seed')
        top10sheet3DSC.write(0, 0, 'seed')

        generationSpeedSheetRICI.write(0, 0, 'seed')
        generationSpeedSheetSI.write(0, 0, 'seed')
        generationSpeedSheet3DSC.write(0, 0, 'seed')

        comparisonSpeedSheetRICI.write(0, 0, 'seed')
        comparisonSpeedSheetSI.write(0, 0, 'seed')
        comparisonSpeedSheet3DSC.write(0, 0, 'seed')

        vertexCountSheet.write(0, 0, 'seed')
        totalVertexCountSheet.write(0, 0, 'seed')
        totalTriangleCountSheet.write(0, 0, 'seed')

        for seedIndex, seed in enumerate(seedList):
            top0sheetRICI.write(seedIndex + 1, 0, seed)
            top0sheetSI.write(seedIndex + 1, 0, seed)
            top0sheet3DSC.write(seedIndex + 1, 0, seed)

            top10sheetRICI.write(seedIndex + 1, 0, seed)
            top10sheetSI.write(seedIndex + 1, 0, seed)
            top10sheet3DSC.write(seedIndex + 1, 0, seed)

            generationSpeedSheetRICI.write(seedIndex + 1, 0, seed)
            generationSpeedSheetSI.write(seedIndex + 1, 0, seed)
            generationSpeedSheet3DSC.write(seedIndex + 1, 0, seed)

            comparisonSpeedSheetRICI.write(seedIndex + 1, 0, seed)
            comparisonSpeedSheetSI.write(seedIndex + 1, 0, seed)
            comparisonSpeedSheet3DSC.write(seedIndex + 1, 0, seed)

            vertexCountSheet.write(seedIndex + 1, 0, seed)
            totalVertexCountSheet.write(seedIndex + 1, 0, seed)
            totalTriangleCountSheet.write(seedIndex + 1, 0, seed)

    resultSet = loadedResults[directory]
    directoryName, _ = inputDirectories[directory]

    # Writing column headers
    for sampleCountIndex, sampleObjectCount in enumerate(resultSet['settings']['sampleObjectCounts']):
        columnHeader = directoryName + ' (' + str(sampleObjectCount) + ' ' + objects(
            len(resultSet['settings']['sampleObjectCounts'])) + ')'

        top0sheetRICI.write(0, currentColumn + sampleCountIndex, columnHeader)
        top0sheetSI.write(0, currentColumn + sampleCountIndex, columnHeader)
        top0sheet3DSC.write(0, currentColumn + sampleCountIndex, columnHeader)

        top10sheetRICI.write(0, currentColumn + sampleCountIndex, columnHeader)
        top10sheetSI.write(0, currentColumn + sampleCountIndex, columnHeader)
        top10sheet3DSC.write(0, currentColumn + sampleCountIndex, columnHeader)

        generationSpeedSheetRICI.write(0, currentColumn + sampleCountIndex, columnHeader)
        generationSpeedSheetSI.write(0, currentColumn + sampleCountIndex, columnHeader)
        generationSpeedSheet3DSC.write(0, currentColumn + sampleCountIndex, columnHeader)

        comparisonSpeedSheetRICI.write(0, currentColumn + sampleCountIndex, columnHeader)
        comparisonSpeedSheetSI.write(0, currentColumn + sampleCountIndex, columnHeader)
        comparisonSpeedSheet3DSC.write(0, currentColumn + sampleCountIndex, columnHeader)

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
                top0sheetRICI.write(seedIndex + 1, currentColumn + sampleCountIndex, percentageAtPlace0)

                # Top 10 performance
                totalImageCountInTop10 = sum(
                    [entry['QSIhistograms'][str(sampleCountIndex)][str(x)] for x in range(0, 10) if
                     str(x) in entry['QSIhistograms'][str(sampleCountIndex)]])
                percentInTop10 = float(totalImageCountInTop10) / float(totalImageCount)
                top10sheetRICI.write(seedIndex + 1, currentColumn + sampleCountIndex, percentInTop10)

                # generation execution time
                generationTime = entry['runtimes']['QSISampleGeneration']['total'][sampleCountIndex]
                generationSpeedSheetRICI.write(seedIndex + 1, currentColumn + sampleCountIndex, generationTime)

                # search execution time
                comparisonTime = entry['runtimes']['QSISearch']['total'][sampleCountIndex]
                comparisonSpeedSheetRICI.write(seedIndex + 1, currentColumn + sampleCountIndex, comparisonTime)

                # Vertex count sanity check
                vertexCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, entry['imageCounts'][0])
                totalVertexCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex,
                                            sum(entry['imageCounts'][0:sampleObjectCount]))
                totalTriangleCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex,
                                              sum(entry['vertexCounts'][0:sampleObjectCount]) / 3)
        else:
            for sampleCountIndex, sampleObjectCount in enumerate(resultSet['settings']['sampleObjectCounts']):
                top0sheetRICI.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                top10sheetRICI.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                generationSpeedSheetRICI.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                comparisonSpeedSheetRICI.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')

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
                generationSpeedSheetSI.write(seedIndex + 1, currentColumn + sampleCountIndex, generationTime)

                # search execution time
                comparisonTime = entry['runtimes']['SISearch']['total'][sampleCountIndex]
                comparisonSpeedSheetSI.write(seedIndex + 1, currentColumn + sampleCountIndex, comparisonTime)

                # Vertex count sanity check
                vertexCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, entry['imageCounts'][0])
                totalVertexCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex,
                                            sum(entry['imageCounts'][0:sampleObjectCount]))
                totalTriangleCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex,
                                              sum(entry['vertexCounts'][0:sampleObjectCount]) / 3)
        else:
            for sampleCountIndex, sampleObjectCount in enumerate(resultSet['settings']['sampleObjectCounts']):
                top0sheetSI.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                top10sheetSI.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                generationSpeedSheetSI.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                comparisonSpeedSheetSI.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')

        if seed in resultSet['results']['3DSC']:
            for sampleCountIndex, sampleObjectCount in enumerate(resultSet['settings']['sampleObjectCounts']):
                # Top 1 performance
                entry = resultSet['results']['3DSC'][seed]
                totalImageCount = entry['imageCounts'][0]
                experimentIterationCount = len(resultSet['settings']['sampleObjectCounts'])
                percentageAtPlace0 = 0
                if '0' in entry['3DSChistograms'][str(sampleCountIndex)]:
                    percentageAtPlace0 = float(entry['3DSChistograms'][str(sampleCountIndex)]['0']) / float(totalImageCount)
                top0sheet3DSC.write(seedIndex + 1, currentColumn + sampleCountIndex, percentageAtPlace0)

                # Top 10 performance
                totalImageCountInTop10 = sum(
                    [entry['3DSChistograms'][str(sampleCountIndex)][str(x)] for x in range(0, 10) if
                     str(x) in entry['3DSChistograms'][str(sampleCountIndex)]])
                percentInTop10 = float(totalImageCountInTop10) / float(totalImageCount)
                top10sheet3DSC.write(seedIndex + 1, currentColumn + sampleCountIndex, percentInTop10)

                # generation execution time
                generationTime = entry['runtimes']['3DSCSampleGeneration']['total'][sampleCountIndex]
                generationSpeedSheet3DSC.write(seedIndex + 1, currentColumn + sampleCountIndex, generationTime)

                # search execution time
                comparisonTime = entry['runtimes']['3DSCSearch']['total'][sampleCountIndex]
                comparisonSpeedSheet3DSC.write(seedIndex + 1, currentColumn + sampleCountIndex, comparisonTime)

                # Vertex count sanity check
                vertexCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex, entry['imageCounts'][0])
                totalVertexCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex,
                                            sum(entry['imageCounts'][0:sampleObjectCount]))
                totalTriangleCountSheet.write(seedIndex + 1, currentColumn + sampleCountIndex,
                                              sum(entry['vertexCounts'][0:sampleObjectCount]) / 3)
        else:
            for sampleCountIndex, sampleObjectCount in enumerate(resultSet['settings']['sampleObjectCounts']):
                top0sheet3DSC.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                top10sheet3DSC.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                generationSpeedSheet3DSC.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')
                comparisonSpeedSheet3DSC.write(seedIndex + 1, currentColumn + sampleCountIndex, ' ')

    # Moving on to the next column
    currentColumn += len(resultSet['settings']['sampleObjectCounts'])

# beauty addition.. Cuts off final column
for seedIndex, seed in enumerate(seedList + ['dummy entry for final row']):
    top0sheetRICI.write(seedIndex, currentColumn, ' ')
    top10sheetRICI.write(seedIndex, currentColumn, ' ')
    generationSpeedSheetRICI.write(seedIndex, currentColumn, ' ')
    comparisonSpeedSheetRICI.write(seedIndex, currentColumn, ' ')

    top0sheetSI.write(seedIndex, currentColumn, ' ')
    top10sheetSI.write(seedIndex, currentColumn, ' ')
    generationSpeedSheetSI.write(seedIndex, currentColumn, ' ')
    comparisonSpeedSheetSI.write(seedIndex, currentColumn, ' ')

    vertexCountSheet.write(seedIndex, currentColumn, ' ')
    totalVertexCountSheet.write(seedIndex, currentColumn, ' ')
    totalTriangleCountSheet.write(seedIndex, currentColumn, ' ')

book.save(outfile)
print('Complete.')
