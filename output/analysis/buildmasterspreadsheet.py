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
    'SI, 180 support angle, 5 objects',
    'SI, 60 support angle, '

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
    if 'overrideObjectCount' in loadedJson:
        settings['overrideObjectCount'] = loadedJson['overrideObjectCount']
    if 'descriptors' in loadedJson:
        settings['descriptors'] = loadedJson['descriptors']
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
            results['results'][fileContents['seed']] = fileContents
            previousExperimentSettings = currentExperimentSettings
    print()

    results['settings'] = previousExperimentSettings
    pp = pprint.PrettyPrinter(indent=4)
    print(pp.pformat(results['settings']))

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

# -- Dump to spreadsheet --

print('Dumping spreadsheet..')

book = xlwt.Workbook(encoding="utf-8")

# 1. Create data page for dataset settings table
experimentSheet = book.add_sheet("Experiment Overview")



# 2. Create data page for mapping of seed -> % at rank 0
top0sheet = book.add_sheet("Rank 0 results")

# 3. Create data page for mapping of seed -> % at rank 0 - 9
top10sheet = book.add_sheet("Top 10 results")


print('Complete.')