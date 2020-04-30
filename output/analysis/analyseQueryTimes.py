import os
import os.path
import json

def loadOutputFileDirectory(path):
    originalFiles = os.listdir(path)
    results = []
    for fileindex, file in enumerate(originalFiles):
        print(str(fileindex+1) + '/' + str(len(originalFiles)), file)
        if(file == 'raw'):
        	continue
        with open(os.path.join(path, file), 'r') as openFile:
            fileContents = json.loads(openFile.read())
            results.append(fileContents)
    return results

results = loadOutputFileDirectory('../HEIDRUNS/QUICCIndex_results')

counts = [0] * 4096
sums = [0] * 4096

with open('queryTimes.csv', 'w') as outFile:
    outFile.write('Index, End time, Counts\n')
    for i in range(0, len(results)):
        for row in range(0, 4096):
            value = results[i]['indexedQueryResults']['distanceTimes'][row]
            if value != -1:
                counts[row] += 1
                sums[row] += value
    for i in range(0, 4096):
        if counts[i] == 0:
            continue
        outFile.write(str(i) + ', ' + str(float(sums[i]) / float(counts[i])) + ', ' + str(counts[i]) + '\n')