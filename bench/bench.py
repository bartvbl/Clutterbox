import time
import subprocess
from math import floor

iterationCount = 0
crashedFiles = []
startTime = time.time()

while True:
	print("ITERATION", iterationCount, " (" + str(floor(time.time() - startTime)) + " seconds,", len(crashedFiles), "crashes)")
	iterationCount += 1
	output = subprocess.run(['../cmake-build-release/qsiverification --source-directory="/home/bart/Datasets/SHREC2017" --box-size=3 --sample-set-size=5 --spin-image-width=0.3'],
		shell=True, check=False)
	if output.returncode != 0:
		print("PROGRAM CRASHED!")
		crashedFiles.append(str(floor(time.time() - startTime)))