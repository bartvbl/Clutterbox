import os
import random
from shutil import copyfile

count = 1000
indir  = '/media/ntfsHOME/Datasets/SHREC17'
outdir = '/media/ntfsHOME/git/quasispinimageverification/bench/batches'

files = os.listdir(indir)
print('Found', len(files), 'files.')

for i in range(0, count):
	os.mkdir(outdir + '/' + str(i))
	chosen = random.sample(files, 5)
	for file in chosen:
		copyfile(indir + '/' + file, outdir + '/' + str(i) + '/' + file)
		print(indir + '/' + file, '->', outdir + '/' + str(i) + '/' + file)
