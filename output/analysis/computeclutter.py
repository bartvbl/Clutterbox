import os
from datetime import datetime
from os import listdir
from os.path import isfile, join
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import re
from scipy import stats
from PIL import Image

# COMPUTED CLUTTER IS BASED ON 10 OBJECT SCENES!!!
qsi_metafiles_directory = "../COMBINED/qsi_primary"
rici_rawfiles_directory = rici_metafiles_directory + "/raw"
si_metafiles_directory = "../COMBINED/si_primary"
si_rawfiles_directory = si_metafiles_directory + "/raw"
clutterfiles_directory = "../clutter/lotsofobjects"

#metafiles_directory = "/media/ntfsHOME/git/quasispinimageverification/output/HEIDRUNS/output_majorfix_v1/output"
#rawfiles_directory = "/media/ntfsHOME/git/quasispinimageverification/output/HEIDRUNS/output_majorfix_v1/output/raw"
#clutterfiles_directory = "/media/ntfsHOME/git/quasispinimageverification/output/clutter/base"


outFile = 'clutter_out.csv'

use_cached_results = False
size = 256

def readJsonFile(file):
	with open(file, 'r') as openFile:
		return json.loads(openFile.read())

def writeCacheFile(file, matrix):
	with open(file, 'w') as outFile:
		matrix.tofile(outFile)

def readCacheFile(file):
	with open(file, 'rb') as inFile:
		rawMatrix = np.fromfile(inFile, dtype=np.int64)
		return rawMatrix.reshape((size,size))

hist_rici = np.zeros(shape=(size,size), dtype=np.int64)
hist_si  = np.zeros(shape=(size,size), dtype=np.int64)

if not use_cached_results:
	rici_rawfiles = [f for f in listdir(rici_rawfiles_directory) if isfile(join(rici_rawfiles_directory, f))]
	rici_metafiles = [f for f in listdir(rici_metafiles_directory) if isfile(join(rici_metafiles_directory, f))]
	rici_seeds = [x.split("_")[2] for x in rici_metafiles]

	si_rawfiles = [f for f in listdir(si_rawfiles_directory) if isfile(join(si_rawfiles_directory, f))]
	si_metafiles = [f for f in listdir(si_metafiles_directory) if isfile(join(si_metafiles_directory, f))]
	si_seeds = [x.split("_")[2] for x in si_metafiles]

	# Find out which seeds are present in both result sets
	combined_seeds = [x for x in rici_seeds if x in si_seeds]

	clutterfiles = [f for f in listdir(clutterfiles_directory) if isfile(join(clutterfiles_directory, f))]

	print("Found", len(rici_metafiles), " RICI result files.")
	print("Found", len(si_metafiles), " SI result files.")
	print("Found", len(clutterfiles), "clutter files.")
	print("Found", len(combined_seeds), " seeds.")

	clutterfilesContent = {}
	rici_metafilesContent = {}
	rici_rawfilesContent = {}
	si_metafilesContent = {}
	si_rawfilesContent = {}

	for index, clutterfile in enumerate(clutterfiles):
		print('\rLoading clutter files.. (' + str(index+1) + '/' + str(len(clutterfiles)) + ')', end="", flush=True)
		clutterfile_contents = readJsonFile(os.path.join(clutterfiles_directory, clutterfile))
		# hacky, but works for my purposes
		seed = clutterfile_contents['sourceFile'].split("_")[2]
		clutterfilesContent[seed] = clutterfile_contents
	print()

	for fileIndex, rici_metafile in enumerate(rici_metafiles):
		print('\rLoading RICI meta files.. (' + str(fileIndex+1) + '/' + str(len(rici_metafiles)) + ')', end="", flush=True)
		seed = rici_metafile.split("_")[2]
		rici_metafilesContent[seed] = readJsonFile(os.path.join(rici_metafiles_directory, rici_metafile))
	print()

	for fileIndex, si_metafile in enumerate(si_metafiles):
		print('\rLoading SI meta files.. (' + str(fileIndex+1) + '/' + str(len(si_metafiles)) + ')', end="", flush=True)
		seed = si_metafile.split("_")[2]
		si_metafilesContent[seed] = readJsonFile(os.path.join(si_metafiles_directory, si_metafile))
	print()

	for fileIndex, rici_rawfile in enumerate(rici_rawfiles):
		print('\rLoading RICI raw files.. (' + str(fileIndex+1) + '/' + str(len(rici_rawfiles)) + ')', end="", flush=True)
		seed = rici_rawfile.split("_")[2]
		rici_rawfilesContent[seed] = readJsonFile(os.path.join(rici_rawfiles_directory, rici_rawfile))
	print()

	for fileIndex, si_rawfile in enumerate(si_rawfiles):
		print('\rLoading SI raw files.. (' + str(fileIndex+1) + '/' + str(len(si_rawfiles)) + ')', end="", flush=True)
		seed = si_rawfile.split("_")[2]
		si_rawfilesContent[seed] = readJsonFile(os.path.join(si_rawfiles_directory, si_rawfile))
	print()

	clutterValues = []
	indices_rici = []
	indices_si = []
	for index, filetuple in enumerate(corresponding_files):
		print('\rBuilding results.. (' + str(index+1) + '/' + str(len(corresponding_files)) + ')', end="", flush=True)
		
		metafile_contents = metafilesContent[filetuple[0]]
		rawfile_contents = rawfilesContent[filetuple[1]]
		clutterfile_contents = clutterfilesContent[filetuple[2]]

		assert(len(clutterfile_contents['clutterValues']) == len(rawfile_contents['QSI']['5']))
		assert(len(clutterfile_contents['clutterValues']) == len(rawfile_contents['SI']['5']))

		clutterValues += clutterfile_contents['clutterValues']
		indices_qsi += rawfile_contents['QSI']['5']
		indices_si += rawfile_contents['SI']['5']

	print()
	print('Building heatmap.. (' + str(len(indices_rici)) + ' & ' + str(len(indices_si)) + ' values)')

	#for i in range(0, len(indices)):
	#	if indices[i] > 0:
	#		indices[i] = math.log(indices[i], 10)

	#with open(outFile, 'w') as outFile:

	print('Max RICI index:', max(indices_rici))
	print('Max SI index:', max(indices_si))

	hist_indices_rici = np.zeros(shape=(size), dtype=np.int64)
	hist_indices_si  = np.zeros(shape=(size), dtype=np.int64)
	hist_counts = np.zeros(shape=(size), dtype=np.int64)
	
	step = 1.0 / float(size)

	assert(len(indices_rici) == len(indices_si))

	for i in range(0, len(indices_rici)):
		clutterValue = clutterValues[i]
		index_rici = indices_rici[i]
		index_si = indices_si[i]

		# Apparently some NaN in there
		if clutterValue is None:
			continue

		xBin = int((1.0 - clutterValue) * size)
		if xBin >= size:
			continue

		#hist_indices_rici[xBin] += index_rici
		#hist_indices_si[xBin] += index_si
		#hist_counts[xBin] += 1

		yBin_rici = index_rici
		if yBin_rici < size:
			hist_rici[size - 1 - yBin_rici,xBin] += 1

		yBin_si = index_si
		if yBin_si < size:
			hist_si[size - 1 - yBin_si,xBin] += 1

	#print(hist_indices_rici)
	#print(hist_indices_si)
	#print(hist_counts)

	print('Writing cache files..')

	writeCacheFile('hist_rici_cache.txt', hist_rici)
	writeCacheFile('hist_si_cache.txt', hist_si)


else:
	print('Reading cache files..')
	hist_rici = readCacheFile('hist_rici_cache.txt')
	hist_si = readCacheFile('hist_si_cache.txt')

hist_rici = np.log10(np.maximum(hist_rici,0.1))
hist_si = np.log10(np.maximum(hist_si,0.1))



# Create heatmap
#heatmap, xedges, yedges = np.histogram2d(clutterValues, indices, bins=(256,256), range=[[0, 1], [0, 256]])
extent = [0, size, 0, size]

#np.savetxt('hist.txt',hist,delimiter='\n')


# Plot heatmap
plt.clf()	

colorbar_ticks = np.arange(0, 8, 1)
total_minimum_value = min(np.amin(hist_rici), np.amin(hist_si))
total_maximum_value = max(np.amax(hist_rici), np.amax(hist_si))
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
#rici_cbar = plt.colorbar(rici_im, orientation='horizontal', ticks=colorbar_ticks)
#rici_cbar.ax.set_xticklabels(["{:.0E}".format(x) for x in np.power(10, colorbar_ticks)])
#rici_cbar.set_label('Sample count', rotation=0)

siplt = plt.figure(2)
plt.title('')
plt.ylabel('rank')
plt.xlabel('clutter percentage')
si_im = plt.imshow(hist_si, extent=extent, cmap='nipy_spectral', norm=normalisation)
plt.xticks(horizontal_ticks_real_coords, horizontal_ticks_labels)
si_cbar = plt.colorbar(si_im, ticks=colorbar_ticks)
si_cbar.ax.set_yticklabels(["{:.0E}".format(x) for x in np.power(10, colorbar_ticks)])
si_cbar.set_label('Sample count', rotation=90)

riciplt.show()
siplt.show()

input()

def dumpHistogram(histogram, file):
	hist_double = histogram.astype(np.double)
	hist_normalised = np.log10(hist_double)
	hist_normalised = hist_normalised / np.max(hist_normalised)
	hist_colour = hist_normalised * 255.0
	im = Image.fromarray(hist_colour.astype(np.ubyte), 'L')
	im.save(file, "PNG")

dumpHistogram(hist_rici, 'rici_out.png')
dumpHistogram(hist_si, 'si_out.png')

print('Complete.')