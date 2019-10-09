import time
from threading import Thread
import threading
import os, os.path
import subprocess
import sys

from prompt_toolkit import prompt
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter

print("-- Server Manager v0.1 --")

seedfile = "seeds_idun_qsifix_v4_failedseeds.txt"
outputDir = '../combinedoutput/output_qsifix_v4_lotsofobjects_idun_failed/'

 
DAEMONONLY_currentSeedIndex = 0
DAEMONONLY_seedCount = sum(1 for line in open('bench/' + seedfile))
DAEMONONLY_activeseeds = []
DAEMONONLY_gpuIDs = []

SHARED_activegpus = []
SHARED_serverlog = []
SHARED_barrier = threading.Condition()
SHARED_lock = threading.Lock()
SHARED_jobqueue = []

print('Loaded', DAEMONONLY_seedCount, 'seeds.')

def log(message):
	SHARED_serverlog.append(message)


def launchInstance(gpuID):
	global seedfile
	global DAEMONONLY_currentSeedIndex
	global DAEMONONLY_activeseeds
	global DAEMONONLY_gpuIDs

	gpuIndex = DAEMONONLY_gpuIDs.index(gpuID)
	# Don't launch if we have reached the end
	if DAEMONONLY_currentSeedIndex >= DAEMONONLY_seedCount:
		SHARED_activegpus[gpuIndex] = False
		return

	log("Launching job with ID " + str(DAEMONONLY_currentSeedIndex) + ' on GPU ' + str(gpuID))

	cmd = subprocess.run(['nvidia-docker run -ti -d -v "/home/bartiver/SHREC17:/home/bartiver/SHREC17" --rm --device /dev/nvidia3:/dev/nvidia0 bartiver_qsiverification:latest ' + str(gpuID) + " " + str(DAEMONONLY_currentSeedIndex) + " " + str(DAEMONONLY_currentSeedIndex+1) + ' "' + seedfile + '" ' + '"--source-directory=/home/bartiver/SHREC17/ --object-counts=5 --descriptors=qsi --box-size=1 --spin-image-support-angle-degrees=180 --spin-image-width=0.3 --dump-raw-search-results --override-total-object-count=10"'], shell=True, stdout=subprocess.PIPE)
	subprocess.run(['docker rename ' + cmd.stdout.decode('ascii').strip() + ' bartiver_qsiverification_gpu' + str(gpuID)], shell=True)

	DAEMONONLY_activeseeds[gpuIndex] = DAEMONONLY_currentSeedIndex
	DAEMONONLY_currentSeedIndex += 1



def checkInstance(gpuID):
	global outputDir

	result = subprocess.run(['docker ps --filter="name=bartiver_qsiverification_gpu' + str(gpuID) + '"'], shell=True, stdout=subprocess.PIPE)
	containerHasCrashed = len(result.stdout.decode('utf-8').split('\n')) > 3
	if containerHasCrashed:
		log('Job running on GPU ' + str(gpuID) + ' has crashed!')
		return containerHasCrashed

	beforeCount = len([name for name in os.listdir(outputDir + 'output/')])
	subprocess.run(['docker cp bartiver_qsiverification_gpu' + str(gpuID) + ':/qsiverification/output/ ' + outputDir], shell=True)
	afterCount = len([name for name in os.listdir(outputDir + 'output/')])
	return afterCount > beforeCount

def stopInstance(gpuID):
	subprocess.run(['docker stop bartiver_qsiverification_gpu' + str(gpuID)], shell=True, stdout=subprocess.PIPE)






print("Starting daemon thread..")



def wakeDaemon():
	SHARED_barrier.acquire()
	SHARED_barrier.notify_all()
	SHARED_barrier.release()

def pushDaemonJob(item):
	SHARED_lock.acquire()
	SHARED_jobqueue.append(item)
	SHARED_lock.release()
	wakeDaemon()

def popDaemonJob():
	SHARED_lock.acquire()
	if len(SHARED_jobqueue) == 0:
		item = None
	else:
		item = SHARED_jobqueue[0]
		del SHARED_jobqueue[0]
	SHARED_lock.release()
	return item

def gpuDaemon():
	global SHARED_activegpus
	global SHARED_gpuIDs
	global DAEMONONLY_activeseeds

	while True:

		SHARED_barrier.acquire()
		SHARED_barrier.wait(10)
		job = popDaemonJob()
 
		while job is not None:
			if job['command'] == "stop":
				return
			elif job['command'] == 'add':
				gpuID = job['id']
				SHARED_activegpus.append(True)
				DAEMONONLY_gpuIDs.append(gpuID)
				DAEMONONLY_activeseeds.append(-1)
				launchInstance(gpuID)
			elif job['command'] == 'remove':
				gpuID = job['id']
				stopInstance(gpuID)
				gpuIndex = DAEMONONLY_gpuIDs.index(gpuID)
				processingSeed = DAEMONONLY_activeseeds[gpuIndex]
				log("Processing seed index " + str(processingSeed) + " on GPU " + str(gpuIndex) + " failed!")
				del DAEMONONLY_gpuIDs[gpuIndex]
				del DAEMONONLY_activeseeds[gpuIndex]
				del SHARED_activegpus[gpuIndex]

			job = popDaemonJob()

		# do the rounds
		for gpuID in DAEMONONLY_gpuIDs:
			if checkInstance(gpuID):
				stopInstance(gpuID)
				launchInstance(gpuID)


		SHARED_barrier.release()

daemonThread = Thread(None, target=gpuDaemon, name="daemon")
daemonThread.start()

commandCompleter = WordCompleter(['help', 'quit', 'status', 'start', 'list', 'until', 'stop'])

def bottom_toolbar_status():
	return "%i/%i GPU's running, %i/%i batch items complete" % (sum(SHARED_activegpus), len(SHARED_activegpus), DAEMONONLY_currentSeedIndex, DAEMONONLY_seedCount)

def printStatus():
	print('\n'.join(SHARED_serverlog))

try:
	if __name__ == '__main__':
		session = PromptSession(
			history=FileHistory('./history.txt'),
			auto_suggest=AutoSuggestFromHistory(),
			enable_history_search=True,
			completer=commandCompleter,
			bottom_toolbar=bottom_toolbar_status)
		print("Ready.")

		command = ""
		while command != "quit":
			command = session.prompt("servermanager: ")
			if command == "help":
				print("    help: Show this help message")
				print("    quit: Exit the server manager")
				print("    status: Show progress towards current queue")
				print("    start [gpu ID]: Add GPU with [gpu ID] to the pool of available cards")
				print("    list: List GPU's currently in pool")
				print("    until [gpu ID] [time in minutes]: Stop [gpu ID] [time in minutes] from now")
				print("    stop [gpu ID]: Remove GPU with [gpu ID] from the pool of available cards.")

			elif command == "status":
				printStatus()

			elif command.startswith("start"): #duh
				try:
					gpuID = int(command.split(" ")[1])
					print("Adding GPU with ID", gpuID, "to the pool..")
					pushDaemonJob({'command': "add", 'id': gpuID})
				except:
					print("Invalid GPU ID. Put a number after the 'start' command.")

			elif command.startswith("stop"):
				try:
					gpuID = int(command.split(" ")[1])
					print("Removing GPU with ID", gpuID, "from the pool..")
					pushDaemonJob({'command': "remove", 'id': gpuID})
				except:
					print("Invalid GPU ID. Put a number after the 'stop' command.")

			elif command == "list":
				for i in activegpus:
					print("GPU with ID %i" % i)

			elif command == "quit":
				response = session.prompt("Write \'quit\' again to exit. THIS STOPS ALL GPUs!: ")
				if response != "quit":
					command = "lol"


	# Start shutdown sequence
except KeyboardInterrupt:
	pass
finally:
	print("Exiting..")
	pushDaemonJob({'command': "stop"})



	
