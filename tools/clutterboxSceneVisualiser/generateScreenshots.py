seedFilePath = "seeds_lotsofobjects.txt"
outputDir = '../../output/sceneOBJs/'

with open(seedfile) as seedFile:
allseeds = [line.strip() for line in ]

subprocess.run(['./riciverification --source-directory="/home/bart/Datasets/SHREC2017/" --box-size=1 --object-counts=1,5,10 --support-radius=0.3 --3dsc-min-support-radius=0.048 --3dsc-point-density-radius=0.096 --spin-image-support-angle-degrees=180 --dump-raw-search-results --override-total-object-count=10 --descriptors=none --scene-obj-file-dump-directory="../output/sceneOBJs/" --force-seed=2485274722'], shell=True)