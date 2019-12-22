#include <arrrgh.hpp>
#include "cuda.h"
#include "cuda_runtime.h"
#include "nvidia/helper_cuda.h"

#include "experiments/clutterBoxExperiment.hpp"


#include <stdexcept>

cudaDeviceProp setCurrentCUDADevice(bool listOnly, int forceGPU);

const float DEFAULT_SPIN_IMAGE_WIDTH = 1;
const float DEFAULT_SPIN_IMAGE_SUPPORT_ANGLE_DEGREES = 90;

void splitByCharacter(std::vector<std::string>* parts, const std::string &s, char delim) {

    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        parts->push_back(item);
    }
}

int main(int argc, const char **argv)
{
	arrrgh::parser parser("riciverification", "Generates and compares both radial intersection count and spin images on the GPU");
	const auto& showHelp = parser.add<bool>("help", "Show this help message.", 'h', arrrgh::Optional, false);
	const auto& listGPUs = parser.add<bool>("list-gpus", "List all GPU's, used for the --force-gpu parameter.", 'a', arrrgh::Optional, false);
	const auto& forceGPU = parser.add<int>("force-gpu", "Force using the GPU with the given ID", 'b', arrrgh::Optional, -1);
	const auto& boxSize = parser.add<float>("box-size", "Size of the cube box for the clutter box experiment", '\0', arrrgh::Optional, 1);
	const auto& objectDirectory = parser.add<std::string>("source-directory", "Defines the directory from which input objects are read", '\0', arrrgh::Optional, "");
	const auto& spinImageWidth = parser.add<float>("spin-image-width", "The size of the spin image plane in 3D object space", '\0', arrrgh::Optional, DEFAULT_SPIN_IMAGE_WIDTH);
	const auto& spinImageSupportAngle = parser.add<float>("spin-image-support-angle-degrees", "The support angle to use for filtering spin image point samples", '\0', arrrgh::Optional, DEFAULT_SPIN_IMAGE_SUPPORT_ANGLE_DEGREES);
    const auto& forcedSeed = parser.add<std::string>("force-seed", "Specify the seed to use for random generation. Used for reproducing results.", '\0', arrrgh::Optional, "0");
	const auto& dumpRawResults = parser.add<bool>("dump-raw-search-results", "Enable dumping of raw search result index values", '\0', arrrgh::Optional, false);
	const auto& outputDirectory = parser.add<std::string>("output-directory", "Specify the location where output files should be dumped", '\0', arrrgh::Optional, "../output/");
    const auto& objectCounts = parser.add<std::string>("object-counts", "Specify the number of objects the experiment should be performed with, as a comma separated list WITHOUT spaces (e.g. --object-counts=1,2,5)", '\0', arrrgh::Optional, "NONE");
    const auto& overrideObjectCount = parser.add<int>("override-total-object-count", "If you want a specified number of objects to be used for the experiment (for ensuring consistency between seeds)", '\0', arrrgh::Optional, -1);
    const auto& descriptors = parser.add<std::string>("descriptors", "Specify the descriptors that should be used in the experiment, with as valid options \"rici\", \"si\", \"quicci\", and \"all\", as a comma separated list WITHOUT spaces (e.g. --object-counts=rici,si). Use value \"all\" for using all supported descriptors", '\0', arrrgh::Optional, "all");

	try
	{
		parser.parse(argc, argv);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error parsing arguments: " << e.what() << std::endl;
		parser.show_usage(std::cerr);
		exit(1);
	}

	// Show help if desired
	if(showHelp.value())
	{
		return 0;
	}

	// First, we create a CUDA context on the best compute device.
	// This is naturally the device with most memory available, becausewhywouldntit.
	
	cudaDeviceProp device_information = setCurrentCUDADevice(listGPUs.value(), forceGPU.value());
	GPUMetaData gpuMetaData;
	gpuMetaData.name = std::string(device_information.name);
	gpuMetaData.clockRate = device_information.clockRate;
	gpuMetaData.memorySizeMB = device_information.totalGlobalMem / (1024 * 1024);

	if(listGPUs.value()) {
		return 0;
	}

	if(objectCounts.value() == "NONE") {
		std::cout << "Experiment requires the --object-counts parameter to be set" << std::endl;
		exit(0);
	}

	if(objectDirectory.value().empty()) {
		std::cout << "Experiment requires the --source-directory parameter to be set" << std::endl;
		exit(0);
	}

	// Interpret seed value
    std::stringstream sstr(forcedSeed.value());
    size_t randomSeed;
    sstr >> randomSeed;
    if(randomSeed != 0) {
        std::cout << "Using overridden seed: " << randomSeed << std::endl;
    }

    // Interpret the object counts string
    std::vector<std::string> objectCountParts;
	splitByCharacter(&objectCountParts, objectCounts.value(), ',');
    std::vector<int> objectCountList;
    for (const auto &objectCountPart : objectCountParts) {
        objectCountList.push_back(std::stoi(objectCountPart));
    }

    // Interpret the descriptor list string
    std::vector<std::string> descriptorListParts;
    splitByCharacter(&descriptorListParts, descriptors.value(), ',');
    std::vector<std::string> descriptorList;
    bool containsAll = false;
    for (const auto &descriptorPart : descriptorListParts) {
        if(descriptorPart == "all") {
            containsAll = true;
        } else if(descriptorPart != "rici" && descriptorPart != "si" && descriptorPart != "quicci") {
            std::cout << "Error: Unknown descriptor name detected: \"" + descriptorPart + "\". Ignoring." << std::endl;
        } else {
            descriptorList.push_back(descriptorPart);
        }
    }
    if(containsAll /*|| descriptorList.size() == 0 feature, not a bug*/) {
        descriptorList = {"rici", "si", "quicci"};
    }

    if(std::find(descriptorList.begin(), descriptorList.end(), "quicci") != descriptorList.end() &&
       std::find(descriptorList.begin(), descriptorList.end(), "rici") == descriptorList.end()) {
        std::cout << "WARNING: RICI descriptor was not found in list of descriptors, even though QUICCI descriptor was enabled." << std::endl;
        std::cout << "Both must be enabled for QUICCI to work, and therefore RICI is automatically added to the descriptor list." << std::endl;
        descriptorList.emplace_back("rici");
    }

    std::sort(objectCountList.begin(), objectCountList.end());

	runClutterBoxExperiment(objectDirectory.value(), descriptorList, objectCountList, overrideObjectCount.value(), boxSize.value(), spinImageWidth.value(), spinImageSupportAngle.value(), dumpRawResults.value(), outputDirectory.value(), gpuMetaData, randomSeed);

    return 0;
}

cudaDeviceProp setCurrentCUDADevice(bool listOnly, int forceGPU)
{
	int deviceCount;
	checkCudaErrors(cudaGetDeviceCount(&deviceCount));

	if(listOnly) {	
		std::cout << "Found " << deviceCount << " devices:" << std::endl;
	}

	size_t maxAvailableMemory = 0;
	cudaDeviceProp deviceWithMostMemory;
	int chosenDeviceIndex = 0;
	
	for(int i = 0; i < deviceCount; i++)
	{
		cudaDeviceProp deviceProperties;
		checkCudaErrors(cudaGetDeviceProperties(&deviceProperties, i));

		if(listOnly) {	
			std::cout << "\t- " << deviceProperties.name << " (ID " << i << ")" << std::endl;
		}

		if(deviceProperties.totalGlobalMem > maxAvailableMemory)
		{
			maxAvailableMemory = deviceProperties.totalGlobalMem;
			deviceWithMostMemory = deviceProperties;
			chosenDeviceIndex = i;
		}
	}

	if(listOnly) {
		return deviceWithMostMemory;
	}

	if(forceGPU != -1) {
		chosenDeviceIndex = forceGPU;
	}

	checkCudaErrors(cudaGetDeviceProperties(&deviceWithMostMemory, chosenDeviceIndex));

	checkCudaErrors(cudaSetDevice(chosenDeviceIndex));
	std::cout << "Chose " << deviceWithMostMemory.name << " as main device." << std::endl;
#if PRINT_GPU_PROPERTIES

	std::cout << "This device supports CUDA Compute Capability v" << deviceWithMostMemory.major << "." << deviceWithMostMemory.minor << "." << std::endl;
	std::cout << std::endl;
	std::cout << "Other device info:" << std::endl;
	std::cout << "\t- Total global memory: " << deviceWithMostMemory.totalGlobalMem << std::endl;
	std::cout << "\t- Clock rate (KHz): " << deviceWithMostMemory.clockRate << std::endl;
	std::cout << "\t- Number of concurrent kernels: " << deviceWithMostMemory.concurrentKernels << std::endl;
	std::cout << "\t- Max grid size: (" << deviceWithMostMemory.maxGridSize[0] << ", " << deviceWithMostMemory.maxGridSize[1] << ", " << deviceWithMostMemory.maxGridSize[2] << ")" << std::endl;
	std::cout << "\t- Max threads per block dimension: (" << deviceWithMostMemory.maxThreadsDim[0] << ", " << deviceWithMostMemory.maxThreadsDim[1] << ", " << deviceWithMostMemory.maxThreadsDim[2] << ")" << std::endl;
	std::cout << "\t- Max threads per block: " << deviceWithMostMemory.maxThreadsPerBlock << std::endl;
	std::cout << "\t- Max threads per multiprocessor: " << deviceWithMostMemory.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "\t- Number of multiprocessors: " << deviceWithMostMemory.multiProcessorCount << std::endl;
	std::cout << "\t- Number of registers per block: " << deviceWithMostMemory.regsPerBlock << std::endl;
	std::cout << "\t- Number of registers per multiprocessor: " << deviceWithMostMemory.regsPerMultiprocessor << std::endl;
	std::cout << "\t- Total constant memory: " << deviceWithMostMemory.totalConstMem << std::endl;
	std::cout << "\t- Warp size measured in threads: " << deviceWithMostMemory.warpSize << std::endl;
	std::cout << "\t- Single to double precision performance ratio: " << deviceWithMostMemory.singleToDoublePrecisionPerfRatio << std::endl;
	std::cout << "\t- Shared memory per block: " << deviceWithMostMemory.sharedMemPerBlock << std::endl;
	std::cout << "\t- Shared memory per multiprocessor: " << deviceWithMostMemory.sharedMemPerMultiprocessor << std::endl;
	std::cout << "\t- L2 Cache size: " << deviceWithMostMemory.l2CacheSize << std::endl;
	std::cout << std::endl;
#endif

	return deviceWithMostMemory;
}