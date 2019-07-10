#include "arrrgh.hpp"
#include "cuda.h"
#include "cuda_runtime.h"
#include "nvidia/helper_cuda.h"

#include "experiments/clutterBoxExperiment.hpp"


#include <stdexcept>

cudaDeviceProp setCurrentCUDADevice(bool listOnly, int forceGPU);

const float DEFAULT_SPIN_IMAGE_WIDTH = 1;
const float DEFAULT_SPIN_IMAGE_SUPPORT_ANGLE_DEGREES = 90;

int main(int argc, const char **argv)
{
	arrrgh::parser parser("qsiverification", "Generates and compares spin images on the GPU");
	const auto& showHelp = parser.add<bool>("help", "Show this help message.", 'h', arrrgh::Optional, false);
	const auto& forceSpinImageSize = parser.add<float>("force-spin-image-size", "Rather than automatically selecting the spin image size based upon a specific voxel count, force the image to use a specific width.", '\0', arrrgh::Optional, 0);
	const auto& listGPUs = parser.add<bool>("list-gpus", "List all GPU's, used for the --force-gpu parameter.", 'a', arrrgh::Optional, false);
	const auto& forceGPU = parser.add<int>("force-gpu", "Force using the GPU with the given ID", 'b', arrrgh::Optional, -1);
	const auto& sampleSetSize = parser.add<int>("sample-set-size", "How many sample models the clutter box experiment should use", '\0', arrrgh::Optional, -1);
	const auto& boxSize = parser.add<int>("box-size", "Size of the cube box for the clutter box experiment", '\0', arrrgh::Optional, -1);
	const auto& objectDirectory = parser.add<std::string>("source-directory", "Defines the directory from which input objects are read", '\0', arrrgh::Optional, "");
	const auto& spinImageWidth = parser.add<float>("spin-image-width", "The size of the spin image plane in 3D object space", '\0', arrrgh::Optional, DEFAULT_SPIN_IMAGE_WIDTH);
	const auto& spinImageSupportAngle = parser.add<float>("spin-image-support-angle-degrees", "The support angle to use for filtering spin image point samples", '\0', arrrgh::Optional, DEFAULT_SPIN_IMAGE_SUPPORT_ANGLE_DEGREES);
    const auto& forcedSeed = parser.add<std::string>("force-seed", "Specify the seed to use for random generation. Used for reproducing results.", '\0', arrrgh::Optional, "0");
	const auto& dumpRawResults = parser.add<bool>("dump-raw-search-results", "Enable dumping of raw search result index values", '\0', arrrgh::Optional, false);
	const auto& outputDirectory = parser.add<std::string>("output-directory", "Specify the location where output files should be dumped", '\0', arrrgh::Optional, "../output/");

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

	if(listGPUs.value()) {
		return 0;
	}

	if(sampleSetSize.value() == -1) {
		std::cout << "Experiment requires the --sample-set-size parameter to be set" << std::endl;
		exit(0);
	}

	if(boxSize.value() == -1) {
		std::cout << "Experiment requires the --box-size parameter to be set" << std::endl;
		exit(0);
	}

	if(objectDirectory.value().empty()) {
		std::cout << "Experiment requires the --source-directory parameter to be set" << std::endl;
		exit(0);
	}

	// Interpret seed value
    std::stringstream sstr(forcedSeed.value());
    size_t val;
    sstr >> val;
    if(val != 0) {
        std::cout << "Using overridden seed: " << val << std::endl;
    }

	runClutterBoxExperiment(objectDirectory.value(), sampleSetSize.value(), boxSize.value(), spinImageWidth.value(), spinImageSupportAngle.value(), dumpRawResults.value(), outputDirectory.value(), val);

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