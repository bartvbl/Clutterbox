#pragma once

#include "cuda_runtime.h"
#include <string>

void runClutterBoxExperiment(cudaDeviceProp device_information, std::string objectDirectory, unsigned int sampleSetSize, float boxSize, unsigned int experimentRepetitions, int spinImageWidth);