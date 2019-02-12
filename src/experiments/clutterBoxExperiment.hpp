#pragma once

#include "cuda_runtime.h"
#include <string>

void runClutterBoxExperiment(cudaDeviceProp device_information, std::string objectDirectory, int sampleSetSize, float boxSize, int spinImageWidthPixels);