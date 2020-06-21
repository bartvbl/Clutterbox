#pragma once

#include <vector>
#include <string>
#include <experiments/clutterBoxExperiment.hpp>
#include <experimental/filesystem>

void runQuicciDistanceFunctionBenchmark(
        std::experimental::filesystem::path sourceDirectory,
        std::experimental::filesystem::path outputDirectory,
        size_t seed,
        std::vector<int> sphereCountList,
        int sceneSphereCount,
        float clutterSphereRadius,
        GPUMetaData gpuMetaData);