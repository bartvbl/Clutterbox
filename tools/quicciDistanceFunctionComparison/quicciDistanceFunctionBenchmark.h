#pragma once

#include <vector>
#include <string>
#include <clutterbox/clutterBoxExperiment.hpp>
#include <experimental/filesystem>

enum class BenchmarkMode {
    BASELINE,
    SPHERE_CLUTTER
};

void runQuicciDistanceFunctionBenchmark(
        std::experimental::filesystem::path sourceDirectory,
        std::experimental::filesystem::path outputDirectory,
        std::experimental::filesystem::path objDumpFilePath,
        bool dumpOBJ,
        size_t seed,
        std::vector<int> sphereCountList,
        int sceneSphereCount,
        float clutterSphereRadius,
        GPUMetaData gpuMetaData,
        float supportRadius,
        BenchmarkMode mode);