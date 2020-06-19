#pragma once

#include <vector>
#include <string>

void runQuicciDistanceFunctionBenchmark(
        std::string sourceDirectory,
        std::string outputDirectory,
        size_t seed,
        std::vector<int> sphereCountList,
        int sceneSphereCount,
        float clutterSphereRadius);