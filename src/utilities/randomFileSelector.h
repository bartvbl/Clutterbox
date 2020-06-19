#pragma once

#include <vector>
#include <string>
#include <random>

std::vector<std::string> generateRandomFileList(const std::string &objectDirectory,
                                                unsigned int sampleSetSize,
                                                std::default_random_engine &generator);