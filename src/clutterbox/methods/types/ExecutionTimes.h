#pragma once

#include <string>
#include <utility>
#include <vector>

class ExecutionTimeMeasurement {
    std::string name;
    double timeInSeconds;

public:
    ExecutionTimeMeasurement(std::string name, double time) : name(std::move(name)), timeInSeconds(time) {}
};

class ExecutionTimes {
    std::vector<ExecutionTimeMeasurement> measurements;

    void append(const std::string& name, double timeInSeconds) {
        measurements.emplace_back(name, timeInSeconds);
    }
};