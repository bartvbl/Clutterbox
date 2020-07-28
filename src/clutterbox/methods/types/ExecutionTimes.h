#pragma once

#include <string>
#include <utility>
#include <vector>
#include <map>

class ExecutionTimeMeasurement {
    std::string name;
    double timeInSeconds;

public:
    ExecutionTimeMeasurement(std::string name, double time) : name(std::move(name)), timeInSeconds(time) {}
};

class ExecutionTimes {
    std::vector<ExecutionTimeMeasurement> measurements;
    std::map<std::string, double> measurementMap;

public:
    void append(const std::string& name, double timeInSeconds) {
        measurements.emplace_back(name, timeInSeconds);
        measurementMap.at(name) = timeInSeconds;
    }

    double getMeasurementByName(std::string name) {
        return measurementMap.at(name);
    }
};