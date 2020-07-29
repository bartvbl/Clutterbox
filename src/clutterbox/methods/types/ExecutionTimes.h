#pragma once

#include <string>
#include <utility>
#include <vector>
#include <map>

struct ExecutionTimeMeasurement {
    ExecutionTimeMeasurement(std::string name, double time) : name(std::move(name)), timeInSeconds(time) {}

    std::string name;
    double timeInSeconds;
};

class ExecutionTimes {
    std::vector<ExecutionTimeMeasurement> measurements;
    std::map<std::string, double> measurementMap;

public:
    void append(const std::string& name, double timeInSeconds) {
        measurements.emplace_back(name, timeInSeconds);
        measurementMap.at(name) = timeInSeconds;
    }

    std::vector<ExecutionTimeMeasurement>* getAll() {
        return &measurements;
    }

    double getMeasurementByName(std::string name) {
        return measurementMap.at(name);
    }
};