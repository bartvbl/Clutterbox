#pragma once

#include <string>
#include <utility>
#include <spinImage/common/types/array.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/gpu/types/PointCloud.h>
#include "ExecutionTimes.h"

class ClutterboxMethod {

    virtual SpinImage::array<char> generateDescriptors(
            SpinImage::gpu::Mesh device_sceneMesh,
            SpinImage::gpu::PointCloud device_scenePointCloud,
            SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> device_descriptorOrigins,
            float supportRadius,
            ExecutionTimes* executionTimes = nullptr) = 0;

    virtual SpinImage::array<unsigned int> computeSearchResultRanks(
            SpinImage::array<char> device_needleDescriptors,
            SpinImage::array<char> device_haystackDescriptors,
            ExecutionTimes* executionTimes = nullptr) = 0;

    virtual const std::string getMethodCommandLineParameterName() = 0;

    virtual const std::string getMethodDumpFileName() = 0;
};