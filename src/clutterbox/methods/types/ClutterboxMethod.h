#pragma once

#include <string>
#include <utility>
#include <spinImage/common/types/array.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/gpu/types/PointCloud.h>
#include "ExecutionTimes.h"

template<typename DescriptorType>
class ClutterboxMethod {
    std::string methodShorthandIdentifier;
    std::string methodIdentifier;

    ClutterboxMethod(std::string shorthandIdentifier, std::string identifier) :
            methodShorthandIdentifier(std::move(shorthandIdentifier)),
            methodIdentifier(std::move(identifier)) {}

    virtual SpinImage::array<DescriptorType> generateDescriptors(
            SpinImage::gpu::Mesh device_sceneMesh,
            SpinImage::gpu::PointCloud device_scenePointCloud,
            SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> device_origins,
            ExecutionTimes* executionTimes = nullptr) = 0;

    virtual SpinImage::array<unsigned int> computeSearchResultRanks(
            SpinImage::array<DescriptorType> device_needleDescriptors,
            SpinImage::array<DescriptorType> device_haystackDescriptors,
            ExecutionTimes* executionTimes = nullptr) = 0;
};