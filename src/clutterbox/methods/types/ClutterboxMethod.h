#pragma once

#include <string>
#include <utility>
#include <spinImage/common/types/array.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/gpu/types/PointCloud.h>
#include "ExecutionTimes.h"

namespace Clutterbox {
    // Storing additional parameters in structs allows the addition of more parameters without
    // requiring code changes, and does not needlessly expand function signatures

    struct GenerationParameters {
        float supportRadius;
    };

    struct SearchParameters {
        size_t needleDescriptorScenePointCloudPointCount;
        size_t haystackDescriptorScenePointCloudPointCount;
    };

    class Method {
        virtual SpinImage::array<char> generateDescriptors(
                SpinImage::gpu::Mesh device_sceneMesh,
                SpinImage::gpu::PointCloud device_scenePointCloud,
                SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> device_descriptorOrigins,
                Clutterbox::GenerationParameters parameters,
                ExecutionTimes* executionTimes = nullptr) = 0;

        virtual SpinImage::array<unsigned int> computeSearchResultRanks(
                SpinImage::array<char> device_needleDescriptors,
                SpinImage::array<char> device_haystackDescriptors,
                Clutterbox::SearchParameters parameters,
                ExecutionTimes* executionTimes = nullptr) = 0;

        virtual const std::string getMethodCommandLineParameterName() = 0;

        virtual const std::string getMethodDumpFileName() = 0;
    };
}

