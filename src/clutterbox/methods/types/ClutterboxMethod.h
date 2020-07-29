#pragma once

#include <string>
#include <utility>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/gpu/types/PointCloud.h>
#include <spinImage/gpu/types/array.h>
#include <spinImage/cpu/types/array.h>
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
    public:
        virtual SpinImage::gpu::array<char> generateDescriptors(
                SpinImage::gpu::Mesh device_sceneAsMesh,
                SpinImage::gpu::PointCloud device_sceneAsPointCloud,
                SpinImage::gpu::array<SpinImage::gpu::DeviceOrientedPoint> device_descriptorOrigins,
                Clutterbox::GenerationParameters parameters,
                ExecutionTimes* executionTimes = nullptr) = 0;

        virtual SpinImage::cpu::array<unsigned int> computeSearchResultRanks(
                SpinImage::gpu::array<char> device_needleDescriptors,
                SpinImage::gpu::array<char> device_haystackDescriptors,
                Clutterbox::SearchParameters parameters,
                ExecutionTimes* executionTimes = nullptr) = 0;

        virtual const std::string getMethodCommandLineParameterName() = 0;

        virtual const std::string getMethodDumpFileName() = 0;
    };
}

