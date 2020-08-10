#pragma once

#include <string>
#include <utility>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/DeviceOrientedPoint.h>
#include <shapeDescriptor/gpu/types/PointCloud.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/cpu/types/array.h>
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
        virtual ShapeDescriptor::gpu::array<char> generateDescriptors(
                ShapeDescriptor::gpu::Mesh device_sceneAsMesh,
                ShapeDescriptor::gpu::PointCloud device_sceneAsPointCloud,
                ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::DeviceOrientedPoint> device_descriptorOrigins,
                Clutterbox::GenerationParameters parameters,
                ExecutionTimes* executionTimes = nullptr) = 0;

        virtual ShapeDescriptor::cpu::array<unsigned int> computeSearchResultRanks(
                ShapeDescriptor::gpu::array<char> device_needleDescriptors,
                ShapeDescriptor::gpu::array<char> device_haystackDescriptors,
                Clutterbox::SearchParameters parameters,
                ExecutionTimes* executionTimes = nullptr) = 0;

        virtual const std::string getMethodCommandLineParameterName() = 0;

        virtual const std::string getMethodDumpFileName() = 0;
    };
}

