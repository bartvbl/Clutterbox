#pragma once

#include <string>
#include <utility>
#include <json.hpp>

#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <shapeDescriptor/gpu/types/PointCloud.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <tsl/ordered_map.h>
#include "ExecutionTimes.h"

template<class Key, class T, class Ignore, class Allocator,
        class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>,
        class AllocatorPair = typename std::allocator_traits<Allocator>::template rebind_alloc<std::pair<Key, T>>,
        class ValueTypeContainer = std::vector<std::pair<Key, T>, AllocatorPair>>
using ordered_map = tsl::ordered_map<Key, T, Hash, KeyEqual, AllocatorPair, ValueTypeContainer>;

using json = nlohmann::basic_json<ordered_map>;

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
                ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_descriptorOrigins,
                Clutterbox::GenerationParameters parameters,
                ExecutionTimes* executionTimes = nullptr) = 0;

        virtual ShapeDescriptor::cpu::array<unsigned int> computeSearchResultRanks(
                ShapeDescriptor::gpu::array<char> device_needleDescriptors,
                ShapeDescriptor::gpu::array<char> device_haystackDescriptors,
                Clutterbox::SearchParameters parameters,
                ExecutionTimes* executionTimes = nullptr) = 0;

        virtual void dumpMetadata(json* jsonOutput) = 0;

        virtual const std::string getMethodCommandLineParameterName() = 0;

        virtual const std::string getMethodDumpFileName() = 0;
    };
}

