#pragma once

#include <vector>
#include <random>
#include <spinImage/gpu/types/DeviceMesh.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>

struct Transformation {
    float3 position;
    float3 rotation;
};

array<signed long long> computeUniqueIndexMapping(DeviceMesh boxScene, std::vector<DeviceMesh> deviceMeshes, std::vector<size_t> *uniqueVertexCounts, size_t &totalUniqueVertexCount);
array<DeviceOrientedPoint> applyUniqueMapping(DeviceMesh boxScene, array<signed long long> mapping, size_t totalUniqueVertexCount);
array<DeviceOrientedPoint> computeUniqueSpinOrigins(DeviceMesh &mesh);

void randomlyTransformMeshes(DeviceMesh scene, std::vector<DeviceMesh> meshList, std::vector<Transformation> transformations);
void randomlyTransformMeshes(DeviceMesh scene, float maxDistance, std::vector<DeviceMesh> meshList, std::default_random_engine &randomGenerator);