#pragma once

#include <vector>
#include <random>
#include <spinImage/gpu/types/DeviceMesh.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>

array<signed long long> computeUniqueIndexMapping(DeviceMesh boxScene, std::vector<DeviceMesh> deviceMeshes, std::vector<size_t> *uniqueVertexCounts);
array<DeviceOrientedPoint> applyUniqueMapping(DeviceMesh boxScene, array<signed long long> mapping);

void randomlyTransformMeshes(DeviceMesh scene, float maxDistance, std::vector<DeviceMesh> meshList, std::default_random_engine &randomGenerator);