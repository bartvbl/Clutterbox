#pragma once

#include <vector>
#include <random>
#include <spinImage/gpu/types/DeviceMesh.h>

array<DeviceOrientedPoint> removeDuplicates(DeviceMesh mesh);
void randomlyTransformMeshes(DeviceMesh scene, float maxDistance, std::vector<DeviceMesh> meshList, std::default_random_engine &randomGenerator);