#pragma once

#include <vector>
#include <random>
#include <shapeSearch/gpu/types/DeviceMesh.h>

void randomlyTransformMeshes(DeviceMesh scene, float maxDistance, std::vector<DeviceMesh> meshList, std::default_random_engine randomGenerator);