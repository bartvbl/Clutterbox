#pragma once

#include <shapeSearch/gpu/types/DeviceMesh.h>

void scaleMeshOnGPU(DeviceMesh mesh, float targetRadius);
void randomlyTransformMeshes(DeviceMesh scene, std::vector<DeviceMesh> meshList, std::default_random_engine randomGenerator);