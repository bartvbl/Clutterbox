#pragma once

#include <vector>
#include <random>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/DeviceOrientedPoint.h>

struct Transformation {
    float3 position;
    float3 rotation;
};

void randomlyTransformMeshes(SpinImage::gpu::Mesh scene, std::vector<SpinImage::gpu::Mesh> meshList, std::vector<Transformation> transformations);
void randomlyTransformMeshes(SpinImage::gpu::Mesh scene, float maxDistance, std::vector<SpinImage::gpu::Mesh> meshList, std::minstd_rand0 &randomGenerator);