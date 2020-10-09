#pragma once

#include <vector>
#include <random>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/common/types/OrientedPoint.h>

struct Transformation {
    float3 position;
    float3 rotation;
};

void randomlyTransformMeshes(ShapeDescriptor::gpu::Mesh scene, std::vector<ShapeDescriptor::gpu::Mesh> meshList, std::vector<Transformation> transformations);
void randomlyTransformMeshes(ShapeDescriptor::gpu::Mesh scene, float maxDistance, std::vector<ShapeDescriptor::gpu::Mesh> meshList, std::minstd_rand0 &randomGenerator);