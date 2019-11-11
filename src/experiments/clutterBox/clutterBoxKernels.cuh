#pragma once

#include <vector>
#include <random>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>

struct Transformation {
    float3 position;
    float3 rotation;
};

SpinImage::array<signed long long> computeUniqueIndexMapping(SpinImage::gpu::Mesh boxScene, std::vector<SpinImage::gpu::Mesh> deviceMeshes, std::vector<size_t> *uniqueVertexCounts, size_t &totalUniqueVertexCount);
SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> applyUniqueMapping(SpinImage::gpu::Mesh boxScene, SpinImage::array<signed long long> mapping, size_t totalUniqueVertexCount);
SpinImage::array<SpinImage::gpu::DeviceOrientedPoint> computeUniqueSpinOrigins(SpinImage::gpu::Mesh &mesh);

void randomlyTransformMeshes(SpinImage::gpu::Mesh scene, std::vector<SpinImage::gpu::Mesh> meshList, std::vector<Transformation> transformations);
void randomlyTransformMeshes(SpinImage::gpu::Mesh scene, float maxDistance, std::vector<SpinImage::gpu::Mesh> meshList, std::default_random_engine &randomGenerator);