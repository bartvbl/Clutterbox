#pragma once

#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/OrientedPoint.h>
#include <shapeDescriptor/gpu/types/PointCloud.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/utilities/kernels/meshSampler.cuh>
#include <vector>

ShapeDescriptor::cpu::array<float> computeClutter(
        ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::OrientedPoint> array,
        ShapeDescriptor::gpu::PointCloud cloud,
        float spinImageWidth,
        size_t referenceObjectSampleCount,
        size_t referenceObjectOriginCount);

size_t computeReferenceSampleCount(
        ShapeDescriptor::gpu::Mesh referenceMesh,
        size_t totalSceneSampleCount,
        ShapeDescriptor::gpu::array<float> cumulativeAreaArray);