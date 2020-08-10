#pragma once

#include <vector>
#include <shapeDescriptor/gpu/types/Mesh.h>

ShapeDescriptor::gpu::Mesh combineMeshesOnGPU(std::vector<ShapeDescriptor::gpu::Mesh> meshes);