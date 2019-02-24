#pragma once

#include <vector>
#include <shapeSearch/gpu/types/DeviceMesh.h>

DeviceMesh combineMeshesOnGPU(std::vector<DeviceMesh> meshes);