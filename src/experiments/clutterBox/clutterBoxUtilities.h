#pragma once

#include <vector>
#include <spinImage/gpu/types/DeviceMesh.h>

DeviceMesh combineMeshesOnGPU(std::vector<DeviceMesh> meshes);