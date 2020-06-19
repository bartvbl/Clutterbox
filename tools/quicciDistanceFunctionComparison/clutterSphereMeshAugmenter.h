#pragma once

#include <spinImage/cpu/types/Mesh.h>

const unsigned int SPHERE_RESOLUTION_X = 60;
const unsigned int SPHERE_RESOLUTION_Y = 60;


SpinImage::cpu::Mesh applyClutterSpheres(SpinImage::cpu::Mesh mesh, int count, float radius, size_t randomSeed);