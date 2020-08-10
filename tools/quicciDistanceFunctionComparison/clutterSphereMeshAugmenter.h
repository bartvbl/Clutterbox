#pragma once

#include <shapeDescriptor/cpu/types/Mesh.h>

const unsigned int SPHERE_RESOLUTION_X = 15;
const unsigned int SPHERE_RESOLUTION_Y = 15;

const unsigned int SPHERE_TRIANGLE_COUNT = SPHERE_RESOLUTION_X * SPHERE_RESOLUTION_Y * 2;
const unsigned int SPHERE_VERTEX_COUNT = 3 * SPHERE_TRIANGLE_COUNT;


ShapeDescriptor::cpu::Mesh applyClutterSpheres(ShapeDescriptor::cpu::Mesh mesh, int count, float radius, size_t randomSeed);