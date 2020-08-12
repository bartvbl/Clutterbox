# Clutterbox Experiment Implementation Source Code

This source code repository accompanies the paper 'Radial Intersection Count Image: a Clutter Resistant 3D Shape Descriptor', and contains the implementation of the proposed clutterbox experiment described in it.

## Compiling

The project has been developed on Ubuntu Linux 18.04 LTS. It should in principle compile for Windows and MacOS installations too, although I have not personally tested this. 

For compiling on Ubuntu Linux, the following Aptitude packages were used (package versions used for results in paper are listed):

- cmake (version 3.10.2)
- libeigen3-dev (version 3.3.4-4)
- libpcl-dev (version 1.8.1)
- python3 (version 3.6.7-1~18.04)
- nvidia-cuda-toolkit (version 10.1.105)
- nvidia-cuda-dev (version 10.1.105)
- gcc-7 (version 7.4.0-1)

Compiling can be done by running the following from the project's root directory:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j 4
```

The project generates several executables:

- `clutterbox`: Main executable for running the Clutterbox experiment
- `projectionBenchmark`: The paper discusses a microbenchmark for comparing the PCL method for projecting points into cylindrical coordinates relative to our proposed method. This executable performs this benchmark.
- `clutterEstimator`: Tool used to generate the heatmaps comparing search result ranks to clutter levels in the area around a given point. This tool was used to estimate the clutter level around each point in a scene.
- `libShapeDescriptor/imagerenderer`: Tool for using our QUICCI, RICI, and SI implementations to render images of a given input object. Outputs PNG images.

## Credits

- Development and implementation: Bart Iver van Blokland, [NTNU Visual Computing Lab](https://www.idi.ntnu.no/grupper/vis/)
- Supervision: Theoharis Theoharis, [NTNU Visual Computing Lab](https://www.idi.ntnu.no/grupper/vis/)



