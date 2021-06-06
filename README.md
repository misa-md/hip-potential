# hip-potential

[hip](https://github.com/ROCm-Developer-Tools/HIP) support for potential lib.

## Requirements
- CMake version 3.15 or higher.
- [HIP](https://github.com/ROCm-Developer-Tools/HIP) version 3.5 or higher
- CUDA if your target platform is NVIDIA GPU.

## Build
### Build dependencies
```bash
pkg fetch
pkg install
```

### Build hip-potential
For AMD GPU and DCU:
```bash
cmake -B./build-hip-pot -S./
cmake --build ./build-hip-pot/ -j 4
```
For NVIDIA GPU and CUDA:
```bash
pkg install
export CUDA_PATH=/opt/tools/cuda/10.0 # change to your CUDA path
export HIP_PATH=/opt/compilers/rocm/4.2.0 # change to your HIP path
export HIP_PLATFORM=nvidia
cmake -B./build-hip-pot -S./
cmake --build ./build-hip-pot/ -j 4
```
