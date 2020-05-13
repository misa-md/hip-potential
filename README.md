# hip-potential

[hip](https://github.com/ROCm-Developer-Tools/HIP) support for potential lib.

## Build
```bash
# here, we use hipcc as linker, you can also change to use another one.
cmake -B./build-hip-pot -S./ -DCMAKE_LINKER=hipcc \
    -DCMAKE_CXX_LINK_EXECUTABLE="<CMAKE_LINKER> <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>"
cmake --build ./build-hip-pot/ -j 4
```
