# HetuML

An Efficient Machine Learning Library for Hetu.

## Installation

1. Clone the respository.

2. Requirements: HetuML requires you to have the following tools prepared
- CMake >= 3.11
- Protobuf >= 3.0.0
- OpenMP

3. Configuration: Before compilation, please refer to your configuration by `cp cmake/config.cmake.template cmake/config.cmake`. Modify the configuration as needed.

4. Compile HetuML via the following commands
```shell
mkdir build && cd build
cmake ..
make -j 8
```

5. Set environment variables by running `source env.exp`
