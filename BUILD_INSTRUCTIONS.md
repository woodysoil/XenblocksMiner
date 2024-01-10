# Build Instructions

This document provides instructions for building the project on both Linux and Windows systems using vcpkg.

## Prerequisites

- vcpkg installed on your system.
- CMake installed on your system.

## Building on Linux

Execute the following commands in your terminal:

```bash
./vcpkg install argon2
./vcpkg install cryptopp
./vcpkg install curl
./vcpkg install nlohmann-json
./vcpkg install openssl

cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release
```

## Building on Windows

Execute the following commands in Command Prompt or PowerShell:

```bash
./vcpkg.exe install argon2:x64-windows-static
./vcpkg.exe install cryptopp:x64-windows-static
./vcpkg.exe install curl:x64-windows-static
./vcpkg.exe install nlohmann-json:x64-windows-static
./vcpkg.exe install openssl:x64-windows-static

cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=path/to/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows-static -DCMAKE_PREFIX_PATH=path/to/vcpkg/installed/x64-windows-static -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded
cmake --build build --config Release
```

