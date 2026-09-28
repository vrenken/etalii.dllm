cmake_minimum_required(VERSION 3.18...3.31)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_kernels NB_STATIC cpp/kernels.cpp)
target_include_directories(_kernels PRIVATE cpp/include)

# Determinism: never let the compiler reassociate or fuse floating point operations.
if(MSVC)
  # VS 2022 only contracts to FMA when /fp:contract is given explicitly.
  target_compile_options(_kernels PRIVATE /fp:precise)
else()
  target_compile_options(_kernels PRIVATE -ffp-contract=off -fno-fast-math)
endif()

install(TARGETS _kernels LIBRARY DESTINATION etalii_dllm)
