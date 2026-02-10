cmake_minimum_required(VERSION 3.31)

project(PCG)

set(EXPORT_COMPILE_COMMANDS ON)

# Find packages requirements
find_package(Kokkos CONFIG)
if(Kokkos_FOUND)
    message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
else()
    if(EXISTS ${Kokkos_COMMON_SOURCE_DIR})
      add_subdirectory(${Kokkos_COMMON_SOURCE_DIR} Kokkos)
    else()
      include(FetchContent)
      FetchContent_Declare(
        Kokkos
        GIT_REPOSITORY https://github.com/kokkos/kokkos.git
        GIT_TAG        4.7.01
      )
    FetchContent_MakeAvailable(Kokkos)
  endif()
endif()

find_package(KokkosKernels CONFIG)

find_package(CLI11 CONFIG)
if(NOT CLI11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.1.3
  )
  FetchContent_MakeAvailable(CLI11)
endif()

# Build PCG executable
add_executable(pcg pcg.cpp)
target_link_libraries(pcg PRIVATE Kokkos::kokkos Kokkos::kokkoskernels CLI11::CLI11)

# Copy CSV data files to build directory
add_custom_command(TARGET pcg POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/data/poisson_128x128/rowptr.csv
    ${CMAKE_BINARY_DIR}/rowptr.csv
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/data/poisson_128x128/col.csv
    ${CMAKE_BINARY_DIR}/col.csv
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/data/poisson_128x128/val.csv
    ${CMAKE_BINARY_DIR}/val.csv
    COMMENT "Copying CSV files to build directory"
)
