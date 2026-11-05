cmake_minimum_required(VERSION 3.16)
project(kims LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(KIMS_NO_ROS "Build the offline binaries only, ignore catkin" OFF)
if(NOT KIMS_NO_ROS)
  find_package(catkin QUIET COMPONENTS
    roscpp rosbag topic_tools sensor_msgs geometry_msgs nav_msgs
  )
endif()
if(catkin_FOUND)
  catkin_package()
endif()

find_package(Ceres REQUIRED)
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -ffp-contract=off")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-ffp-contract=off")
if(DEFINED ENV{KIMS_PORTABLE})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
endif()

if(NOT DEFINED CUDA_HOME)
  if(EXISTS "/usr/local/cuda-12.8")
    set(CUDA_HOME "/usr/local/cuda-12.8")
  elseif(EXISTS "/usr/local/cuda")
    set(CUDA_HOME "/usr/local/cuda")
  endif()
endif()
include_directories(${CUDA_HOME}/include)
link_directories(${CUDA_HOME}/lib64)

include_directories(/usr/include/eigen3 ${CERES_INCLUDE_DIRS})

function(kims_add TARGET WITH_CERES)
  add_executable(${TARGET} ${ARGN})
  target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(${TARGET} PRIVATE cudart)
  if(${WITH_CERES})
    target_link_libraries(${TARGET} PRIVATE ${CERES_LIBRARIES})
  endif()
  if(OPENMP_FOUND)
    target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
  endif()
  set_target_properties(${TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
  if(catkin_FOUND)
    add_custom_command(TARGET ${TARGET} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory
              ${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              $<TARGET_FILE:${TARGET}> ${CATKIN_DEVEL_PREFIX}/lib/${PROJECT_NAME}/)
  endif()
endfunction()

kims_add(kims   FALSE src/kims_node.cu src/kims_solver.cu)
kims_add(sw_pgo TRUE  src/sw_pgo.cpp)
kims_add(pf     FALSE src/pf.cpp)

if(catkin_FOUND)
  kims_add(kims_online FALSE src/kims_online.cpp src/kims_node.cu src/kims_solver.cu)
  target_compile_definitions(kims_online PRIVATE KIMS_NO_MAIN)
  target_include_directories(kims_online PRIVATE ${catkin_INCLUDE_DIRS})
  target_link_libraries(kims_online PRIVATE ${catkin_LIBRARIES})
endif()
