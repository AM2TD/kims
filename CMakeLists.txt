cmake_minimum_required(VERSION 3.16)
project(kims LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# The published results were produced by a Release build (-O3 -DNDEBUG). Neither
# a bare `cmake` nor `catkin_make` sets a build type, so pin one here; without
# -DNDEBUG the Eigen-heavy baselines keep their assertions and UKF-RTS in
# particular converges to slightly different numbers.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The offline tools need no ROS: a plain `cmake .. && make` builds them and they
# read CSV. Inside a catkin workspace the online node is built as well.
# -DKIMS_NO_ROS=ON forces the offline-only path even in a shell where ROS is
# sourced (CMake would otherwise pick /opt/ros/<distro> up from PATH).
option(KIMS_NO_ROS "Build the offline binaries only, ignore catkin" OFF)
if(NOT KIMS_NO_ROS)
  find_package(catkin QUIET COMPONENTS
    roscpp rospy roslib std_msgs sensor_msgs geometry_msgs nav_msgs tf2_ros
    message_generation nlink_parser
  )
endif()
if(catkin_FOUND)
  add_message_files(FILES UwbRange.msg ImuSample.msg)
  generate_messages(DEPENDENCIES std_msgs geometry_msgs)
  catkin_package(CATKIN_DEPENDS roscpp rospy roslib std_msgs sensor_msgs geometry_msgs
                                nav_msgs tf2_ros message_runtime nlink_parser)
endif()

find_package(Ceres REQUIRED)
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Identical floating-point contraction everywhere, so the same expression gives
# the same bits whether it is compiled for the offline tool or the online node.
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

# ── Offline: one executable per method, dataset chosen at run time ───────────
# kims_add(<target> <needs_ceres> <src...>)
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
  # One predictable location in every configuration: <build>/bin. With catkin the
  # binaries are additionally copied into devel/lib/kims, so rosrun and the
  # devel-space fallback in scripts/run.py keep working.
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

kims_add(kims               FALSE src/offline/kims_node.cu src/offline/kims_solver.cu)
kims_add(ekf_rts            FALSE src/offline/ekf_rts.cpp)
kims_add(ukf_rts            FALSE src/offline/ukf_rts.cpp)
kims_add(particle_smoothing FALSE src/offline/particle_smoothing.cpp)
kims_add(sw_map             TRUE  src/offline/sw_map.cpp)
# IULUI solves a different SW-MAP problem: attitude on a quaternion manifold
# with a dynamic autodiff cost, so it keeps its own node.
kims_add(sw_map_iului       TRUE  src/offline/sw_map_iului.cpp)

# ── Online: real-time node, built only when ROS is present ──────────────────
if(catkin_FOUND)
  include_directories(${catkin_INCLUDE_DIRS})
  add_executable(kims_online src/online/kims_online_node.cpp src/offline/kims_solver.cu)
  target_include_directories(kims_online PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(kims_online PRIVATE ${catkin_LIBRARIES} cudart)
  add_dependencies(kims_online ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
  if(OPENMP_FOUND)
    target_link_libraries(kims_online PRIVATE OpenMP::OpenMP_CXX)
  endif()

  catkin_install_python(PROGRAMS scripts/run.py scripts/plot.py scripts/replay.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
endif()
