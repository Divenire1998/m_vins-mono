cmake_minimum_required(VERSION 2.8.3)
project(feature_tracker)

############################################################
# Build Type
############################################################
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()
MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})



############################################################
#  C/C++ Flags
############################################################
# -O3 编译等级优化为3上 开启了这个就不能用debug,会丢失信息了
# -Wall 开启所有警告
# -march=native, which automatically detects the features your CPU supports and sets the options appropriately.
if(CMAKE_BUILD_TYPE MATCHES Release)
   MESSAGE("开启编译器优化")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3  ")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
   MESSAGE("Debug 模式")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall ")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  ")
endif()


find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    camera_model
    )

find_package(OpenCV REQUIRED)

catkin_package()

include_directories(
    ${catkin_INCLUDE_DIRS}
    )

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(feature_tracker
    src/feature_tracker_node.cpp
    src/parameters.cpp
    src/feature_tracker.cpp
    )

target_link_libraries(feature_tracker ${catkin_LIBRARIES} ${OpenCV_LIBS})
