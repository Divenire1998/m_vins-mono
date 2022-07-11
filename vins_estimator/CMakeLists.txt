cmake_minimum_required(VERSION 2.8.3)
project(vins_estimator)


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
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    )

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


# 包含一些头文件
find_package(OpenCV REQUIRED)
# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")
find_package(Eigen3)
find_package(Ceres REQUIRED)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  # src/initial
  # src/
)

catkin_package()


add_executable(vins_estimator

    # 节点
    src/estimator_node.cpp
    # 后端
    src/estimator.cpp

    src/parameters.cpp
    src/feature_manager.cpp

    # 各自约束关系
    src/factor/pose_local_parameterization.cpp
    src/factor/projection_factor.cpp
    src/factor/projection_td_factor.cpp
    src/factor/marginalization_factor.cpp

    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp

    # 初始化
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    )

# 添加依赖库
target_link_libraries(vins_estimator ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES}) 


