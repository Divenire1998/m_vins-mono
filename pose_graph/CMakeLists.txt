cmake_minimum_required(VERSION 2.8.3)
project(pose_graph)

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
    nav_msgs
    camera_model
    cv_bridge
    roslib
    )

find_package(OpenCV)


find_package(Ceres REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS}  ${EIGEN3_INCLUDE_DIR})

catkin_package()

add_executable(pose_graph
    src/pose_graph_node.cpp
    src/pose_graph.cpp
    src/keyframe.cpp
    src/utility/CameraPoseVisualization.cpp
    src/ThirdParty/DBoW/BowVector.cpp
    src/ThirdParty/DBoW/FBrief.cpp
    src/ThirdParty/DBoW/FeatureVector.cpp
    src/ThirdParty/DBoW/QueryResults.cpp
    src/ThirdParty/DBoW/ScoringObject.cpp
    src/ThirdParty/DUtils/Random.cpp
    src/ThirdParty/DUtils/Timestamp.cpp
    src/ThirdParty/DVision/BRIEF.cpp
    src/ThirdParty/VocabularyBinary.cpp
    )

target_link_libraries(pose_graph ${catkin_LIBRARIES}  ${OpenCV_LIBS} ${CERES_LIBRARIES}) 
# message("catkin_lib  ${catkin_LIBRARIES}")
