cmake_minimum_required( VERSION 2.8 )

#set( CMAKE_BUILD_TYPE Debug )
 SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# opencv 
set(OpenCV_DIR /home/mameng/software/opencv-3.4.5/install/share/OpenCV)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

##-----------------------------------------------------------------------------
## OpenMP
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
    message("OPENMP_FOUND")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF(OPENMP_FOUND)

# eigen 
include_directories( "/usr/include/eigen3/" )

# pcl 
set(PCL_DIR /home/mameng/software/pcl/install/share/pcl-1.8)
find_package( PCL REQUIRED)
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

# octomap 
#find_package( octomap REQUIRED )
#include_directories( ${OCTOMAP_INCLUDE_DIRS} )
add_subdirectory(lib)

add_executable( pointcloud_mapping pointcloud_mapping.cpp )
target_link_libraries( pointcloud_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES} )

add_executable(generate_cloud generate_cloud.cpp)
target_link_libraries( generate_cloud ${OpenCV_LIBS} ${PCL_LIBRARIES} )

add_executable(generate_sfm generate_sfm.cpp)
target_link_libraries( generate_sfm ${OpenCV_LIBS} ${PCL_LIBRARIES} )

add_executable(computeNeighbor computeNeighbor.cpp)
target_link_libraries( computeNeighbor ${OpenCV_LIBS} ${PCL_LIBRARIES} Line3D)

add_executable(match_line match_line.cpp)
target_link_libraries( match_line ${OpenCV_LIBS} ${PCL_LIBRARIES} Line3D)
add_executable(deb debug.cpp)
target_link_libraries( deb ${OpenCV_LIBS} ${PCL_LIBRARIES} Line3D)
#add_executable( octomap_mapping octomap_mapping.cpp )
#target_link_libraries( octomap_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES})
