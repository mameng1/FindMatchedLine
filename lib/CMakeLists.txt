add_library(Line3D
            commons.h
            view.cc)
target_link_libraries(Line3D
    ${PCL_LIBRARIES}
    ${OpenCV_LIBRARIES})
