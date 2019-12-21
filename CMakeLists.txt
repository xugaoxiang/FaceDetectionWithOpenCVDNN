cmake_minimum_required(VERSION 3.10)

project(fd)
#project(FaceDetectionLib)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories( ${OpenCV_INCLUDE_DIRS} )

#add_library(FaceDetectionLib SHARED fd.cpp fd.h)
#target_link_libraries(FaceDetectionLib LINK_PRIVATE ${OpenCV_LIBS})

ADD_EXECUTABLE(fd test.cpp)
target_link_libraries(fd LINK_PRIVATE ${OpenCV_LIBS})
