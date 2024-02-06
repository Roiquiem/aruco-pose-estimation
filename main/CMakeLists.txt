cmake_minimum_required(VERSION 3.0) 
PROJECT (aruco)

# Set the C++ compiler
set(CMAKE_CXX_STANDARD 14)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

# Define source and header files
set( NAME_SRC
    poseEstimateAruco.cpp   
)

set( NAME_HEADERS       
    #  include/header.h
)

# Create the executable
add_executable(aruco ${ARUCO_SRC} ${ARUCO_HEADERS})

# Link libraries
target_link_libraries(aruco PUBLIC ${OpenCV_LIBS} nlohmann_json::nlohmann_json)
