# Copyright: (C) 2013 iCub Facility - Istituto Italiano di Tecnologia
# Author: Alessandro Roncone
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

cmake_minimum_required(VERSION 2.8)
project(demoINNOROBO)

find_package(YARP REQUIRED)
find_package(ICUB REQUIRED)
find_package(ICUBcontrib REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${YARP_MODULE_PATH})
list(APPEND CMAKE_MODULE_PATH ${ICUB_MODULE_PATH})
include(YarpInstallationHelpers)

list(APPEND CMAKE_MODULE_PATH ${ICUBCONTRIB_MODULE_PATH})
include(ICUBcontribHelpers)
include(ICUBcontribOptions)
icubcontrib_set_default_prefix()

find_package(OpenCV REQUIRED)

include_directories(${YARP_INCLUDE_DIRS})
include_directories(${ICUB_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_definitions(${YARP_DEFINES})   # add yarp definitions
include(YarpInstallationHelpers)   # pick up yarp's cmake scripts

file(GLOB source src/*.cpp)
file(GLOB header src/*.h)

source_group("Source Files" FILES ${source})
source_group("Header Files" FILES ${header})

add_executable(demoINNOROBO ${source} ${header} )

target_link_libraries(demoINNOROBO icubmod ${YARP_LIBRARIES} ${OpenCV_LIBRARIES})

install(TARGETS demoINNOROBO DESTINATION bin)

## Add apps
add_subdirectory(app)

icubcontrib_add_uninstall_target()
