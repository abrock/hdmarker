cmake_minimum_required(VERSION 2.8.12)
project(hdmarker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost COMPONENTS system filesystem iostreams REQUIRED)

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

FIND_PACKAGE(Eigen3 REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(CMakeParseArguments)

include_directories(/usr/local/include)

#####################################################
## SET COMPILER ENVIRONMENT
#####################################################
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
  add_definitions(-DCOMPILER_CLANG_X86)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmmx")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -fno-omit-frame-pointer -fPIC -march=native -fopenmp -mmmx")
  add_definitions(-DCOMPILER_GCC_X86)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /noOy /W2 /EHsc /openmp")
endif()

# default to static on win shared on others
if (NOT DEFINED BUILD_SHARED_LIBS)
  if (WIN32)
    option(BUILD_SHARED_LIBS "build shared libraries" OFF)
  else()
    option(BUILD_SHARED_LIBS "build shared libraries" ON)
  endif()
endif()



#####################################################
## DEFINE make doc
#####################################################
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY IMMEDIATE)
    add_custom_target(doc-hdmarker ALL COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        SOURCES ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM
        )
endif()

find_package(Boost COMPONENTS system filesystem REQUIRED)

#####################################################
## COMPILE
#####################################################
if (BUILD_SHARED_LIBS)
  add_definitions(-D${BUILD_SHARED_LIBS})
endif()

include_directories(src/lib/)

add_executable(scale-pyramid src/bin/scale-pyramid.cpp)
target_link_libraries(scale-pyramid PUBLIC
    ceres opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_calib3d)

target_link_libraries(scale-pyramid PUBLIC
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_IOSTREAMS_LIBRARY}
    )

add_library(hdmarker src/lib/hdmarker.cpp src/lib/gridstore.cpp src/lib/timebench.cpp src/lib/subpattern.cpp)

target_link_libraries(hdmarker PUBLIC
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${Boost_FILESYSTEM_LIBRARY}
    ${Boost_SYSTEM_LIBRARY}
    ${Boost_IOSTREAMS_LIBRARY}
    gzstream
    )

set(HDMARKER_LIBRARY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}hdmarker${CMAKE_SHARED_LIBRARY_SUFFIX} CACHE INTERNAL "" FORCE)

find_package(ZLIB REQUIRED)

add_executable(marker_gen src/bin/marker_gen.cpp)
target_link_libraries(marker_gen hdmarker podofo jpeg ${ZLIB_LIBRARIES}
    fontconfig freetype idn crypto crypto ssl jpeg pthread freetype png z
    tiff unistring
    )

find_package(GTest REQUIRED)
add_executable(tests src/tests.cpp)
target_link_libraries(tests
  hdmarker
  ${GTEST_BOTH_LIBRARIES}
  )

add_dependencies(hdmarker hdmarker-header-export)
set(CONF_INCLUDE_DIRS ${CMAKE_BINARY_DIR}/include)
export(TARGETS hdmarker FILE ${hdmarker_BINARY_DIR}/hdmarkerTargets.cmake)
include(cmake/GenConfig.cmake)

#####################################################
## COMPILE DEMO
#####################################################
add_subdirectory(demo/)


#####################################################
## when install then also install pkgconfig
#####################################################
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/hdmarker.pc.in ${CMAKE_CURRENT_BINARY_DIR}/hdmarker.pc @ONLY)
message("Path: " ${CMAKE_INSTALL_DATAROOTDIR})
install(FILES ${CMAKE_BINARY_DIR}/hdmarker.pc DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pkgconfig")
install(FILES ${CMAKE_BINARY_DIR}/hdmarker.pc DESTINATION "${CMAKE_INSTALL_PREFIX}/share/pkgconfig")

install (FILES src/lib/gridstore.hpp src/lib/hdmarker.hpp src/lib/loess.hpp src/lib/subpattern.hpp src/lib/timebench.hpp DESTINATION ${CMAKE_INSTALL_PREFIX}/include/hdmarker)

install (TARGETS hdmarker DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)



