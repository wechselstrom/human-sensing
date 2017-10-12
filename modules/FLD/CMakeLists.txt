add_executable(FLD FLD.cpp)

# Local libraries
include_directories(${CLM_SOURCE_DIR}/include)


include_directories(${YARP_INCLUDE_DIRS})

			

target_link_libraries(FLD dlib)



if(WIN32)
    target_link_libraries(FLD ${OpenCVLibraries})
endif(WIN32)
if(UNIX)
    target_link_libraries(FLD ${OpenCV_LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES} ${YARP_LIBRARIES})
endif(UNIX)

install (TARGETS FLD DESTINATION ${CMAKE_BINARY_DIR}/bin)
