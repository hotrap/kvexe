cmake_minimum_required(VERSION 3.15)
project(rocksdb-kvexe CXX)

add_subdirectory(3rdparty/rusty-cpp)
add_subdirectory(3rdparty/rcu-vector)
add_subdirectory(3rdparty/counter-timer)

add_compile_options(-Wall -Wextra -fexceptions)
IF (CMAKE_BUILD_TYPE STREQUAL Debug)
    ADD_DEFINITIONS(-DDEBUG)
ENDIF()

aux_source_directory(src SRCS)
add_executable(${PROJECT_NAME} ${SRCS})

include_directories(
    ${ROCKSDB_INCLUDE}
    ${VISCNTS_INCLUDE}
)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_directories(${PROJECT_NAME} PUBLIC ${ROCKSDB_LIB})
target_link_libraries(${PROJECT_NAME}
	PUBLIC
		dl
		rocksdb
		pthread
		boost_fiber
	PRIVATE
		counter-timer
		boost_program_options
)
IF (DEFINED VISCNTS_LIB)
    target_link_directories(${PROJECT_NAME} PUBLIC ${VISCNTS_LIB})
    target_link_libraries(${PROJECT_NAME} PUBLIC viscnts)
ENDIF()

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

install(TARGETS ${RPOJECT_NAME} DESTINATION "."
	RUNTIME DESTINATION bin
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
)
