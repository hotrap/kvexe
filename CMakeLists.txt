cmake_minimum_required(VERSION 3.15)
project(rocksdb-kvexe CXX)

find_package(rusty-cpp CONFIG REQUIRED)
find_package(rcu-vector CONFIG REQUIRED COMPONENTS rcu_vector_bp)

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
		rocksdb
	PRIVATE
		rusty-cpp
		rcu_vector_bp
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
