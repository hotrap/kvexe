cmake_minimum_required(VERSION 3.2)
project(rocksdb-kvexe)

IF (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()
add_compile_options(-Wall -Wextra -fexceptions)
IF (CMAKE_BUILD_TYPE STREQUAL Debug)
    ADD_DEFINITIONS(-DDEBUG)
ENDIF()

add_subdirectory(rcu_vector)

aux_source_directory(src SRCS)
add_executable(${PROJECT_NAME} ${SRCS})

include_directories(${ROCKSDB_INCLUDE} ${VISCNTS_INCLUDE})
target_include_directories(${PROJECT_NAME}
	PRIVATE
		${PROJECT_SOURCE_DIR}/include
)
target_link_directories(${PROJECT_NAME} PUBLIC ${ROCKSDB_LIB})
target_link_libraries(${PROJECT_NAME}
	PUBLIC
        rocksdb
    PRIVATE
        rcu_vector_bp
        boost_fiber
)
IF (DEFINED VISCNTS_LIB)
    target_link_directories(${PROJECT_NAME} PUBLIC ${VISCNTS_LIB})
    target_link_libraries(${PROJECT_NAME} PUBLIC viscnts)
ENDIF()

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
