cmake_minimum_required(VERSION 3.15)
project(rocksdb-kvexe CXX)

if(NOT DEFINED CACHELIB_HOME)
	if(NOT DEFINED ENV{CACHELIB_HOME})
		message(FATAL_ERROR "CACHELIB_HOME not set")  
	endif()
	set(CACHELIB_HOME $ENV{CACHELIB_HOME})
endif()

add_subdirectory(3rdparty/rusty-cpp)
add_subdirectory(3rdparty/counter-timer)
add_subdirectory(3rdparty/RocksCachelibWrapper)

add_compile_options(-Wall -Wextra -fexceptions)
IF (CMAKE_BUILD_TYPE STREQUAL Debug)
    ADD_DEFINITIONS(-DDEBUG)
ENDIF()

aux_source_directory(src SRCS)
add_executable(${PROJECT_NAME} ${SRCS})

include_directories(
    ${ROCKSDB_INCLUDE}
)
target_include_directories(${PROJECT_NAME} PUBLIC include 3rdparty/ycsbgen)
target_link_directories(${PROJECT_NAME} PUBLIC ${ROCKSDB_LIB})
target_link_libraries(${PROJECT_NAME}
	PRIVATE
		RocksCachelibWrapper
		counter-timer
		pthread
		boost_program_options
)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

install(TARGETS ${RPOJECT_NAME} DESTINATION "."
	RUNTIME DESTINATION bin
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
)
