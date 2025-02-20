cmake_minimum_required(VERSION 3.15)
project(rocksdb-kvexe CXX)

add_subdirectory(3rdparty/rusty-cpp)
add_subdirectory(3rdparty/counter-timer)
add_subdirectory(3rdparty/rcu-vector)
add_subdirectory(3rdparty/counter-timer-vec)
add_subdirectory(3rdparty/AutoTuner)

add_compile_options(-fno-omit-frame-pointer -Wall -Wextra -fexceptions)

aux_source_directory(src SRCS)
add_executable(${PROJECT_NAME} ${SRCS})

option(WITH_ASAN "build with ASAN" OFF)
if(WITH_ASAN)
	add_compile_options(-fsanitize=address)
	# Should come first in initial library list
	target_link_libraries(${PROJECT_NAME} PRIVATE asan)
endif()

include_directories(
    ${ROCKSDB_INCLUDE_DIR}
    ${RALT_INCLUDE_DIR}
)
target_include_directories(${PROJECT_NAME} PUBLIC include 3rdparty/ycsbgen)
target_link_directories(${PROJECT_NAME} PUBLIC ${ROCKSDB_LIB_DIR})
target_link_libraries(${PROJECT_NAME}
	PUBLIC
		dl
		rocksdb
		pthread
		xxhash
	PRIVATE
		counter-timer-vec
		boost_program_options
		autotuner
)
IF (DEFINED RALT_LIB_DIR)
    target_link_directories(${PROJECT_NAME} PUBLIC ${RALT_LIB_DIR})
    target_link_libraries(${PROJECT_NAME} PUBLIC ralt)
ENDIF()

# target_compile_features can omit -std flag in compile_commands.json
# https://gitlab.kitware.com/cmake/cmake/-/issues/23397
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

install(TARGETS ${RPOJECT_NAME} DESTINATION "."
	RUNTIME DESTINATION bin
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
)
