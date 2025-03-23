# backend/cpp/src/CUDAoperators/CMakeLists.txt
file(GLOB_RECURSE OPERATOR_SOURCES "*.cu" "*.cpp")
add_library(operators STATIC ${OPERATOR_SOURCES})
target_include_directories(operators PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
