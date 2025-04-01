file(GLOB_RECURSE OPERATOR_SOURCES "*.cu" "*.cpp")
add_library(operators STATIC ${OPERATOR_SOURCES})




target_include_directories(operators PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}  
    ${CMAKE_SOURCE_DIR}/backend/cpp/include
)

# 如果你要手动设置 CUTLASS 的路径




set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/cutlass")
target_include_directories(operators PUBLIC   ${CUTLASS_DIR}/include
${CUTLASS_DIR}/tools/util/include
)

