cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project("CudaIsinIzleyici" LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(
    "./include/"
)
add_executable(main_cuda.out
    "src/main.cu"
    )

add_executable(main_c.out
    "src/main.cpp"
    )



set (CMAKE_CXX_FLAGS "-lstdc++ -ldl -ggdb -Wall -Wextra -pedantic -Wextra-tokens --std=c++17")



set(CMAKE_CUDA_FLAGS "-ccbin clang -g -gencode arch=compute_50,code=sm_50")

#target_compile_features(main PUBLIC cxx_std_11)

#set_target_properties( main_cuda.out
#    PROPERTIES CUDA_SEPARABLE_COMPILATION ON 
#                     )
set_target_properties(main_cuda.out
    PROPERTIES LINKER_LANGUAGE CUDA)

install(TARGETS main_cuda.out DESTINATION "${PROJECT_SOURCE_DIR}/bin")
install(TARGETS main_c.out DESTINATION "${PROJECT_SOURCE_DIR}/bin")
