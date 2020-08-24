cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project("CudaIsinIzleyici" LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_COMPILER "clang")

include_directories(
    "./include/"
)
add_executable(
    oneweekend.out
    "src/oneweekend/main.cu"
)


add_executable(
    main_c.out
    "src/main.cpp"
)

find_package(CUDA REQUIRED)
#find_package(Thrust REQUIRED)
#thrust_create_target(Thrust)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-lstdc++ -ldl -ggdb -Wall -Wextra -pedantic -Wextra-tokens")


set(CMAKE_CUDA_FLAGS "-ccbin clang++ -g -gencode arch=compute_50,code=sm_50")

#target_compile_features(main PUBLIC cxx_std_11)

set_target_properties( oneweekend.out
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON 
                     )
set_target_properties(oneweekend.out
    PROPERTIES LINKER_LANGUAGE CUDA)


install(TARGETS oneweekend.out DESTINATION "${PROJECT_SOURCE_DIR}/bin")
install(TARGETS main_c.out DESTINATION "${PROJECT_SOURCE_DIR}/bin")
