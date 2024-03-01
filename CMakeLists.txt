cmake_minimum_required(VERSION 3.26.4)

project(gTest)

# GoogleTest requires at least C++17
set(CMAKE_CXX_STANDARD 17)

# BLOOM
add_compile_options(-std=gnu++17  -lpthread)

add_library(bloom ../bloom/bloom.cpp)
add_library(torch ../torch/torch.cpp)
add_library(utils ../utils/utils.cpp)
add_library(tokenizer ../../bloom_tokenizer/tokenization.cpp)

include(FetchContent)
FetchContent_Declare(
    googletest 
    URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest) 

enable_testing()

add_executable(
    hello_test
    hello_test.cc
)

target_link_libraries(
    hello_test
    bloom
    torch
    utils
    tokenizer
    gtest_main
)

# add_executable(
#     hello_test
#     hello_test.cc
# )

# target_link_libraries(
#     hello_test
#     gtest_main
# )

include(GoogleTest)
gtest_discover_tests(hello_test)