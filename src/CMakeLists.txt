# cmake -S . -B ../build
# cmake --build ../build --config Debug --target bruteforce -j 10
cmake_minimum_required(VERSION 3.0.0)
project(sudoku VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(bruteforce bruteforce.c bitmap.c matrix.c)
target_compile_features(bruteforce PRIVATE c_std_11)
target_compile_options(bruteforce PRIVATE -Wall)
target_compile_definitions(bruteforce PRIVATE DEBUG)

add_test(bf_test COMMAND ../build/bruteforce CONFIGULATIONS Debug)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
