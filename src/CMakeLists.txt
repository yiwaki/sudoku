# cmake -S . -B ../build
# cmake --build ../build --config Debug --target bruteforce -j 10
cmake_minimum_required(VERSION 3.8.0)
project(sudoku VERSION 1.0.0)

add_executable(bruteforce bruteforce.c bitmap.c matrix.c)
target_compile_features(bruteforce PRIVATE c_std_11)
target_compile_options(bruteforce PRIVATE -Wall)
target_compile_definitions(bruteforce PRIVATE DEBUG)
