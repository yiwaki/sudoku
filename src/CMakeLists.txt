# cmake -S . -B ../build
# cmake --build ../build --config Debug --target bruteforce -j 10
cmake_minimum_required(VERSION 3.0.0)
project(sudoku VERSION 0.1.0)

set(CMAKE_C_FLAGS "-std=c11 -DDEBUG -Wall")
add_executable(bruteforce bruteforce.c bitmap.c matrix.c)
