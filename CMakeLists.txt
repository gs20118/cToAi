cmake_minimum_required(VERSION 3.25)
project(CLionProject)

set(CMAKE_CXX_STANDARD 20)

add_executable(CLionProject main.cpp tensor.h cailib.h operator.h cai.h range.h operators.h layer.h network.h optimizer.h)
