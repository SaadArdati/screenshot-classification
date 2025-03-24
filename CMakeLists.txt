cmake_minimum_required(VERSION 3.30)
project(image_classification C)

set(CMAKE_C_STANDARD 11)

add_executable(training training.c
        stb_image.h)

add_executable(predict predict.c
        stb_image.h)

target_link_libraries(training m)
target_link_libraries(predict m)

