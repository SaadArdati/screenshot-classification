# Sequential implementation CMakeLists.txt
# Only requires C compiler, no CUDA dependencies

# Ensure stb_image.h is available
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add executable targets
add_executable(train_seq training.c)
add_executable(predict_seq predict.c)

# Link with math library (needed for pow, sqrt, etc.)
target_link_libraries(train_seq m)
target_link_libraries(predict_seq m)

# Install targets
install(TARGETS train_seq predict_seq
        RUNTIME DESTINATION bin)

