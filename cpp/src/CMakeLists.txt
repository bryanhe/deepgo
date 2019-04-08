find_package(Torch REQUIRED)

add_executable(train train.cpp datasets/kgs.cpp)
target_link_libraries(train "${TORCH_LIBRARIES}" stdc++fs pthread)
set_property(TARGET train PROPERTY CXX_STANDARD 11)
