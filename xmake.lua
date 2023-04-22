set_project("AIRuntime")
set_version("0.0.1")
set_languages("c++17")
add_rules("mode.debug", "mode.release")

add_requires("doctest", "opencv", "fmt", "spdlog", "nlohmann_json")
add_packages("doctest", "opencv", "fmt", "spdlog", "nlohmann_json")

if is_mode("release") then
    add_defines("LOG_LEVEL=\"info\"")
else
    add_defines("LOG_LEVEL=\"info\"")
end

-- include CUDA
add_includedirs("$(env CUDA_PATH)/include")
add_linkdirs("$(env CUDA_PATH)/lib/x64")
add_links("cudart")

-- include TensorRT
add_includedirs("$(env TensorRTDir)/include")
add_linkdirs("$(env TensorRTDir)/lib")
add_links("nvinfer")
add_links("nvonnxparser")

-- cuda
add_cugencodes("native")
add_cugencodes("compute_75")
add_cugencodes("compute_86")

add_includedirs("src")
add_includedirs("src/trt")
add_files("src/**.cpp")


target("Main")
    set_kind("binary")
    add_files("main.cpp")

target("TestLogger")
    set_kind("binary")
    add_files("./test/logger_test.cpp")


