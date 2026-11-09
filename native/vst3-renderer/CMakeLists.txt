cmake_minimum_required(VERSION 3.25)

project(tone_workbench_vst3_renderer LANGUAGES C CXX OBJCXX)

if(NOT APPLE)
    message(FATAL_ERROR "The Soundshed renderer currently requires macOS.")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(NOT VST3_SDK_ROOT OR NOT EXISTS "${VST3_SDK_ROOT}/CMakeLists.txt")
    message(FATAL_ERROR "Set VST3_SDK_ROOT to a checked-out Steinberg VST3 SDK.")
endif()

set(SMTG_ENABLE_VSTGUI_SUPPORT OFF CACHE BOOL "" FORCE)
set(SMTG_ENABLE_VST3_PLUGIN_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SMTG_ENABLE_VST3_HOSTING_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SMTG_RUN_VST_VALIDATOR OFF CACHE BOOL "" FORCE)

add_subdirectory("${VST3_SDK_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/vst3-sdk" EXCLUDE_FROM_ALL)

set(SDK_ROOT "${VST3_SDK_ROOT}")
add_executable(tone-vst3-renderer
    src/main.cpp
    src/state_validation.mm
    "${SDK_ROOT}/public.sdk/source/common/memorystream.cpp"
    "${SDK_ROOT}/public.sdk/source/vst/hosting/plugprovider.cpp"
    "${SDK_ROOT}/public.sdk/source/vst/hosting/module_mac.mm"
)

set_source_files_properties(
    "${SDK_ROOT}/public.sdk/source/vst/hosting/module_mac.mm"
    PROPERTIES COMPILE_FLAGS "-fobjc-arc"
)

target_compile_features(tone-vst3-renderer PRIVATE cxx_std_17)
target_link_libraries(tone-vst3-renderer PRIVATE sdk_hosting sdk "-framework Cocoa")
target_include_directories(tone-vst3-renderer PRIVATE "${SDK_ROOT}")
set_target_properties(tone-vst3-renderer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
