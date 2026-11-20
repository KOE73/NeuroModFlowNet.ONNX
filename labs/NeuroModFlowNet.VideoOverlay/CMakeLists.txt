cmake_minimum_required(VERSION 3.25)
project(NeuroModFlowNetVideoOverlay LANGUAGES CXX)

if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "This lab requires Windows x64.")
endif()

set(FFMPEG_ROOT "C:/ffmpeg/ffmpeg-9.0.2-full_build-shared" CACHE PATH "Shared FFmpeg SDK root")
find_path(FFMPEG_INCLUDE libavcodec/avcodec.h PATHS "${FFMPEG_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
foreach(component IN ITEMS avcodec avformat avutil avdevice)
    find_library(${component}_library ${component} PATHS "${FFMPEG_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
    list(APPEND ffmpeg_libraries "${${component}_library}")
endforeach()

add_executable(VideoOverlay main.cpp GpuOverlay.cpp GpuFrameTiming.cpp OverlayBenchmark.cpp
    VideoCodecs.cpp CompressedReplay.cpp CodecBenchmark.cpp ParallelCodecBenchmark.cpp CodecPreview.cpp)
target_compile_features(VideoOverlay PRIVATE cxx_std_20)
target_compile_definitions(VideoOverlay PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE)
target_include_directories(VideoOverlay SYSTEM PRIVATE "${FFMPEG_INCLUDE}")
target_link_libraries(VideoOverlay PRIVATE ${ffmpeg_libraries} d3d11 dxgi d2d1 dwrite ole32)
if(MSVC)
    target_compile_options(VideoOverlay PRIVATE /W4 /permissive- /utf-8)
endif()

file(GLOB ffmpeg_runtime "${FFMPEG_ROOT}/bin/*.dll")
add_custom_command(TARGET VideoOverlay POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ffmpeg_runtime} "$<TARGET_FILE_DIR:VideoOverlay>"
    VERBATIM)
