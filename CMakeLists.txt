cmake_minimum_required(VERSION 3.10)
project(faceID CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -fPIC -Wall -Wno-unused-function -Wno-unused-variable")

option(ENABLE_CROSSCOMPILE "Enable aarch64(ARM) cross compile" OFF)
message(STATUS "============ ENABLE_CROSSCOMPILE: ${ENABLE_CROSSCOMPILE}")

# Output dirs (mirror the aimvision_prediction layout)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# 仅依赖以下 OpenCV 模块, x86 与 arm 通用
set(FACE_OPENCV_LIBS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_objdetect
)

if(NOT ${ENABLE_CROSSCOMPILE})
    # ---------- x86_64: 使用系统 OpenCV (find_package) ----------
    message(STATUS "build for x86_64")
    find_package(OpenCV REQUIRED)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    set(FACE_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
    set(FACE_LINK_LIBS ${OpenCV_LIBS})
    set(BIN_NAME face_detect)
else()
    # ---------- aarch64: 复用参考工程 thor 内的 ARM OpenCV 3.4.12 ----------
    message(STATUS "build for aarch64 (ARM)")
    # 可通过 -DARM_OPENCV_ROOT=/path/to/opencv 覆盖
    set(ARM_OPENCV_ROOT "/home/xugang/aimvision_prediction/thirdparty/thor/opencv"
        CACHE PATH "ARM prebuilt OpenCV root (include/ and lib/)")
    if(NOT EXISTS ${ARM_OPENCV_ROOT}/include/opencv2/objdetect.hpp)
        message(FATAL_ERROR "ARM OpenCV 未找到: ${ARM_OPENCV_ROOT} (用 -DARM_OPENCV_ROOT= 指定)")
    endif()
    message(STATUS "ARM_OPENCV_ROOT: ${ARM_OPENCV_ROOT}")
    set(FACE_INCLUDE_DIRS ${ARM_OPENCV_ROOT}/include)
    link_directories(${ARM_OPENCV_ROOT}/lib)
    # 运行时在目标板上从同目录加载 .so
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(FACE_LINK_LIBS ${FACE_OPENCV_LIBS} pthread dl)
    set(BIN_NAME face_detect_arm)
endif()

add_executable(${BIN_NAME} src/face_detect.cc)
target_include_directories(${BIN_NAME} PRIVATE ${FACE_INCLUDE_DIRS})
target_link_libraries(${BIN_NAME} ${FACE_LINK_LIBS})

# face_embed: YuNet 检测 + SFace 128 维特征提取 + JSON 存储。
# 依赖 OpenCV >= 4.5 的 FaceDetectorYN/FaceRecognizerSF(dnn),仅 x86 构建。
if(NOT ${ENABLE_CROSSCOMPILE})
    add_executable(face_embed src/face_embed.cc)
    target_include_directories(face_embed PRIVATE ${FACE_INCLUDE_DIRS})
    target_link_libraries(face_embed ${OpenCV_LIBS})

    # face_register: 摄像头实时人脸注册 + 余弦相似度识别(需 highgui/videoio)
    add_executable(face_register src/face_register.cc)
    target_include_directories(face_register PRIVATE ${FACE_INCLUDE_DIRS})
    target_link_libraries(face_register ${OpenCV_LIBS})
endif()
