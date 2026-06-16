#!/bin/bash
set -e

# 用法:
#   ./build_local.sh          # 默认编译 x86
#   ./build_local.sh x86      # 编译 x86_64
#   ./build_local.sh cross    # 交叉编译 aarch64 (ARM)
#   ./build_local.sh clean    # 清理所有构建目录

export HOME_DIR=$(dirname $(readlink -f $0))

CROSS_COMPILE=OFF
CLEAN_FLAG=OFF
for opt in "$@"; do
    case "${opt}" in
        cross) CROSS_COMPILE=ON ;;
        x86)   CROSS_COMPILE=OFF ;;
        clean) CLEAN_FLAG=ON ;;
    esac
done

if [ "${CLEAN_FLAG}" == "ON" ]; then
    rm -rf ${HOME_DIR}/build_x86 ${HOME_DIR}/build_cross ${HOME_DIR}/bin ${HOME_DIR}/lib
    echo "已清理构建目录"
    exit 0
fi

COMM_ARGS="-DCMAKE_BUILD_TYPE=Release"

if [ "${CROSS_COMPILE}" == "ON" ]; then
    BUILD_DIR=${HOME_DIR}/build_cross
    TOOLCHAIN=${HOME_DIR}/cmake/arm_toolchain.cmake
    COMM_ARGS="${COMM_ARGS} -DENABLE_CROSSCOMPILE=ON -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN}"
    BIN=face_detect_arm
    echo "============> 交叉编译 aarch64 (ARM)"
else
    BUILD_DIR=${HOME_DIR}/build_x86
    COMM_ARGS="${COMM_ARGS} -DENABLE_CROSSCOMPILE=OFF"
    BIN=face_detect
    echo "============> 本地编译 x86_64"
fi

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}
echo "cmake ${COMM_ARGS}"
cmake ${COMM_ARGS} ${HOME_DIR}
make -j$(nproc)

echo "================================================"
echo "编译完成, 可执行文件: ${HOME_DIR}/bin/${BIN}"
echo "用法: ${HOME_DIR}/bin/${BIN} <输入图像> [输出图像] [级联文件.xml]"
echo "示例: ${HOME_DIR}/bin/${BIN} ${HOME_DIR}/data/lena.jpg ${HOME_DIR}/data/lena_detected.jpg"
