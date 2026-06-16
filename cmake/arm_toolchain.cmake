# ARM (aarch64) 交叉编译工具链 —— 与 aimvision_prediction 原始工程一致
# 在 docker 内编译, 该 buildroot 工具链已就位。

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER  /opt/aarch64_toolchain/aarch64_gun_13.x/bin/aarch64-buildroot-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /opt/aarch64_toolchain/aarch64_gun_13.x/bin/aarch64-buildroot-linux-gnu-g++)
set(CMAKE_FIND_ROOT_PATH  /opt/aarch64_toolchain/aarch64_gun_13.x/aarch64-buildroot-linux-gnu/sysroot)
