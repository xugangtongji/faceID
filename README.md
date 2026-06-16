# faceID

x86 人脸识别工具集，CMake 结构参考 `aimvision_prediction`。

- **face_register**（摄像头实时注册 + 识别，x86）：OpenCV YuNet + SFace（128 维）。打开本地摄像头，实时检测人脸并按余弦相似度检索人脸库；命中显示人名，未命中按回车注册。
- **face_register_isf**（同上，InspireFace 版，x86 CPU）：用 InspireFace（MNN/CPU）做检测 + 512 维特征，便于与 SFace 版**对比效果**。需先编译 InspireFace SDK（见下）。
- **face_embed**（单图特征提取，x86）：YuNet 检测 + SFace 提取 **128 维特征向量** + 绘框标注 + 向量化存储为 JSON。
- **face_detect**：基础版，OpenCV Haar 级联检测 + 绘框保存（x86/ARM 通用）。

> ARM 版本暂时搁置，当前聚焦 x86。

## 依赖

- x86_64：系统 OpenCV 4.x（本机 `/usr/local`，4.10.0），通过 `find_package(OpenCV)` 查找。
- aarch64：buildroot 工具链 `/opt/aarch64_toolchain/aarch64_gun_13.x` + 预编译 ARM OpenCV 3.4.12
  （默认复用 `aimvision_prediction/thirdparty/thor/opencv`，可用 `-DARM_OPENCV_ROOT=` 覆盖）。
  仅依赖 `opencv_core / imgproc / imgcodecs / objdetect`，3.4 与 4.x API 一致。
- CMake >= 3.10，C++17。

## 编译

```bash
./build_local.sh          # 默认 x86
./build_local.sh x86      # x86_64 -> bin/face_detect
./build_local.sh cross    # aarch64(ARM, docker 内) -> bin/face_detect_arm
./build_local.sh clean    # 清理
```

ARM OpenCV 路径覆盖示例：

```bash
cd build_cross && cmake -DENABLE_CROSSCOMPILE=ON \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/arm_toolchain.cmake \
  -DARM_OPENCV_ROOT=/your/arm/opencv ..
```

## 运行 face_register（摄像头实时注册 + 识别）

```bash
bin/face_register [摄像头索引] [人脸库.json]
```

- `摄像头索引` 默认 `0`（对应 `/dev/video0`）
- `人脸库` 默认 `../data/face_db.json`（OpenCV FileStorage JSON：每条含 `name` 标签 + 128 维 `feature`）

**交互**（窗口需有焦点接收按键，姓名在终端输入）：

| 操作 | 说明 |
|------|------|
| 实时 | 每张人脸按余弦相似度检索人脸库，命中→绿框+人名+相似度；未命中→红框 `Unknown` + 底部提示 `Unrecognized. Register? press ENTER` |
| 回车 | 对**最大的未识别人脸**启动**引导式注册**：终端输入姓名/ID（留空取消）后进入分步采集 |
| 空格 | 引导注册中，对当前步骤**手动抓拍**（自动判定不灵时用） |
| ESC | 引导注册中=取消本次注册；非注册时=退出 |
| q | 退出 |

**引导式注册流程**（按 5 关键点估计头部姿态，到位自动抓拍，凑齐 6 张后入库）：

1. 正面 → 左转头 → 右转头（左右转头，按定位点抓拍 正/左/右）
2. 抬头 → 低头
3. 张嘴（YuNet 仅 5 点、无上下唇，张嘴判定较弱，必要时用空格手动抓拍）

完成后提示 `Registration complete!`，6 张样本以同一姓名写入人脸库（多样本提升识别稳健性）。
屏幕底部显示实时 `yaw/pitch/mouth` 数值，便于现场校准 `src/face_register.cc` 顶部的姿态阈值
（`kYawFront/kYawTurn/kPitchUp/kPitchDown/kMouthOpen`；左右判反则把 `kYawSign` 改为 -1）。

阈值：SFace 余弦相似度 `>= 0.5` 判为同一人（`kCosineThreshold`）。

```bash
./build_local.sh x86
bin/face_register          # 打开摄像头0，加载/新建 data/face_db.json
```

## 运行 face_register_isf（InspireFace CPU 版，用于对比）

操作与 `face_register` 完全一致（含**引导式自动注册**：正面/左转/右转/抬头/低头/张嘴 6 步、空格手动、ESC 取消、完成提示），区别是引擎换成 InspireFace（MNN，纯 CPU），特征 **512 维**，人脸库存为 `data/face_db_isf.json`。

> 头部姿态判定更准：左右/抬低头直接用 InspireFace 输出的 **3D 头部角度**（yaw/pitch，单位度，开启了 `enable_face_pose`），而非 OpenCV 版的 5 点几何估计；张嘴仍用 5 关键点估计（偏弱，必要时空格手动）。
> 角度阈值在 `src/face_register_isf.cpp` 顶部（`kYawTurnDeg/kPitchUDDeg/...`；左右或上下判反则改 `kYawSign/kPitchSign`）。屏幕底部显示实时 `yaw/pitch/mouth`。

**前置：编译 InspireFace SDK**（一次性，CPU 版）：

```bash
cd insightface/cpp-package/inspireface
# 需要 cmake >= 3.20（可 pip install --user "cmake>=3.20"）
# 首次会克隆 3rdparty(含 MNN) 并需要模型包 Pikachu
bash command/download_models_general.sh          # 下载模型包到 test_res/pack/
PATH="$HOME/.local/bin:$PATH" bash command/build_linux_ubuntu18.sh
# 产物: build/inspireface-linux-x86-ubuntu18/InspireFace/{include,lib}
cp test_res/pack/Pikachu ../../../models/Pikachu  # 模型放到 faceID/models/
```

回到项目根目录重新 cmake，会自动检测到 SDK 并编译 `face_register_isf`：

```bash
./build_local.sh x86
bin/face_register_isf            # 摄像头0，模型默认 models/Pikachu
```

> SDK 安装目录可用 `-DISF_ROOT=` 覆盖；模型包路径可用环境变量 `FACE_ISF_MODEL` 覆盖。
> InspireFace 自带相似度阈值 0.48，本程序统一用 0.5。

## 运行 face_embed（人脸特征提取 + 向量化存储）

需要 `models/` 下两个 ONNX 模型（已随仓库下载）：
- `face_detection_yunet_2023mar.onnx`（YuNet 检测）
- `face_recognition_sface_2021dec.onnx`（SFace 识别，128 维）

```bash
bin/face_embed <输入图像> [输出JSON] [标注图]
```

- `输出JSON` 默认 `<输入图像>.faces.json`
- `标注图`   默认 `<输入图像>_detected.<后缀>`
- 模型路径默认取可执行文件同级的 `../models`，可用环境变量 `FACE_YUNET_MODEL` / `FACE_SFACE_MODEL` 覆盖。

示例：

```bash
bin/face_embed data/lena.jpg
# → data/lena_detected.jpg     (标注图: 检测框 + 5 个关键点)
# → data/lena_aligned_0.jpg    (Umeyama 对齐到 112x112 的标准姿态人脸, 每张脸一份)
# → data/lena.jpg.faces.json   (128 维特征向量)
```

对齐说明：用 YuNet 检出的 **5 个关键点**(右眼/左眼/鼻尖/右嘴角/左嘴角)对 ArcFace
112×112 标准模板做 **Umeyama 相似变换**(旋转+统一缩放+平移),`warpAffine` 得到摆正的
112×112 人脸并保存。标注图上会用 5 种颜色画出这 5 个关键点(编号 0~4)。
> SFace 的 `alignCrop` 内部用的是同一套 5 点相似变换,这里把它显式化并存图以便查看。

JSON 结构（每张人脸含框、分数、128 维 embedding）：

```json
{
  "image": "data/lena.jpg", "width": 512, "height": 512, "feature_dim": 128,
  "faces": [
    { "id": 0, "box": [208,183,146,207], "score": 0.9090, "embedding": [ ... 128 个 float ... ] }
  ]
}
```

## 运行 face_detect（基础 Haar 检测）

```bash
bin/face_detect <输入图像> [输出图像] [级联文件.xml]
```

- `输出图像` 默认 `<输入图像>_detected.<后缀>`
- `级联文件` 默认 `/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml`

示例（使用自带的 lena 图）：

```bash
bin/face_detect data/lena.jpg data/lena_detected.jpg
```

## 程序流程（src/face_detect.cc）

1. `cv::imread` 加载图像
2. 加载 Haar 级联分类器
3. 转灰度 + 直方图均衡
4. `detectMultiScale` 检测人脸
5. `cv::rectangle` 绘框 + 序号
6. `cv::imwrite` 保存结果
