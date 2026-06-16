# faceID

x86 人脸识别工具集，CMake 结构参考 `aimvision_prediction`。

- **face_register**（摄像头实时注册 + 识别，x86）：打开本地摄像头，实时检测人脸并按余弦相似度检索人脸库；命中显示人名，未命中按回车注册。
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
| 实时 | 每张人脸按余弦相似度检索人脸库，命中→绿框+人名+相似度；未命中→红框 `Unknown (Enter=register)` |
| 回车 | 对当前画面中**最大的未知人脸**注册：在终端输入姓名/ID，回车确认（留空取消）。**同名可多次注册以追加样本**（不同角度/光照），识别时取该人所有样本的最高相似度 |
| q / ESC | 退出 |

阈值：SFace 余弦相似度 `>= 0.5` 判为同一人（在 `src/face_register.cc` 的 `kCosineThreshold` 调整）。

```bash
./build_local.sh x86
bin/face_register          # 打开摄像头0，加载/新建 data/face_db.json
```

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
# → data/lena_detected.jpg (标注图) + data/lena.jpg.faces.json (特征向量)
```

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
