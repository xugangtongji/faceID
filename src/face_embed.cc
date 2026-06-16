// face_embed.cc
//
// 加载一幅图像 -> YuNet 人脸检测 -> SFace 对齐并提取 128 维特征向量
//   -> 在图上绘制检测框 -> 保存标注图 -> 把特征向量+元信息写入 JSON 文件。
//
// 用法:
//   face_embed <输入图像> [输出JSON] [标注图]
//
// 默认:
//   输出JSON = <输入图像>.faces.json
//   标注图   = <输入图像>_detected.<后缀>
//
// 模型默认从可执行文件同级的 ../models 加载, 可用环境变量覆盖:
//   FACE_YUNET_MODEL, FACE_SFACE_MODEL

#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/face.hpp>

namespace {

// 在 path 的扩展名前插入 suffix。
std::string InsertBeforeExt(const std::string& path, const std::string& suffix) {
  size_t slash = path.find_last_of("/\\");
  size_t dot = path.find_last_of('.');
  if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
    return path + suffix;
  }
  return path.substr(0, dot) + suffix + path.substr(dot);
}

// 取可执行文件所在目录(用于定位默认模型)。
std::string ExeDir() {
  char buf[4096];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return ".";
  std::string p(buf, n);
  size_t slash = p.find_last_of('/');
  return (slash == std::string::npos) ? "." : p.substr(0, slash);
}

std::string EnvOr(const char* key, const std::string& fallback) {
  const char* v = std::getenv(key);
  return (v && *v) ? std::string(v) : fallback;
}

// 将一行特征写成 JSON 数组字符串: [0.1,-0.2,...]
std::string FeatureToJson(const cv::Mat& feat) {
  std::string s = "[";
  for (int i = 0; i < feat.cols; ++i) {
    if (i) s += ",";
    char buf[32];
    snprintf(buf, sizeof(buf), "%.6f", feat.at<float>(0, i));
    s += buf;
  }
  s += "]";
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "用法: " << argv[0] << " <输入图像> [输出JSON] [标注图]\n";
    return 1;
  }

  const std::string input_path = argv[1];
  const std::string json_path =
      (argc >= 3) ? argv[2] : (input_path + ".faces.json");
  const std::string annotated_path =
      (argc >= 4) ? argv[3] : InsertBeforeExt(input_path, "_detected");

  const std::string exe_dir = ExeDir();
  const std::string yunet_model = EnvOr(
      "FACE_YUNET_MODEL", exe_dir + "/../models/face_detection_yunet_2023mar.onnx");
  const std::string sface_model = EnvOr(
      "FACE_SFACE_MODEL", exe_dir + "/../models/face_recognition_sface_2021dec.onnx");

  // 1. 加载图像
  cv::Mat image = cv::imread(input_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "错误: 无法加载图像: " << input_path << "\n";
    return 1;
  }
  std::cout << "已加载图像: " << input_path << " (" << image.cols << "x"
            << image.rows << ")\n";

  // 2. 创建 YuNet 检测器与 SFace 识别器
  cv::Ptr<cv::FaceDetectorYN> detector;
  cv::Ptr<cv::FaceRecognizerSF> recognizer;
  try {
    detector = cv::FaceDetectorYN::create(yunet_model, "",
                                          cv::Size(image.cols, image.rows),
                                          /*score_threshold=*/0.9f,
                                          /*nms_threshold=*/0.3f,
                                          /*top_k=*/5000);
    recognizer = cv::FaceRecognizerSF::create(sface_model, "");
  } catch (const cv::Exception& e) {
    std::cerr << "错误: 模型加载失败: " << e.what() << "\n"
              << "  YuNet: " << yunet_model << "\n"
              << "  SFace: " << sface_model << "\n";
    return 1;
  }

  // 3. 检测人脸。faces 每行: [x, y, w, h, 5个关键点(10), score]
  cv::Mat faces;
  detector->detect(image, faces);
  const int n = faces.rows;
  std::cout << "检测到人脸数量: " << n << "\n";

  // 4. 逐张人脸: 对齐裁剪 -> 提取 128 维特征
  std::vector<cv::Mat> features;  // 每个为 1x128 float
  features.reserve(n);
  for (int i = 0; i < n; ++i) {
    cv::Mat aligned;
    recognizer->alignCrop(image, faces.row(i), aligned);
    cv::Mat feat;
    recognizer->feature(aligned, feat);
    features.push_back(feat.clone());  // feature() 复用内部缓冲, 必须 clone

    // 绘制检测框 + 序号
    const float* d = faces.ptr<float>(i);
    cv::Rect box(cvRound(d[0]), cvRound(d[1]), cvRound(d[2]), cvRound(d[3]));
    cv::rectangle(image, box, cv::Scalar(0, 255, 0), 2);
    cv::putText(image, std::to_string(i + 1),
                cv::Point(box.x, std::max(box.y - 6, 12)),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
  }

  // 5. 保存标注图
  if (!cv::imwrite(annotated_path, image)) {
    std::cerr << "错误: 无法保存标注图: " << annotated_path << "\n";
    return 1;
  }
  std::cout << "标注图已保存: " << annotated_path << "\n";

  // 6. 写出 JSON: 图像信息 + 每张人脸的框/分数/128维特征
  std::ofstream ofs(json_path);
  if (!ofs) {
    std::cerr << "错误: 无法写入 JSON: " << json_path << "\n";
    return 1;
  }
  ofs << "{\n";
  ofs << "  \"image\": \"" << input_path << "\",\n";
  ofs << "  \"width\": " << image.cols << ",\n";
  ofs << "  \"height\": " << image.rows << ",\n";
  ofs << "  \"feature_dim\": "
      << (features.empty() ? 0 : features[0].cols) << ",\n";
  ofs << "  \"faces\": [\n";
  for (int i = 0; i < n; ++i) {
    const float* d = faces.ptr<float>(i);
    ofs << "    {\n";
    ofs << "      \"id\": " << i << ",\n";
    ofs << "      \"box\": [" << cvRound(d[0]) << "," << cvRound(d[1]) << ","
        << cvRound(d[2]) << "," << cvRound(d[3]) << "],\n";
    char score[32];
    snprintf(score, sizeof(score), "%.4f", d[14]);
    ofs << "      \"score\": " << score << ",\n";
    ofs << "      \"embedding\": " << FeatureToJson(features[i]) << "\n";
    ofs << "    }" << (i + 1 < n ? "," : "") << "\n";
  }
  ofs << "  ]\n";
  ofs << "}\n";
  ofs.close();
  std::cout << "特征已写入: " << json_path << " (" << n << " 张人脸, "
            << (features.empty() ? 0 : features[0].cols) << " 维)\n";

  return 0;
}
