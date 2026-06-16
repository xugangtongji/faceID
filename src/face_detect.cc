// face_detect.cc
//
// 加载一幅图像 -> OpenCV Haar 级联人脸检测 -> 在原图上绘制检测框 -> 保存结果。
//
// 用法:
//   face_detect <输入图像> [输出图像] [级联文件.xml]
//
// 默认:
//   输出图像   = <输入图像>_detected.<后缀>
//   级联文件   = /usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect.hpp>

namespace {

const char* kDefaultCascade =
    "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";

// 在 path 的扩展名前插入 suffix，生成默认输出路径。
std::string MakeDefaultOutput(const std::string& path, const std::string& suffix) {
  size_t slash = path.find_last_of("/\\");
  size_t dot = path.find_last_of('.');
  // 点必须在最后一个路径分隔符之后才算扩展名
  if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) {
    return path + suffix;
  }
  return path.substr(0, dot) + suffix + path.substr(dot);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "用法: " << argv[0]
              << " <输入图像> [输出图像] [级联文件.xml]\n";
    return 1;
  }

  const std::string input_path = argv[1];
  const std::string output_path =
      (argc >= 3) ? argv[2] : MakeDefaultOutput(input_path, "_detected");
  const std::string cascade_path = (argc >= 4) ? argv[3] : kDefaultCascade;

  // 1. 加载图像
  cv::Mat image = cv::imread(input_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "错误: 无法加载图像: " << input_path << "\n";
    return 1;
  }
  std::cout << "已加载图像: " << input_path << " (" << image.cols << "x"
            << image.rows << ")\n";

  // 2. 加载级联分类器
  cv::CascadeClassifier face_cascade;
  if (!face_cascade.load(cascade_path)) {
    std::cerr << "错误: 无法加载级联文件: " << cascade_path << "\n";
    return 1;
  }

  // 3. 转灰度 + 直方图均衡，提升检测稳定性
  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  cv::equalizeHist(gray, gray);

  // 4. 人脸检测
  std::vector<cv::Rect> faces;
  face_cascade.detectMultiScale(gray, faces,
                                /*scaleFactor=*/1.1,
                                /*minNeighbors=*/5,
                                /*flags=*/0,
                                /*minSize=*/cv::Size(30, 30));
  std::cout << "检测到人脸数量: " << faces.size() << "\n";

  // 5. 绘制检测框与序号
  for (size_t i = 0; i < faces.size(); ++i) {
    const cv::Rect& r = faces[i];
    cv::rectangle(image, r, cv::Scalar(0, 255, 0), 2);
    cv::putText(image, std::to_string(i + 1),
                cv::Point(r.x, std::max(r.y - 6, 12)),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
  }

  // 6. 保存结果
  if (!cv::imwrite(output_path, image)) {
    std::cerr << "错误: 无法保存图像: " << output_path << "\n";
    return 1;
  }
  std::cout << "结果已保存: " << output_path << "\n";

  return 0;
}
