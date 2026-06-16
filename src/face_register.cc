// face_register.cc
//
// 本地摄像头人脸注册 + 实时识别（小型人脸库）。
//
//   - YuNet 检测 + SFace 提取 128 维特征。
//   - 人脸库: 每条 = {人名/ID 标签, 128 维特征}, 持久化为 JSON(OpenCV FileStorage)。
//   - 实时识别: 对每张人脸按余弦相似度检索人脸库, 命中则绿框显示人名+相似度;
//     未命中则红框显示 "Unknown (Enter=register)"。
//   - 按【回车】对当前画面中最大的未知人脸进行注册(在终端输入人名)。
//   - 按【q / ESC】退出。
//
// 用法:
//   face_register [摄像头索引] [人脸库.json]
// 默认:
//   摄像头索引 = 0
//   人脸库     = <可执行文件>/../data/face_db.json
//
// 模型路径可用环境变量覆盖: FACE_YUNET_MODEL, FACE_SFACE_MODEL

#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/objdetect/face.hpp>

namespace {

// SFace 余弦相似度阈值: >= 该值判为同一人。
constexpr double kCosineThreshold = 0.5;

struct FaceEntry {
  std::string name;
  cv::Mat feature;  // 1x128 float
};

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

// 从 JSON 载入人脸库(文件不存在则返回空库)。
std::vector<FaceEntry> LoadDB(const std::string& path) {
  std::vector<FaceEntry> db;
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) return db;
  cv::FileNode faces = fs["faces"];
  for (cv::FileNodeIterator it = faces.begin(); it != faces.end(); ++it) {
    FaceEntry e;
    e.name = (std::string)(*it)["name"];
    (*it)["feature"] >> e.feature;
    if (!e.feature.empty()) db.push_back(e);
  }
  fs.release();
  return db;
}

// 把人脸库写回 JSON。
bool SaveDB(const std::string& path, const std::vector<FaceEntry>& db) {
  cv::FileStorage fs(path, cv::FileStorage::WRITE);
  if (!fs.isOpened()) return false;
  fs << "faces" << "[";
  for (const auto& e : db) {
    fs << "{" << "name" << e.name << "feature" << e.feature << "}";
  }
  fs << "]";
  fs.release();
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  const int cam_index = (argc >= 2) ? std::atoi(argv[1]) : 0;
  const std::string exe_dir = ExeDir();
  const std::string db_path =
      (argc >= 3) ? argv[2] : (exe_dir + "/../data/face_db.json");

  const std::string yunet_model = EnvOr(
      "FACE_YUNET_MODEL", exe_dir + "/../models/face_detection_yunet_2023mar.onnx");
  const std::string sface_model = EnvOr(
      "FACE_SFACE_MODEL", exe_dir + "/../models/face_recognition_sface_2021dec.onnx");

  // 模型
  cv::Ptr<cv::FaceDetectorYN> detector;
  cv::Ptr<cv::FaceRecognizerSF> recognizer;
  try {
    detector = cv::FaceDetectorYN::create(yunet_model, "", cv::Size(320, 320),
                                          0.9f, 0.3f, 5000);
    recognizer = cv::FaceRecognizerSF::create(sface_model, "");
  } catch (const cv::Exception& e) {
    std::cerr << "错误: 模型加载失败: " << e.what() << "\n";
    return 1;
  }

  // 人脸库
  std::vector<FaceEntry> db = LoadDB(db_path);
  std::cout << "已载入人脸库: " << db_path << " (" << db.size() << " 条)\n";

  // 摄像头
  cv::VideoCapture cap(cam_index);
  if (!cap.isOpened()) {
    std::cerr << "错误: 无法打开摄像头 " << cam_index << "\n";
    return 1;
  }
  std::cout << "操作: [回车]=注册当前未知人脸  [q/ESC]=退出\n";

  const std::string win = "faceID - register & recognize";
  cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

  cv::Mat frame;
  while (true) {
    if (!cap.read(frame) || frame.empty()) {
      std::cerr << "警告: 读取摄像头帧失败\n";
      break;
    }

    // 检测
    detector->setInputSize(frame.size());
    cv::Mat faces;
    detector->detect(frame, faces);

    // 记录当前帧中最大的未知人脸(用于注册)
    int unknown_idx = -1;
    int unknown_area = 0;
    cv::Mat unknown_feature;

    for (int i = 0; i < faces.rows; ++i) {
      const float* d = faces.ptr<float>(i);
      cv::Rect box(cvRound(d[0]), cvRound(d[1]), cvRound(d[2]), cvRound(d[3]));

      // 对齐 + 提取特征
      cv::Mat aligned, feat;
      recognizer->alignCrop(frame, faces.row(i), aligned);
      recognizer->feature(aligned, feat);
      feat = feat.clone();

      // 检索人脸库, 取最高余弦相似度
      double best_sim = -1.0;
      int best_j = -1;
      for (size_t j = 0; j < db.size(); ++j) {
        double sim = recognizer->match(feat, db[j].feature,
                                       cv::FaceRecognizerSF::FR_COSINE);
        if (sim > best_sim) {
          best_sim = sim;
          best_j = static_cast<int>(j);
        }
      }

      std::string label;
      cv::Scalar color;
      if (best_j >= 0 && best_sim >= kCosineThreshold) {
        char buf[128];
        snprintf(buf, sizeof(buf), "%s (%.2f)", db[best_j].name.c_str(), best_sim);
        label = buf;
        color = cv::Scalar(0, 255, 0);  // 命中: 绿
      } else {
        label = "Unknown (Enter=register)";
        color = cv::Scalar(0, 0, 255);  // 未知: 红
        int area = box.width * box.height;
        if (area > unknown_area) {
          unknown_area = area;
          unknown_idx = i;
          unknown_feature = feat;
        }
      }

      cv::rectangle(frame, box, color, 2);
      cv::putText(frame, label, cv::Point(box.x, std::max(box.y - 8, 14)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }

    // 左上角: 已注册名单(去重, 带样本数)
    {
      std::vector<std::string> names;
      std::vector<int> counts;
      for (const auto& e : db) {
        auto it = std::find(names.begin(), names.end(), e.name);
        if (it == names.end()) {
          names.push_back(e.name);
          counts.push_back(1);
        } else {
          counts[it - names.begin()]++;
        }
      }
      cv::putText(frame, "Registered:", cv::Point(10, 24),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
      for (size_t k = 0; k < names.size(); ++k) {
        char line[160];
        snprintf(line, sizeof(line), "%zu. %s (x%d)", k + 1, names[k].c_str(),
                 counts[k]);
        cv::putText(frame, line, cv::Point(10, 48 + static_cast<int>(k) * 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 0), 1);
      }
    }

    // 顶部状态栏
    char status[128];
    snprintf(status, sizeof(status), "DB: %zu faces  |  Enter=register  q=quit",
             db.size());
    cv::putText(frame, status, cv::Point(10, frame.rows - 12),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);

    cv::imshow(win, frame);
    int key = cv::waitKey(1) & 0xFF;

    if (key == 'q' || key == 27) {  // q / ESC
      break;
    } else if (key == '\r' || key == '\n' || key == 13 || key == 10) {  // 回车
      if (unknown_idx < 0 || unknown_feature.empty()) {
        std::cout << "当前没有可注册的未知人脸。\n";
        continue;
      }
      std::cout << "请输入该人脸的姓名/ID(同名可多次注册以追加样本; 回车确认, 留空取消): "
                << std::flush;
      std::string name;
      std::getline(std::cin, name);
      if (name.empty()) {
        std::cout << "已取消注册。\n";
        continue;
      }
      // 统计该姓名已有多少样本(同名追加多张)
      int same = 0;
      for (const auto& e : db) {
        if (e.name == name) ++same;
      }
      db.push_back(FaceEntry{name, unknown_feature.clone()});
      if (SaveDB(db_path, db)) {
        std::cout << "已注册: " << name << "  (该人第 " << (same + 1)
                  << " 个样本, 人脸库共 " << db.size() << " 条)\n";
      } else {
        std::cerr << "错误: 写入人脸库失败: " << db_path << "\n";
        db.pop_back();
      }
    }
  }

  cap.release();
  cv::destroyAllWindows();
  std::cout << "退出。人脸库共 " << db.size() << " 条, 文件: " << db_path << "\n";
  return 0;
}
