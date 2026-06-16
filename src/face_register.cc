// face_register.cc
//
// 本地摄像头人脸注册 + 实时识别（小型人脸库），带【引导式自动注册】。
//
//   - YuNet 检测 + SFace 提取 128 维特征。
//   - 人脸库: 每条 = {人名/ID 标签, 128 维特征}, 持久化为 JSON(OpenCV FileStorage)。
//   - 实时识别: 每张人脸按余弦相似度检索人脸库; 命中绿框显示人名+相似度,
//     未命中红框提示 "Unknown (ENTER=register)"。
//   - 引导式注册(对未识别人脸按回车启动):
//       依次引导 正面 / 左转 / 右转 / 抬头 / 低头 / 张嘴,
//       用 5 个关键点估计头部姿态, 姿态到位自动抓拍(也可按空格手动抓拍),
//       凑齐 6 张样本后写入人脸库并提示“完成注册”。
//   - 按 q / ESC 退出(注册过程中 ESC 取消本次注册)。
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
#include <cmath>
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

// 引导注册参数
constexpr int kHoldNeed = 6;   // 姿态需稳定保持的帧数后才抓拍
constexpr int kCooldown = 12;  // 每抓拍一张后的冷却帧数(防连拍)

// 姿态判定阈值(可按现场微调)。若左右判反, 把 kYawSign 改为 -1.0f。
constexpr float kYawSign = 1.0f;
constexpr float kYawFront = 0.15f;   // 正面: |yaw| 小于此值
constexpr float kYawTurn = 0.28f;    // 左/右转: |yaw| 大于此值
constexpr float kPitchUp = 0.40f;    // 抬头: pitch 小于此值
constexpr float kPitchDown = 0.66f;  // 低头: pitch 大于此值
// 张嘴: 嘴角中点到鼻尖纵距/瞳距 大于此值。注意 YuNet 仅有左右嘴角(无上下唇),
// 张嘴检测较弱, 阈值偏经验; 必要时按【空格】手动抓拍该步。
constexpr float kMouthOpen = 0.80f;

enum Step { S_FRONT = 0, S_LEFT, S_RIGHT, S_UP, S_DOWN, S_MOUTH, S_NUM };
const char* kStepInstr[S_NUM] = {
    "Look at CAMERA (frontal)", "Turn head LEFT", "Turn head RIGHT",
    "Look UP", "Look DOWN", "Open your MOUTH"};
const char* kStepCN[S_NUM] = {"正面", "左转头", "右转头", "抬头", "低头", "张嘴"};

struct FaceEntry {
  std::string name;
  cv::Mat feature;  // 1x128 float
};

// 由 5 关键点估计的简易头部姿态
struct Pose {
  float yaw;    // 水平: 左右转头(正面≈0)
  float pitch;  // 纵向比例: 抬头(小)/低头(大)
  float mouth;  // 嘴-鼻纵距/瞳距: 张嘴时变大
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

std::vector<FaceEntry> LoadDB(const std::string& path) {
  std::vector<FaceEntry> db;
  if (access(path.c_str(), R_OK) != 0) return db;  // 文件不存在: 空库
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

// YuNet 行: [x,y,w,h, 右眼(4,5),左眼(6,7),鼻尖(8,9),右嘴角(10,11),左嘴角(12,13), score]
Pose EstimatePose(const float* d) {
  cv::Point2f reye(d[4], d[5]), leye(d[6], d[7]), nose(d[8], d[9]);
  cv::Point2f rmou(d[10], d[11]), lmou(d[12], d[13]);
  cv::Point2f eye_mid = (reye + leye) * 0.5f;
  cv::Point2f mou_mid = (rmou + lmou) * 0.5f;
  float iod = static_cast<float>(cv::norm(leye - reye)) + 1e-6f;
  float span = mou_mid.y - eye_mid.y;
  if (std::fabs(span) < 1e-3f) span = 1e-3f;
  Pose p;
  p.yaw = (nose.x - eye_mid.x) / iod;
  p.pitch = (nose.y - eye_mid.y) / span;
  p.mouth = (mou_mid.y - nose.y) / iod;
  return p;
}

bool PoseMatches(int step, const Pose& p) {
  const float yaw = p.yaw * kYawSign;
  switch (step) {
    case S_FRONT:
      return std::fabs(p.yaw) < kYawFront && p.pitch > 0.42f && p.pitch < 0.63f;
    case S_LEFT:
      return yaw > kYawTurn;
    case S_RIGHT:
      return yaw < -kYawTurn;
    case S_UP:
      return p.pitch < kPitchUp;
    case S_DOWN:
      return p.pitch > kPitchDown;
    case S_MOUTH:
      return p.mouth > kMouthOpen && std::fabs(p.yaw) < 0.20f;
  }
  return false;
}

void DrawLandmarks(cv::Mat& img, const float* d) {
  const cv::Scalar c[5] = {{0, 0, 255},   {0, 255, 255}, {255, 0, 255},
                           {255, 0, 0}, {0, 255, 0}};
  for (int k = 0; k < 5; ++k)
    cv::circle(img, cv::Point2f(d[4 + k * 2], d[5 + k * 2]), 2, c[k], -1);
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

  std::vector<FaceEntry> db = LoadDB(db_path);
  std::cout << "已载入人脸库: " << db_path << " (" << db.size() << " 条)\n";

  cv::VideoCapture cap(cam_index);
  if (!cap.isOpened()) {
    std::cerr << "错误: 无法打开摄像头 " << cam_index << "\n";
    return 1;
  }
  std::cout << "操作: [回车]=对未识别人脸启动引导注册  [空格]=手动抓拍  [q/ESC]=退出\n";

  const std::string win = "faceID - register & recognize";
  cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

  // 引导注册状态
  bool reg = false;
  std::string reg_name;
  int reg_step = 0, hold = 0, cooldown = 0, done_flash = 0;
  std::vector<cv::Mat> reg_samples;

  cv::Mat frame;
  while (true) {
    if (!cap.read(frame) || frame.empty()) {
      std::cerr << "警告: 读取摄像头帧失败\n";
      break;
    }

    detector->setInputSize(frame.size());
    cv::Mat faces;
    detector->detect(frame, faces);

    // 找最大人脸(注册时作为目标)
    int big = -1, big_area = -1;
    for (int i = 0; i < faces.rows; ++i) {
      const float* d = faces.ptr<float>(i);
      int area = cvRound(d[2]) * cvRound(d[3]);
      if (area > big_area) {
        big_area = area;
        big = i;
      }
    }

    int key = -1;  // 在末尾统一 waitKey

    if (!reg) {
      // ---------- 识别模式 ----------
      bool big_unknown = false;
      for (int i = 0; i < faces.rows; ++i) {
        const float* d = faces.ptr<float>(i);
        cv::Rect box(cvRound(d[0]), cvRound(d[1]), cvRound(d[2]), cvRound(d[3]));

        cv::Mat aligned, feat;
        recognizer->alignCrop(frame, faces.row(i), aligned);
        recognizer->feature(aligned, feat);
        feat = feat.clone();

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
          snprintf(buf, sizeof(buf), "%s (%.2f)", db[best_j].name.c_str(),
                   best_sim);
          label = buf;
          color = cv::Scalar(0, 255, 0);
        } else {
          label = "Unknown (ENTER=register)";
          color = cv::Scalar(0, 0, 255);
          if (i == big) big_unknown = true;
        }
        cv::rectangle(frame, box, color, 2);
        cv::putText(frame, label, cv::Point(box.x, std::max(box.y - 8, 14)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
      }

      // 未识别提示
      if (big_unknown) {
        cv::putText(frame, "Unrecognized. Register? press ENTER",
                    cv::Point(10, frame.rows - 40), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0, 0, 255), 2);
      }
      if (done_flash > 0) {
        cv::putText(frame, "Registration complete!",
                    cv::Point(frame.cols / 2 - 160, frame.rows / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 3);
        --done_flash;
      }

      // 左上角: 已注册名单
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
      char status[128];
      snprintf(status, sizeof(status),
               "DB: %zu faces | ENTER=register  q=quit", db.size());
      cv::putText(frame, status, cv::Point(10, frame.rows - 12),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);

      cv::imshow(win, frame);
      key = cv::waitKey(1) & 0xFF;

      if (key == 'q' || key == 27) break;
      if ((key == '\r' || key == '\n' || key == 13 || key == 10) && big_unknown) {
        std::cout << "检测到未注册人脸。请输入姓名/ID 开始引导注册(留空取消): "
                  << std::flush;
        std::getline(std::cin, reg_name);
        if (reg_name.empty()) {
          std::cout << "已取消注册。\n";
        } else {
          reg = true;
          reg_step = 0;
          hold = 0;
          cooldown = 0;
          reg_samples.clear();
          std::cout << "开始为 [" << reg_name
                    << "] 引导注册: 正面/左转/右转/抬头/低头/张嘴, 按提示移动头部。\n";
        }
      }
    } else {
      // ---------- 引导注册模式 ----------
      bool captured_now = false;
      if (big >= 0) {
        const float* d = faces.ptr<float>(big);
        cv::Rect box(cvRound(d[0]), cvRound(d[1]), cvRound(d[2]), cvRound(d[3]));
        Pose pose = EstimatePose(d);
        bool match = PoseMatches(reg_step, pose);

        cv::Scalar bcolor = match ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
        cv::rectangle(frame, box, bcolor, 2);
        DrawLandmarks(frame, d);

        if (cooldown > 0) {
          --cooldown;
        } else if (match) {
          ++hold;
          if (hold >= kHoldNeed) {
            cv::Mat aligned, feat;
            recognizer->alignCrop(frame, faces.row(big), aligned);
            recognizer->feature(aligned, feat);
            reg_samples.push_back(feat.clone());
            std::cout << "  已抓拍[" << kStepCN[reg_step] << "] ("
                      << reg_samples.size() << "/" << S_NUM << ")\n";
            ++reg_step;
            hold = 0;
            cooldown = kCooldown;
            captured_now = true;
          }
        } else {
          hold = 0;
        }

        // 实时姿态数值(便于现场校准阈值)
        char m[96];
        snprintf(m, sizeof(m), "yaw=%.2f pitch=%.2f mouth=%.2f", pose.yaw,
                 pose.pitch, pose.mouth);
        cv::putText(frame, m, cv::Point(10, frame.rows - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
      } else {
        cv::putText(frame, "No face detected", cv::Point(10, frame.rows - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
      }

      // 完成判定
      if (reg_step >= S_NUM) {
        for (const auto& f : reg_samples) db.push_back(FaceEntry{reg_name, f});
        if (SaveDB(db_path, db)) {
          std::cout << "完成注册: " << reg_name << " (+" << reg_samples.size()
                    << " 张, 人脸库共 " << db.size() << " 条)\n";
        } else {
          std::cerr << "错误: 写入人脸库失败: " << db_path << "\n";
        }
        reg = false;
        done_flash = 30;
      } else {
        // 引导提示
        char head[160];
        snprintf(head, sizeof(head), "REGISTER [%s]  step %d/%d", reg_name.c_str(),
                 reg_step + 1, S_NUM);
        cv::putText(frame, head, cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 255), 2);
        cv::putText(frame, kStepInstr[reg_step], cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 255), 2);
        std::string tip = (hold > 0) ? "Hold steady..." : "SPACE=manual  ESC=cancel";
        cv::putText(frame, tip, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 0), 2);
      }

      cv::imshow(win, frame);
      key = cv::waitKey(1) & 0xFF;

      if (reg && key == 27) {  // ESC 取消本次注册
        std::cout << "已取消注册 [" << reg_name << "] (放弃 " << reg_samples.size()
                  << " 张抓拍)。\n";
        reg = false;
        reg_samples.clear();
      } else if (reg && !captured_now && key == ' ' && big >= 0 &&
                 reg_step < S_NUM) {  // 空格手动抓拍
        const float* d = faces.ptr<float>(big);
        cv::Mat aligned, feat;
        recognizer->alignCrop(frame, faces.row(big), aligned);
        recognizer->feature(aligned, feat);
        reg_samples.push_back(feat.clone());
        std::cout << "  手动抓拍[" << kStepCN[reg_step] << "] ("
                  << reg_samples.size() << "/" << S_NUM << ")\n";
        ++reg_step;
        hold = 0;
        cooldown = kCooldown;
      }
    }
  }

  cap.release();
  cv::destroyAllWindows();
  std::cout << "退出。人脸库共 " << db.size() << " 条, 文件: " << db_path << "\n";
  return 0;
}
