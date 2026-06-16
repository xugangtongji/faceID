// face_register_isf.cpp
//
// InspireFace 版本: 本地摄像头人脸注册 + 实时识别(小型人脸库), 带【引导式自动注册】。
// 与 face_register.cc(OpenCV YuNet+SFace 版)功能一致, 便于对比效果。
//
//   - InspireFace(MNN/CPU) 检测+跟踪, 提取 512 维特征。
//   - 人脸库: 每条 = {人名/ID 标签, 512 维特征}, 持久化为 JSON(OpenCV FileStorage)。
//   - 实时识别: 余弦相似度检索人脸库; 命中绿框显示人名+相似度, 未命中红框提示注册。
//   - 引导式注册(对未识别人脸按回车启动): 依次引导 正面/左转/右转/抬头/低头/张嘴,
//     左右/抬低头用 InspireFace 的 3D 头部角度(yaw/pitch)判定, 张嘴用 5 关键点估计;
//     到位自动抓拍(也可按空格手动), 凑齐 6 张样本写入人脸库并提示“完成注册”。
//   - q / ESC 退出(注册中 ESC 取消本次注册)。
//
// 用法:
//   face_register_isf [摄像头索引] [人脸库.json]
// 默认:
//   摄像头索引 = 0
//   人脸库     = <可执行文件>/../data/face_db_isf.json
//   模型包     = <可执行文件>/../models/Pikachu  (环境变量 FACE_ISF_MODEL 可覆盖)

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <inspirecv/inspirecv.h>
#include <inspireface/inspireface.hpp>

namespace {

constexpr float kCosineThreshold = 0.5f;

constexpr int kHoldNeed = 6;
constexpr int kCooldown = 12;

// 姿态阈值。yaw/pitch 单位为度(来自 InspireFace 3D 角度)。
// 若左右/上下判反, 把对应 sign 改为 -1。
constexpr float kYawSign = 1.0f;
// 实测: 抬头=正 pitch, 低头=负 pitch, 故取 -1 使"抬头"对应 S_UP。
constexpr float kPitchSign = -1.0f;
constexpr float kYawFrontDeg = 8.0f;    // 正面: |yaw| 小于此
constexpr float kYawTurnDeg = 30.0f;    // 左/右转: |yaw| 大于此(±30°)
constexpr float kPitchFrontDeg = 10.0f; // 正面: |pitch| 小于此
constexpr float kPitchUpDeg = 15.0f;    // 抬头: pitch 小于 -此(-15°)
constexpr float kPitchDownDeg = 25.0f;  // 低头: pitch 大于 +此(+25°)
// 张嘴: 用人脸框高宽比(h/w)。张嘴下巴下垂使框变高 -> 比值变大。
// 以【正面步骤】抓拍时的高宽比为基线, 超过 baseline*kMouthRatio 判为张嘴(自校准)。
constexpr float kMouthRatio = 1.15f;        // 比正面基线高 15%
constexpr float kMouthAbsFallback = 1.25f;  // 未取得基线时的绝对回退阈值

enum Step { S_FRONT = 0, S_LEFT, S_RIGHT, S_UP, S_DOWN, S_MOUTH, S_NUM };
const char* kStepInstr[S_NUM] = {
    "Look at CAMERA (frontal)", "Turn head LEFT", "Turn head RIGHT",
    "Look UP", "Look DOWN", "Open your MOUTH"};
const char* kStepCN[S_NUM] = {"正面", "左转头", "右转头", "抬头", "低头", "张嘴"};

struct FaceEntry {
  std::string name;
  std::vector<float> feature;  // 512 维
};

struct Pose {
  float yaw;    // 度
  float pitch;  // 度
  float mouth;  // 嘴-鼻纵距/瞳距
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
  if (access(path.c_str(), R_OK) != 0) return db;
  cv::FileStorage fs(path, cv::FileStorage::READ);
  if (!fs.isOpened()) return db;
  cv::FileNode faces = fs["faces"];
  for (cv::FileNodeIterator it = faces.begin(); it != faces.end(); ++it) {
    FaceEntry e;
    e.name = (std::string)(*it)["name"];
    cv::Mat feat;
    (*it)["feature"] >> feat;
    if (!feat.empty()) {
      e.feature.assign((float*)feat.datastart, (float*)feat.dataend);
      db.push_back(e);
    }
  }
  fs.release();
  return db;
}

bool SaveDB(const std::string& path, const std::vector<FaceEntry>& db) {
  cv::FileStorage fs(path, cv::FileStorage::WRITE);
  if (!fs.isOpened()) return false;
  fs << "faces" << "[";
  for (const auto& e : db) {
    cv::Mat feat(1, static_cast<int>(e.feature.size()), CV_32F,
                 const_cast<float*>(e.feature.data()));
    fs << "{" << "name" << e.name << "feature" << feat << "}";
  }
  fs << "]";
  fs.release();
  return true;
}

// p.mouth 现为人脸框高宽比 h/w(张嘴下巴下垂时变大)。keyPoints 仍用于绘制。
Pose EstimatePose(const inspire::FaceTrackWrap& f) {
  Pose p;
  p.yaw = f.face3DAngle.yaw;
  p.pitch = f.face3DAngle.pitch;
  p.mouth = static_cast<float>(f.rect.height) /
            static_cast<float>(f.rect.width > 0 ? f.rect.width : 1);
  return p;
}

// baseline: 正面步骤抓拍到的高宽比基线(<=0 表示尚未取得, 用绝对回退阈值)。
bool PoseMatches(int step, const Pose& p, float baseline) {
  const float yaw = p.yaw * kYawSign;
  const float pitch = p.pitch * kPitchSign;
  switch (step) {
    case S_FRONT:
      return std::fabs(p.yaw) < kYawFrontDeg && std::fabs(p.pitch) < kPitchFrontDeg;
    case S_LEFT:
      return yaw > kYawTurnDeg;
    case S_RIGHT:
      return yaw < -kYawTurnDeg;
    case S_UP:
      return pitch < -kPitchUpDeg;
    case S_DOWN:
      return pitch > kPitchDownDeg;
    case S_MOUTH: {
      float thr = (baseline > 0.0f) ? baseline * kMouthRatio : kMouthAbsFallback;
      return p.mouth > thr && std::fabs(p.yaw) < 15.0f;
    }
  }
  return false;
}

void DrawLandmarks(cv::Mat& img, const inspire::FaceTrackWrap& f) {
  const cv::Scalar c[5] = {{0, 255, 255}, {0, 0, 255}, {255, 0, 255},
                           {0, 255, 0}, {255, 0, 0}};
  for (int k = 0; k < 5; ++k)
    cv::circle(img, cv::Point2f(f.keyPoints[k].x, f.keyPoints[k].y), 2, c[k], -1);
}

}  // namespace

int main(int argc, char** argv) {
  const int cam_index = (argc >= 2) ? std::atoi(argv[1]) : 0;
  const std::string exe_dir = ExeDir();
  const std::string db_path =
      (argc >= 3) ? argv[2] : (exe_dir + "/../data/face_db_isf.json");
  const std::string model_path =
      EnvOr("FACE_ISF_MODEL", exe_dir + "/../models/Pikachu");

  if (INSPIREFACE_CONTEXT->Reload(model_path) != 0) {
    std::cerr << "错误: 模型包加载失败: " << model_path << "\n";
    return 1;
  }

  inspire::CustomPipelineParameter param;
  param.enable_recognition = true;
  param.enable_face_pose = true;  // 开启 3D 头部角度(yaw/pitch)
  std::shared_ptr<inspire::Session> session(inspire::Session::CreatePtr(
      inspire::DETECT_MODE_ALWAYS_DETECT, /*max_detect_face=*/5, param,
      /*detect_level_px=*/320));
  if (!session) {
    std::cerr << "错误: 会话创建失败\n";
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

  const std::string win = "faceID (InspireFace) - register & recognize";
  cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

  bool reg = false;
  std::string reg_name;
  int reg_step = 0, hold = 0, cooldown = 0, done_flash = 0;
  float front_aspect = 0.0f;  // 正面步骤抓拍到的高宽比基线
  std::vector<std::vector<float>> reg_samples;

  // 姿态日志: 记录最大脸的 yaw/pitch/高宽比, 便于事后定阈值。
  const std::string log_path = exe_dir + "/../data/pose_log.csv";
  std::ofstream plog(log_path);
  if (plog) plog << "frame,mode,step,yaw,pitch,hw_ratio\n";
  long frame_no = 0;
  std::cout << "姿态日志: " << log_path << "\n";

  cv::Mat frame;
  while (true) {
    if (!cap.read(frame) || frame.empty()) {
      std::cerr << "警告: 读取摄像头帧失败\n";
      break;
    }

    inspirecv::FrameProcess process = inspirecv::FrameProcess::Create(
        frame.data, frame.rows, frame.cols, inspirecv::BGR, inspirecv::ROTATION_0);
    std::vector<inspire::FaceTrackWrap> results;
    session->FaceDetectAndTrack(process, results);

    // 最大人脸
    int big = -1, big_area = -1;
    for (size_t i = 0; i < results.size(); ++i) {
      int area = results[i].rect.width * results[i].rect.height;
      if (area > big_area) {
        big_area = area;
        big = static_cast<int>(i);
      }
    }

    // 记录最大脸的姿态(yaw/pitch/高宽比)到日志
    ++frame_no;
    if (plog && big >= 0) {
      Pose lp = EstimatePose(results[big]);
      plog << frame_no << "," << (reg ? "register" : "recognize") << ","
           << (reg ? reg_step : -1) << "," << lp.yaw << "," << lp.pitch << ","
           << lp.mouth << "\n";
      plog.flush();
    }

    int key = -1;

    if (!reg) {
      // ---------- 识别模式 ----------
      bool big_unknown = false;
      for (size_t i = 0; i < results.size(); ++i) {
        const inspire::FaceRect& r = results[i].rect;
        cv::Rect box(r.x, r.y, r.width, r.height);

        inspire::FaceEmbedding emb;
        std::string label;
        cv::Scalar color;
        if (session->FaceFeatureExtract(process, results[i], emb) == 0 &&
            !emb.embedding.empty()) {
          float best_sim = -1.0f;
          int best_j = -1;
          for (size_t j = 0; j < db.size(); ++j) {
            float sim = 0.0f;
            inspire::FeatureHubDB::CosineSimilarity(emb.embedding, db[j].feature, sim);
            if (sim > best_sim) {
              best_sim = sim;
              best_j = static_cast<int>(j);
            }
          }
          if (best_j >= 0 && best_sim >= kCosineThreshold) {
            char buf[128];
            snprintf(buf, sizeof(buf), "%s (%.2f)", db[best_j].name.c_str(), best_sim);
            label = buf;
            color = cv::Scalar(0, 255, 0);
          } else {
            label = "Unknown (ENTER=register)";
            color = cv::Scalar(0, 0, 255);
            if (static_cast<int>(i) == big) big_unknown = true;
          }
        } else {
          label = "?";
          color = cv::Scalar(0, 0, 255);
        }
        cv::rectangle(frame, box, color, 2);
        cv::putText(frame, label, cv::Point(box.x, std::max(box.y - 8, 14)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

        // 显示该脸的 3D 头部角度
        char ang[80];
        snprintf(ang, sizeof(ang), "yaw=%.1f pitch=%.1f roll=%.1f",
                 results[i].face3DAngle.yaw, results[i].face3DAngle.pitch,
                 results[i].face3DAngle.roll);
        cv::putText(frame, ang, cv::Point(box.x, box.y + box.height + 18),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
      }

      if (big_unknown) {
        cv::putText(frame, "Unrecognized. Register? press ENTER",
                    cv::Point(10, frame.rows - 40), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 0, 255), 2);
      }
      if (done_flash > 0) {
        cv::putText(frame, "Registration complete!",
                    cv::Point(frame.cols / 2 - 160, frame.rows / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 3);
        --done_flash;
      }

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
      cv::putText(frame, "Registered (InspireFace):", cv::Point(10, 24),
                  cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
      for (size_t k = 0; k < names.size(); ++k) {
        char line[160];
        snprintf(line, sizeof(line), "%zu. %s (x%d)", k + 1, names[k].c_str(),
                 counts[k]);
        cv::putText(frame, line, cv::Point(10, 48 + static_cast<int>(k) * 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 0), 1);
      }
      char status[128];
      snprintf(status, sizeof(status), "DB: %zu faces | ENTER=register  q=quit",
               db.size());
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
          front_aspect = 0.0f;
          reg_samples.clear();
          std::cout << "开始为 [" << reg_name
                    << "] 引导注册: 正面/左转/右转/抬头/低头/张嘴, 按提示移动头部。\n";
        }
      }
    } else {
      // ---------- 引导注册模式 ----------
      bool captured_now = false;
      if (big >= 0) {
        const inspire::FaceTrackWrap& f = results[big];
        const inspire::FaceRect& r = f.rect;
        cv::Rect box(r.x, r.y, r.width, r.height);
        Pose pose = EstimatePose(f);
        bool match = PoseMatches(reg_step, pose, front_aspect);

        cv::rectangle(frame, box,
                      match ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255), 2);
        DrawLandmarks(frame, f);

        if (cooldown > 0) {
          --cooldown;
        } else if (match) {
          ++hold;
          // 张嘴框增高信号弱、峰值短暂, 该步降低保持帧数要求
          int need = (reg_step == S_MOUTH) ? 2 : kHoldNeed;
          if (hold >= need) {
            inspire::FaceEmbedding emb;
            if (session->FaceFeatureExtract(process, results[big], emb) == 0 &&
                !emb.embedding.empty()) {
              reg_samples.push_back(emb.embedding);
              if (reg_step == S_FRONT) front_aspect = pose.mouth;  // 记基线
              std::cout << "  已抓拍[" << kStepCN[reg_step] << "] ("
                        << reg_samples.size() << "/" << S_NUM << ")\n";
              ++reg_step;
              hold = 0;
              cooldown = kCooldown;
              captured_now = true;
            }
          }
        } else {
          hold = 0;
        }

        char m[120];
        snprintf(m, sizeof(m), "yaw=%.1f pitch=%.1f hw=%.2f base=%.2f", pose.yaw,
                 pose.pitch, pose.mouth, front_aspect);
        cv::putText(frame, m, cv::Point(10, frame.rows - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
      } else {
        cv::putText(frame, "No face detected", cv::Point(10, frame.rows - 12),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
      }

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

      if (reg && key == 27) {
        std::cout << "已取消注册 [" << reg_name << "] (放弃 " << reg_samples.size()
                  << " 张抓拍)。\n";
        reg = false;
        reg_samples.clear();
      } else if (reg && !captured_now && key == ' ' && big >= 0 &&
                 reg_step < S_NUM) {
        inspire::FaceEmbedding emb;
        if (session->FaceFeatureExtract(process, results[big], emb) == 0 &&
            !emb.embedding.empty()) {
          reg_samples.push_back(emb.embedding);
          if (reg_step == S_FRONT)
            front_aspect = static_cast<float>(results[big].rect.height) /
                           static_cast<float>(results[big].rect.width > 0
                                                  ? results[big].rect.width
                                                  : 1);
          std::cout << "  手动抓拍[" << kStepCN[reg_step] << "] ("
                    << reg_samples.size() << "/" << S_NUM << ")\n";
          ++reg_step;
          hold = 0;
          cooldown = kCooldown;
        }
      }
    }
  }

  cap.release();
  cv::destroyAllWindows();
  std::cout << "退出。人脸库共 " << db.size() << " 条, 文件: " << db_path << "\n";
  return 0;
}
