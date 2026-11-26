// Core value types shared by the whole pipeline.
#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <vector>

namespace autoaim {

/// One light bar (a glowing vertical strip on an armour plate).
struct LightBar {
  cv::Point2f center;            // bar centre, full-frame pixels
  float length = 0.f;            // extent along the long axis, px
  float width = 0.f;             // extent along the short axis, px
  cv::Point2f axis{0.f, -1.f};   // unit vector along the long axis, pointing "up"
  cv::Point2f half{1.f, 0.f};    // unit vector along the short axis
  float angle_deg = 0.f;         // long-axis direction normalised to [-90, 90)
  float mean_val = 0.f;          // mean dominant-channel value along the centre line
  // Original minAreaRect of this bar. Kept so armourCornersFromLightBars() can be
  // fed the documented RotatedRect inputs without re-fitting anything.
  cv::RotatedRect rect;

  float ratio() const { return length / std::max(width, 1e-6f); }
};

/// An armour plate candidate: two bars plus the four plate corners.
struct Armor {
  // Corner order is fixed, clockwise from the top-left, and is what the overlay
  // labels P1..P4. It also defines the PnP object-point order, so the two must
  // always be changed together:
  //   0 / P1 = top-left      1 / P2 = top-right
  //   2 / P3 = bottom-right  3 / P4 = bottom-left
  std::array<cv::Point2f, 4> corners{};
  // Mean of the four corners. Overlay + tracker only: it is NEVER passed to PnP,
  // because handing the solver the centre instead of the corners would throw away
  // the perspective information and make the pose unobservable.
  cv::Point2f center{0.f, 0.f};
  int left_bar = -1;
  int right_bar = -1;
  float score = 0.f;
  float center_ratio = 0.f;  // centre distance / mean bar length
  float length_ratio = 0.f;  // longer bar / shorter bar
  float angle_diff = 0.f;    // degrees between the two bar axes
  float aspect = 0.f;        // corner-quad width / height, px
  // 四边形在图像上的像素宽度（板高外扩之前的值）。用来做"最小可信尺寸"判断：
  // 真实装甲板在 2 m 处约 160 px，背景误检往往只有 25~40 px，差一个数量级。
  float width_px = 0.f;
  bool has_pose = false;
};

/// Result of the PnP solve.
struct Pose {
  cv::Vec3d rvec{0, 0, 0};
  cv::Vec3d tvec{0, 0, 0};
  double reproj_err_px = 1e9;
  bool ok = false;

  double x() const { return tvec[0]; }
  double y() const { return tvec[1]; }
  double z() const { return tvec[2]; }
  double range() const { return cv::norm(tvec); }

  /// Plate normal expressed as yaw/pitch in the camera frame, degrees.
  void yawPitchDeg(double* yaw_deg, double* pitch_deg) const;
};

/// Per-frame timing breakdown, in milliseconds.
struct FrameTiming {
  double decode_ms = 0.0;
  double detect_ms = 0.0;   // colour split + threshold + morphology + contours
  double match_ms = 0.0;    // pairing + corner construction
  double pnp_ms = 0.0;
  double draw_ms = 0.0;
  double total_ms = 0.0;
};

/// Everything the pipeline produced for one frame.
struct FrameResult {
  std::vector<LightBar> bars;
  std::vector<Armor> armors;
  std::vector<Pose> poses;
  FrameTiming timing;
  // Temporal-confirmation state of the reported target (see temporal_filter.hpp).
  // A detection not seen for `confirm_hits` consecutive frames stays "tentative":
  // it is still drawn, but its coordinates are not treated as a fireable target.
  bool target_confirmed = false;
  int confirm_hits = 0;
  // 时序层选定的主目标下标；-1 表示"按原来的 score 顺序取"。
  // primaryIndex() 会优先返回它，于是绘制和 CSV 都自动跟随时序判定。
  int primary_override = -1;
  // 时序状态码（0 无 / 1 确认 / 2 续帧 / 3 次候选），写进 CSV 便于复盘
  int track_state = 0;

  int primaryIndex() const;  // best armour, or -1
  const Pose* primaryPose() const;
};

}  // namespace autoaim
