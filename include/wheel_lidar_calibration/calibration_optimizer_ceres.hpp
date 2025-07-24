#ifndef WHEEL_LIDAR_CALIBRATION_CALIBRATION_OPTIMIZER_CERES_HPP_
#define WHEEL_LIDAR_CALIBRATION_CALIBRATION_OPTIMIZER_CERES_HPP_

#include <rclcpp/rclcpp.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "wheel_lidar_calibration/data_synchronizer.hpp"
#include "wheel_lidar_calibration/calibration_optimizer.hpp"

namespace wheel_lidar_calibration
{

// Cost function for position alignment
struct PositionAlignmentCost
{
  PositionAlignmentCost(const Eigen::Vector3d& wheel_pos, 
                       const Eigen::Vector3d& lidar_pos)
    : wheel_pos_(wheel_pos), lidar_pos_(lidar_pos) {}
  
  template <typename T>
  bool operator()(const T* const rotation, const T* const translation, T* residual) const
  {
    // Convert rotation parameters to rotation matrix
    T rotation_matrix[9];
    ceres::AngleAxisToRotationMatrix(rotation, rotation_matrix);
    
    // Apply transformation: wheel = R * lidar + t
    T transformed_lidar[3];
    transformed_lidar[0] = rotation_matrix[0] * T(lidar_pos_[0]) + 
                          rotation_matrix[1] * T(lidar_pos_[1]) + 
                          rotation_matrix[2] * T(lidar_pos_[2]) + translation[0];
    transformed_lidar[1] = rotation_matrix[3] * T(lidar_pos_[0]) + 
                          rotation_matrix[4] * T(lidar_pos_[1]) + 
                          rotation_matrix[5] * T(lidar_pos_[2]) + translation[1];
    transformed_lidar[2] = rotation_matrix[6] * T(lidar_pos_[0]) + 
                          rotation_matrix[7] * T(lidar_pos_[1]) + 
                          rotation_matrix[8] * T(lidar_pos_[2]) + translation[2];
    
    // Compute residuals
    residual[0] = transformed_lidar[0] - T(wheel_pos_[0]);
    residual[1] = transformed_lidar[1] - T(wheel_pos_[1]);
    residual[2] = transformed_lidar[2] - T(wheel_pos_[2]);
    
    return true;
  }
  
private:
  Eigen::Vector3d wheel_pos_;
  Eigen::Vector3d lidar_pos_;
};

// Motion-based constraint for relative transformations
struct MotionConstraintCost
{
  MotionConstraintCost(const Eigen::Matrix4d& wheel_relative,
                      const Eigen::Matrix4d& lidar_relative)
    : wheel_relative_(wheel_relative), lidar_relative_(lidar_relative) {}
  
  template <typename T>
  bool operator()(const T* const rotation, const T* const translation, T* residual) const
  {
    // Convert parameters to transformation matrix
    T R[9];
    ceres::AngleAxisToRotationMatrix(rotation, R);
    
    // Build transformation matrix T_wheel_lidar
    Eigen::Matrix<T, 4, 4> T_wl = Eigen::Matrix<T, 4, 4>::Identity();
    T_wl(0,0) = R[0]; T_wl(0,1) = R[1]; T_wl(0,2) = R[2]; T_wl(0,3) = translation[0];
    T_wl(1,0) = R[3]; T_wl(1,1) = R[4]; T_wl(1,2) = R[5]; T_wl(1,3) = translation[1];
    T_wl(2,0) = R[6]; T_wl(2,1) = R[7]; T_wl(2,2) = R[8]; T_wl(2,3) = translation[2];
    
    // Compute: wheel_relative - T_wl * lidar_relative * T_wl^(-1)
    Eigen::Matrix<T, 4, 4> T_wl_inv = T_wl.inverse();
    Eigen::Matrix<T, 4, 4> error = wheel_relative_.cast<T>() - 
                                   T_wl * lidar_relative_.cast<T>() * T_wl_inv;
    
    // Extract rotation and translation errors
    Eigen::Matrix<T, 3, 3> R_error = error.block<3,3>(0,0);
    Eigen::AngleAxis<T> aa_error(R_error);
    
    residual[0] = aa_error.angle() * aa_error.axis()[0];
    residual[1] = aa_error.angle() * aa_error.axis()[1];
    residual[2] = aa_error.angle() * aa_error.axis()[2];
    residual[3] = error(0,3);
    residual[4] = error(1,3);
    residual[5] = error(2,3);
    
    return true;
  }
  
private:
  Eigen::Matrix4d wheel_relative_;
  Eigen::Matrix4d lidar_relative_;
};

class CalibrationOptimizerCeres : public CalibrationOptimizer
{
public:
  CalibrationOptimizerCeres();
  
  // Override the base class optimize method
  virtual CalibrationResult optimize(const std::vector<OdometryPair>& odom_pairs) override;
  
  // Enable/disable motion constraints
  void setUseMotionConstraints(bool use) { use_motion_constraints_ = use; }
  
  // Set robustifier (Huber, Cauchy, etc.)
  void setLossFunction(const std::string& loss_type, double scale = 1.0);

private:
  bool use_motion_constraints_;
  std::string loss_function_type_;
  double loss_function_scale_;
  
  std::vector<std::pair<Eigen::Matrix4d, Eigen::Matrix4d>> 
  extractMotionConstraints(const std::vector<OdometryPair>& odom_pairs);
  
  Eigen::Matrix4d poseToTransform(const geometry_msgs::msg::Pose& pose);
};

} // namespace wheel_lidar_calibration

#endif // WHEEL_LIDAR_CALIBRATION_CALIBRATION_OPTIMIZER_CERES_HPP_