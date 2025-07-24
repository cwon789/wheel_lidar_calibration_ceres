#include "wheel_lidar_calibration/calibration_optimizer_ceres.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace wheel_lidar_calibration
{

CalibrationOptimizerCeres::CalibrationOptimizerCeres()
  : CalibrationOptimizer(),
    use_motion_constraints_(true),
    loss_function_type_("huber"),
    loss_function_scale_(0.1)
{
}

CalibrationResult CalibrationOptimizerCeres::optimize(const std::vector<OdometryPair>& odom_pairs)
{
  if (odom_pairs.size() < 10) {
    RCLCPP_ERROR(logger_, "Not enough odometry pairs for calibration. Need at least 10, got %zu", 
                 odom_pairs.size());
    result_.converged = false;
    return result_;
  }
  
  RCLCPP_INFO(logger_, "Starting Ceres-based calibration with %zu odometry pairs", odom_pairs.size());
  
  // Initial estimate using SVD (from base class)
  std::vector<Eigen::Vector3d> wheel_points = extractTrajectoryPoints(odom_pairs, true);
  std::vector<Eigen::Vector3d> lidar_points = extractTrajectoryPoints(odom_pairs, false);
  Eigen::Matrix4d initial_transform = computeTransformSVD(lidar_points, wheel_points);
  
  // Convert to angle-axis representation
  Eigen::AngleAxisd aa(initial_transform.block<3,3>(0,0));
  double rotation[3] = {aa.angle() * aa.axis()[0], 
                        aa.angle() * aa.axis()[1], 
                        aa.angle() * aa.axis()[2]};
  double translation[3] = {initial_transform(0,3), 
                          initial_transform(1,3), 
                          initial_transform(2,3)};
  
  // Setup Ceres problem
  ceres::Problem problem;
  ceres::LossFunction* loss_function = nullptr;
  
  // Configure loss function for robustness
  if (loss_function_type_ == "huber") {
    loss_function = new ceres::HuberLoss(loss_function_scale_);
  } else if (loss_function_type_ == "cauchy") {
    loss_function = new ceres::CauchyLoss(loss_function_scale_);
  } else if (loss_function_type_ == "arctan") {
    loss_function = new ceres::ArctanLoss(loss_function_scale_);
  }
  
  // Add position alignment constraints
  for (size_t i = 0; i < odom_pairs.size(); ++i) {
    ceres::CostFunction* cost_function = 
      new ceres::AutoDiffCostFunction<PositionAlignmentCost, 3, 3, 3>(
        new PositionAlignmentCost(wheel_points[i], lidar_points[i]));
    
    problem.AddResidualBlock(cost_function, loss_function, rotation, translation);
  }
  
  // Add motion constraints if enabled
  if (use_motion_constraints_ && odom_pairs.size() > 1) {
    auto motion_constraints = extractMotionConstraints(odom_pairs);
    
    RCLCPP_INFO(logger_, "Adding %zu motion constraints", motion_constraints.size());
    
    for (const auto& constraint : motion_constraints) {
      ceres::CostFunction* cost_function = 
        new ceres::AutoDiffCostFunction<MotionConstraintCost, 6, 3, 3>(
          new MotionConstraintCost(constraint.first, constraint.second));
      
      // Motion constraints typically have different weight
      ceres::LossFunction* motion_loss = new ceres::HuberLoss(loss_function_scale_ * 2.0);
      problem.AddResidualBlock(cost_function, motion_loss, rotation, translation);
    }
  }
  
  // Configure solver
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = max_iterations_;
  options.function_tolerance = convergence_threshold_;
  options.gradient_tolerance = convergence_threshold_ * 0.1;
  options.parameter_tolerance = convergence_threshold_ * 0.01;
  
  // Solve
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  
  RCLCPP_INFO(logger_, "Ceres optimization completed:");
  RCLCPP_INFO(logger_, "  Initial cost: %f", summary.initial_cost);
  RCLCPP_INFO(logger_, "  Final cost: %f", summary.final_cost);
  RCLCPP_INFO(logger_, "  Iterations: %d", static_cast<int>(summary.iterations.size()));
  RCLCPP_INFO(logger_, "  Termination: %s", 
              ceres::TerminationTypeToString(summary.termination_type));
  
  // Convert back to transformation matrix
  Eigen::AngleAxisd aa_result(Eigen::Vector3d(rotation[0], rotation[1], rotation[2]).norm(),
                              Eigen::Vector3d(rotation[0], rotation[1], rotation[2]).normalized());
  
  result_.transform_matrix = Eigen::Matrix4d::Identity();
  result_.transform_matrix.block<3,3>(0,0) = aa_result.toRotationMatrix();
  result_.transform_matrix.block<3,1>(0,3) = Eigen::Vector3d(translation[0], translation[1], translation[2]);
  
  result_.translation = result_.transform_matrix.block<3,1>(0,3);
  result_.rotation = Eigen::Quaterniond(result_.transform_matrix.block<3,3>(0,0));
  result_.rmse = std::sqrt(summary.final_cost / summary.num_residuals * 2.0);  // Approximate RMSE
  result_.iterations = summary.iterations.size();
  result_.converged = (summary.termination_type == ceres::CONVERGENCE);
  
  RCLCPP_INFO(logger_, "Calibration result:");
  RCLCPP_INFO(logger_, "  Translation: [%f, %f, %f]", 
              result_.translation.x(), result_.translation.y(), result_.translation.z());
  RCLCPP_INFO(logger_, "  Rotation (quaternion): [%f, %f, %f, %f]",
              result_.rotation.x(), result_.rotation.y(), result_.rotation.z(), result_.rotation.w());
  RCLCPP_INFO(logger_, "  Final RMSE: %f", result_.rmse);
  
  return result_;
}

std::vector<std::pair<Eigen::Matrix4d, Eigen::Matrix4d>> 
CalibrationOptimizerCeres::extractMotionConstraints(const std::vector<OdometryPair>& odom_pairs)
{
  std::vector<std::pair<Eigen::Matrix4d, Eigen::Matrix4d>> constraints;
  
  // Skip if motion is too small to avoid numerical issues
  const double min_translation = 0.05;  // 5cm
  const double min_rotation = 0.05;     // ~3 degrees
  
  for (size_t i = 1; i < odom_pairs.size(); ++i) {
    // Compute relative transformations
    Eigen::Matrix4d wheel_prev = poseToTransform(odom_pairs[i-1].wheel_odom.pose.pose);
    Eigen::Matrix4d wheel_curr = poseToTransform(odom_pairs[i].wheel_odom.pose.pose);
    Eigen::Matrix4d wheel_relative = wheel_prev.inverse() * wheel_curr;
    
    Eigen::Matrix4d lidar_prev = poseToTransform(odom_pairs[i-1].lidar_odom.pose.pose);
    Eigen::Matrix4d lidar_curr = poseToTransform(odom_pairs[i].lidar_odom.pose.pose);
    Eigen::Matrix4d lidar_relative = lidar_prev.inverse() * lidar_curr;
    
    // Check if motion is significant
    double translation_norm = wheel_relative.block<3,1>(0,3).norm();
    Eigen::AngleAxisd aa(wheel_relative.block<3,3>(0,0));
    double rotation_angle = std::abs(aa.angle());
    
    if (translation_norm > min_translation || rotation_angle > min_rotation) {
      constraints.push_back(std::make_pair(wheel_relative, lidar_relative));
    }
  }
  
  return constraints;
}

Eigen::Matrix4d CalibrationOptimizerCeres::poseToTransform(const geometry_msgs::msg::Pose& pose)
{
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  
  // Translation
  transform(0,3) = pose.position.x;
  transform(1,3) = pose.position.y;
  transform(2,3) = pose.position.z;
  
  // Rotation
  Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x, 
                       pose.orientation.y, pose.orientation.z);
  transform.block<3,3>(0,0) = q.toRotationMatrix();
  
  return transform;
}

void CalibrationOptimizerCeres::setLossFunction(const std::string& loss_type, double scale)
{
  loss_function_type_ = loss_type;
  loss_function_scale_ = scale;
}

} // namespace wheel_lidar_calibration