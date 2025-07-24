#include "wheel_lidar_calibration/observability_analyzer.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <numeric>
#include <sstream>

namespace wheel_lidar_calibration
{

std::string ObservabilityMetrics::getSummary() const
{
  std::stringstream ss;
  ss << "=== Observability Analysis ===\n";
  ss << "Overall observability: " << (is_observable ? "GOOD" : "POOR") << "\n";
  ss << "Rotation observability: " << std::fixed << std::setprecision(3) << rotation_observability << "\n";
  ss << "Translation observability: " << translation_observability << "\n";
  ss << "Condition number: " << condition_number << "\n";
  ss << "Total rotation: " << total_rotation.norm() * 180.0 / M_PI << " degrees\n";
  ss << "Total translation: " << total_translation.norm() << " meters\n";
  ss << "Path length: " << path_length << " meters\n";
  
  if (!recommendations.empty()) {
    ss << "\nRecommendations:\n";
    for (const auto& rec : recommendations) {
      ss << "  - " << rec << "\n";
    }
  }
  
  return ss.str();
}

ObservabilityAnalyzer::ObservabilityAnalyzer()
  : rotation_threshold_(0.3),
    translation_threshold_(0.3),
    condition_threshold_(100.0),
    logger_(rclcpp::get_logger("observability_analyzer"))
{
}

ObservabilityMetrics ObservabilityAnalyzer::analyze(const std::vector<OdometryPair>& odom_pairs)
{
  ObservabilityMetrics metrics;
  
  if (odom_pairs.size() < 10) {
    metrics.is_observable = false;
    metrics.recommendations.push_back("Collect more data points (at least 10 required)");
    return metrics;
  }
  
  // Compute Fisher Information Matrix
  metrics.fisher_information_matrix = computeFisherInformation(odom_pairs);
  
  // Perform SVD on Fisher Information Matrix
  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(
    metrics.fisher_information_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  
  metrics.singular_values.resize(6);
  for (int i = 0; i < 6; ++i) {
    metrics.singular_values[i] = svd.singularValues()[i];
  }
  
  // Compute condition number
  metrics.condition_number = metrics.singular_values[0] / metrics.singular_values[5];
  
  // Analyze motion diversity
  analyzeMotionDiversity(odom_pairs, metrics);
  
  // Compute overall observability score
  double svd_score = 1.0 / (1.0 + std::log10(metrics.condition_number));
  metrics.overall_observability = 0.4 * metrics.rotation_observability + 
                                 0.4 * metrics.translation_observability + 
                                 0.2 * svd_score;
  
  // Determine if calibration is observable
  metrics.is_observable = (metrics.rotation_observability > rotation_threshold_ &&
                          metrics.translation_observability > translation_threshold_ &&
                          metrics.condition_number < condition_threshold_);
  
  // Generate recommendations
  generateRecommendations(metrics);
  
  RCLCPP_INFO(logger_, "Observability analysis complete:");
  RCLCPP_INFO(logger_, "%s", metrics.getSummary().c_str());
  
  return metrics;
}

Eigen::Matrix<double, 6, 6> ObservabilityAnalyzer::computeFisherInformation(
  const std::vector<OdometryPair>& odom_pairs)
{
  Eigen::Matrix<double, 6, 6> FIM = Eigen::Matrix<double, 6, 6>::Zero();
  
  // Extract trajectory points
  std::vector<Eigen::Vector3d> lidar_points;
  for (const auto& pair : odom_pairs) {
    lidar_points.push_back(Eigen::Vector3d(
      pair.lidar_odom.pose.pose.position.x,
      pair.lidar_odom.pose.pose.position.y,
      pair.lidar_odom.pose.pose.position.z
    ));
  }
  
  // For each measurement, compute Jacobian and add to FIM
  for (const auto& point : lidar_points) {
    // Jacobian of measurement w.r.t calibration parameters [rx, ry, rz, tx, ty, tz]
    Eigen::Matrix<double, 3, 6> J;
    
    // Partial derivatives w.r.t rotation (using small angle approximation)
    J(0, 0) = 0;          J(0, 1) = point.z();  J(0, 2) = -point.y();
    J(1, 0) = -point.z(); J(1, 1) = 0;          J(1, 2) = point.x();
    J(2, 0) = point.y();  J(2, 1) = -point.x(); J(2, 2) = 0;
    
    // Partial derivatives w.r.t translation
    J(0, 3) = 1.0; J(0, 4) = 0.0; J(0, 5) = 0.0;
    J(1, 3) = 0.0; J(1, 4) = 1.0; J(1, 5) = 0.0;
    J(2, 3) = 0.0; J(2, 4) = 0.0; J(2, 5) = 1.0;
    
    // Add to Fisher Information Matrix (assuming unit measurement noise)
    FIM += J.transpose() * J;
  }
  
  return FIM;
}

void ObservabilityAnalyzer::analyzeMotionDiversity(
  const std::vector<OdometryPair>& odom_pairs,
  ObservabilityMetrics& metrics)
{
  auto relative_motions = extractRelativeMotions(odom_pairs);
  
  // Initialize accumulators
  metrics.total_rotation = Eigen::Vector3d::Zero();
  metrics.total_translation = Eigen::Vector3d::Zero();
  metrics.path_length = 0.0;
  
  std::vector<Eigen::Matrix3d> rotations;
  std::vector<Eigen::Vector3d> translations;
  
  for (const auto& motion : relative_motions) {
    rotations.push_back(motion.first);
    translations.push_back(motion.second);
    
    // Accumulate total motion
    Eigen::AngleAxisd aa(motion.first);
    metrics.total_rotation += aa.angle() * aa.axis();
    metrics.total_translation += motion.second;
    metrics.path_length += motion.second.norm();
  }
  
  // Compute motion coverage
  metrics.rotation_space_coverage = computeRotationCoverage(rotations);
  metrics.translation_space_coverage = computeTranslationCoverage(translations);
  
  // Normalize observability scores
  double rotation_magnitude = metrics.total_rotation.norm();
  metrics.rotation_observability = std::min(1.0, rotation_magnitude / MIN_ROTATION_RAD);
  
  double translation_magnitude = metrics.total_translation.norm();
  metrics.translation_observability = std::min(1.0, translation_magnitude / MIN_TRANSLATION_M);
  
  // Consider path length for translation observability
  double path_score = std::min(1.0, metrics.path_length / MIN_PATH_LENGTH_M);
  metrics.translation_observability = 0.5 * metrics.translation_observability + 0.5 * path_score;
  
  // Combine with coverage scores
  metrics.rotation_observability = 0.7 * metrics.rotation_observability + 
                                  0.3 * metrics.rotation_space_coverage;
  metrics.translation_observability = 0.7 * metrics.translation_observability + 
                                     0.3 * metrics.translation_space_coverage;
}

double ObservabilityAnalyzer::computeRotationCoverage(
  const std::vector<Eigen::Matrix3d>& rotations)
{
  if (rotations.empty()) return 0.0;
  
  // Compute principal axes of rotation
  Eigen::Matrix3d rotation_covariance = Eigen::Matrix3d::Zero();
  
  for (const auto& R : rotations) {
    Eigen::AngleAxisd aa(R);
    Eigen::Vector3d axis_angle = aa.angle() * aa.axis();
    rotation_covariance += axis_angle * axis_angle.transpose();
  }
  
  rotation_covariance /= rotations.size();
  
  // Compute eigenvalues
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(rotation_covariance);
  Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
  
  // Coverage metric: ratio of smallest to largest eigenvalue
  double coverage = eigenvalues.minCoeff() / (eigenvalues.maxCoeff() + 1e-6);
  
  return std::min(1.0, coverage * 3.0); // Scale to 0-1
}

double ObservabilityAnalyzer::computeTranslationCoverage(
  const std::vector<Eigen::Vector3d>& translations)
{
  if (translations.empty()) return 0.0;
  
  // Compute covariance of translations
  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const auto& t : translations) {
    mean += t;
  }
  mean /= translations.size();
  
  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const auto& t : translations) {
    Eigen::Vector3d diff = t - mean;
    covariance += diff * diff.transpose();
  }
  covariance /= translations.size();
  
  // Compute eigenvalues
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(covariance);
  Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
  
  // Coverage metric
  double coverage = eigenvalues.minCoeff() / (eigenvalues.maxCoeff() + 1e-6);
  
  return std::min(1.0, coverage * 3.0);
}

void ObservabilityAnalyzer::generateRecommendations(ObservabilityMetrics& metrics)
{
  // Clear existing recommendations
  metrics.recommendations.clear();
  
  // Check rotation observability
  if (metrics.rotation_observability < rotation_threshold_) {
    metrics.recommendations.push_back(
      "Insufficient rotation motion. Perform more turning maneuvers (target: " +
      std::to_string(static_cast<int>(MIN_ROTATION_RAD * 180.0 / M_PI)) + 
      " degrees total rotation)");
  }
  
  if (metrics.rotation_space_coverage < 0.3) {
    metrics.recommendations.push_back(
      "Rotation motion is too uni-directional. Try rotating in different axes");
  }
  
  // Check translation observability
  if (metrics.translation_observability < translation_threshold_) {
    metrics.recommendations.push_back(
      "Insufficient translation motion. Drive longer distances (target: " +
      std::to_string(static_cast<int>(MIN_TRANSLATION_M)) + 
      " meters displacement, " +
      std::to_string(static_cast<int>(MIN_PATH_LENGTH_M)) + 
      " meters path length)");
  }
  
  if (metrics.translation_space_coverage < 0.3) {
    metrics.recommendations.push_back(
      "Translation motion is too linear. Try moving in different directions");
  }
  
  // Check condition number
  if (metrics.condition_number > condition_threshold_) {
    metrics.recommendations.push_back(
      "Poor numerical conditioning. The calibration problem is ill-conditioned");
  }
  
  // Check for specific motion patterns
  if (metrics.path_length > MIN_PATH_LENGTH_M && 
      metrics.total_rotation.norm() < MIN_ROTATION_RAD * 0.5) {
    metrics.recommendations.push_back(
      "Motion is mostly straight. Include more curved trajectories");
  }
  
  // If everything is good
  if (metrics.recommendations.empty()) {
    metrics.recommendations.push_back("Motion diversity is sufficient for reliable calibration");
  }
}

std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> 
ObservabilityAnalyzer::extractRelativeMotions(const std::vector<OdometryPair>& odom_pairs)
{
  std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> motions;
  
  for (size_t i = 1; i < odom_pairs.size(); ++i) {
    // Extract poses
    tf2::Quaternion q_prev, q_curr;
    tf2::fromMsg(odom_pairs[i-1].wheel_odom.pose.pose.orientation, q_prev);
    tf2::fromMsg(odom_pairs[i].wheel_odom.pose.pose.orientation, q_curr);
    
    Eigen::Quaterniond eigen_q_prev(q_prev.w(), q_prev.x(), q_prev.y(), q_prev.z());
    Eigen::Quaterniond eigen_q_curr(q_curr.w(), q_curr.x(), q_curr.y(), q_curr.z());
    
    // Compute relative rotation
    Eigen::Matrix3d R_relative = eigen_q_prev.toRotationMatrix().transpose() * 
                                eigen_q_curr.toRotationMatrix();
    
    // Compute relative translation
    Eigen::Vector3d t_prev(odom_pairs[i-1].wheel_odom.pose.pose.position.x,
                          odom_pairs[i-1].wheel_odom.pose.pose.position.y,
                          odom_pairs[i-1].wheel_odom.pose.pose.position.z);
    
    Eigen::Vector3d t_curr(odom_pairs[i].wheel_odom.pose.pose.position.x,
                          odom_pairs[i].wheel_odom.pose.pose.position.y,
                          odom_pairs[i].wheel_odom.pose.pose.position.z);
    
    Eigen::Vector3d t_relative = eigen_q_prev.toRotationMatrix().transpose() * (t_curr - t_prev);
    
    motions.push_back(std::make_pair(R_relative, t_relative));
  }
  
  return motions;
}

} // namespace wheel_lidar_calibration