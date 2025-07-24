#include "wheel_lidar_calibration/calibration_validator.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <chrono>

namespace wheel_lidar_calibration
{

std::string ValidationResult::getSummary() const
{
  std::stringstream ss;
  ss << "=== Validation Result ===\n";
  ss << "Status: " << (is_valid ? "VALID" : "INVALID") << "\n";
  ss << "Confidence score: " << std::fixed << std::setprecision(3) << confidence_score << "\n";
  ss << "Mean error: " << mean_error << " m\n";
  ss << "RMSE: " << rmse << " m\n";
  ss << "Max error: " << max_error << " m\n";
  ss << "Std deviation: " << std_error << " m\n";
  ss << "Outlier ratio: " << outlier_ratio * 100 << "%\n";
  
  if (!warnings.empty()) {
    ss << "\nWarnings:\n";
    for (const auto& warning : warnings) {
      ss << "  ⚠ " << warning << "\n";
    }
  }
  
  if (!errors.empty()) {
    ss << "\nErrors:\n";
    for (const auto& error : errors) {
      ss << "  ✗ " << error << "\n";
    }
  }
  
  return ss.str();
}

CalibrationValidator::CalibrationValidator()
  : outlier_threshold_(0.2),  // 20cm default
    min_confidence_score_(0.7),
    max_acceptable_error_(0.1),  // 10cm
    enable_cross_validation_(true),
    enable_outlier_detection_(true),
    logger_(rclcpp::get_logger("calibration_validator")),
    max_history_size_(100)
{
}

ValidationResult CalibrationValidator::validate(
  const CalibrationResult& calibration,
  const std::vector<OdometryPair>& test_pairs)
{
  ValidationResult result;
  auto start_time = std::chrono::high_resolution_clock::now();
  
  if (test_pairs.empty()) {
    result.is_valid = false;
    result.errors.push_back("No test data provided for validation");
    return result;
  }
  
  // Compute errors for all test pairs
  result.error_values.reserve(test_pairs.size());
  for (const auto& pair : test_pairs) {
    double error = computePairError(calibration, pair);
    result.error_values.push_back(error);
  }
  
  // Compute error statistics
  computeErrorStatistics(result.error_values, result);
  
  // Detect outliers
  if (enable_outlier_detection_) {
    detectOutliers(result.error_values, result);
  }
  
  // Check consistency
  checkConsistency(calibration, result);
  
  // Create error histogram
  result.error_histogram = createErrorHistogram(result.error_values);
  
  // Compute confidence score
  result.confidence_score = computeConfidenceScore(result);
  
  // Determine overall validity
  result.is_valid = (result.confidence_score >= min_confidence_score_ &&
                    result.rmse <= max_acceptable_error_ &&
                    result.outlier_ratio < 0.2);  // Less than 20% outliers
  
  // Add warnings and errors
  if (result.rmse > max_acceptable_error_) {
    result.errors.push_back("RMSE exceeds acceptable threshold (" + 
                           std::to_string(max_acceptable_error_) + "m)");
  }
  
  if (result.outlier_ratio > 0.1) {
    result.warnings.push_back("High outlier ratio detected (" + 
                             std::to_string(result.outlier_ratio * 100) + "%)");
  }
  
  if (result.max_error > 3 * result.rmse) {
    result.warnings.push_back("Maximum error is significantly larger than RMSE");
  }
  
  if (!result.rotation_consistency || !result.translation_consistency) {
    result.warnings.push_back("Calibration parameters may be inconsistent");
  }
  
  auto end_time = std::chrono::high_resolution_clock::now();
  result.computation_time = std::chrono::duration<double>(end_time - start_time).count();
  result.validation_timestamp = rclcpp::Clock().now().seconds();
  
  RCLCPP_INFO(logger_, "Validation complete: %s", result.getSummary().c_str());
  
  return result;
}

ValidationResult CalibrationValidator::validateRealtime(
  const CalibrationResult& calibration,
  const OdometryPair& new_pair)
{
  ValidationResult result;
  
  // Compute error for the new pair
  double error = computePairError(calibration, new_pair);
  
  // Update running statistics
  recent_errors_.push_back(error);
  if (recent_errors_.size() > max_history_size_) {
    recent_errors_.pop_front();
  }
  
  // Convert deque to vector for statistics computation
  std::vector<double> error_vector(recent_errors_.begin(), recent_errors_.end());
  result.error_values = error_vector;
  
  // Compute statistics on recent errors
  computeErrorStatistics(error_vector, result);
  
  // Quick outlier check
  double median = result.mean_error;  // Approximate with mean for speed
  result.outlier_ratio = (std::abs(error - median) > outlier_threshold_) ? 1.0 : 0.0;
  
  // Simple confidence score
  result.confidence_score = std::exp(-result.rmse / max_acceptable_error_);
  
  // Validity check
  result.is_valid = (error < max_acceptable_error_ && 
                    result.rmse < max_acceptable_error_);
  
  if (!result.is_valid) {
    result.warnings.push_back("Real-time validation failed for current measurement");
  }
  
  result.validation_timestamp = rclcpp::Clock().now().seconds();
  
  return result;
}

ValidationResult CalibrationValidator::crossValidate(
  const std::vector<OdometryPair>& all_pairs,
  int k_folds)
{
  ValidationResult result;
  
  if (all_pairs.size() < static_cast<size_t>(k_folds)) {
    result.is_valid = false;
    result.errors.push_back("Not enough data for " + std::to_string(k_folds) + "-fold cross-validation");
    return result;
  }
  
  std::vector<double> fold_errors;
  int fold_size = all_pairs.size() / k_folds;
  
  CalibrationOptimizer optimizer;
  
  for (int k = 0; k < k_folds; ++k) {
    // Split data into training and validation sets
    std::vector<OdometryPair> training_set;
    std::vector<OdometryPair> validation_set;
    
    for (size_t i = 0; i < all_pairs.size(); ++i) {
      if (i >= static_cast<size_t>(k * fold_size) && 
          i < static_cast<size_t>((k + 1) * fold_size)) {
        validation_set.push_back(all_pairs[i]);
      } else {
        training_set.push_back(all_pairs[i]);
      }
    }
    
    // Train on training set
    CalibrationResult fold_calibration = optimizer.optimize(training_set);
    
    // Validate on validation set
    ValidationResult fold_result = validate(fold_calibration, validation_set);
    fold_errors.push_back(fold_result.rmse);
  }
  
  // Compute cross-validation statistics
  result.cross_validation_error = std::accumulate(fold_errors.begin(), fold_errors.end(), 0.0) / k_folds;
  
  // Use cross-validation error as main metric
  result.rmse = result.cross_validation_error;
  result.mean_error = result.cross_validation_error * 0.8;  // Approximate
  
  result.confidence_score = std::exp(-result.cross_validation_error / max_acceptable_error_);
  result.is_valid = (result.cross_validation_error < max_acceptable_error_);
  
  RCLCPP_INFO(logger_, "%d-fold cross-validation error: %f", k_folds, result.cross_validation_error);
  
  return result;
}

double CalibrationValidator::computePairError(
  const CalibrationResult& calibration,
  const OdometryPair& pair)
{
  // Extract positions
  Eigen::Vector3d wheel_pos(
    pair.wheel_odom.pose.pose.position.x,
    pair.wheel_odom.pose.pose.position.y,
    pair.wheel_odom.pose.pose.position.z
  );
  
  Eigen::Vector3d lidar_pos(
    pair.lidar_odom.pose.pose.position.x,
    pair.lidar_odom.pose.pose.position.y,
    pair.lidar_odom.pose.pose.position.z
  );
  
  // Transform lidar position using calibration
  Eigen::Vector4d lidar_homo(lidar_pos.x(), lidar_pos.y(), lidar_pos.z(), 1.0);
  Eigen::Vector4d transformed = calibration.transform_matrix * lidar_homo;
  Eigen::Vector3d transformed_pos = transformed.head<3>();
  
  // Compute error
  return (transformed_pos - wheel_pos).norm();
}

void CalibrationValidator::computeErrorStatistics(
  const std::vector<double>& errors,
  ValidationResult& result)
{
  if (errors.empty()) {
    result.mean_error = 0.0;
    result.rmse = 0.0;
    result.max_error = 0.0;
    result.std_error = 0.0;
    return;
  }
  
  // Mean
  result.mean_error = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
  
  // RMSE
  double sum_squared = 0.0;
  for (double error : errors) {
    sum_squared += error * error;
  }
  result.rmse = std::sqrt(sum_squared / errors.size());
  
  // Max
  result.max_error = *std::max_element(errors.begin(), errors.end());
  
  // Standard deviation
  double variance = 0.0;
  for (double error : errors) {
    variance += std::pow(error - result.mean_error, 2);
  }
  result.std_error = std::sqrt(variance / errors.size());
}

void CalibrationValidator::detectOutliers(
  const std::vector<double>& errors,
  ValidationResult& result)
{
  if (errors.size() < 3) {
    result.outlier_ratio = 0.0;
    return;
  }
  
  // Compute median
  std::vector<double> sorted_errors = errors;
  std::sort(sorted_errors.begin(), sorted_errors.end());
  double median = sorted_errors[sorted_errors.size() / 2];
  
  // Compute MAD (Median Absolute Deviation)
  std::vector<double> absolute_deviations;
  for (double error : errors) {
    absolute_deviations.push_back(std::abs(error - median));
  }
  std::sort(absolute_deviations.begin(), absolute_deviations.end());
  double mad = absolute_deviations[absolute_deviations.size() / 2];
  
  // Detect outliers (modified Z-score > 3.5)
  double threshold = median + 3.5 * 1.4826 * mad;  // 1.4826 is consistency constant
  
  result.outlier_indices.clear();
  for (size_t i = 0; i < errors.size(); ++i) {
    if (errors[i] > threshold) {
      result.outlier_indices.push_back(i);
    }
  }
  
  result.outlier_ratio = static_cast<double>(result.outlier_indices.size()) / errors.size();
}

void CalibrationValidator::checkConsistency(
  const CalibrationResult& calibration,
  ValidationResult& result)
{
  // Check rotation consistency (should be a valid rotation matrix)
  Eigen::Matrix3d R = calibration.transform_matrix.block<3,3>(0,0);
  double det = R.determinant();
  Eigen::Matrix3d should_be_identity = R.transpose() * R;
  double orthogonality_error = (should_be_identity - Eigen::Matrix3d::Identity()).norm();
  
  result.rotation_consistency = (std::abs(det - 1.0) < 0.01 && orthogonality_error < 0.01);
  
  // Check translation consistency (reasonable bounds)
  double translation_norm = calibration.translation.norm();
  result.translation_consistency = (translation_norm < 10.0);  // Less than 10 meters
  
  if (!result.rotation_consistency) {
    result.warnings.push_back("Rotation matrix is not orthonormal");
  }
  
  if (!result.translation_consistency) {
    result.warnings.push_back("Translation magnitude seems unreasonable");
  }
}

std::map<std::string, int> CalibrationValidator::createErrorHistogram(
  const std::vector<double>& errors,
  int num_bins)
{
  std::map<std::string, int> histogram;
  
  if (errors.empty()) return histogram;
  
  double min_error = *std::min_element(errors.begin(), errors.end());
  double max_error = *std::max_element(errors.begin(), errors.end());
  double bin_width = (max_error - min_error) / num_bins;
  
  // Initialize bins
  for (int i = 0; i < num_bins; ++i) {
    double bin_start = min_error + i * bin_width;
    double bin_end = min_error + (i + 1) * bin_width;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3) << bin_start << "-" << bin_end;
    histogram[ss.str()] = 0;
  }
  
  // Fill histogram
  for (double error : errors) {
    int bin_idx = std::min(static_cast<int>((error - min_error) / bin_width), num_bins - 1);
    auto it = histogram.begin();
    std::advance(it, bin_idx);
    it->second++;
  }
  
  return histogram;
}

double CalibrationValidator::computeConfidenceScore(const ValidationResult& result)
{
  double score = 1.0;
  
  // Penalize based on RMSE
  score *= std::exp(-result.rmse / max_acceptable_error_);
  
  // Penalize based on outlier ratio
  score *= (1.0 - result.outlier_ratio);
  
  // Penalize based on max error
  if (result.max_error > 3 * result.rmse) {
    score *= 0.8;
  }
  
  // Penalize for consistency issues
  if (!result.rotation_consistency) score *= 0.7;
  if (!result.translation_consistency) score *= 0.7;
  
  return std::max(0.0, std::min(1.0, score));
}

// ValidationMonitor implementation
ValidationMonitor::ValidationMonitor(std::shared_ptr<CalibrationValidator> validator)
  : validator_(validator),
    max_history_(100),
    error_increase_threshold_(0.5),  // 50% increase
    consecutive_failures_(3),
    failure_count_(0)
{
}

bool ValidationMonitor::updateAndCheck(
  const CalibrationResult& current_calibration,
  const OdometryPair& new_pair)
{
  ValidationResult result = validator_->validateRealtime(current_calibration, new_pair);
  
  validation_history_.push_back(result);
  if (validation_history_.size() > max_history_) {
    validation_history_.erase(validation_history_.begin());
  }
  
  // Check for degradation
  if (!result.is_valid) {
    failure_count_++;
  } else {
    failure_count_ = 0;
  }
  
  // Need recalibration if consecutive failures
  if (failure_count_ >= consecutive_failures_) {
    RCLCPP_WARN(rclcpp::get_logger("validation_monitor"), 
                "Calibration degradation detected. Recalibration recommended.");
    return true;
  }
  
  return false;
}

bool ValidationMonitor::isCalibrationDegrading() const
{
  if (validation_history_.size() < 10) return false;
  
  // Compare recent errors with older errors
  double recent_avg = 0.0;
  double older_avg = 0.0;
  
  size_t split_point = validation_history_.size() / 2;
  
  for (size_t i = 0; i < split_point; ++i) {
    older_avg += validation_history_[i].rmse;
  }
  older_avg /= split_point;
  
  for (size_t i = split_point; i < validation_history_.size(); ++i) {
    recent_avg += validation_history_[i].rmse;
  }
  recent_avg /= (validation_history_.size() - split_point);
  
  return (recent_avg > older_avg * (1.0 + error_increase_threshold_));
}

void ValidationMonitor::reset()
{
  validation_history_.clear();
  failure_count_ = 0;
}

} // namespace wheel_lidar_calibration