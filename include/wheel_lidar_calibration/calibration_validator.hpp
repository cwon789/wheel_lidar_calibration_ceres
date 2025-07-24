#ifndef WHEEL_LIDAR_CALIBRATION_CALIBRATION_VALIDATOR_HPP_
#define WHEEL_LIDAR_CALIBRATION_CALIBRATION_VALIDATOR_HPP_

#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include <vector>
#include <map>
#include "wheel_lidar_calibration/data_synchronizer.hpp"
#include "wheel_lidar_calibration/calibration_optimizer.hpp"

namespace wheel_lidar_calibration
{

struct ValidationResult
{
  // Overall validation status
  bool is_valid;
  double confidence_score;  // 0-1, higher is better
  
  // Error metrics
  double mean_error;
  double rmse;
  double max_error;
  double std_error;
  
  // Error distribution
  std::vector<double> error_values;
  std::map<std::string, int> error_histogram;  // Binned error counts
  
  // Cross-validation results
  double cross_validation_error;
  double leave_one_out_error;
  
  // Outlier analysis
  double outlier_ratio;
  std::vector<size_t> outlier_indices;
  
  // Consistency checks
  bool rotation_consistency;
  bool translation_consistency;
  
  // Time-based validation (for real-time)
  double validation_timestamp;
  double computation_time;
  
  // Detailed report
  std::vector<std::string> warnings;
  std::vector<std::string> errors;
  
  std::string getSummary() const;
};

class CalibrationValidator
{
public:
  CalibrationValidator();
  
  // Validate calibration result with test data
  ValidationResult validate(
    const CalibrationResult& calibration,
    const std::vector<OdometryPair>& test_pairs);
  
  // Real-time validation with new data
  ValidationResult validateRealtime(
    const CalibrationResult& calibration,
    const OdometryPair& new_pair);
  
  // Cross-validation with k-fold
  ValidationResult crossValidate(
    const std::vector<OdometryPair>& all_pairs,
    int k_folds = 5);
  
  // Configure validation parameters
  void setOutlierThreshold(double threshold) { outlier_threshold_ = threshold; }
  void setMinConfidenceScore(double score) { min_confidence_score_ = score; }
  void setMaxAcceptableError(double error) { max_acceptable_error_ = error; }
  
  // Enable/disable specific checks
  void enableCrossValidation(bool enable) { enable_cross_validation_ = enable; }
  void enableOutlierDetection(bool enable) { enable_outlier_detection_ = enable; }

private:
  double outlier_threshold_;
  double min_confidence_score_;
  double max_acceptable_error_;
  bool enable_cross_validation_;
  bool enable_outlier_detection_;
  
  rclcpp::Logger logger_;
  
  // Running statistics for real-time validation
  std::deque<double> recent_errors_;
  size_t max_history_size_;
  
  // Compute error for a single pair
  double computePairError(
    const CalibrationResult& calibration,
    const OdometryPair& pair);
  
  // Compute error statistics
  void computeErrorStatistics(
    const std::vector<double>& errors,
    ValidationResult& result);
  
  // Detect outliers using MAD (Median Absolute Deviation)
  void detectOutliers(
    const std::vector<double>& errors,
    ValidationResult& result);
  
  // Check calibration consistency
  void checkConsistency(
    const CalibrationResult& calibration,
    ValidationResult& result);
  
  // Create error histogram
  std::map<std::string, int> createErrorHistogram(
    const std::vector<double>& errors,
    int num_bins = 10);
  
  // Compute confidence score
  double computeConfidenceScore(const ValidationResult& result);
};

// Real-time validation monitor
class ValidationMonitor
{
public:
  ValidationMonitor(std::shared_ptr<CalibrationValidator> validator);
  
  // Update with new data and check if recalibration is needed
  bool updateAndCheck(
    const CalibrationResult& current_calibration,
    const OdometryPair& new_pair);
  
  // Get validation history
  std::vector<ValidationResult> getHistory() const { return validation_history_; }
  
  // Check if calibration is degrading
  bool isCalibrationDegrading() const;
  
  // Reset monitor
  void reset();

private:
  std::shared_ptr<CalibrationValidator> validator_;
  std::vector<ValidationResult> validation_history_;
  size_t max_history_;
  
  // Thresholds for triggering recalibration
  double error_increase_threshold_;
  int consecutive_failures_;
  int failure_count_;
};

} // namespace wheel_lidar_calibration

#endif // WHEEL_LIDAR_CALIBRATION_CALIBRATION_VALIDATOR_HPP_