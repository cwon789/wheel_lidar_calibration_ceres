#ifndef WHEEL_LIDAR_CALIBRATION_OBSERVABILITY_ANALYZER_HPP_
#define WHEEL_LIDAR_CALIBRATION_OBSERVABILITY_ANALYZER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include "wheel_lidar_calibration/data_synchronizer.hpp"

namespace wheel_lidar_calibration
{

struct ObservabilityMetrics
{
  // Motion diversity metrics
  double rotation_observability;    // 0-1, higher is better
  double translation_observability; // 0-1, higher is better
  double overall_observability;     // Combined metric
  
  // Information matrix analysis
  Eigen::Matrix<double, 6, 6> fisher_information_matrix;
  std::vector<double> singular_values;
  double condition_number;
  
  // Motion coverage
  double rotation_space_coverage;   // How well rotations span SO(3)
  double translation_space_coverage; // How well translations span R^3
  
  // Detailed motion statistics
  Eigen::Vector3d total_rotation;    // Total rotation performed
  Eigen::Vector3d total_translation; // Total translation performed
  double path_length;               // Total path length
  
  // Recommendations
  std::vector<std::string> recommendations;
  bool is_observable;
  
  // Helper methods
  bool isWellConditioned(double threshold = 0.1) const {
    return condition_number < (1.0 / threshold);
  }
  
  std::string getSummary() const;
};

class ObservabilityAnalyzer
{
public:
  ObservabilityAnalyzer();
  
  // Analyze observability of the calibration problem
  ObservabilityMetrics analyze(const std::vector<OdometryPair>& odom_pairs);
  
  // Set thresholds for observability criteria
  void setRotationThreshold(double threshold) { rotation_threshold_ = threshold; }
  void setTranslationThreshold(double threshold) { translation_threshold_ = threshold; }
  void setConditionNumberThreshold(double threshold) { condition_threshold_ = threshold; }
  
  // Get minimum required motion for good observability
  static constexpr double MIN_ROTATION_RAD = 0.5;      // ~30 degrees total
  static constexpr double MIN_TRANSLATION_M = 2.0;     // 2 meters total
  static constexpr double MIN_PATH_LENGTH_M = 5.0;     // 5 meters path

private:
  double rotation_threshold_;
  double translation_threshold_;
  double condition_threshold_;
  
  rclcpp::Logger logger_;
  
  // Compute Fisher Information Matrix for extrinsic calibration
  Eigen::Matrix<double, 6, 6> computeFisherInformation(
    const std::vector<OdometryPair>& odom_pairs);
  
  // Analyze motion diversity
  void analyzeMotionDiversity(
    const std::vector<OdometryPair>& odom_pairs,
    ObservabilityMetrics& metrics);
  
  // Compute rotation and translation coverage
  double computeRotationCoverage(
    const std::vector<Eigen::Matrix3d>& rotations);
  
  double computeTranslationCoverage(
    const std::vector<Eigen::Vector3d>& translations);
  
  // Generate recommendations based on analysis
  void generateRecommendations(ObservabilityMetrics& metrics);
  
  // Helper to extract relative motions
  std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> 
  extractRelativeMotions(const std::vector<OdometryPair>& odom_pairs);
};

} // namespace wheel_lidar_calibration

#endif // WHEEL_LIDAR_CALIBRATION_OBSERVABILITY_ANALYZER_HPP_