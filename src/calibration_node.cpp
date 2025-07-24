#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <std_msgs/msg/bool.hpp>
#include "std_srvs/srv/trigger.hpp"
#include <fstream>
#include <iomanip>

#include "wheel_lidar_calibration/data_synchronizer.hpp"
#include "wheel_lidar_calibration/calibration_optimizer.hpp"
#include "wheel_lidar_calibration/calibration_optimizer_ceres.hpp"
#include "wheel_lidar_calibration/observability_analyzer.hpp"
#include "wheel_lidar_calibration/calibration_validator.hpp"

namespace wheel_lidar_calibration
{

class CalibrationNode : public rclcpp::Node
{
public:
  CalibrationNode() : Node("wheel_lidar_calibration_node")
  {
    // Declare parameters
    this->declare_parameter("wheel_odom_topic", "/wheel/odom");
    this->declare_parameter("lidar_odom_topic", "/lidar/odom");
    this->declare_parameter("max_time_diff", 0.02);
    this->declare_parameter("buffer_size", 1000);
    this->declare_parameter("min_pairs_for_calibration", 100);
    this->declare_parameter("max_iterations", 100);
    this->declare_parameter("convergence_threshold", 1e-6);
    this->declare_parameter("auto_start", false);
    this->declare_parameter("output_file", "calibration_result.txt");
    this->declare_parameter("publish_tf", true);
    this->declare_parameter("use_ceres", true);
    this->declare_parameter("use_motion_constraints", true);
    this->declare_parameter("enable_observability_check", true);
    this->declare_parameter("enable_realtime_validation", true);
    this->declare_parameter("validation_interval", 10);
    
    // Get parameters
    wheel_odom_topic_ = this->get_parameter("wheel_odom_topic").as_string();
    lidar_odom_topic_ = this->get_parameter("lidar_odom_topic").as_string();
    max_time_diff_ = this->get_parameter("max_time_diff").as_double();
    buffer_size_ = this->get_parameter("buffer_size").as_int();
    min_pairs_ = this->get_parameter("min_pairs_for_calibration").as_int();
    max_iterations_ = this->get_parameter("max_iterations").as_int();
    convergence_threshold_ = this->get_parameter("convergence_threshold").as_double();
    auto_start_ = this->get_parameter("auto_start").as_bool();
    output_file_ = this->get_parameter("output_file").as_string();
    publish_tf_ = this->get_parameter("publish_tf").as_bool();
    use_ceres_ = this->get_parameter("use_ceres").as_bool();
    use_motion_constraints_ = this->get_parameter("use_motion_constraints").as_bool();
    enable_observability_check_ = this->get_parameter("enable_observability_check").as_bool();
    enable_realtime_validation_ = this->get_parameter("enable_realtime_validation").as_bool();
    validation_interval_ = this->get_parameter("validation_interval").as_int();
    
    // Initialize components
    data_sync_ = std::make_unique<DataSynchronizer>(max_time_diff_, buffer_size_);
    
    // Choose optimizer based on parameter
    if (use_ceres_) {
      auto ceres_optimizer = std::make_unique<CalibrationOptimizerCeres>();
      ceres_optimizer->setUseMotionConstraints(use_motion_constraints_);
      optimizer_ = std::move(ceres_optimizer);
    } else {
      optimizer_ = std::make_unique<CalibrationOptimizer>();
    }
    optimizer_->setMaxIterations(max_iterations_);
    optimizer_->setConvergenceThreshold(convergence_threshold_);
    
    // Initialize observability analyzer and validator
    observability_analyzer_ = std::make_unique<ObservabilityAnalyzer>();
    validator_ = std::make_shared<CalibrationValidator>();
    
    if (enable_realtime_validation_) {
      validation_monitor_ = std::make_unique<ValidationMonitor>(validator_);
    }
    
    // Create subscriptions
    wheel_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      wheel_odom_topic_, 10,
      std::bind(&CalibrationNode::wheelOdomCallback, this, std::placeholders::_1));
    
    lidar_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      lidar_odom_topic_, 10,
      std::bind(&CalibrationNode::lidarOdomCallback, this, std::placeholders::_1));
    
    // Create service for triggering calibration
    calibrate_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "calibrate",
      std::bind(&CalibrationNode::calibrateService, this, std::placeholders::_1, std::placeholders::_2));
    
    // Create TF broadcaster
    if (publish_tf_) {
      tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
    }
    
    // Create timer for status updates
    status_timer_ = this->create_wall_timer(
      std::chrono::seconds(5),
      std::bind(&CalibrationNode::statusTimerCallback, this));
    
    RCLCPP_INFO(this->get_logger(), "Calibration node initialized");
    RCLCPP_INFO(this->get_logger(), "Wheel odometry topic: %s", wheel_odom_topic_.c_str());
    RCLCPP_INFO(this->get_logger(), "LiDAR odometry topic: %s", lidar_odom_topic_.c_str());
    
    if (auto_start_) {
      RCLCPP_INFO(this->get_logger(), "Auto-start enabled. Will calibrate when enough data is collected.");
    } else {
      RCLCPP_INFO(this->get_logger(), "Call the '/calibrate' service to start calibration.");
    }
  }

private:
  void wheelOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    data_sync_->addWheelOdometry(msg);
    
    if (auto_start_ && !calibration_done_) {
      checkAndCalibrate();
    }
    
    // Real-time validation
    if (enable_realtime_validation_ && validation_enabled_ && validation_monitor_) {
      validation_counter_++;
      if (validation_counter_ >= validation_interval_) {
        validation_counter_ = 0;
        performRealtimeValidation();
      }
    }
  }
  
  void lidarOdomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    data_sync_->addLidarOdometry(msg);
  }
  
  void calibrateService(
    const std_srvs::srv::Trigger::Request::SharedPtr request,
    std_srvs::srv::Trigger::Response::SharedPtr response)
  {
    (void)request;  // Unused
    
    RCLCPP_INFO(this->get_logger(), "Calibration service called");
    
    bool success = performCalibration();
    
    response->success = success;
    if (success) {
      response->message = "Calibration completed successfully. Results saved to " + output_file_;
    } else {
      response->message = "Calibration failed. Check logs for details.";
    }
  }
  
  void checkAndCalibrate()
  {
    std::vector<OdometryPair> pairs;
    if (data_sync_->getSynchronizedPairs(pairs, min_pairs_)) {
      RCLCPP_INFO(this->get_logger(), "Collected enough data. Starting automatic calibration...");
      performCalibration();
      calibration_done_ = true;
    }
  }
  
  bool performCalibration()
  {
    std::vector<OdometryPair> pairs;
    
    if (!data_sync_->getSynchronizedPairs(pairs, min_pairs_)) {
      RCLCPP_ERROR(this->get_logger(), 
                   "Not enough synchronized pairs. Got %zu, need at least %d",
                   pairs.size(), min_pairs_);
      return false;
    }
    
    RCLCPP_INFO(this->get_logger(), "Starting calibration with %zu synchronized pairs", pairs.size());
    
    // Perform observability analysis first
    if (enable_observability_check_) {
      RCLCPP_INFO(this->get_logger(), "Performing observability analysis...");
      ObservabilityMetrics obs_metrics = observability_analyzer_->analyze(pairs);
      
      if (!obs_metrics.is_observable) {
        RCLCPP_WARN(this->get_logger(), "Calibration problem has poor observability!");
        RCLCPP_WARN(this->get_logger(), "%s", obs_metrics.getSummary().c_str());
        
        // Continue with warning or abort based on severity
        if (obs_metrics.overall_observability < 0.2) {
          RCLCPP_ERROR(this->get_logger(), "Observability too low. Aborting calibration.");
          return false;
        }
      }
    }
    
    // Perform calibration
    CalibrationResult result = optimizer_->optimize(pairs);
    
    if (!result.converged) {
      RCLCPP_ERROR(this->get_logger(), "Calibration did not converge");
      return false;
    }
    
    // Store result for real-time validation
    current_calibration_ = result;
    
    // Validate the calibration result
    RCLCPP_INFO(this->get_logger(), "Validating calibration result...");
    
    // Split data for validation (use last 20% for validation)
    size_t train_size = pairs.size() * 0.8;
    std::vector<OdometryPair> train_pairs(pairs.begin(), pairs.begin() + train_size);
    std::vector<OdometryPair> test_pairs(pairs.begin() + train_size, pairs.end());
    
    ValidationResult validation = validator_->validate(result, test_pairs);
    
    if (!validation.is_valid) {
      RCLCPP_ERROR(this->get_logger(), "Calibration validation failed!");
      RCLCPP_ERROR(this->get_logger(), "%s", validation.getSummary().c_str());
      return false;
    }
    
    RCLCPP_INFO(this->get_logger(), "Calibration validated successfully");
    RCLCPP_INFO(this->get_logger(), "%s", validation.getSummary().c_str());
    
    // Perform cross-validation if enabled
    RCLCPP_INFO(this->get_logger(), "Performing 5-fold cross-validation...");
    ValidationResult cv_result = validator_->crossValidate(pairs, 5);
    RCLCPP_INFO(this->get_logger(), "Cross-validation RMSE: %f", cv_result.cross_validation_error);
    
    // Save results
    saveCalibrationResult(result, validation);
    
    // Publish TF if enabled
    if (publish_tf_) {
      publishTransform();
    }
    
    // Clear buffers after successful calibration
    data_sync_->clearBuffers();
    
    // Enable real-time validation
    if (enable_realtime_validation_ && validation_monitor_) {
      validation_monitor_->reset();
      validation_enabled_ = true;
      RCLCPP_INFO(this->get_logger(), "Real-time validation enabled");
    }
    
    return true;
  }
  
  void saveCalibrationResult(const CalibrationResult& result, const ValidationResult& validation)
  {
    std::ofstream file(output_file_);
    
    if (!file.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open output file: %s", output_file_.c_str());
      return;
    }
    
    file << std::fixed << std::setprecision(6);
    file << "# Wheel-LiDAR Extrinsic Calibration Result\n";
    file << "# Generated: " << rclcpp::Clock().now().seconds() << "\n\n";
    
    file << "# Translation (x, y, z) [meters]\n";
    file << "translation: [" 
         << result.translation.x() << ", " 
         << result.translation.y() << ", " 
         << result.translation.z() << "]\n\n";
    
    file << "# Rotation (quaternion: x, y, z, w)\n";
    file << "rotation: [" 
         << result.rotation.x() << ", " 
         << result.rotation.y() << ", " 
         << result.rotation.z() << ", " 
         << result.rotation.w() << "]\n\n";
    
    file << "# Transformation Matrix (4x4)\n";
    file << "transform_matrix:\n";
    for (int i = 0; i < 4; ++i) {
      file << "  - [";
      for (int j = 0; j < 4; ++j) {
        file << result.transform_matrix(i, j);
        if (j < 3) file << ", ";
      }
      file << "]\n";
    }
    
    file << "\n# Calibration Statistics\n";
    file << "rmse: " << result.rmse << "\n";
    file << "iterations: " << result.iterations << "\n";
    file << "converged: " << (result.converged ? "true" : "false") << "\n";
    
    file << "\n# Validation Results\n";
    file << "validation_rmse: " << validation.rmse << "\n";
    file << "validation_mean_error: " << validation.mean_error << "\n";
    file << "validation_max_error: " << validation.max_error << "\n";
    file << "validation_confidence: " << validation.confidence_score << "\n";
    file << "validation_outlier_ratio: " << validation.outlier_ratio << "\n";
    
    file.close();
    
    RCLCPP_INFO(this->get_logger(), "Calibration results saved to: %s", output_file_.c_str());
  }
  
  void publishTransform()
  {
    geometry_msgs::msg::TransformStamped tf_msg = 
      optimizer_->getTransformMsg("wheel_odom", "lidar_odom");
    
    tf_broadcaster_->sendTransform(tf_msg);
    
    RCLCPP_INFO(this->get_logger(), "Published static transform from wheel_odom to lidar_odom");
  }
  
  void statusTimerCallback()
  {
    RCLCPP_INFO(this->get_logger(), 
                "Buffer status - Wheel: %zu, LiDAR: %zu messages",
                data_sync_->getWheelBufferSize(),
                data_sync_->getLidarBufferSize());
  }
  
  void performRealtimeValidation()
  {
    std::vector<OdometryPair> recent_pairs;
    if (data_sync_->getSynchronizedPairs(recent_pairs, 1)) {
      if (!recent_pairs.empty()) {
        bool needs_recalibration = validation_monitor_->updateAndCheck(
          current_calibration_, recent_pairs.back());
        
        if (needs_recalibration) {
          RCLCPP_WARN(this->get_logger(), 
                      "Real-time validation detected calibration degradation. "
                      "Consider recalibrating.");
        }
      }
    }
  }

  // ROS2 interfaces
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr lidar_odom_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr calibrate_srv_;
  rclcpp::TimerBase::SharedPtr status_timer_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_broadcaster_;
  
  // Core components
  std::unique_ptr<DataSynchronizer> data_sync_;
  std::unique_ptr<CalibrationOptimizer> optimizer_;
  std::unique_ptr<ObservabilityAnalyzer> observability_analyzer_;
  std::shared_ptr<CalibrationValidator> validator_;
  std::unique_ptr<ValidationMonitor> validation_monitor_;
  
  // Parameters
  std::string wheel_odom_topic_;
  std::string lidar_odom_topic_;
  double max_time_diff_;
  size_t buffer_size_;
  int min_pairs_;
  int max_iterations_;
  double convergence_threshold_;
  bool auto_start_;
  std::string output_file_;
  bool publish_tf_;
  bool use_ceres_;
  bool use_motion_constraints_;
  bool enable_observability_check_;
  bool enable_realtime_validation_;
  int validation_interval_;
  
  // State
  bool calibration_done_ = false;
  bool validation_enabled_ = false;
  int validation_counter_ = 0;
  CalibrationResult current_calibration_;
};

} // namespace wheel_lidar_calibration

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<wheel_lidar_calibration::CalibrationNode>());
  rclcpp::shutdown();
  return 0;
}