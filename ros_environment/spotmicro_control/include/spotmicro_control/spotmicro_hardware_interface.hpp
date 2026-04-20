#ifndef SPOTMICRO_HARDWARE_INTERFACE_HPP
#define SPOTMICRO_HARDWARE_INTERFACE_HPP

#include <vector>
#include <string>
#include <map>
#include <memory>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/node_interfaces/lifecycle_node_interface.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "std_msgs/msg/float64.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace spotmicro_control
{

class SpotmicroHardwareInterface : public hardware_interface::SystemInterface
{
public:
  SpotmicroHardwareInterface();

  CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;
  
  hardware_interface::return_type read(const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  std::vector<double> hw_commands_;
  std::vector<double> hw_states_;
  std::vector<double> hw_states_velocity_;
  std::vector<std::string> joint_names_;
  
  // ROS communication
  rclcpp::Node::SharedPtr node_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr> joint_state_subscribers_;
  std::map<std::string, rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr> joint_publishers_;
};

}  // namespace spotmicro_control

#endif  // SPOTMICRO_HARDWARE_INTERFACE_HPP