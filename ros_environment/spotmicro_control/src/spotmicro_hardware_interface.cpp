#include "spotmicro_control/spotmicro_hardware_interface.hpp"
#include <hardware_interface/types/hardware_interface_return_values.hpp>
#include <hardware_interface/types/hardware_interface_type_values.hpp>

using hardware_interface::CallbackReturn;
using hardware_interface::return_type;

namespace spotmicro_control
{

SpotmicroHardwareInterface::SpotmicroHardwareInterface()
: node_(std::make_shared<rclcpp::Node>("spotmicro_hardware_interface"))
{
}

CallbackReturn SpotmicroHardwareInterface::on_init(const hardware_interface::HardwareInfo & info)
{
  if (hardware_interface::SystemInterface::on_init(info) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }

  joint_names_ = {
    "front_left_shoulder", "front_left_leg", "front_left_foot",
    "front_right_shoulder", "front_right_leg", "front_right_foot",
    "rear_left_shoulder", "rear_left_leg", "rear_left_foot",
    "rear_right_shoulder", "rear_right_leg", "rear_right_foot"
  };

  hw_commands_.resize(joint_names_.size(), 0.0);
  hw_states_.resize(joint_names_.size(), 0.0);
  hw_states_velocity_.resize(joint_names_.size(), 0.0);

  // Initialize ROS publishers for each joint command
  for (const auto& joint_name : joint_names_) {
    joint_publishers_[joint_name] = node_->create_publisher<std_msgs::msg::Float64>(
      joint_name + "/command", 10);
  }

  // Initialize ROS subscriber for joint states
  auto joint_state_callback = [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
    for (size_t i = 0; i < joint_names_.size(); ++i) {
      for (size_t j = 0; j < msg->name.size(); ++j) {
        if (msg->name[j] == joint_names_[i]) {
          hw_states_[i] = msg->position[j];
          hw_states_velocity_[i] = msg->velocity[j];
          break;
        }
      }
    }
  };

  joint_state_subscribers_.push_back(
    node_->create_subscription<sensor_msgs::msg::JointState>(
      "joint_states",
      10,
      joint_state_callback
    )
  );

  return CallbackReturn::SUCCESS;
}

CallbackReturn SpotmicroHardwareInterface::on_configure(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("SpotmicroHardwareInterface"), "Configuring hardware interface...");
  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> SpotmicroHardwareInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> state_interfaces;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    state_interfaces.emplace_back(
      joint_names_[i],
      hardware_interface::HW_IF_POSITION,
      &hw_states_[i]);
    state_interfaces.emplace_back(
      joint_names_[i],
      hardware_interface::HW_IF_VELOCITY,
      &hw_states_velocity_[i]);
  }
  return state_interfaces;
}

std::vector<hardware_interface::CommandInterface> SpotmicroHardwareInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> command_interfaces;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    command_interfaces.emplace_back(
      joint_names_[i],
      hardware_interface::HW_IF_EFFORT,
      &hw_commands_[i]);
  }
  return command_interfaces;
}

CallbackReturn SpotmicroHardwareInterface::on_activate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("SpotmicroHardwareInterface"), "Activating hardware interface...");
  return CallbackReturn::SUCCESS;
}

CallbackReturn SpotmicroHardwareInterface::on_deactivate(const rclcpp_lifecycle::State &)
{
  RCLCPP_INFO(rclcpp::get_logger("SpotmicroHardwareInterface"), "Deactivating hardware interface...");
  return CallbackReturn::SUCCESS;
}

return_type SpotmicroHardwareInterface::read(const rclcpp::Time &, const rclcpp::Duration &)
{
  // Joint states are updated via the subscriber callback
  return return_type::OK;
}

return_type SpotmicroHardwareInterface::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  // Publish force commands to Gazebo
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    auto msg = std_msgs::msg::Float64();
    msg.data = hw_commands_[i];
    joint_publishers_[joint_names_[i]]->publish(msg);
  }
  return return_type::OK;
}

}  // namespace spotmicro_control

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(spotmicro_control::SpotmicroHardwareInterface, hardware_interface::SystemInterface)