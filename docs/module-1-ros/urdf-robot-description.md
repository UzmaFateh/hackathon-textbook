---
sidebar_label: 'Chapter 4: Understanding URDF for Humanoid Robots'
description: 'Unified Robot Description Format for humanoid robot modeling'
---

# Chapter 4: Understanding URDF for Humanoid Robots

## Introduction

Unified Robot Description Format (URDF) is the standard XML-based format for representing robot models in ROS-based systems. For humanoid robots, URDF serves as the digital blueprint that defines the robot's physical structure, kinematic relationships, and visual properties. This chapter explores the intricacies of URDF specifically for humanoid robotics applications, covering the essential elements needed to create accurate and functional robot models.

## URDF Fundamentals

### XML Structure and Components

URDF is an XML-based format that describes robot structure through a tree of links connected by joints. The basic structure includes:

- **Links**: Represent rigid bodies of the robot
- **Joints**: Define the connections between links
- **Visual**: Defines how the robot appears visually
- **Collision**: Defines collision boundaries for physics simulation
- **Inertial**: Specifies mass properties for dynamics simulation

### Basic URDF Example

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base link (torso) -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
</robot>
```

## Links in URDF

### Link Definition

A link represents a rigid body in the robot structure. Each link must have:

- **Name**: Unique identifier for the link
- **Visual**: How the link appears visually
- **Collision**: Collision boundaries for physics simulation
- **Inertial**: Mass properties for dynamics

### Visual Properties

The visual element defines how a link appears in simulation and visualization:

```xml
<visual>
  <!-- Position and orientation offset from link origin -->
  <origin xyz="0 0 0" rpy="0 0 0"/>

  <!-- Geometry definition -->
  <geometry>
    <!-- Options: box, cylinder, sphere, mesh -->
    <box size="0.1 0.1 0.3"/>
  </geometry>

  <!-- Material properties -->
  <material name="red">
    <color rgba="1 0 0 1"/>
  </material>
</visual>
```

### Collision Properties

The collision element defines boundaries for physics simulation:

```xml
<collision>
  <!-- Position and orientation offset from link origin -->
  <origin xyz="0 0 0" rpy="0 0 0"/>

  <!-- Geometry definition -->
  <geometry>
    <!-- Often simplified compared to visual geometry for performance -->
    <box size="0.1 0.1 0.3"/>
  </geometry>
</collision>
```

### Inertial Properties

The inertial element defines mass properties for dynamics simulation:

```xml
<inertial>
  <mass value="1.0"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <!-- Inertia tensor -->
  <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
</inertial>
```

## Joints in URDF

### Joint Types

URDF supports several joint types for different kinematic relationships:

1. **Revolute**: Rotational joint with limits
2. **Continuous**: Rotational joint without limits
3. **Prismatic**: Linear sliding joint with limits
4. **Fixed**: No movement (rigid connection)
5. **Floating**: 6-DOF movement (for base)
6. **Planar**: Movement in a plane

### Joint Definition

```xml
<joint name="hip_joint" type="revolute">
  <!-- Parent link (the link closer to the base) -->
  <parent link="base_link"/>

  <!-- Child link (the link further from the base) -->
  <child link="thigh_link"/>

  <!-- Origin: position and orientation of joint relative to parent -->
  <origin xyz="0 0.1 -0.3" rpy="0 0 0"/>

  <!-- Axis of rotation/translation -->
  <axis xyz="0 1 0"/>

  <!-- Joint limits (for revolute and prismatic joints) -->
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>

  <!-- Joint dynamics -->
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

## Humanoid Robot URDF Structure

### Complete Humanoid Example

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Torso (base link) -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.5"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="0.9 0.8 0.7 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="arm_color">
        <color rgba="0.6 0.6 0.8 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="3.14" effort="30" velocity="2"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
      <material name="arm_color"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Left Leg -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_thigh"/>
    <origin xyz="0.05 -0.1 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="left_thigh">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
      <material name="leg_color">
        <color rgba="0.4 0.4 0.6 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.36" effort="100" velocity="1"/>
  </joint>

  <link name="left_shin">
    <visual>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
      <material name="leg_color"/>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.5"/>
      <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.04"/>
    </inertial>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_shin"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="50" velocity="1"/>
  </joint>

  <link name="left_foot">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="foot_color">
        <color rgba="0.2 0.2 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
  </link>
</robot>
```

## Advanced URDF Features for Humanoids

### Transmission Elements

Transmissions define how actuators connect to joints:

```xml
<transmission name="left_hip_transmission" type="transmission_interface/SimpleTransmission">
  <joint name="left_hip_joint">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_hip_motor">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Gazebo-Specific Elements

For simulation in Gazebo, additional elements are needed:

```xml
<!-- Gazebo material -->
<gazebo reference="head">
  <material>Gazebo/Blue</material>
</gazebo>

<!-- Gazebo plugin -->
<gazebo>
  <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
    <joint_name>neck_joint</joint_name>
  </plugin>
</gazebo>

<!-- Gazebo dynamics -->
<gazebo reference="left_knee_joint">
  <implicitSpringDamper>1</implicitSpringDamper>
</gazebo>
```

### Safety Controllers

Safety limits for joints:

```xml
<joint name="shoulder_joint" type="revolute">
  <parent link="torso"/>
  <child link="upper_arm"/>
  <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-2.0" upper="2.0" effort="50" velocity="2"/>
  <!-- Safety limits -->
  <safety_controller k_position="10" k_velocity="10" soft_lower_limit="-1.9" soft_upper_limit="1.9"/>
</joint>
```

## URDF Best Practices for Humanoid Robots

### 1. Proper Mass Distribution

Ensure realistic mass properties for stable simulation:

```xml
<!-- Good: Realistic mass distribution -->
<inertial>
  <mass value="3.0"/>  <!-- Realistic for thigh -->
  <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/>
</inertial>

<!-- Avoid: Unrealistic values that cause simulation instability -->
<inertial>
  <mass value="0.001"/>  <!-- Too light -->
  <inertia ixx="0.000001" ixy="0" ixz="0" iyy="0.000001" iyz="0" izz="0.000001"/>
</inertial>
```

### 2. Joint Limit Considerations

Set realistic joint limits based on human anatomy or robot capabilities:

```xml
<!-- Human-like limits -->
<limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>  <!-- ±90° for hip -->

<!-- Avoid overly restrictive or permissive limits -->
<limit lower="-10" upper="10" effort="1000" velocity="10"/>  <!-- Too wide, too high effort -->
```

### 3. Collision Geometry Optimization

Balance accuracy with performance:

```xml
<!-- Good: Simplified but accurate collision geometry -->
<collision>
  <geometry>
    <cylinder length="0.4" radius="0.06"/>  <!-- Simple cylinder for leg -->
  </geometry>
</collision>

<!-- Avoid: Overly complex collision geometry -->
<collision>
  <geometry>
    <mesh filename="complex_leg.dae"/>  <!-- Too complex, affects performance -->
  </geometry>
</collision>
```

## Tools for URDF Development

### 1. xacro (XML Macros)

xacro allows parameterization and reusability:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="link_length" value="0.4" />
  <xacro:property name="link_radius" value="0.05" />

  <!-- Macro for creating a limb -->
  <xacro:macro name="limb" params="name parent xyz_origin">
    <joint name="${name}_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${name}_link"/>
      <origin xyz="${xyz_origin}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
    </joint>

    <link name="${name}_link">
      <visual>
        <geometry>
          <cylinder length="${link_length}" radius="${link_radius}"/>
        </geometry>
        <material name="limb_color">
          <color rgba="0.6 0.6 0.8 1.0"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder length="${link_length}" radius="${link_radius}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.5"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
      <material name="body_color">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.5"/>
    </inertial>
  </link>

  <!-- Use the macro to create limbs -->
  <xacro:limb name="left_arm" parent="base_link" xyz_origin="0.2 0 0.1"/>
  <xacro:limb name="right_arm" parent="base_link" xyz_origin="-0.2 0 0.1"/>

</robot>
```

### 2. Validation Tools

Always validate URDF files:

```bash
# Check URDF syntax
check_urdf my_robot.urdf

# Visualize the robot
urdf_to_graphiz my_robot.urdf

# Launch in RViz
roslaunch urdf_tutorial display.launch model:=my_robot.urdf
```

## Integration with ROS 2 and Simulation

### Robot State Publisher

URDF models are used with the robot state publisher to broadcast transforms:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Subscribe to joint states
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Create transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize joint positions
        self.joint_positions = {}

    def joint_state_callback(self, msg):
        # Update joint positions
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

        # Broadcast transforms
        self.broadcast_transforms()

    def broadcast_transforms(self):
        # Calculate and broadcast transforms for each joint
        # This would involve forward kinematics calculations
        pass
```

## Exercises

1. **URDF Creation**: Create a complete URDF file for a simple humanoid robot with at least 12 degrees of freedom (head, 2 arms, 2 legs). Include proper mass properties, visual geometry, and collision models.

2. **xacro Exercise**: Convert your URDF from Exercise 1 to use xacro macros for the arms and legs, making the model parameterized and reusable.

3. **Simulation Integration**: Create a launch file that loads your URDF into Gazebo with appropriate plugins for joint state publishing and robot state publishing.

## Summary

URDF is fundamental to humanoid robotics in ROS-based systems, providing the digital representation of robot structure and properties. Understanding URDF's elements, best practices, and integration with simulation environments is crucial for developing effective Physical AI systems. Proper URDF modeling ensures realistic simulation, stable control, and successful transfer from simulation to reality.

The next chapter will explore Gazebo simulation environments and how to create digital twins of physical robots.

## References

- Bohren, J., & Cousins, S. (2010). "slaw: A successor to the robot manipulation software component of pr2_apps." *Robot Operating System*, 2010, 241-262.
- Koenig, N., & Howard, A. (2004). "Design and use paradigms for Gazebo, an open-source multi-robot simulator." *IEEE/RSJ International Conference on Intelligent Robots and Systems*, 2149-2154.
- ROS URDF Documentation. (2024). Retrieved from http://wiki.ros.org/urdf
- Siciliano, B., & Khatib, O. (2016). *Springer Handbook of Robotics*. Springer.