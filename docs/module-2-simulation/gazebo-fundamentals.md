---
sidebar_label: 'Chapter 6: Gazebo Simulation Fundamentals'
description: 'Physics simulation and environment building for humanoid robots'
---

# Chapter 6: Gazebo Simulation Fundamentals

## Introduction

Gazebo is a powerful physics simulation environment that enables the creation of realistic digital twins for robotic systems. For Physical AI and humanoid robotics, Gazebo provides the essential capability to test algorithms, validate control systems, and generate synthetic training data in a safe, controlled environment. This chapter explores the fundamentals of Gazebo simulation, focusing on physics modeling, environment creation, and sensor simulation for humanoid robotic applications.

## Understanding Gazebo Architecture

### The Simulation Engine

Gazebo is built on top of the Open Dynamics Engine (ODE), Bullet Physics, or Simbody for physics simulation. It provides:

- **Realistic Physics**: Accurate modeling of forces, collisions, and dynamics
- **3D Visualization**: Real-time rendering of simulated environments
- **Sensor Simulation**: Emulation of various sensor types (LiDAR, cameras, IMUs)
- **Plugin System**: Extensible architecture for custom functionality
- **ROS Integration**: Seamless communication with ROS-based systems

### Core Components

Gazebo's architecture includes several key components:

1. **Server (gzserver)**: Handles physics simulation and environment management
2. **Client (gzclient)**: Provides the 3D visualization interface
3. **World Files**: Define environments, objects, and initial conditions
4. **Model Files**: Describe robot and object geometries
5. **Plugins**: Extend functionality for sensors, controllers, and communication

## Installing and Setting Up Gazebo

### Installation

Gazebo can be installed in various configurations depending on your needs:

```bash
# Install Gazebo Harmonic (recommended for ROS 2 Humble)
sudo apt update
sudo apt install gazebo
# Or for specific version
sudo apt install gazebo-harmonic
```

### Basic Launch

Starting Gazebo with a basic empty world:

```bash
# Launch Gazebo with empty world
gz sim

# Launch with a specific world file
gz sim -r my_world.sdf
```

## World Files and Environment Building

### SDF (Simulation Description Format)

Gazebo uses SDF (Simulation Description Format) to define simulation environments. SDF is an XML-based format that describes:

- World properties (gravity, magnetic field, etc.)
- Models and their initial positions
- Lighting and rendering settings
- Physics engine parameters

### Basic World Structure

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_world">
    <!-- World properties -->
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Lighting -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.1 -0.1 -1</direction>
    </light>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>1.0</mu>
                <mu2>1.0</mu2>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.3 0.3 0.3 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Humanoid robot model -->
    <include>
      <uri>model://simple_humanoid</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>

    <!-- Obstacles -->
    <model name="table">
      <pose>2 0 0 0 0 0</pose>
      <link name="table_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.6 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.6 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.6 0.4 1</ambient>
            <diffuse>0.8 0.6 0.4 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10.0</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>1.0</iyy>
            <iyz>0.0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

## Physics Simulation in Gazebo

### Physics Engine Configuration

Gazebo supports multiple physics engines, each with different characteristics:

```xml
<!-- ODE Physics Engine (default) -->
<physics type="ode">
  <gravity>0 0 -9.8</gravity>
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>

<!-- Bullet Physics Engine (more stable for complex interactions) -->
<physics type="bullet">
  <gravity>0 0 -9.8</gravity>
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <bullet>
    <solver>
      <type>sequential_impulse</type>
      <iterations>50</iterations>
      <sor>1.3</sor>
    </solver>
  </bullet>
</physics>
```

### Material Properties and Friction

Realistic physics simulation requires accurate material properties:

```xml
<collision name="foot_collision">
  <geometry>
    <box>
      <size>0.2 0.1 0.05</size>
    </box>
  </geometry>
  <surface>
    <friction>
      <ode>
        <mu>0.8</mu>    <!-- Coefficient of friction -->
        <mu2>0.8</mu2>  <!-- Secondary friction coefficient -->
        <fdir1>0 0 1</fdir1>  <!-- Friction direction -->
      </ode>
    </friction>
    <bounce>
      <restitution_coefficient>0.1</restitution_coefficient>  <!-- Bounciness -->
      <threshold>100000</threshold>  <!-- Velocity threshold for bounce -->
    </bounce>
    <contact>
      <ode>
        <soft_cfm>0</soft_cfm>
        <soft_erp>0.2</soft_erp>
        <kp>1000000000000</kp>  <!-- Contact stiffness -->
        <kd>1</kd>  <!-- Damping coefficient -->
        <max_vel>100</max_vel>
        <min_depth>0</min_depth>
      </ode>
    </contact>
  </surface>
</collision>
```

## Sensor Simulation

### Camera Sensors

Simulating vision systems for humanoid robots:

```xml
<sensor name="head_camera" type="camera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
  <camera name="head_camera">
    <pose>0.05 0 0.05 0 0 0</pose>  <!-- Offset from parent link -->
    <horizontal_fov>1.047</horizontal_fov>  <!-- Field of view in radians (60 degrees) -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10</far>
    </clip>
  </camera>
</sensor>
```

### LiDAR Sensors

Simulating range-finding sensors:

```xml
<sensor name="lidar" type="gpu_lidar">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
  <pose>0 0 0.2 0 0 0</pose>
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>  <!-- -π radians -->
        <max_angle>3.14159</max_angle>   <!-- π radians -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>10.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
</sensor>
```

### IMU Sensors

Simulating inertial measurement units:

```xml
<sensor name="imu_sensor" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <pose>0 0 0 0 0 0</pose>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.001</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.001</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.001</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>0.017</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

## Gazebo Plugins for Humanoid Robotics

### ROS 2 Integration Plugins

Gazebo provides plugins for seamless ROS 2 integration:

```xml
<!-- Joint state publisher -->
<gazebo>
  <plugin filename="libgazebo_ros_joint_state_publisher.so" name="gazebo_ros_joint_state_publisher">
    <ros>
      <namespace>/humanoid</namespace>
    </ros>
    <update_rate>30</update_rate>
    <joint_name>hip_joint</joint_name>
    <joint_name>knee_joint</joint_name>
    <joint_name>ankle_joint</joint_name>
  </plugin>
</gazebo>

<!-- Joint trajectory controller -->
<gazebo>
  <plugin filename="libgazebo_ros_joint_trajectory.so" name="gazebo_ros_joint_trajectory">
    <ros>
      <namespace>/humanoid</namespace>
    </ros>
    <command_topic>joint_trajectory</command_topic>
    <update_rate>100</update_rate>
  </plugin>
</gazebo>

<!-- IMU sensor plugin -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>/humanoid</namespace>
        <remapping>~/out:=imu/data</remapping>
      </ros>
      <frame_id>imu_link</frame_id>
      <topic>~/imu/data</topic>
      <gaussian_noise>0.0005</gaussian_noise>
    </plugin>
  </sensor>
</gazebo>
```

### Custom Controller Plugins

Creating custom plugins for humanoid-specific control:

```cpp
// Example custom plugin for balance control
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math/Pose3.hh>

namespace gazebo
{
  class HumanoidBalancePlugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
    {
      // Store the model pointer for convenience
      this->model = _parent;

      // Listen to the update event. This event is broadcast every
      // simulation iteration.
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&HumanoidBalancePlugin::OnUpdate, this));
    }

    // Called by the world update start event
    public: void OnUpdate()
    {
      // Apply balance control logic here
      // This is where you would implement humanoid balance algorithms
    }

    // Pointer to the model
    private: physics::ModelPtr model;

    // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(HumanoidBalancePlugin)
}
```

## Creating Complex Environments

### Indoor Environments

Creating realistic indoor environments for humanoid testing:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="indoor_humanoid">
    <physics type="ode">
      <gravity>0 0 -9.8</gravity>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
    </physics>

    <!-- Lighting -->
    <light name="ceiling_light_1" type="point">
      <pose>0 0 3 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
    </light>

    <!-- Floor -->
    <model name="floor">
      <static>true</static>
      <link name="floor_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>0.8</mu>
              </ode>
            </friction>
          </surface>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Walls -->
    <model name="wall_north">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>10 0.2 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.6 0.6 0.6 1</ambient>
            <diffuse>0.6 0.6 0.6 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Furniture -->
    <model name="chair">
      <pose>-2 1 0 0 0 0</pose>
      <link name="seat">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.4 0.4 0.05</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.4 0.4 0.05</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.3 0.1 1</ambient>
            <diffuse>0.5 0.3 0.1 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>5.0</mass>
          <inertia>
            <ixx>0.2</ixx>
            <iyy>0.2</iyy>
            <izz>0.2</izz>
          </inertia>
        </inertial>
      </link>
      <!-- Additional links for chair back, legs, etc. -->
    </model>

    <!-- Humanoid robot -->
    <include>
      <uri>model://simple_humanoid</uri>
      <pose>0 0 0.8 0 0 0</pose>
    </include>
  </world>
</sdf>
```

### Outdoor Environments

Creating outdoor environments with terrain variations:

```xml
<!-- Terrain model -->
<model name="terrain">
  <static>true</static>
  <link name="terrain_link">
    <collision name="collision">
      <geometry>
        <heightmap>
          <uri>file://meshes/terrain.png</uri>
          <size>20 20 3</size>  <!-- width, depth, height -->
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <heightmap>
          <uri>file://meshes/terrain.png</uri>
          <size>20 20 3</size>
        </heightmap>
      </geometry>
    </visual>
  </link>
</model>
```

## Performance Optimization

### Simulation Parameters

Optimizing Gazebo performance for complex humanoid simulations:

```xml
<physics type="ode">
  <gravity>0 0 -9.8</gravity>
  <!-- Smaller step size for stability, larger for performance -->
  <max_step_size>0.001</max_step_size>
  <!-- Real-time factor: 1.0 = real-time, < 1.0 = slower, > 1.0 = faster -->
  <real_time_factor>1.0</real_time_factor>
  <!-- Update rate: higher = more accurate but slower -->
  <real_time_update_rate>1000</real_time_update_rate>

  <ode>
    <solver>
      <!-- Quick solver is faster but less accurate -->
      <type>quick</type>
      <!-- More iterations = more accurate but slower -->
      <iters>20</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <!-- Contact parameters affect stability -->
      <contact_surface_layer>0.001</contact_surface_layer>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
    </constraints>
  </ode>
</physics>
```

### Visual Optimization

Reducing visual complexity for better performance:

```xml
<!-- Reduce visual complexity for faster simulation -->
<model name="humanoid_robot">
  <link name="body">
    <!-- Use simpler collision geometry than visual geometry -->
    <collision>
      <geometry>
        <cylinder>
          <length>0.5</length>
          <radius>0.1</radius>
        </cylinder>
      </geometry>
    </collision>
    <visual>
      <!-- More detailed visual geometry -->
      <geometry>
        <mesh>
          <uri>meshes/detailed_body.dae</uri>
        </mesh>
      </geometry>
    </visual>
  </link>
</model>
```

## Integration with ROS 2

### Launch Files for Simulation

Creating launch files that start both Gazebo and ROS 2 nodes:

```python
# launch/humanoid_simulation.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    return LaunchDescription([
        # Launch Gazebo with a world file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('gazebo_ros'),
                    'launch',
                    'gazebo.launch.py'
                ])
            ]),
            launch_arguments={
                'world': PathJoinSubstitution([
                    FindPackageShare('my_robot_gazebo'),
                    'worlds',
                    'humanoid_world.sdf'
                ])
            }.items()
        ),

        # Launch robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'use_sim_time': True}]
        ),

        # Launch joint state publisher
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[{'use_sim_time': True}]
        )
    ])
```

## Exercises

1. **Environment Creation**: Create a Gazebo world file that includes a humanoid robot in an indoor environment with furniture, obstacles, and appropriate lighting. Include at least one sensor (camera or LiDAR) on the robot.

2. **Physics Tuning**: Experiment with different physics parameters to optimize simulation stability for a humanoid walking. Document the parameters that provide the best balance between accuracy and performance.

3. **Sensor Integration**: Create a launch file that starts Gazebo with your humanoid robot and integrates it with ROS 2, publishing joint states and sensor data to ROS topics.

## Summary

Gazebo provides a powerful simulation environment for Physical AI and humanoid robotics, enabling safe testing of algorithms and control systems. Understanding world files, physics simulation, sensor modeling, and ROS integration is crucial for creating effective digital twins of physical robots. Proper configuration and optimization ensure realistic simulation while maintaining performance.

The next chapter will explore Unity integration and high-fidelity rendering for advanced humanoid simulation.

## References

- Koenig, N., & Howard, A. (2004). "Design and use paradigms for Gazebo, an open-source multi-robot simulator." *IEEE/RSJ International Conference on Intelligent Robots and Systems*, 2149-2154.
- Gazebo Documentation. (2024). Retrieved from http://gazebosim.org/
- O'Flaherty, R., et al. (2019). "Robot Operating System 2: Design, architecture, and uses in the wild." *arXiv preprint arXiv:1906.06049*.
- Siciliano, B., & Khatib, O. (2016). *Springer Handbook of Robotics*. Springer.