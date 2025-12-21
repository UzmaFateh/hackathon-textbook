import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    'index',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1-ros/introduction-physical-ai',
        'module-1-ros/concepts',
        'module-1-ros/ros2-fundamentals',
        'module-1-ros/urdf-robot-description'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-simulation/robot-description',
        'module-2-simulation/gazebo-fundamentals',
        'module-2-simulation/unity-integration',
        'module-2-simulation/integration'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        'module-3-ai-brain/nvidia-isaac-sim',
        'module-3-ai-brain/isaac-ros-perception',
        'module-3-ai-brain/navigation',
        'module-3-ai-brain/rl-sim-to-real'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        'module-4-vla/whisper',
        'module-4-vla/vision-language-action',
        'module-4-vla/llm-planning',
        'module-4-vla/plans-to-actions',
        'module-4-vla/capstone',
        'module-4-vla/autonomous-humanoid-capstone'
      ],
    },
  ],
};

export default sidebars;