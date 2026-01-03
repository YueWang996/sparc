import mujoco as mj
import numpy as np
from typing import Tuple


class MuJoCoInterface:
    """Interface for interacting with MuJoCo simulation."""
    
    def __init__(self, model: mj.MjModel, data: mj.MjData):
        """
        Initialize the MuJoCo interface.
        
        Args:
            model: MuJoCo model object
            data: MuJoCo data object
        """
        self.model = model
        self.data = data
        
    def get_body_pose(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the position and quaternion orientation of a body.
        
        Args:
            name: Name of the body
            
        Returns:
            position: 3D position array [x, y, z]
            quaternion: Quaternion orientation [w, x, y, z]
        """
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            raise ValueError(f"Body '{name}' not found in model")
        
        position = self.data.xpos[body_id].copy()
        quaternion = self.data.xquat[body_id].copy()
        
        return position, quaternion
    
    def get_joint_pos(self, name: str) -> float:
        """
        Get the position of a joint.
        
        Args:
            name: Name of the joint
            
        Returns:
            Joint position (angle for revolute, displacement for prismatic)
        """
        joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            raise ValueError(f"Joint '{name}' not found in model")
        
        joint_addr = self.model.jnt_qposadr[joint_id]
        return self.data.qpos[joint_addr]
    
    def set_joint_pos(self, name: str, position: float) -> None:
        """
        Set the position of a joint.
        
        Args:
            name: Name of the joint
            position: Desired joint position (angle for revolute, displacement for prismatic)
        """
        joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            raise ValueError(f"Joint '{name}' not found in model")
        
        joint_addr = self.model.jnt_qposadr[joint_id]
        self.data.qpos[joint_addr] = position

    def set_joint_vel(self, name: str, velocity: float) -> None:
        """
        Set the velocity of a joint.
        
        Args:
            name: Name of the joint
            velocity: Desired joint velocity (angular velocity for revolute, linear for prismatic)
        """
        joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            raise ValueError(f"Joint '{name}' not found in model")
        
        joint_addr = self.model.jnt_dofadr[joint_id]
        self.data.qvel[joint_addr] = velocity
    
    def get_joint_vel(self, name: str) -> float:
        """
        Get the velocity of a joint.
        
        Args:
            name: Name of the joint
            
        Returns:
            Joint velocity (angular velocity for revolute, linear for prismatic)
        """
        joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            raise ValueError(f"Joint '{name}' not found in model")
        
        joint_addr = self.model.jnt_dofadr[joint_id]
        return self.data.qvel[joint_addr]
    
    def set_joint_torque(self, motor_name: str, torque: float) -> None:
        """
        Set the control input (torque) for a motor/actuator.
        
        Args:
            motor_name: Name of the actuator
            torque: Desired torque value
        """
        actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, motor_name)
        if actuator_id == -1:
            raise ValueError(f"Actuator '{motor_name}' not found in model")
        
        self.data.ctrl[actuator_id] = torque
    
    def get_body_velocity(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the linear and angular velocity of a body.
        
        Args:
            name: Name of the body
            
        Returns:
            linear_velocity: 3D linear velocity [vx, vy, vz]
            angular_velocity: 3D angular velocity [wx, wy, wz]
        """
        body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            raise ValueError(f"Body '{name}' not found in model")
        
        linear_vel = self.data.cvel[body_id, 3:6].copy()
        angular_vel = self.data.cvel[body_id, 0:3].copy()
        
        return linear_vel, angular_vel