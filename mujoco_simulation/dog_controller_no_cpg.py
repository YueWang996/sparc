import numpy as np
import torch
from spine_impedance_controller import SpineImpedanceController

# ------------------------- Gait Generator (Pure NumPy) -------------------------
class GaitTrajectoryGenerator:
    """
    Generates a time-based 8D joint angle trajectory for a quadruped's legs.
    Supports Trot and Bound gaits with SMOOTH frequency transitions.
    Supports Directional Steering via Stride Length asymmetry.
    """

    def __init__(self, step_length, step_height, frequency,
                 thigh_length, tibia_length, gait_type='trot'):
        self.step_length = step_length
        self.step_height = step_height
        self.l1 = thigh_length
        self.l2 = tibia_length
        
        # Initial joint positions
        self.q0_ = np.array([
            -0.785398, 1.5708,   # Hind Left
            -0.785398, 1.5708,   # Hind Right
            0.785398, -1.5708,   # Front Left
            0.785398, -1.5708    # Front Right
        ])
        
        # Phase Continuity State
        self.global_phase_offset = 0.0
        
        # Initialize frequency
        self.frequency = frequency
        self.angular_freq = 2 * np.pi * frequency
        self.set_gait_type(gait_type)
        
        # Steering State (1.0 = Nominal)
        self.left_stride_scale = 1.0
        self.right_stride_scale = 1.0

    def set_frequency(self, frequency_hz, current_time=0.0):
        """
        Modifies gait frequency while preserving phase continuity.
        """
        # 1. Calculate the accumulated phase right before the switch
        current_phase = self.angular_freq * current_time + self.global_phase_offset
        
        # 2. Update the frequency
        self.frequency = frequency_hz
        self.angular_freq = 2 * np.pi * frequency_hz
        
        # 3. Calculate new offset to start from the exact same phase
        self.global_phase_offset = current_phase - (self.angular_freq * current_time)

    def set_gait_type(self, gait_type):
        """
        Interface to switch between 'trot' and 'bound'.
        Leg Order: HL, HR, FL, FR
        """
        self.gait_type = gait_type
        if gait_type == 'trot':
            # Trot: Diagonal pairs move together.
            self.phase_offsets = np.array([0, np.pi, np.pi, 0])
        elif gait_type == 'bound':
            # Bound: Front legs move together, Hind legs move together.
            self.phase_offsets = np.array([0, 0, np.pi, np.pi])
        else:
            raise ValueError(f"Unknown gait type: {gait_type}")
        # print(f"Gait generator switched to: {gait_type}")

    def set_steering_offset(self, offset):
        """
        Adjusts stride length asymmetry for steering.
        offset > 0: Turn Left (Right stride > Left stride)
        offset < 0: Turn Right (Left stride > Right stride)
        Range clamped to +/- 0.4 (40% asymmetry)
        """
        offset = np.clip(offset, -0.4, 0.4)
        self.right_stride_scale = 1.0 + offset
        self.left_stride_scale = 1.0 - offset

    def get_current_phase(self, t):
        """Returns global phase in [0, 2pi)"""
        return (self.angular_freq * t + self.global_phase_offset) % (2 * np.pi)

    def get_joint_angles(self, t):
        """
        Returns 8D array: [HL_u, HL_l, HR_u, HR_l, FL_u, FL_l, FR_u, FR_l]
        """
        # Calculate continuous global phase
        global_phase = self.angular_freq * t + self.global_phase_offset
        
        # Apply leg-specific offsets
        phases = (global_phase + self.phase_offsets) % (2 * np.pi)
        
        joint_angles = np.zeros(8)
        
        for i in range(4):
            # Determine leg side (0=HL, 1=HR, 2=FL, 3=FR) -> Evens are Left, Odds are Right
            is_left = (i % 2 == 0)
            stride_scale = self.left_stride_scale if is_left else self.right_stride_scale
            
            phase = phases[i]
            if np.sin(phase) > 0:  # Swing
                x = self.step_length * stride_scale * np.cos(phase)
                z = self.step_height * np.sin(phase)
            else:  # Stance
                x = self.step_length * stride_scale * np.cos(phase)
                z = 0.0

            z -= 0.23 # Vertical offset

            # IK Logic
            reference_q_robot = self.q0_[i * 2: i * 2 + 2]
            reference_q_generator = reference_q_robot.copy()
            reference_q_generator[0] += np.pi / 2 

            q1_gen, q2_gen = self._leg_inverse_kinematics(x, z, reference_q_generator)

            q1_robot = q1_gen + np.pi / 2 

            joint_angles[i * 2] = q1_robot
            joint_angles[i * 2 + 1] = q2_gen

        return joint_angles

    def _leg_inverse_kinematics(self, x, z, reference_q):
        dist_sq = x ** 2 + z ** 2
        dist = np.sqrt(dist_sq)

        if dist > (self.l1 + self.l2) or dist < abs(self.l1 - self.l2):
            return reference_q

        q2_cos_arg = np.clip((dist_sq - self.l1 ** 2 - self.l2 ** 2) / (2 * self.l1 * self.l2), -1.0, 1.0)
        q2_sol1 = np.arccos(q2_cos_arg)
        q2_sol2 = -q2_sol1

        psi = np.arctan2(z, x)
        phi_cos_arg = np.clip((dist_sq + self.l1 ** 2 - self.l2 ** 2) / (2 * dist * self.l1), -1.0, 1.0)
        phi = np.arccos(phi_cos_arg)

        q1_sol1 = psi - phi
        q1_sol2 = psi + phi

        sol1 = np.array([q1_sol1, q2_sol1])
        sol2 = np.array([q1_sol2, q2_sol2])

        dist1 = np.sum((sol1 - reference_q) ** 2)
        dist2 = np.sum((sol2 - reference_q) ** 2)

        return sol1 if dist1 < dist2 else sol2


# ------------------------- Dog Controller -------------------------
class DogController:
    """
    Single-robot controller with numpy interface.
    Includes:
    1. Analytic Gait Generator
    2. Spine Impedance Control
    3. Directional PD Controller (Heading correction)
    """

    def __init__(
        self,
        urdf_path: str,
        control_period: float = 0.005,
        gait_frequency_hz: float = 3.0,
        gait_type: str = 'trot',
        device: str = "cpu",
    ):
        self.device = device
        self.control_period = control_period

        # Leg PD gains
        self.leg_Kp = 8.0
        self.leg_Kd = 0.1
        
        # Track simulation time for smooth transitions
        self.last_known_time = 0.0

        # --- Initialize Gait Generator ---
        self.gait_generator = GaitTrajectoryGenerator(
            step_length=0.12,
            step_height=0.06,
            frequency=gait_frequency_hz,
            thigh_length=0.151, 
            tibia_length=0.151,
            gait_type=gait_type
        )
        
        # --- Initialize Spine Controller ---
        self.spine_controller = SpineImpedanceController(
            urdf_path, num_envs=1, device=device
        )
        
        # --- Direction Control State ---
        self.target_yaw = 0.0  # Target heading (rad)
        self.last_yaw_error = 0.0 # For D-term
        self.last_phase = 0.0
        
        # Direction PD Gains
        # Kp: Stiffness (Correction strength)
        # Kd: Damping (Prevents oscillation)
        self.dir_kp = -0.3
        self.dir_kd = -0.1

    # --- Setters for dynamic modification ---
    def set_gait_frequency(self, frequency_hz: float):
        """
        Updates gait frequency smoothly using the last known simulation time.
        """
        self.gait_generator.set_frequency(frequency_hz, self.last_known_time)
    
    def set_gait_type(self, gait_type: str):
        self.gait_generator.set_gait_type(gait_type)

    def reset(self):
        self.last_yaw_error = 0.0
        self.last_phase = 0.0

    def _quat_to_yaw(self, q):
        """
        Converts MuJoCo quaternion [w, x, y, z] to Yaw (Z-axis rotation).
        """
        w, x, y, z = q
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def compute_torques(
        self,
        time: float,          # current time
        q_spine_full: np.ndarray,
        v_spine_full: np.ndarray, 
        q_legs_actual: np.ndarray, 
        v_legs_actual: np.ndarray, 
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute control torques for the robot.
        """
        
        # Update internal time tracker for frequency switching
        self.last_known_time = time
        
        # --- DIRECTION CONTROL LOGIC (Start of Gait Cycle) ---
        current_phase = self.gait_generator.get_current_phase(time)
        
        # Check for phase wrap-around (2pi -> 0)
        if current_phase < self.last_phase:
            # 1. Get Current Yaw
            # q_spine_full structure: [pos(3), quat(4), joints...]
            quat = q_spine_full[3:7] 
            current_yaw = self._quat_to_yaw(quat)
            
            # 2. Calculate Error
            error = self.target_yaw - current_yaw
            
            # 3. Calculate Derivative (D-term)
            # Since this runs once per gait cycle, dt is the gait period.
            # We use the raw delta for simplicity as the interval is roughly constant.
            derivative = error - self.last_yaw_error
            self.last_yaw_error = error
            
            # 4. Calculate PD Output (Steering Offset)
            # Positive Output -> Turn Left -> Increase Right Stride
            steering_cmd = (self.dir_kp * error) + (self.dir_kd * derivative)
            
            # 5. Apply to Gait Generator
            # print(f"[DirControl t={time:.2f}] Yaw: {current_yaw:.3f}, Err: {error:.3f}, Cmd: {steering_cmd:.3f}")
            self.gait_generator.set_steering_offset(steering_cmd)
            
        self.last_phase = current_phase

        # --- STANDARD CONTROL LOOP ---
        
        # 1. Generate desired leg angles
        q_des_legs_np = self.gait_generator.get_joint_angles(time)
        v_des_legs_np = np.zeros_like(q_des_legs_np)

        # 2. Leg PD Control
        q_err = q_des_legs_np - q_legs_actual
        v_err = v_des_legs_np - v_legs_actual
        leg_torques_np = self.leg_Kp * q_err + self.leg_Kd * v_err

        # 3. Spine Impedance Control
        spine_torques_np = self.spine_controller.compute_torque(q_spine_full, v_spine_full)

        # 4. Assemble full torque vector
        full_tau = np.zeros(11)
        full_tau[0:2] = leg_torques_np[0:2]   # Hind Left
        full_tau[2:4] = leg_torques_np[2:4]   # Hind Right
        full_tau[4:7] = spine_torques_np      # Spine
        full_tau[7:9] = leg_torques_np[4:6]   # Front Left
        full_tau[9:11] = leg_torques_np[6:8]  # Front Right

        return full_tau, q_des_legs_np