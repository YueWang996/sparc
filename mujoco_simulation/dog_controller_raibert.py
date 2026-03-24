import numpy as np
from dataclasses import dataclass
from spine_impedance_controller import SpineImpedanceController

@dataclass
class LegTrajectoryState:
    """Trajectory state for each leg"""
    x_liftoff: float = 0.0
    x_touchdown: float = 0.0
    last_phase: float = 0.0
    in_swing: bool = False

class GaitTrajectoryGenerator:
    def __init__(self, step_height, frequency,
                 thigh_length, tibia_length, gait_type='trot'):
        self.step_height = step_height
        self.l1 = thigh_length
        self.l2 = tibia_length
        
        # Initial robot joint angles (Robot Frame)
        self.q0_ = np.array([
            -0.785398, 1.5708,   # Hind Left
            -0.785398, 1.5708,   # Hind Right
            0.785398, -1.5708,   # Front Left
            0.785398, -1.5708    # Front Right
        ])
        
        self.global_phase_offset = 0.0
        self.frequency = frequency
        self.angular_freq = 2 * np.pi * frequency
        self.set_gait_type(gait_type)
        
        # Raibert parameters
        self.raibert_gain_hind = 0.08
        self.raibert_gain_front = 0.12
        self.feedforward_gain = 0.5
        self.max_stride = 0.2
        
        self.pitch_gain_front = 0.0
        self.pitch_rate_gain_front = 0.0
        
        self.leg_states = [LegTrajectoryState() for _ in range(4)]
        self.standing_height = -0.23

    def set_gait_type(self, gait_type):
        self.gait_type = gait_type
        if gait_type == 'trot':
            self.phase_offsets = np.array([0, np.pi, np.pi, 0])
        elif gait_type == 'bound':
            self.phase_offsets = np.array([0, 0, np.pi, np.pi])
        else:
            raise ValueError(f"Unknown gait type: {gait_type}")

    def get_current_phase(self, t):
        return (self.angular_freq * t + self.global_phase_offset) % (2 * np.pi)

    def _compute_raibert_touchdown(self, v_use, v_cmd, raibert_gain, T_stance):
        touchdown_x = self.feedforward_gain * T_stance * v_cmd + raibert_gain * (v_use - v_cmd)
        return np.clip(touchdown_x, -self.max_stride, self.max_stride)

    def _swing_trajectory_cosine(self, t_normalized, x_start, x_end, z_apex):
        alpha = (1 - np.cos(np.pi * t_normalized)) / 2
        x = x_start + alpha * (x_end - x_start)
        z = z_apex * np.sin(np.pi * t_normalized)
        return x, z

    def get_swing_target(self, t, leg_idx, v_hind_filt, v_front_filt, v_cmd, pitch=0.0, pitch_rate=0.0):
        """
        Get swing target position for a single leg.
        """
        global_phase = self.angular_freq * t + self.global_phase_offset
        phase = (global_phase + self.phase_offsets[leg_idx]) % (2 * np.pi)
        
        state = self.leg_states[leg_idx]
        is_front = (leg_idx >= 2)
        
        T_stance = 0.5 / self.frequency
        
        # Select velocity and gain based on leg position
        if is_front:
            v_use = v_front_filt
            raibert_gain = self.raibert_gain_front
        else:
            v_use = v_hind_filt
            raibert_gain = self.raibert_gain_hind
        
        in_swing_now = np.sin(phase) > 0
        
        # Stance -> Swing
        if in_swing_now and not state.in_swing:
            state.x_liftoff = -state.x_touchdown
            
            # Calculate touchdown point using selected torso velocity
            touchdown_x = self._compute_raibert_touchdown(v_use, v_cmd, raibert_gain, T_stance)
            
            if is_front:
                pitch_comp = self.pitch_gain_front * pitch + self.pitch_rate_gain_front * pitch_rate
                touchdown_x += pitch_comp
                touchdown_x = np.clip(touchdown_x, -self.max_stride, self.max_stride)
                touchdown_x = -touchdown_x
            
            state.x_touchdown = touchdown_x
        
        state.in_swing = in_swing_now
        state.last_phase = phase
        
        if in_swing_now:
            t_normalized = np.clip(phase / np.pi, 0, 1)
            x, z = self._swing_trajectory_cosine(
                t_normalized, state.x_liftoff, state.x_touchdown, self.step_height
            )
        else:
            return 0.0, 0.0, False
        
        z += self.standing_height
        return x, z, True

    def leg_inverse_kinematics(self, x, z, reference_q):
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


class LegController:
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2
        self.swing_kp = 10.0
        self.swing_kd = 0.1
        self.stance_Fz = 0.0
        self.stance_Fx = 0.0
    
    def compute_jacobian(self, q1_robot, q2):
        q1_math = q1_robot - np.pi / 2
        s1 = np.sin(q1_math)
        c1 = np.cos(q1_math)
        s12 = np.sin(q1_math + q2)
        c12 = np.cos(q1_math + q2)
        J11 = -self.l1 * s1 - self.l2 * s12
        J12 = -self.l2 * s12
        J21 = self.l1 * c1 + self.l2 * c12
        J22 = self.l2 * c12
        J = np.array([[J11, J12], [J21, J22]])
        return J
    
    def compute_torque(self, q, v, q_des, in_swing, is_front=False, Fx_cmd=None, Fz_cmd=None):
        q1, q2 = q
        if in_swing:
            q_err = q_des - q
            v_err = -v
            tau = self.swing_kp * q_err + self.swing_kd * v_err
        else:
            Fx = Fx_cmd if Fx_cmd is not None else self.stance_Fx
            Fz = Fz_cmd if Fz_cmd is not None else self.stance_Fz
            F = np.array([Fx, Fz])
            J = self.compute_jacobian(q1, q2)
            tau = J.T @ F
        return tau


class DogControllerRaibert:
    def __init__(self, urdf_path, device="cpu"):
        self.spine_controller = SpineImpedanceController(urdf_path, num_envs=1, device=device)
        
        self.gait_generator = GaitTrajectoryGenerator(
            step_height=0.08,
            frequency=3.1,
            thigh_length=0.151,
            tibia_length=0.151,
            gait_type='bound'
        )
        
        self.leg_controllers = [LegController(l1=0.151, l2=0.151) for _ in range(4)]
        self.v_cmd_x = 0.0
        self.last_time = 0.0
        
        # Base feedforward forces
        self.hind_Fz = -25.0 
        self.hind_Fx = 10.0   
        self.front_Fz = -25.0 
        self.front_Fx = 0.0

        self.target_pitch = -0.1
        self.kp_pitch = 50.0
        self.kd_pitch = 5.0
        
        # Velocity tracking controller (hind legs)
        self.kp_vel_x = 40.0
        self.ki_vel_x = 0.1
        self.vel_x_integral = 0.0
        # Acceleration limit to prevent tipping during startup
        self.v_cmd_filtered = 0.3  # Smoothed target velocity
        self.max_accel = 1.0       # Max acceleration 1.0 m/s^2 (tunable: smaller = more stable but slower)
        
        # Height maintenance controller (all legs)
        self.desired_height = 0.23
        self.filtered_height = 0.25
        self.hind_kp_z = 80.0   
        self.hind_kd_z = 4.0
        self.front_kp_z = 80.0  
        self.front_kd_z = 10.0
        
        # Attitude PI controller
        self.target_roll = 0.0
        self.kp_roll = 30.0
        self.ki_roll = 0.1
        self.roll_integral = 0.0
        
        self.target_yaw = 0.0
        self.kp_yaw = 20.0
        self.ki_yaw = 0.05
        self.yaw_integral = 0.0
        
        # State estimation: separate velocity estimates for front and hind legs
        # Hind legs
        self.hind_v_accumulator = 0.0
        self.hind_v_sample_count = 0
        self.hind_v_stance_avg = 1.0
        self.last_hind_in_stance = False
        self.filtered_v_hind = 0.0
        
        # Front legs
        self.front_v_accumulator = 0.0
        self.front_v_sample_count = 0
        self.front_v_stance_avg = 1.0
        self.last_front_in_stance = False
        self.filtered_v_front = 0.0
        
        self.filtered_pitch_rate = 0.0
        self.last_pitch = 0.0

    def set_command(self, v_x, yaw_target=0.0):
        self.v_cmd_x = v_x
        self.target_yaw = yaw_target

    def _quat_to_rpy(self, q):
        w, x, y, z = q
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2
        else:
            pitch = np.arcsin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def compute_torques(self, time, q_full, v_full, q_legs, v_legs, 
                        v_body_local_front, v_body_local_hind, pitch=0.0):
        dt = time - self.last_time
        if dt <= 0: dt = 0.001
        self.last_time = time
        
        # State computation
        current_z_raw = q_full[2] 
        current_vz = v_full[2]    
        quat = q_full[3:7]
        roll_curr, pitch_curr, yaw_curr = self._quat_to_rpy(quat)
        
        # Low-pass filter (height)
        alpha_h = 0.2
        self.filtered_height = (1 - alpha_h) * self.filtered_height + alpha_h * current_z_raw
        
        # Velocity estimation: separate front and hind legs
        current_phase = self.gait_generator.get_current_phase(time)
        
        # Hind leg velocity estimation (phase offset typically 0 or pi)
        # Use left hind leg (Leg 0) phase as reference
        hind_phase = (current_phase + self.gait_generator.phase_offsets[0]) % (2 * np.pi)
        hind_in_stance = np.sin(hind_phase) <= 0
        
        if hind_in_stance:
            self.hind_v_accumulator += v_body_local_hind[0]
            self.hind_v_sample_count += 1
        
        if not hind_in_stance and self.last_hind_in_stance:
            if self.hind_v_sample_count > 0:
                self.hind_v_stance_avg = self.hind_v_accumulator / self.hind_v_sample_count
            self.hind_v_accumulator = 0.0
            self.hind_v_sample_count = 0
        self.last_hind_in_stance = hind_in_stance
        
        # Front leg velocity estimation (different phase offset)
        # Use left front leg (Leg 2) phase as reference
        front_phase = (current_phase + self.gait_generator.phase_offsets[2]) % (2 * np.pi)
        front_in_stance = np.sin(front_phase) <= 0
        
        if front_in_stance:
            self.front_v_accumulator += v_body_local_front[0]
            self.front_v_sample_count += 1
            
        if not front_in_stance and self.last_front_in_stance:
            if self.front_v_sample_count > 0:
                self.front_v_stance_avg = self.front_v_accumulator / self.front_v_sample_count
            self.front_v_accumulator = 0.0
            self.front_v_sample_count = 0
        self.last_front_in_stance = front_in_stance
        
        # Filtering
        alpha_v = 0.5
        self.filtered_v_hind = (1 - alpha_v) * self.filtered_v_hind + alpha_v * v_body_local_hind[0]
        self.filtered_v_front = (1 - alpha_v) * self.filtered_v_front + alpha_v * v_body_local_front[0]
        
        # Select velocity to use (stance average if available, otherwise filtered value)
        v_hind_final = self.hind_v_stance_avg if self.hind_v_stance_avg != 0 else self.filtered_v_hind
        v_front_final = self.front_v_stance_avg if self.front_v_stance_avg != 0 else self.filtered_v_front
        
        # Pitch Rate
        pitch_rate = (pitch_curr - self.last_pitch) / dt
        self.filtered_pitch_rate = (1 - alpha_v) * self.filtered_pitch_rate + alpha_v * pitch_rate
        self.last_pitch = pitch_curr
        
        # Controller computation (Global Adjustments)

        # Target velocity smoothing (Acceleration Limiter)
        # Limit velocity command changes per step to simulate realistic physical acceleration
        diff = self.v_cmd_x - self.v_cmd_filtered
        limit = self.max_accel * dt
        # Clip ensures changes don't exceed physical limits
        change = np.clip(diff, -limit, limit)
        self.v_cmd_filtered += change
        
        # Velocity P control (Velocity Tracking) - only for hind legs, so use hind leg velocity for error
        vel_error = self.v_cmd_filtered - v_hind_final
        self.vel_x_integral += vel_error * dt
        self.vel_x_integral = np.clip(self.vel_x_integral, -20.0, 20.0)
        Fx_vel_adjustment = self.kp_vel_x * vel_error + self.ki_vel_x * self.vel_x_integral
        
        # Height PD control
        z_error = self.desired_height - self.filtered_height
        Fz_hind_height_adjustment = -(self.hind_kp_z * z_error - self.hind_kd_z * current_vz)
        Fz_front_height_adjustment = -(self.front_kp_z * z_error - self.front_kd_z * current_vz)

        pitch_error = self.target_pitch - pitch_curr
        pitch_rate = self.filtered_pitch_rate
        Fz_pitch_adjustment = self.kp_pitch * pitch_error - self.kd_pitch * pitch_rate
        
        # Attitude PI control
        roll_error = self.target_roll - roll_curr
        self.roll_integral += roll_error * dt
        self.roll_integral = np.clip(self.roll_integral, -0.5, 0.5)
        Fz_roll_adjustment = self.kp_roll * roll_error + self.ki_roll * self.roll_integral
        
        yaw_error = self.target_yaw - yaw_curr
        if yaw_error > np.pi: yaw_error -= 2*np.pi
        if yaw_error < -np.pi: yaw_error += 2*np.pi
        self.yaw_integral += yaw_error * dt
        self.yaw_integral = np.clip(self.yaw_integral, -0.5, 0.5)
        Fx_yaw_adjustment = self.kp_yaw * yaw_error + self.ki_yaw * self.yaw_integral
        
        # Leg loop
        leg_torques = np.zeros(8)
        
        for i in range(4):
            is_front = (i >= 2)
            is_left = (i % 2 == 0)
            
            q_leg = q_legs[i*2 : i*2+2]
            v_leg = v_legs[i*2 : i*2+2]
            
            # Pass both front and hind leg velocities, automatically selected internally
            x_target, z_target, in_swing = self.gait_generator.get_swing_target(
                time, i, v_hind_final, v_front_final, self.v_cmd_filtered,
                pitch=pitch_curr, pitch_rate=self.filtered_pitch_rate
            )
            
            if in_swing:
                ref_q_robot = self.gait_generator.q0_[i*2 : i*2+2]
                ref_q_math = ref_q_robot.copy()
                ref_q_math[0] += np.pi / 2 
                q_sol_math = self.gait_generator.leg_inverse_kinematics(x_target, z_target, ref_q_math)
                q_des = q_sol_math.copy()
                q_des[0] += np.pi / 2
                
                tau = self.leg_controllers[i].compute_torque(
                    q_leg, v_leg, q_des, in_swing=True, is_front=is_front
                )
            else:
                curr_Fx = self.front_Fx if is_front else self.hind_Fx
                curr_Fz = self.front_Fz if is_front else self.hind_Fz
                
                if is_front:
                    # Front legs: apply braking force when velocity is too high
                    front_vel_gain = 20.0  # tunable
                    curr_Fx -= front_vel_gain * vel_error  # Note: when vel_error > 0, reduce braking to accelerate
                    curr_Fz += Fz_pitch_adjustment
                else:
                    # Hind legs: original logic
                    curr_Fx += Fx_vel_adjustment
                    curr_Fz -= Fz_pitch_adjustment
                
                curr_Fz += Fz_hind_height_adjustment if not is_front else Fz_front_height_adjustment
                
                if is_left:
                    curr_Fz -= Fz_roll_adjustment 
                else:
                    curr_Fz += Fz_roll_adjustment 
                
                if not is_front:
                    if is_left:
                        curr_Fx -= Fx_yaw_adjustment
                    else:
                        curr_Fx += Fx_yaw_adjustment 
                
                curr_Fz = np.clip(curr_Fz, -80.0, -2.0) 
                curr_Fx = np.clip(curr_Fx, -20.0, 40.0)

                tau = self.leg_controllers[i].compute_torque(
                    q_leg, v_leg, q_des=None, in_swing=False, is_front=is_front,
                    Fx_cmd=curr_Fx, Fz_cmd=curr_Fz
                )
            
            leg_torques[i*2 : i*2+2] = tau
        
        spine_torques = self.spine_controller.compute_torque(q_full, v_full)
        
        full_tau = np.zeros(11)
        full_tau[0:2] = leg_torques[0:2]
        full_tau[2:4] = leg_torques[2:4]
        full_tau[4:7] = spine_torques
        full_tau[7:9] = leg_torques[4:6]
        full_tau[9:11] = leg_torques[6:8]
        
        return full_tau, leg_torques