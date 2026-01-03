import mujoco as mj
import numpy as np
from scipy.signal import find_peaks
from dog_controller_no_cpg import DogController
from mujoco_interface import MuJoCoInterface
import mujoco.viewer
import time
import torch

# --- Helper Class: Low Pass Filter ---
class LowPassFilter:
    def __init__(self, cutoff_freq, dt):
        self.rc = 1.0 / (2 * np.pi * cutoff_freq)
        self.dt = dt
        self.alpha = self.dt / (self.rc + self.dt)
        self.last_val = None

    def filter(self, value):
        if self.last_val is None:
            self.last_val = value
            return value
        filtered_val = self.alpha * value + (1 - self.alpha) * self.last_val
        self.last_val = filtered_val
        return filtered_val

# --- Helper Function: Quaternion to Euler ---
def quat_to_rpy(q):
    w, x, y, z = q
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])

def get_relative_state(sim, base_name="hind_body", tip_name="front_body"):
    base_id = sim.model.body(base_name).id
    p_base = sim.data.xpos[base_id]
    R_base = sim.data.xmat[base_id].reshape(3, 3)
    tip_id = sim.model.body(tip_name).id
    p_tip = sim.data.xpos[tip_id]
    R_tip = sim.data.xmat[tip_id].reshape(3, 3)
    diff_world = p_tip - p_base
    pos_rel = R_base.T @ diff_world
    R_rel = R_base.T @ R_tip
    pitch_rel = np.arctan2(-R_rel[2, 0], np.sqrt(R_rel[2, 1]**2 + R_rel[2, 2]**2))
    return np.array([pos_rel[0], pos_rel[2], pitch_rel])

# --- Calculate Shortest Phase Diff ---
def get_phase_diff(current, target):
    """
    Calculates the shortest distance between two angles in radians.
    Returns value between -pi and pi.
    Positive result means 'current' is ahead of 'target' (or target is lagging).
    """
    diff = (current - target + np.pi) % (2 * np.pi) - np.pi
    return diff

# --- MuJoCoSimulation Class ---
class MuJoCoSimulation:
    def __init__(self, model_path: str, control_frequency: float = 200.0):
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        self.interface = MuJoCoInterface(self.model, self.data)
        self.control_frequency = control_frequency
        self.control_period = 1.0 / control_frequency
        self.last_control_time = 0.0
        
    def control_callback(self):
        self.data.ctrl[:] = 0.0
        
    def run(self, max_time: float = None):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Set camera view
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            viewer.cam.distance = 3.0
            viewer.cam.lookat = np.array([0.0, 0.0, 0.5])
            
            start_time = self.data.time
            
            while viewer.is_running():
                if max_time is not None and (self.data.time - start_time) >= max_time:
                    print(f"\nSimulation reached max time: {max_time}s")
                    break
                
                if self.data.time - self.last_control_time >= self.control_period:
                    self.control_callback()
                    self.last_control_time = self.data.time
                
                mj.mj_step(self.model, self.data)
                viewer.sync()

# =========================================================================
#  Unified Mass Update Function
# =========================================================================
def update_robot_mass_properties(sim, controller, body_names, percent_change):
    """
    Scales the mass and inertia of specified bodies in:
    1. The MuJoCo simulation model (for Physics).
    2. The Bard Kinematic Chains (for Model-Based Control).
    """
    scale_factor = 1.0 + (percent_change / 100.0)
    print(f"\n[Configuration] Scaling Mass & Inertia by {percent_change}% (Factor: {scale_factor:.4f})")
    
    # ---------------------------------------------------------
    # 1. Update MuJoCo Model (Physics Engine)
    # ---------------------------------------------------------
    for name in body_names:
        bid = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, name)
        if bid != -1:
            sim.model.body_mass[bid] *= scale_factor
            sim.model.body_inertia[bid] *= scale_factor
            print(f"  [MuJoCo] Scaled body '{name}'")
        else:
            print(f"  [MuJoCo] Warning: Body '{name}' not found.")

    # ---------------------------------------------------------
    # 2. Update Bard Chains (Controller)
    # ---------------------------------------------------------
    # We must update both chains: one used for Dynamics (floating) 
    # and one used for Kinematics (body).
    chains_to_update = [
        controller.spine_controller.chain_floating,
        controller.spine_controller.chain_body
    ]
    
    for chain in chains_to_update:
        # A. Modify the Link objects in the chain structure
        updates_made = False
        for name in body_names:
            try:
                # Find the frame and its link
                frame = chain.find_frame(name)
                if frame is None:
                    continue
                    
                link = frame.link
                if link.inertial is None:
                    continue
                
                # Unpack current properties (tuple: origin, mass, inertia)
                # Note: origin is a Transform3d object, mass is float, inertia is Tensor
                origin, mass_val, inertia_tensor = link.inertial
                
                # Apply scaling
                new_mass = mass_val * scale_factor
                new_inertia = inertia_tensor * scale_factor
                
                # Update the Link object with a NEW tuple (tuples are immutable)
                link.inertial = (origin, new_mass, new_inertia)
                
                updates_made = True
                # print(f"  [Bard]   Updated Link properties for '{name}'")

            except Exception as e:
                print(f"  [Bard]   Error updating '{name}': {e}")

        # B. Trigger Chain Recomputation
        # This calls the internal method of the Chain class to rebuild the 
        # spatial inertia tensor using the updated Link properties.
        # This ensures all math (parallel axis theorem, rotations) is consistent.
        if updates_made:
            chain.spatial_inertias = chain._precompute_all_spatial_inertias(chain.n_nodes)
            print(f"  [Bard]   Recomputed spatial inertias for chain (nodes: {chain.n_nodes})")

    total_mass = mj.mj_getTotalmass(sim.model)
    print(f"  > New Total Robot Mass (MuJoCo): {total_mass:.4f} kg")
    print("---------------------------------------------------\n")

def main():
    # ---------------- Setup ----------------
    model_path = "/Users/justin/PycharmProjects/spine-sim/asset/spined_dog.xml"
    urdf_path = "/Users/justin/PycharmProjects/spine-sim/asset/spined_dog_spine_dyn.urdf"

    CONTROL_FREQUENCY = 200.0
    CONTROL_PERIOD = 1.0 / CONTROL_FREQUENCY
    
    sim = MuJoCoSimulation(model_path, control_frequency=CONTROL_FREQUENCY)
    
    # --- Controller Setup ---
    INITIAL_FREQ = 3.1
    current_gait_freq = INITIAL_FREQ
    
    controller = DogController(
        urdf_path=urdf_path,
        control_period=CONTROL_PERIOD,
        gait_frequency_hz=INITIAL_FREQ,
    )
    controller.gait_generator.extra_phase_shift = np.pi

    # --- Mass Modification Parameters ---
    MASS_MOD_PERCENT = 0.0  # +50% Mass Increase
    TARGET_BODIES = ["front_body", "hind_body"]

    # --- Initialization (Joints) ---
    sim.interface.set_joint_pos("joint_hind_spine", -0.6)
    sim.interface.set_joint_pos("joint_front_spine", 1.2)
    sim.interface.set_joint_pos("joint_front_body", -0.6)
    
    q_legs_init = controller.gait_generator.get_joint_angles(0.0)

    init_positions = {
        "joint_hind_left_leg_upper": q_legs_init[0],
        "joint_hind_left_leg_lower": q_legs_init[1],
        "joint_hind_right_leg_upper": q_legs_init[2],
        "joint_hind_right_leg_lower": q_legs_init[3],
        "joint_front_left_leg_upper": q_legs_init[4],
        "joint_front_left_leg_lower": q_legs_init[5],
        "joint_front_right_leg_upper": q_legs_init[6],
        "joint_front_right_leg_lower": q_legs_init[7]
    }
    for name, pos in init_positions.items():
        sim.interface.set_joint_pos(name, pos)
    
    controller.set_gait_type('bound')
    controller.set_gait_frequency(INITIAL_FREQ)
    
    # --- Controller Parameters ---
    target_kp = np.array([800.0, 2000.0, 15.0])
    target_kd = np.array([8.0, 5.0, 0.5])
    target_x_des = np.array([0.268, 0.0, 0.0])
    
    initial_kp = np.array([800.0, 2000.0, 15.0])
    initial_kd = np.array([15.0, 5.0, 1.5])

    controller.spine_controller.x_des = np.array([0.273, 0.0, 0.0])
    controller.spine_controller.kp = initial_kp
    controller.spine_controller.kd = initial_kd

    # --- Experiment Timing & State ---
    params_switched = False
    PARAM_SWITCH_TIME = 4.0 
    
    MAX_TILT_ANGLE = np.deg2rad(90.0) 
    start_pos = sim.interface.get_body_pose("hind_body")[0].copy()

    # --- Phase-Based Adaptation Setup ---
    spine_lpf = LowPassFilter(cutoff_freq=6.0, dt=CONTROL_PERIOD)
    spine_x_buffer = []
    spine_t_buffer = []
    
    baseline_phases = []     # To store phases during calibration
    target_peak_phase = None # The "Ideal" phase we learn
    calibration_done = False
    
    PHASE_GAIN = 1.0         # Hz per Radian of lag.
    MIN_FREQ = 2.8           # Floor for safety
    MAX_FREQ = 3.2           # Ceiling
    
    # ** NEW ** Delayed Update State
    last_gen_phase = 0.0
    pending_freq_update = None

    # ---------------- Control Loop ----------------
    def custom_control():
        nonlocal params_switched, spine_x_buffer, spine_t_buffer
        nonlocal current_gait_freq, baseline_phases, target_peak_phase, calibration_done
        nonlocal last_gen_phase, pending_freq_update
        
        current_time = sim.data.time
        
        # ----------------------------------------------------
        # 0. Check for Gait Cycle Completion (Wrap-Around)
        # ----------------------------------------------------
        now_phase = controller.gait_generator.get_current_phase(current_time)
        
        # If current phase is LESS than last phase, we wrapped from 2pi -> 0
        if now_phase < last_gen_phase:
            # We are at the start of a new cycle. Apply pending updates now.
            if pending_freq_update is not None:
                print(f"  [Cycle Sync] Updating Freq: {current_gait_freq:.2f} -> {pending_freq_update:.2f} Hz")
                controller.set_gait_frequency(pending_freq_update)
                current_gait_freq = pending_freq_update
                pending_freq_update = None
        
        last_gen_phase = now_phase

        # ----------------------------------------------------
        # 1. Apply Dynamic Mass Switch
        # ----------------------------------------------------
        if current_time >= PARAM_SWITCH_TIME and not params_switched:
            # Update Controller Gains
            controller.spine_controller.kp = target_kp
            controller.spine_controller.kd = target_kd
            controller.spine_controller.x_des = target_x_des
            
            # Update Mass Physically
            update_robot_mass_properties(sim, controller, TARGET_BODIES, MASS_MOD_PERCENT)
            params_switched = True
            print(f"[t={current_time:.2f}] Mass increased by {MASS_MOD_PERCENT}%. Phase Adaptation Active.")
            
        # 2. Safety Check
        base_pos, base_quat = sim.interface.get_body_pose("hind_body")
        rpy = quat_to_rpy(base_quat)
        if abs(rpy[0]) > MAX_TILT_ANGLE or abs(rpy[1]) > MAX_TILT_ANGLE:
            raise RuntimeError(f"Fall Detected! Roll: {np.rad2deg(rpy[0]):.1f}, Pitch: {np.rad2deg(rpy[1]):.1f}")
        
        # 3. State Gathering
        base_vel, base_angular_vel = sim.interface.get_body_velocity("hind_body")
        q_legs = np.array([
            sim.interface.get_joint_pos("joint_hind_left_leg_upper"), sim.interface.get_joint_pos("joint_hind_left_leg_lower"),
            sim.interface.get_joint_pos("joint_hind_right_leg_upper"), sim.interface.get_joint_pos("joint_hind_right_leg_lower"),
            sim.interface.get_joint_pos("joint_front_left_leg_upper"), sim.interface.get_joint_pos("joint_front_left_leg_lower"),
            sim.interface.get_joint_pos("joint_front_right_leg_upper"), sim.interface.get_joint_pos("joint_front_right_leg_lower"),
        ])
        v_legs = np.array([
            sim.interface.get_joint_vel("joint_hind_left_leg_upper"), sim.interface.get_joint_vel("joint_hind_left_leg_lower"),
            sim.interface.get_joint_vel("joint_hind_right_leg_upper"), sim.interface.get_joint_vel("joint_hind_right_leg_lower"),
            sim.interface.get_joint_vel("joint_front_left_leg_upper"), sim.interface.get_joint_vel("joint_front_left_leg_lower"),
            sim.interface.get_joint_vel("joint_front_right_leg_upper"), sim.interface.get_joint_vel("joint_front_right_leg_lower"),
        ])
        q_spine = np.array([sim.interface.get_joint_pos("joint_hind_spine"), sim.interface.get_joint_pos("joint_front_spine"), sim.interface.get_joint_pos("joint_front_body")])
        v_spine = np.array([sim.interface.get_joint_vel("joint_hind_spine"), sim.interface.get_joint_vel("joint_front_spine"), sim.interface.get_joint_vel("joint_front_body")])
        q_full = np.concatenate([base_pos, base_quat, q_spine])
        v_full = np.concatenate([base_vel, base_angular_vel, v_spine])

        # 4. Compute Control
        full_tau, _ = controller.compute_torques(sim.data.time, q_full, v_full, q_legs, v_legs)
        
        motor_names = [
            "motor_hind_left_leg_upper", "motor_hind_left_leg_lower",
            "motor_hind_right_leg_upper", "motor_hind_right_leg_lower",
            "motor_hind_spine", "motor_front_spine", "motor_front_body",
            "motor_front_left_leg_upper", "motor_front_left_leg_lower",
            "motor_front_right_leg_upper", "motor_front_right_leg_lower"
        ]
        for i, name in enumerate(motor_names):
            sim.interface.set_joint_torque(name, full_tau[i])

        # --------------------------------------------------------
        # 5. PHASE-BASED ADAPTATION LOGIC (Detection)
        # --------------------------------------------------------
        
        # A. Filter Spine State (Get Relative X Extension)
        spine_state = get_relative_state(sim)
        filtered_rel_x = spine_lpf.filter(spine_state[0])
        
        # B. Buffer for Peak Detection
        spine_x_buffer.append(filtered_rel_x)
        spine_t_buffer.append(current_time)
        if len(spine_x_buffer) > 200: 
            spine_x_buffer.pop(0)
            spine_t_buffer.pop(0)
            
        # C. Detect Peaks
        if len(spine_x_buffer) > 5:
            val_prev = spine_x_buffer[-2]
            val_curr = spine_x_buffer[-1]
            val_old  = spine_x_buffer[-3]
            
            # Simple local maxima detection
            if val_prev > val_curr and val_prev > val_old and val_prev > 0.02:
                peak_time = spine_t_buffer[-2]
                
                # Get the Gait Generator Phase at the exact moment of physical peak
                current_gen_phase = controller.gait_generator.get_current_phase(peak_time)
                
                # --- PHASE 1: CALIBRATION (Before Switch) ---
                if current_time < PARAM_SWITCH_TIME:
                    # Ignore startup transient (t < 2.0)
                    if current_time > 2.0:
                        baseline_phases.append(current_gen_phase)
                
                # --- PHASE 2: CALCULATE TARGET (At Switch) ---
                elif not calibration_done:
                    if len(baseline_phases) > 0:
                        # Circular Mean
                        sin_sum = np.sum(np.sin(baseline_phases))
                        cos_sum = np.sum(np.cos(baseline_phases))
                        target_peak_phase = np.arctan2(sin_sum, cos_sum)
                        print(f"--> Calibration Complete. Target Phase: {np.rad2deg(target_peak_phase):.1f} deg")
                    else:
                        target_peak_phase = 0.0 
                        print("--> Warning: Calibration failed (no peaks), defaulting to 0.0")
                    calibration_done = True
                
                # --- PHASE 3: ADAPTATION (After Switch) ---
                else:
                    # Calculate Lag: Positive if Generator > Target (Body is late)
                    phase_lag = get_phase_diff(current_gen_phase, target_peak_phase)
                    
                    # Correction: Drop freq if lag is positive
                    freq_correction = PHASE_GAIN * phase_lag
                    
                    if phase_lag > 0.1: # Threshold ~5.7 degrees
                        new_freq = current_gait_freq - freq_correction
                        new_freq = np.clip(new_freq, MIN_FREQ, MAX_FREQ)
                        
                        # Smooth update
                        alpha = 0.5
                        smoothed_freq = alpha * new_freq + (1 - alpha) * current_gait_freq
                        
                        # QUEUE THE UPDATE instead of applying immediately
                        if abs(smoothed_freq - current_gait_freq) > 0.01:
                            print(f"  [Adapt] Lag: {np.rad2deg(phase_lag):.1f}° | Queued Freq: {smoothed_freq:.2f} Hz")
                            pending_freq_update = smoothed_freq

    sim.control_callback = custom_control
    
    # Run simulation
    RUN_TIME = 20.0
    print(f"Starting simulation for {RUN_TIME} seconds...")
    try:
        sim.run(max_time=RUN_TIME)
    except RuntimeError as e:
        print(f"Simulation terminated early: {e}")
    
    # ---------------- Post-Processing ----------------
    end_pos = sim.interface.get_body_pose("hind_body")[0]
    distance = np.linalg.norm(end_pos[:2] - start_pos[:2]) 
    print(f"Simulation Complete. Total Distance: {distance:.4f} m")

if __name__ == "__main__":
    main()