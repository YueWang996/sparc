import mujoco as mj
import numpy as np
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


# =========================================================================
#  Main Function
# =========================================================================
def main():
    # --- Configuration ---
    model_path = "/Users/justin/PycharmProjects/spine-sim/asset/spined_dog.xml"
    urdf_path = "/Users/justin/PycharmProjects/spine-sim/asset/spined_dog_spine_dyn.urdf"
    
    GAIT_TYPE = 'bound'
    GAIT_FREQUENCY = 3.12
    CONTROL_FREQ = 200.0
    MAX_TILT_ANGLE = np.deg2rad(60)
    
    # Foot body names for contact detection [HL, HR, FL, FR]
    # !!! UPDATE THESE TO MATCH YOUR URDF !!!
    FOOT_BODY_NAMES = [
        "hind_left_foot",   # HL foot body
        "hind_right_foot",  # HR foot body  
        "front_left_foot",  # FL foot body
        "front_right_foot"  # FR foot body
    ]
    
    # Dynamic Parameter Switch
    PARAM_SWITCH_TIME = 5.0
    # --- Mass Modification Parameters ---
    MASS_MOD_PERCENT = 0.0  # +0% Mass Increase
    TARGET_BODIES = ["front_body", "hind_body"]

    # --- Initialize Simulation ---
    sim = MuJoCoSimulation(model_path, control_frequency=CONTROL_FREQ)

    # 在你的仿真或提取脚本中运行一次以验证
    # print("Motor Names Order:", motor_names)

    # 获取 MuJoCo 模型中的关节名称顺序 (排除前6个自由度的基座)
    # joint_names_in_xml = [mj.mj_id2name(sim.model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(sim.model.njnt)]
    # actuated_joint_names = joint_names_in_xml[1:] # 假设第0个是 'root' 或 'free' joint

    # print("XML Joint Order:", actuated_joint_names)
    
    # --- Initialize Controller (使用原始参数) ---
    controller = DogController(
        urdf_path=urdf_path,
        control_period=1.0 / CONTROL_FREQ,
        gait_frequency_hz=GAIT_FREQUENCY,
        gait_type=GAIT_TYPE,
        device="cpu"
    )

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

    # --- Controller Parameters ---
    target_kp = np.array([300.0, 2000.0, 10.0])
    target_kd = np.array([8.0, 5.0, 1.5])
    target_x_des = np.array([0.268, 0.0, 0.0])
    
    initial_kp = np.array([1500.0, 2000.0, 10.0])
    initial_kd = np.array([8.0, 5.0, 1.5])

    controller.spine_controller.x_des = target_x_des
    controller.spine_controller.kp = initial_kp
    controller.spine_controller.kd = initial_kd
    
    # Event-based 参数预设（启用后才会生效）
    controller.gait_generator.contact_threshold = 30.0      # 提高阈值，过滤PD弹性噪声
    controller.gait_generator.phase_correction_gain = 0.1
    controller.gait_generator.max_phase_adjustment = 0.1
    controller.gait_generator.sync_strength = 0.1
    controller.gait_generator.phase_window = 0.2            # ~46度，只在预期相位附近检测
    controller.gait_generator.event_debounce_time = 0.1    # 100ms debounce
    
    # --- Record Initial State ---
    start_pos = sim.interface.get_body_pose("hind_body")[0].copy()
    print(f"Starting Position: {start_pos}")
    
    # ---------------- State Variables ----------------
    params_switched = False

    # ---------------- Control Loop ----------------
    def custom_control():
        nonlocal params_switched
        
        current_time = sim.data.time

        # ----------------------------------------------------
        # 1. Apply Dynamic Parameter Switch & Enable Event-Based
        # ----------------------------------------------------
        if current_time >= PARAM_SWITCH_TIME and not params_switched:
            # 更新脊柱阻抗参数
            controller.spine_controller.kp = target_kp
            controller.spine_controller.kd = target_kd
            controller.spine_controller.x_des = target_x_des
            
            # 更新质量
            update_robot_mass_properties(sim, controller, TARGET_BODIES, MASS_MOD_PERCENT)
            
            # 启用 event-based 步态控制
            controller.enable_event_based_control()
            
            # 启用脊柱-步态相位同步
            # desired_phase=π 表示后腿着地时脊柱最大压缩（能量存储最大）
            controller.enable_spine_gait_sync(desired_phase=1.0*np.pi, sync_gain=0.01)
            
            params_switched = True
            print(f"[t={current_time:.2f}] Params switched. Event-based + Spine sync ENABLED.")
            
        # ----------------------------------------------------
        # 2. Safety Check
        # ----------------------------------------------------
        base_pos, base_quat = sim.interface.get_body_pose("hind_body")
        rpy = quat_to_rpy(base_quat)
        if abs(rpy[0]) > MAX_TILT_ANGLE or abs(rpy[1]) > MAX_TILT_ANGLE:
            raise RuntimeError(f"Fall Detected! Roll: {np.rad2deg(rpy[0]):.1f}, Pitch: {np.rad2deg(rpy[1]):.1f}")
        
        # ----------------------------------------------------
        # 3. State Gathering
        # ----------------------------------------------------
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
        q_spine = np.array([
            sim.interface.get_joint_pos("joint_hind_spine"), 
            sim.interface.get_joint_pos("joint_front_spine"), 
            sim.interface.get_joint_pos("joint_front_body")
        ])
        v_spine = np.array([
            sim.interface.get_joint_vel("joint_hind_spine"), 
            sim.interface.get_joint_vel("joint_front_spine"), 
            sim.interface.get_joint_vel("joint_front_body")
        ])
        q_full = np.concatenate([base_pos, base_quat, q_spine])
        v_full = np.concatenate([base_vel, base_angular_vel, v_spine])

        # ----------------------------------------------------
        # 4. Get Contact Forces (only after event-based enabled)
        # ----------------------------------------------------
        if controller.gait_generator.event_based_active:
            contact_forces = sim.interface.get_foot_contact_forces(FOOT_BODY_NAMES)
        else:
            contact_forces = None
        
        # ----------------------------------------------------
        # 4b. Get Spine Relative State (for spine-gait sync)
        # ----------------------------------------------------
        spine_x_relative = None
        if controller.gait_generator.spine_sync_enabled:
            spine_state = get_relative_state(sim)  # [x_rel, z_rel, pitch_rel]
            spine_x_relative = spine_state[0]

        # ----------------------------------------------------
        # 5. Compute Control
        # ----------------------------------------------------
        full_tau, _ = controller.compute_torques(
            sim.data.time, 
            q_full, 
            v_full, 
            q_legs, 
            v_legs,
            contact_forces=contact_forces,
            spine_x_relative=spine_x_relative
        )
        
        # ----------------------------------------------------
        # 6. Apply Torques
        # ----------------------------------------------------
        motor_names = [
            "motor_hind_left_leg_upper", "motor_hind_left_leg_lower",
            "motor_hind_right_leg_upper", "motor_hind_right_leg_lower",
            "motor_hind_spine", "motor_front_spine", "motor_front_body",
            "motor_front_left_leg_upper", "motor_front_left_leg_lower",
            "motor_front_right_leg_upper", "motor_front_right_leg_lower"
        ]
        for i, name in enumerate(motor_names):
            sim.interface.set_joint_torque(name, full_tau[i])

    sim.control_callback = custom_control
    
    # Run simulation
    RUN_TIME = 20.0
    print(f"Starting simulation for {RUN_TIME} seconds...")
    print(f"Gait: {GAIT_TYPE} @ {GAIT_FREQUENCY} Hz")
    print(f"Event-based control will be enabled at t={PARAM_SWITCH_TIME}s")
    
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