from pathlib import Path
import mujoco as mj
import numpy as np
from dog_controller_raibert import DogControllerRaibert
from mujoco_interface import MuJoCoInterface
import mujoco.viewer
import time

# --- Helper Function: World Vel to Local Vel ---
def get_local_velocity(sim, body_name):
    """
    Get rigid body linear velocity in local frame (vx, vy, vz)
    """
    body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, body_name)
    
    R_flat = sim.data.xmat[body_id]
    R = R_flat.reshape(3, 3)
    vel_world = sim.data.cvel[body_id, 3:6]
    vel_local = R.T @ vel_world
    return vel_local

# --- Helper Function: Quaternion to Pitch ---
def quat_to_pitch(q):
    """
    Extract pitch angle from quaternion [w, x, y, z]
    Positive value indicates forward tilt
    """
    w, x, y, z = q
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    return pitch

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
                    break
                if self.data.time - self.last_control_time >= self.control_period:
                    self.control_callback()
                    self.last_control_time = self.data.time
                mj.mj_step(self.model, self.data)
                viewer.sync()

# =========================================================================
#  Main Function
# =========================================================================
def main():
    # --- Configuration ---
    current_script_path = Path(__file__).resolve()
    # model_path = str(current_script_path.parent / "asset" / "spined_dog_locked_spine.xml")
    model_path = str(current_script_path.parent / "asset" / "spined_dog.xml")
    urdf_path = str(current_script_path.parent / "asset" / "spined_dog_spine_dyn.urdf")
    
    CONTROL_FREQ = 500.0
    
    # --- Initialize Simulation ---
    sim = MuJoCoSimulation(model_path, control_frequency=CONTROL_FREQ)
    
    # --- Initialize Raibert Controller ---
    controller = DogControllerRaibert(urdf_path=urdf_path, device="cpu")
    
    # Set desired velocity (m/s)
    DESIRED_VELOCITY = 1.3
    controller.set_command(DESIRED_VELOCITY)
    
    # Raibert gain tuning
    controller.gait_generator.raibert_gain_hind = 0.08
    controller.gait_generator.raibert_gain_front = 0.12
    
    # Spine settings
    initial_kp = np.array([2400.0, 3000.0, 15.0])
    initial_kd = np.array([0.0, 30.0, 1.0])
    target_x_des = np.array([0.367, 0.02, -0.3])
    controller.spine_controller.kp = initial_kp
    controller.spine_controller.kd = initial_kd
    controller.spine_controller.x_des = target_x_des

    # --- Initialization (Joints) ---
    sim.interface.set_joint_pos("joint_hind_spine", -0.6)
    sim.interface.set_joint_pos("joint_front_spine", 1.2)
    sim.interface.set_joint_pos("joint_front_body", -0.6)
    
    init_positions = {
        "joint_hind_left_leg_upper": -0.79, "joint_hind_left_leg_lower": 1.57,
        "joint_hind_right_leg_upper": -0.79, "joint_hind_right_leg_lower": 1.57,
        "joint_front_left_leg_upper": 0.79, "joint_front_left_leg_lower": -1.57,
        "joint_front_right_leg_upper": 0.79, "joint_front_right_leg_lower": -1.57
    }
    for name, pos in init_positions.items():
        sim.interface.set_joint_pos(name, pos)

    # ---------------- Control Loop ----------------
    def custom_control():
        current_time = sim.data.time
        
        # Get velocities
        v_front_local = get_local_velocity(sim, "front_body")
        v_hind_local = get_local_velocity(sim, "hind_body")
        
        # Get orientation (pitch) using hind_body as reference
        base_pos, base_quat = sim.interface.get_body_pose("hind_body")
        pitch = quat_to_pitch(base_quat)
        
        # State Gathering
        base_vel, base_ang_vel = sim.interface.get_body_velocity("hind_body")
        
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
        v_full = np.concatenate([base_vel, base_ang_vel, v_spine])

        # Compute Control
        full_tau, _ = controller.compute_torques(
            current_time, 
            q_full, v_full, 
            q_legs, v_legs,
            v_front_local, v_hind_local,
            pitch=pitch
        )
        
        # 5. Apply Torques
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
    
    try:
        sim.run(max_time=20.0)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()