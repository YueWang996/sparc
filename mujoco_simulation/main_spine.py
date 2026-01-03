import mujoco as mj
import numpy as np
from spine_impedance_controller import SpineImpedanceController
from mujoco_interface import MuJoCoInterface
# import bard
import mujoco.viewer
import numpy as np
import csv
import time


class DataLogger:
    def __init__(self):
        self.times = []
        self.positions = [] # Stores [x, z, pitch]

    def log(self, t, pos_array):
        """
        Args:
            t (float): Simulation time
            pos_array (np.array): Shape (3,) -> [x, z, pitch]
        """
        self.times.append(t)
        self.positions.append(pos_array.copy())
        
    def save(self, filename="sim_k300_b0.csv"):
        data = np.column_stack((self.times, self.positions))
        header = "time,rel_x,rel_z,rel_pitch"
        np.savetxt(filename, data, delimiter=",", header=header, comments="")
        print(f"Data saved to {filename}")


def get_relative_state(sim, base_name="hind_body", tip_name="front_body"):
    """
    Calculates the Tip position relative to the Base frame.
    Matches the logic used inside the Impedance Controller.
    """
    # 1. Get Base (Hind) Position and Rotation Matrix
    base_id = sim.model.body(base_name).id
    p_base = sim.data.xpos[base_id]
    # xmat gives the 3x3 rotation matrix (flattened to 9)
    R_base = sim.data.xmat[base_id].reshape(3, 3)

    # 2. Get Tip (Front) Position
    tip_id = sim.model.body(tip_name).id
    p_tip = sim.data.xpos[tip_id]
    R_tip = sim.data.xmat[tip_id].reshape(3, 3)

    # 3. Compute Relative Position in Base Frame
    # Vector from Base to Tip (World Frame)
    diff_world = p_tip - p_base
    
    # Rotate into Base Frame: v_local = R_base.T @ v_world
    pos_rel = R_base.T @ diff_world

    # 4. Compute Relative Pitch (Approximation)
    # R_rel = R_base.T @ R_tip
    # For simple pitch, we can just grab the relative vector components calculated above.
    # However, to match controller exactly, we need Euler pitch from R_rel.
    R_rel = R_base.T @ R_tip
    
    # Extract Pitch from Rotation Matrix (XYZ convention)
    # pitch = arcsin(R[0, 2]) ... standard conversion
    # Simpler approach: Just return the pitch angle
    pitch_rel = np.arctan2(-R_rel[2, 0], np.sqrt(R_rel[2, 1]**2 + R_rel[2, 2]**2))

    # Return [x, z, pitch]
    return np.array([pos_rel[0] + 0.117, pos_rel[2], pitch_rel])


class MuJoCoSimulation:
    """MuJoCo simulation with passive viewer."""
    
    def __init__(self, model_path: str):
        """
        Initialize the simulation.
        
        Args:
            model_path: Path to the MuJoCo XML model file
        """
        # Load model and data
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        
        # Create interface
        self.interface = MuJoCoInterface(self.model, self.data)
        
    def control_callback(self):
        """
        Control callback called before each simulation step.
        Override this method to implement custom control logic.
        """
        # Example: Set zero torque for all actuators
        self.data.ctrl[:] = 0.0
        
    def run(self, max_time: float = None, max_wall_time: float = None):
        """
        Run the simulation loop with passive viewer.
        
        Args:
            max_time: Maximum simulation time in seconds (None for infinite)
            max_wall_time: Maximum wall clock time in seconds (None for infinite)
        """
        # Launch passive viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Set camera position
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -20
            viewer.cam.distance = 3.0
            viewer.cam.lookat = np.array([0.0, 0.0, 0.5])
            
            start_time = self.data.time
            start_wall_time = time.time()
            
            # Simulation loop
            while viewer.is_running():
                # Check time limit
                if max_time is not None and (self.data.time - start_time) >= max_time:
                    break
                
                # Check wall time limit
                if max_wall_time is not None and (time.time() - start_wall_time) >= max_wall_time:
                    print(f"\nReached wall time limit of {max_wall_time}s")
                    break
                
                # Apply control
                self.control_callback()
                
                # Step simulation
                mj.mj_step(self.model, self.data)
                
                # Sync viewer (updates at 60 Hz by default)
                viewer.sync()


def main():
    """Main entry point."""
    # Paths
    model_path = "/Users/justin/PycharmProjects/spine-sim/asset/spine.xml"
    urdf_path = "/Users/justin/PycharmProjects/spine-sim/asset/spine.urdf"
    
    # Create simulation
    sim = MuJoCoSimulation(model_path)

    # ---------------------------------------------------------
    # Initialize the new Pinocchio-based PD Controller
    # ---------------------------------------------------------
    # We specify 'hind_spine' as the base_link because that is the body
    # we are using to get the floating base state in the control loop below.
    controller = SpineImpedanceController(
        urdf_path=urdf_path,
        end_effector_name="front_body",
        base_link_name="hind_body"
    )

    # Set Desired Task Space Position (Relative to Base)
    # Format: [x, z, pitch]
    controller.x_des = np.array([0.176, 0.0, 0.0])
    
    controller.kp = np.array([300.0, 8000.0, 5.0])
    controller.kd = np.array([0.0, 5.0, 0.1])

    # Initialize Logger
    logger = DataLogger()

    # Initialize joint positions in MuJoCo
    sim.interface.set_joint_pos("joint_hind_spine", -0.65)
    sim.interface.set_joint_pos("joint_front_spine", 1.3)
    sim.interface.set_joint_pos("joint_front_body", -0.65)
    
    # ---------------------------------------------------------
    # Control Loop
    # ---------------------------------------------------------
    def custom_control():
        # Get Base State (Floating Base)
        # MuJoCo returns Quaternions as [w, x, y, z]
        base_pos, base_quat = sim.interface.get_body_pose("hind_body")
        base_vel, base_angular_vel = sim.interface.get_body_velocity("hind_body")
        
        # Get Spine Joint States
        spine_q = np.array([
            sim.interface.get_joint_pos("joint_hind_spine"),
            sim.interface.get_joint_pos("joint_front_spine"),
            sim.interface.get_joint_pos("joint_front_body"),
        ])
        spine_v = np.array([
            sim.interface.get_joint_vel("joint_hind_spine"),
            sim.interface.get_joint_vel("joint_front_spine"),
            sim.interface.get_joint_vel("joint_front_body"),
        ])
        
        q_full = np.concatenate([base_pos, base_quat, spine_q])
        v_full = np.concatenate([base_vel, base_angular_vel, spine_v])

        tau = controller.compute_torque(q_full, v_full)
        
        current_state = get_relative_state(sim)
        logger.log(sim.data.time, current_state)

        sim.interface.set_joint_torque("motor_hind_spine", tau[0])
        sim.interface.set_joint_torque("motor_front_spine", tau[1])
        sim.interface.set_joint_torque("motor_front_body", tau[2])

    # Set control callback
    sim.control_callback = custom_control
    
    # Run simulation with 5 second wall time limit
    print("Starting simulation...")
    sim.run()
    
    # Save logged data
    logger.save()

if __name__ == "__main__":
    main()