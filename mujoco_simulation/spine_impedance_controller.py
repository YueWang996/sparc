import numpy as np
import torch
import bard.transforms
from bard.parsers.urdf import build_chain_from_urdf
from bard.core.dynamics import RNEA, CRBA
from bard.core.jacobian import Jacobian
from bard.core.kinematics import ForwardKinematics, SpatialAcceleration

class SpineImpedanceController:
    """
    Impedance controller for the spine using Bard (PyTorch).
    
    Regulates the position/orientation of 'end_effector_name' 
    RELATIVE to 'base_link_name'.
    """

    def __init__(self, urdf_path: str, num_envs: int = 1, device: str = "cpu", 
                 end_effector_name: str = "front_body", base_link_name: str = "hind_body"):
        self.device = device
        self.num_envs = num_envs

        # 1. Build Chains
        # Floating base chain (for Dynamics: Mass Matrix & h)
        chain_floating = build_chain_from_urdf(urdf_path, floating_base=True) 
        self.chain_floating = chain_floating.to(device=self.device, dtype=torch.float32)

        # Body chain (for Kinematics: Relative transforms)
        # We use floating_base=False so it treats the root as fixed (0,0,0).
        chain_body = build_chain_from_urdf(urdf_path, floating_base=False)
        self.chain_body = chain_body.to(device=self.device, dtype=torch.float32)

        # 2. Initialize Bard Modules
        self.dyn_rnea = RNEA(chain_floating, max_batch_size=self.num_envs)
        self.dyn_crba = CRBA(chain_floating, max_batch_size=self.num_envs)
        
        self.jac_body = Jacobian(chain_body, max_batch_size=self.num_envs)
        self.fk_body = ForwardKinematics(chain_body, max_batch_size=self.num_envs)
        self.spatial_acc_body = SpatialAcceleration(chain_body, max_batch_size=self.num_envs)
        
        # Get Frame IDs for BOTH the front and hind spine
        self.front_id = self.chain_body.get_frame_id(end_effector_name)
        self.hind_id = self.chain_body.get_frame_id(base_link_name)

        # 3. Gains & Targets
        self.Kp = torch.diag(torch.tensor([300.0, 2000.0, 5.0], device=self.device))
        self.Kd = torch.diag(torch.tensor([1.0, 5.0, 0.1], device=self.device))

        self.X_des = torch.tensor([0.16, 0.0, 0.0], device=self.device)
        self.dX_des = torch.zeros(3, device=self.device)

        # Selection matrix [x, z, pitch]
        self.S = torch.zeros((3, 6), device=self.device)
        self.S[0, 0] = 1.0 # x
        self.S[1, 2] = 1.0 # z
        self.S[2, 4] = 1.0 # pitch 

    # Getter and Setter for X_des
    @property
    def x_des(self):
        """Get the desired task-space position [x, z, pitch]."""
        return self.X_des.detach().cpu().numpy()

    @x_des.setter
    def x_des(self, value):
        """Set the desired task-space position [x, z, pitch]."""
        if isinstance(value, np.ndarray):
            self.X_des = torch.from_numpy(value).float().to(self.device)
        elif isinstance(value, torch.Tensor):
            self.X_des = value.float().to(self.device)
        else:
            self.X_des = torch.tensor(value, dtype=torch.float32, device=self.device)

    # Getter and Setter for Kp
    @property
    def kp(self):
        """Get the proportional gain matrix."""
        return self.Kp.detach().cpu().numpy()

    @kp.setter
    def kp(self, value):
        """Set the proportional gain matrix. Can be a 3x3 matrix or a 3-element vector (for diagonal)."""
        if isinstance(value, np.ndarray):
            tensor_value = torch.from_numpy(value).float().to(self.device)
        elif isinstance(value, torch.Tensor):
            tensor_value = value.float().to(self.device)
        else:
            tensor_value = torch.tensor(value, dtype=torch.float32, device=self.device)
        
        # Handle both diagonal vector and full matrix
        if tensor_value.dim() == 1:
            self.Kp = torch.diag(tensor_value)
        else:
            self.Kp = tensor_value

    # Getter and Setter for Kd
    @property
    def kd(self):
        """Get the derivative gain matrix."""
        return self.Kd.detach().cpu().numpy()

    @kd.setter
    def kd(self, value):
        """Set the derivative gain matrix. Can be a 3x3 matrix or a 3-element vector (for diagonal)."""
        if isinstance(value, np.ndarray):
            tensor_value = torch.from_numpy(value).float().to(self.device)
        elif isinstance(value, torch.Tensor):
            tensor_value = value.float().to(self.device)
        else:
            tensor_value = torch.tensor(value, dtype=torch.float32, device=self.device)
        
        # Handle both diagonal vector and full matrix
        if tensor_value.dim() == 1:
            self.Kd = torch.diag(tensor_value)
        else:
            self.Kd = tensor_value

    def compute_torque(self, q_full_np: np.ndarray, v_full_np: np.ndarray) -> np.ndarray:
        q_full = torch.from_numpy(q_full_np).float().to(self.device).unsqueeze(0)
        v_full = torch.from_numpy(v_full_np).float().to(self.device).unsqueeze(0)

        q_base = q_full[:, :7]
        q_spine = q_full[:, 7:]
        v_base = v_full[:, :6]
        v_spine = v_full[:, 6:]

        # -------------------------------------------------
        # 2. Relative Kinematics (Explicit)
        # -------------------------------------------------
        # Calculate Transform of Front AND Hind relative to the chain root
        T_front = self.fk_body.calc(q_spine, frame_id=self.front_id) # (B, 4, 4)
        T_hind  = self.fk_body.calc(q_spine, frame_id=self.hind_id)  # (B, 4, 4)

        # Compute Relative Transform: T_rel = T_hind_inv @ T_front
        # This gives the pose of Front Spine expressed in Hind Spine frame
        T_hind_inv = torch.linalg.inv(T_hind)
        T_rel = T_hind_inv @ T_front

        # Extract Relative Position
        pos_rel = T_rel[:, :3, 3] # (B, 3)

        # Extract Relative Rotation (R_hind_to_front)
        R_rel = T_rel[:, :3, :3] # (B, 3, 3)
        rot_euler = bard.transforms.matrix_to_euler_angles(R_rel, convention="XYZ")
        pitch_rel = rot_euler[:, 1]

        # Current Task State X [x, z, pitch] (in Hind Spine Frame)
        X_curr = torch.stack([pos_rel[:, 0], pos_rel[:, 2], pitch_rel], dim=1)

        # -------------------------------------------------
        # 3. Jacobian (Aligned to Hind Spine Frame)
        # -------------------------------------------------
        # Get Jacobian in LOCAL frame (Tip frame)
        J_spatial_local = self.jac_body.calc(
            q_spine, frame_id=self.front_id, reference_frame="local"
        ) # (B, 6, 3)

        # Split Linear and Angular
        J_lin_local = J_spatial_local[:, :3, :]
        J_ang_local = J_spatial_local[:, 3:, :]
        
        # Rotate Jacobian from Tip Frame to Hind Spine Frame
        # J_hind = R_rel @ J_local
        J_lin_hind = R_rel @ J_lin_local
        J_ang_hind = R_rel @ J_ang_local
        
        J_spatial_hind = torch.cat([J_lin_hind, J_ang_hind], dim=1) # (B, 6, 3)

        # Apply Selection Matrix
        J_task = self.S @ J_spatial_hind

        # Task Velocity (Relative Velocity in Hind Frame)
        # Since base is "fixed" in this chain, v_spine produces v_rel directly
        dX_curr = torch.squeeze(J_task @ v_spine.unsqueeze(-1), -1)

        # -------------------------------------------------
        # 4. Dynamics (h & M)
        # -------------------------------------------------
        # Pass Gravity Vector [0, 0, -9.81]
        g_vec = torch.tensor([0.0, 0.0, -9.81], device=self.device).expand(q_full.shape[0], 3)
        
        h_full = self.dyn_rnea.calc(q_full, v_full, torch.zeros_like(v_full), gravity=g_vec)
        h_spine = h_full[:, 6:] 

        M_full = self.dyn_crba.calc(q_full)
        M_spine = M_full[:, 6:9, 6:9]

        # -------------------------------------------------
        # 5. Impedance Math
        # -------------------------------------------------
        M_spine_reg = M_spine + 1e-4 * torch.eye(3, device=self.device)
        
        X_inter = torch.linalg.solve(M_spine_reg, J_task.transpose(1, 2))
        JMJt = J_task @ X_inter
        
        Lambda = torch.linalg.solve(JMJt + 1e-6 * torch.eye(3, device=self.device), 
                                    torch.eye(3, device=self.device).unsqueeze(0))

        # -------------------------------------------------
        # 6. Drift Compensation
        # -------------------------------------------------
        acc_local = self.spatial_acc_body.calc(
            q_spine, v_spine, torch.zeros_like(v_spine), 
            frame_id=self.front_id, reference_frame="local"
        ) 
        if acc_local.dim() == 1: acc_local = acc_local.unsqueeze(0)

        # Rotate Drift from Tip to Hind Frame
        acc_lin_local = acc_local[:, :3]
        acc_ang_local = acc_local[:, 3:]
        
        acc_lin_hind = (R_rel @ acc_lin_local.unsqueeze(-1)).squeeze(-1)
        acc_ang_hind = (R_rel @ acc_ang_local.unsqueeze(-1)).squeeze(-1)
        
        acc_hind = torch.cat([acc_lin_hind, acc_ang_hind], dim=1)
        
        Jdot_v = (self.S @ acc_hind.unsqueeze(-1)).squeeze(-1) 

        # -------------------------------------------------
        # 7. Final Forces
        # -------------------------------------------------
        e_pos = self.X_des - X_curr
        e_vel = self.dX_des - dX_curr
        
        F_imp = (self.Kp @ e_pos.unsqueeze(-1)).squeeze(-1) + \
                (self.Kd @ e_vel.unsqueeze(-1)).squeeze(-1)
                
        F_comp = (Lambda @ Jdot_v.unsqueeze(-1)).squeeze(-1)
        
        F_task = F_imp - F_comp
        
        tau_task = (J_task.transpose(1, 2) @ F_task.unsqueeze(-1)).squeeze(-1)
        
        tau_total = h_spine + tau_task
        tau_total = torch.clamp(tau_total, -100.0, 100.0)

        return tau_total.squeeze(0).detach().cpu().numpy()
    