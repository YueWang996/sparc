import numpy as np
import torch
import bard
import bard.transforms

class SpineImpedanceController:
    """
    Impedance controller for the spine using Bard (PyTorch).
    
    Regulates the position/orientation of 'end_effector_name' 
    RELATIVE to 'base_link_name'.
    
    NEW: Supports geometric constraints with soft barrier forces.
    """

    def __init__(self, urdf_path: str, num_envs: int = 1, device: str = "cpu", 
                 end_effector_name: str = "spine_end_effector", base_link_name: str = "hind_body"):
        self.device = device
        self.num_envs = num_envs

        # 1. Build Models
        self.model_floating = bard.build_model_from_urdf(
            urdf_path, floating_base=True, dtype=torch.float32, device=self.device
        )
        self.model_body = bard.build_model_from_urdf(
            urdf_path, floating_base=False, dtype=torch.float32, device=self.device
        )

        # 2. Create Data workspaces
        self.data_floating = bard.create_data(self.model_floating, max_batch_size=self.num_envs)
        self.data_body = bard.create_data(self.model_body, max_batch_size=self.num_envs)

        self.front_id = self.model_body.get_frame_id(end_effector_name)
        self.hind_id = self.model_body.get_frame_id(base_link_name)

        # 3. Gains & Targets
        self.Kp = torch.diag(torch.tensor([5000.0, 3000.0, 30.0], device=self.device))
        self.Kd = torch.diag(torch.tensor([140.0, 50.0, 3.0], device=self.device))

        self.X_des = torch.tensor([0.35, 0.0, 0.0], device=self.device)
        self.dX_des = torch.zeros(3, device=self.device)

        # Selection matrix [x, z, pitch]
        self.S = torch.zeros((3, 6), device=self.device)
        self.S[0, 0] = 1.0  # x
        self.S[1, 2] = 1.0  # z
        self.S[2, 4] = 1.0  # pitch 

        # ============ Geometric Constraints (NEW) ============
        # Constraint format: [x_min, x_max, z_min, z_max, pitch_min, pitch_max]
        # Set to None for no constraint in that direction
        self.constraints_enabled = False
        
        # Default constraint ranges (modifiable via setter)
        self.x_limits = (0.18, 0.35)
        self.z_limits = (-0.2, 0.2)
        self.pitch_limits = (-0.8, 0.8)
        
        # Barrier parameters - set independently for each axis [x, z, pitch]
        self.barrier_stiffness = torch.tensor([1000.0, 2000.0, 15.0], device=self.device)
        self.barrier_buffer = torch.tensor([0.01, 0.01, 0.01], device=self.device)
        self.barrier_damping = torch.tensor([10.0, 5.0, 0.1], device=self.device)

    # ============ Constraint Setters ============
    
    def set_constraints(self, x_limits=None, z_limits=None, pitch_limits=None,
                        stiffness=(800.0, 2000.0, 15.0), 
                        buffer=(0.01, 0.01, 0.01), 
                        damping=(5.0, 5.0, 0.1)):
        self.z_limits = z_limits
        self.pitch_limits = pitch_limits
        
        self.barrier_stiffness = torch.tensor(stiffness, device=self.device, dtype=torch.float32)
        self.barrier_buffer = torch.tensor(buffer, device=self.device, dtype=torch.float32)
        self.barrier_damping = torch.tensor(damping, device=self.device, dtype=torch.float32)
        
        self.constraints_enabled = (x_limits is not None or 
                                    z_limits is not None or 
                                    pitch_limits is not None)
        
        if self.constraints_enabled:
            print(f"[SpineController] Constraints enabled:")
            if x_limits: 
                print(f"  x: [{x_limits[0]:.3f}, {x_limits[1]:.3f}] m, k={stiffness[0]:.0f}, buf={buffer[0]:.3f}, d={damping[0]:.1f}")
            if z_limits: 
                print(f"  z: [{z_limits[0]:.3f}, {z_limits[1]:.3f}] m, k={stiffness[1]:.0f}, buf={buffer[1]:.3f}, d={damping[1]:.1f}")
            if pitch_limits: 
                print(f"  pitch: [{np.rad2deg(pitch_limits[0]):.1f}, {np.rad2deg(pitch_limits[1]):.1f}] deg, k={stiffness[2]:.0f}, buf={np.rad2deg(buffer[2]):.1f}deg, d={damping[2]:.1f}")

    def _compute_barrier_force(self, x, x_min, x_max, dx, buffer, k, d):
        F = torch.zeros_like(x)
        
        if x_min is not None:
            soft_min = x_min + buffer
            in_buffer_low = (x > x_min) & (x < soft_min)
            penetration_low = soft_min - x
            F = torch.where(in_buffer_low, 
                           k * penetration_low - d * dx,
                           F)
            beyond_low = x <= x_min
            violation_low = x_min - x
            F = torch.where(beyond_low,
                           k * buffer + 2 * k * violation_low - d * dx,
                           F)
        if x_max is not None:
            soft_max = x_max - buffer
            in_buffer_high = (x < x_max) & (x > soft_max)
            penetration_high = x - soft_max
            F = torch.where(in_buffer_high,
                           -k * penetration_high - d * dx,
                           F)
            beyond_high = x >= x_max
            violation_high = x - x_max
            F = torch.where(beyond_high,
                           -k * buffer - 2 * k * violation_high - d * dx,
                           F)
        
        return F

    def _compute_constraint_forces(self, X_curr, dX_curr):
        F_constraint = torch.zeros_like(X_curr)
        
        if not self.constraints_enabled:
            return F_constraint
        if self.x_limits is not None:
            F_constraint[:, 0] = self._compute_barrier_force(
                X_curr[:, 0], self.x_limits[0], self.x_limits[1],
                dX_curr[:, 0], 
                self.barrier_buffer[0], 
                self.barrier_stiffness[0], 
                self.barrier_damping[0]
            )
        if self.z_limits is not None:
            F_constraint[:, 1] = self._compute_barrier_force(
                X_curr[:, 1], self.z_limits[0], self.z_limits[1],
                dX_curr[:, 1], 
                self.barrier_buffer[1], 
                self.barrier_stiffness[1], 
                self.barrier_damping[1]
            )
        if self.pitch_limits is not None:
            F_constraint[:, 2] = self._compute_barrier_force(
                X_curr[:, 2], self.pitch_limits[0], self.pitch_limits[1],
                dX_curr[:, 2], 
                self.barrier_buffer[2], 
                self.barrier_stiffness[2], 
                self.barrier_damping[2]
            )
        
        return F_constraint

    # ============ Getters/Setters ============
    
    @property
    def x_des(self):
        return self.X_des.detach().cpu().numpy()

    @x_des.setter
    def x_des(self, value):
        if isinstance(value, np.ndarray):
            self.X_des = torch.from_numpy(value).float().to(self.device)
        elif isinstance(value, torch.Tensor):
            self.X_des = value.float().to(self.device)
        else:
            self.X_des = torch.tensor(value, dtype=torch.float32, device=self.device)

    @property
    def kp(self):
        return self.Kp.detach().cpu().numpy()

    @kp.setter
    def kp(self, value):
        if isinstance(value, np.ndarray):
            tensor_value = torch.from_numpy(value).float().to(self.device)
        elif isinstance(value, torch.Tensor):
            tensor_value = value.float().to(self.device)
        else:
            tensor_value = torch.tensor(value, dtype=torch.float32, device=self.device)
        
        if tensor_value.dim() == 1:
            self.Kp = torch.diag(tensor_value)
        else:
            self.Kp = tensor_value

    @property
    def kd(self):
        return self.Kd.detach().cpu().numpy()

    @kd.setter
    def kd(self, value):
        if isinstance(value, np.ndarray):
            tensor_value = torch.from_numpy(value).float().to(self.device)
        elif isinstance(value, torch.Tensor):
            tensor_value = value.float().to(self.device)
        else:
            tensor_value = torch.tensor(value, dtype=torch.float32, device=self.device)
        
        if tensor_value.dim() == 1:
            self.Kd = torch.diag(tensor_value)
        else:
            self.Kd = tensor_value

    # ============ Main Compute Function ============
    
    def compute_torque(self, q_full_np: np.ndarray, v_full_np: np.ndarray) -> np.ndarray:
        q_full = torch.from_numpy(q_full_np).float().to(self.device).unsqueeze(0)
        v_full = torch.from_numpy(v_full_np).float().to(self.device).unsqueeze(0)

        q_base = q_full[:, :7]
        q_spine = q_full[:, 7:]
        v_base = v_full[:, :6]
        v_spine = v_full[:, 6:]

        # -------------------------------------------------
        # 2. Relative Kinematics
        # -------------------------------------------------
        bard.update_kinematics(self.model_body, self.data_body, q_spine, v_spine)

        T_front = bard.forward_kinematics(self.model_body, self.data_body, self.front_id)
        T_hind = bard.forward_kinematics(self.model_body, self.data_body, self.hind_id)

        T_hind_inv = torch.linalg.inv(T_hind)
        T_rel = T_hind_inv @ T_front

        pos_rel = T_rel[:, :3, 3]
        R_rel = T_rel[:, :3, :3]
        rot_euler = bard.transforms.matrix_to_euler_angles(R_rel, convention="XYZ")
        pitch_rel = rot_euler[:, 1]

        X_curr = torch.stack([pos_rel[:, 0], pos_rel[:, 2], pitch_rel], dim=1)

        # -------------------------------------------------
        # 3. Jacobian
        # -------------------------------------------------
        J_spatial_local = bard.jacobian(
            self.model_body, self.data_body, self.front_id, reference_frame="local"
        )

        J_lin_local = J_spatial_local[:, :3, :]
        J_ang_local = J_spatial_local[:, 3:, :]
        
        J_lin_hind = R_rel @ J_lin_local
        J_ang_hind = R_rel @ J_ang_local
        
        J_spatial_hind = torch.cat([J_lin_hind, J_ang_hind], dim=1)
        J_task = self.S @ J_spatial_hind

        dX_curr = torch.squeeze(J_task @ v_spine.unsqueeze(-1), -1)

        # -------------------------------------------------
        # 4. Dynamics
        # -------------------------------------------------
        bard.update_kinematics(self.model_floating, self.data_floating, q_full, v_full)

        h_full = bard.rnea(self.model_floating, self.data_floating, torch.zeros_like(v_full))
        h_spine = h_full[:, 6:]

        M_full = bard.crba(self.model_floating, self.data_floating)
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
        acc_local = bard.spatial_acceleration(
            self.model_body, self.data_body, torch.zeros_like(v_spine),
            frame_id=self.front_id, reference_frame="local"
        )
        if acc_local.dim() == 1: acc_local = acc_local.unsqueeze(0)

        acc_lin_local = acc_local[:, :3]
        acc_ang_local = acc_local[:, 3:]
        
        acc_lin_hind = (R_rel @ acc_lin_local.unsqueeze(-1)).squeeze(-1)
        acc_ang_hind = (R_rel @ acc_ang_local.unsqueeze(-1)).squeeze(-1)
        
        acc_hind = torch.cat([acc_lin_hind, acc_ang_hind], dim=1)
        Jdot_v = (self.S @ acc_hind.unsqueeze(-1)).squeeze(-1) 

        # -------------------------------------------------
        # 7. Impedance + Constraint Forces
        # -------------------------------------------------
        e_pos = self.X_des - X_curr
        e_vel = self.dX_des - dX_curr
        
        F_imp = (self.Kp @ e_pos.unsqueeze(-1)).squeeze(-1) + \
                (self.Kd @ e_vel.unsqueeze(-1)).squeeze(-1)
        
        F_constraint = self._compute_constraint_forces(X_curr, dX_curr)
        
        F_comp = (Lambda @ Jdot_v.unsqueeze(-1)).squeeze(-1)
        
        F_task = F_imp + F_constraint - F_comp
        
        tau_task = (J_task.transpose(1, 2) @ F_task.unsqueeze(-1)).squeeze(-1)
        
        tau_total = h_spine + tau_task
        tau_total = torch.clamp(tau_total, -12.0, 12.0)

        # print(f"x_curr: {np.array2string(np.round(X_curr.detach().cpu().numpy()[0], 2), formatter={'float_kind': lambda x: f'{x:+.2f}'})}, "
        #       f"F_imp: {np.array2string(np.round(F_imp.detach().cpu().numpy()[0], 2), formatter={'float_kind': lambda x: f'{x:+.2f}'})}, "
        #       f"tau_task: {np.array2string(np.round(tau_task.detach().cpu().numpy()[0], 2), formatter={'float_kind': lambda x: f'{x:+.2f}'})}, "
        #       f"tau_total: {np.array2string(np.round(tau_total.detach().cpu().numpy()[0], 2), formatter={'float_kind': lambda x: f'{x:+.2f}'})}")

        # =========================================================================
        # [DEBUG] Print constraint status
        # =========================================================================
        if self.constraints_enabled:
            x_val = X_curr[0, 0].item()
            z_val = X_curr[0, 1].item()
            pitch_val = X_curr[0, 2].item()
            Fc = F_constraint[0].detach().cpu().numpy()
            print(f"[Constraint] x={x_val:.3f}, z={z_val:.3f}, pitch={np.rad2deg(pitch_val):.1f}° | F={Fc}")
        # =========================================================================

        return tau_total.squeeze(0).detach().cpu().numpy()