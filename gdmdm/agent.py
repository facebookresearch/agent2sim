# dims: diffT, bs, nsamp, TK3

import os, sys
import cv2
import pdb
import numpy as np
import torch
from copy import deepcopy


sys.path.insert(0, os.getcwd())
from lab4d.utils.quat_transform import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    symmetric_orthogonalization,
)
from utils import GDMDMMotion, get_grid_xyz

class Agent:
    def __init__(self, model, data, config, visualizer, meta, agent_type="full", agent_class="lab4d", agent_idx=0, record_grad = False):
        self.nsamp = config.eval_batch_size
        self.drop_cam = config.drop_cam
        self.drop_past = config.drop_past
        self.num_timesteps = config.num_timesteps
        self.agent_type = agent_type
        self.agent_class = agent_class
        self.agent_idx = agent_idx
        self.meta = meta
        self.visualizer = visualizer

        self.data_all = data
        self.reset(config.sample_idx)
        self.model = model
        self.model.eval() # avoid ddp error with bn

        self.prepare_vis(record_grad)
    

    def prepare_vis(self, record_grad):
        """
        
        """
        self.xsize, self.ysize, self.zsize = 30, 5, 30
        xyz, self.xzmin, self.xzmax = get_grid_xyz(self.data.x0, self.xsize, self.ysize, self.zsize)
        if record_grad:    
            self.xyz_grid = torch.tensor(xyz, dtype=torch.float32, device="cuda")
            self.xyz_grid_wp = (
                    self.xyz_grid[:, None]
                    .repeat(1, self.model.forecast_size, 1)
                    .view(self.xyz_grid.shape[0], -1)
            )
        else:
            self.xyz_grid = None
            self.xyz_grid_wp = None


    def reset(self, sample_idx):
        if not isinstance(sample_idx, list):
            sample_idx = [sample_idx]
        self.sample_idx = sample_idx
        data_all = deepcopy(self.data_all)
        self.data = GDMDMMotion(
            x0 = data_all.x0[sample_idx],
            x0_goal = data_all.x0_goal[sample_idx],
            past = data_all.past[sample_idx],
            cam = data_all.cam[sample_idx],
            x0_to_world = data_all.x0_to_world[sample_idx],
            x0_joints = data_all.x0_joints[sample_idx],
            past_joints = data_all.past_joints[sample_idx],
            x0_angles = data_all.x0_angles[sample_idx],
            past_angles = data_all.past_angles[sample_idx],
            x0_angles_to_world = data_all.x0_angles_to_world[sample_idx],
        )
        self.accumulated_traj = self.data.past[:1].clone()  # 1,T',1, 3 in the latest ego coordinate

        if self.visualizer is not None:
            # display user interaction
            self.add_cam_to_visualizer()
            self.render_fullbody(self.data.x0_to_world, self.data.past, self.data.past_angles, self.data.past_joints)

    def extract_local_feature(self):
        if self.model.env_model is not None:
            self.feat_volume = self.model.extract_env_feat(self.data.x0_to_world[:, 0])
        else:
            self.feat_volume = None

    def extract_global_feature(self):
        if self.model.env_model is not None:
            self.feat_volume = self.model.extract_feature_grid()
        else:
            self.feat_volume = None

    def update_goal(self, goal_list, replace_goal=True):
        if len(goal_list) > 0:
            selected_goal = torch.tensor(goal_list, device="cuda", dtype=torch.float32)
            selected_goal = selected_goal - self.data.x0_to_world[0]
            selected_goal = selected_goal.view(1, 1, 3)
            reverse_goal = selected_goal[None, None]
        else:
            reverse_goal, self.reverse_goal_grad = self.model.goal_model.reverse_diffusion(
                self.nsamp,
                self.num_timesteps,
                self.model.noise_scheduler,
                self.data.past,
                self.data.cam,
                self.data.x0_to_world,
                None,
                self.feat_volume,
                self.model.voxel_grid,
                self.drop_cam,
                self.drop_past,
                None,
                xyz_grid=self.xyz_grid,
                visualizer=self.visualizer,
            )
            reverse_goal = reverse_goal.view(self.num_timesteps + 1, -1, self.nsamp, 1, 1, 3)
            selected_goal = reverse_goal[-1, :, 0]  # # bs, T(1),K(1),3

        if replace_goal:
            self.data.x0_goal = selected_goal 
        self.reverse_goal = reverse_goal

        if self.visualizer is not None:
            if self.agent_idx==0:
                color_samp = [1, 0, 0]
            else:
                color_samp = [1, 1, 0]
            self.visualizer.render_goal_viser(
                self.reverse_goal[-1, 0, :, 0, 0] + self.data.x0_to_world[0, 0], color=color_samp
            )

    def update_waypoint(self, replace_wp=True):
        # waypoint | goal conditioning
        reverse_wp_guide, self.reverse_wp_guide_grad = self.model.waypoint_model.reverse_diffusion(
            self.nsamp,
            self.num_timesteps,
            self.model.noise_scheduler,
            self.data.past,
            self.data.cam,
            self.data.x0_to_world,
            None,
            self.feat_volume,
            self.model.voxel_grid,
            self.drop_cam,
            self.drop_past,
            self.data.x0_goal,
            xyz_grid=self.xyz_grid_wp,
            visualizer=self.visualizer,
        )
        if self.visualizer is not None:
            if self.agent_idx==0:
                cm_name="cool"
            else:
                cm_name="hot"
            self.visualizer.render_roottraj_viser(
                reverse_wp_guide[-1].view(self.nsamp, -1, 3) + self.data.x0_to_world, cm_name=cm_name, prefix="waypoints"
            )
        self.reverse_wp_guide = reverse_wp_guide.view(
            self.num_timesteps + 1, -1, self.nsamp, self.model.forecast_size, 1, 3
            )
        if replace_wp:
            self.data.x0 = self.reverse_wp_guide[-1, :, 0]

    def update_fullbody(self):
        # full body | wp conditioning
        bs = self.data.past.shape[0]
        past_joints_cond = torch.cat(
            [
                self.data.past.reshape(bs, self.model.memory_size, -1),
                self.data.past_angles.reshape(bs, self.model.memory_size, -1),
                self.data.past_joints.reshape(bs, self.model.memory_size, -1),
            ],
            -1,
        )
        reverse_joints, self.reverse_joints_grad = self.model.fullbody_model.reverse_diffusion(
            self.nsamp,
            self.num_timesteps,
            self.model.noise_scheduler,
            past_joints_cond,
            self.data.cam * 0,
            self.data.x0_to_world,
            None,
            None,
            self.model.voxel_grid,
            self.drop_cam,
            self.drop_past,
            self.data.x0,
            visualizer=None,
        )
        reverse_joints = reverse_joints.view(
            reverse_joints.shape[:3] + (self.model.forecast_size, -1)
        )
        reverse_wp = reverse_joints[..., :3]
        reverse_angles = reverse_joints[..., 3:12]
        reverse_joints = reverse_joints[..., 12:]

        reverse_joints = reverse_joints.view(
            -1, self.nsamp, self.model.forecast_size, self.model.num_kps, 3
        )
        reverse_angles = reverse_angles.view(-1, self.nsamp, self.model.forecast_size, 1, 9)
        # SVD and get valid rotation
        reverse_angles = symmetric_orthogonalization(reverse_angles)

        reverse_wp = reverse_wp.view(-1, bs, self.nsamp, self.model.forecast_size, 1, 3)
        reverse_joints = reverse_joints.view(
            -1, bs, self.nsamp, self.model.forecast_size, self.model.num_kps, 3
        )
        reverse_angles = reverse_angles.view(
            -1, bs, self.nsamp, self.model.forecast_size, 1, 3, 3
        )

        if self.visualizer is not None:
            # visualize full body
            self.visualizer.render_roottraj_viser(
                reverse_wp[-1].view(self.nsamp, -1, 3) + self.data.x0_to_world,
                prefix="roottraj",
                cm_name="jet",
            )
            self.render_fullbody(self.data.x0_to_world, reverse_wp[-1, :, 0], reverse_angles[-1, :, 0], reverse_joints[-1, :, 0])

        self.reverse_wp = reverse_wp
        self.reverse_angles = reverse_angles
        self.reverse_joints = reverse_joints

    def render_fullbody(self, x0_to_world_sub, x0_sub, x0_angles_sub, x0_joints_sub):
        """
        input: bs,T,K,3
        """
        x0_to_world_sub = x0_to_world_sub[0]
        x0_sub = x0_sub[0]
        x0_angles_sub = x0_angles_sub[0]
        x0_joints_sub = x0_joints_sub[0]

        if self.agent_class=="lab4d":
            x0_angles_sub = matrix_to_axis_angle(x0_angles_sub).view(-1, 3)
            x0_sub = (x0_sub + x0_to_world_sub).view(-1, 3)
            self.visualizer.render_fullbody_viser(
                x0_angles_sub, x0_sub, x0_joints_sub
            )
        elif self.agent_class=="smpl":
            # flip yz
            x0_angles_sub[...,[1,2]] *= -1
            self.visualizer.visualze_smpl(
                x0_joints_sub, x0_angles_sub, x0_sub, x0_to_world_sub
            )
        else:
            raise ValueError

    def update(self):
        if self.agent_type=="root":
            pred_traj = self.reverse_wp_guide[-1, :1, 0]
        elif self.agent_type=="full":
            pred_traj = self.reverse_wp[-1, :1, 0]
        else:
            raise ValueError
        self.accumulated_traj = torch.cat([self.accumulated_traj, pred_traj], 1)
        self.accumulated_traj = self.accumulated_traj - pred_traj[:, -1:]

        self.data.x0_to_world = self.data.x0_to_world + pred_traj[:, -1:]
        self.data.past = self.accumulated_traj[:, -self.model.memory_size :]

        if self.agent_type=="full":
            self.data.past_joints = self.reverse_joints[-1, :, 0, -self.model.memory_size :]
            self.data.x0_joints = self.reverse_joints[-1, :, 0, -1:]

            self.data.past_angles = self.reverse_angles[-1, :, 0, -self.model.memory_size :]
            self.data.x0_angles = self.reverse_angles[-1, :, 0, -1:]

    def save_to_file(self, out_path, round_idx, batch_idx = 0, sample_idx = 0):
        # render to video in a separate thread
        wp = self.data.x0_to_world[batch_idx] + self.reverse_wp[-1, batch_idx, sample_idx] # TK3
        sample = torch.cat(
            [
                matrix_to_axis_angle(self.reverse_angles[-1, batch_idx, sample_idx]),
                wp,
                self.reverse_joints[-1, batch_idx, sample_idx],
            ],
            dim=-2,
        )
        sample = sample.reshape(sample.shape[0], -1).cpu().numpy()
        # T,81
        os.makedirs("%s/sample" % (out_path), exist_ok=True)
        np.save("%s/sample/%04d.npy" % (out_path, round_idx), sample)

    def add_cam_to_visualizer(self):
        hit_list = torch.stack([self.data.cam[0, 0, 0], self.data.cam[0, -1, 0]], 0)  # 2,3
        hit_list = hit_list + self.data.x0_to_world[0, :, 0]
        self.visualizer.userwp_list = hit_list.cpu().numpy().tolist()
        self.visualizer.show_control_points()