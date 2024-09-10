# CUDA_VISIBLE_DEVICES=0 python projects/gdmdm/demo.py --logname a2s-fix --sample_idx 0 --eval_batch_size 192 --use_test_data
import os, sys
import cv2
import pdb
import numpy as np
import torch
import trimesh
import argparse
import shutil

import ddpm
from utils import get_lab4d_data, get_grid_xyz
from denoiser import reverse_diffusion, simulate_forward_diffusion
from eval import eval_ADE, eval_all
from visualizer import DiffusionVisualizer, run_eval
from config import get_config

sys.path.insert(0, os.getcwd())
from lab4d.utils.quat_transform import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    symmetric_orthogonalization,
)
from projects.csim.voxelize import BGField
from denoiser import TotalDenoiserThreeStageFull
from trainer import GDMDMTrainer
from utils import get_xzbounds


if __name__ == "__main__":
    # params
    config = get_config()

    rt_dict = GDMDMTrainer.construct_test_model(config)
    data = rt_dict["data"]
    model = rt_dict["model"]
    meta = rt_dict["meta"]

    # visualize
    xzmin, xzmax, _, _ = get_xzbounds(data.x0)
    visualizer = DiffusionVisualizer(
        xzmax=xzmax,
        xzmin=xzmin,
        num_timesteps=config.num_timesteps,
        bg_field=model.bg_field,
        logdir="projects/gdmdm/exps/%s-%s/" % (config.load_logname, config.logname_gd),
        lab4d_dir=meta["lab4d_dir"],
        num_seq=len(data.x0)
    )
    config.sample_idx = list(range(len(data.x0)))[4:6]
    agent = run_eval(None, config, rt_dict)

    # compute numbers
    eval_all(
        agent.reverse_goal[-1],
        agent.data.x0_goal,
        agent.reverse_wp_guide[-1],
        agent.data.x0,
        agent.reverse_angles[-1],
        agent.data.x0_angles,
        agent.reverse_joints[-1],
        agent.data.x0_joints,
        ntrial=config.eval_ntrial,
    )

    bs = agent.reverse_goal.shape[1]
    for i in range(bs):
        if i==0:
            reverse_grad_grid_goal = agent.reverse_goal_grad[:, i]
            reverse_grad_grid_wp = agent.reverse_wp_guide_grad[:, i]
        else:
            reverse_grad_grid_goal = None
            reverse_grad_grid_wp = None

        reverse_goal = agent.reverse_goal[:, i]
        reverse_wp = agent.reverse_wp[:, i]
        reverse_angles = matrix_to_axis_angle(agent.reverse_angles[:, i])
        reverse_joints = agent.reverse_joints[:, i]
        ############# visualization
        # goal visualization
        save_prefix = "goal-%d" % i
        # forward process
        forward_samples_goal = agent.model.goal_model.simulate_forward_diffusion(
            agent.data.x0_goal, agent.model.noise_scheduler
        )
        visualizer.render_trajectory(
            forward_samples_goal,
            reverse_goal,
            agent.data.past[i],
            agent.data.x0_to_world[i],
            prefix=save_prefix,
        )
        visualizer.plot_trajectory_2d(
            forward_samples_goal,
            reverse_goal,
            reverse_grad_grid_goal,
            agent.data.x0_goal[i],
            agent.xyz_grid.cpu().numpy(),
            agent.ysize,
            agent.data.past[i],
            agent.data.cam[i],
            prefix=save_prefix,
        )

        # waypoint visualization
        save_prefix = "wp-%d" % i
        # forward process
        forward_samples_waypoint = agent.model.waypoint_model.simulate_forward_diffusion(
            agent.data.x0, agent.model.noise_scheduler
        )
        visualizer.render_trajectory(
            forward_samples_waypoint,
            reverse_wp,
            agent.data.past[i],
            agent.data.x0_to_world[i],
            prefix=save_prefix,
        )
        visualizer.plot_trajectory_2d(
            forward_samples_waypoint,
            reverse_wp,
            reverse_grad_grid_wp,
            agent.data.x0[i],
            agent.xyz_grid.cpu().numpy(),
            agent.ysize,
            agent.data.past[i],
            agent.data.cam[i],
            prefix=save_prefix,
        )

        # full body visualization
        save_prefix = "joints-%d" % i
        # forward process
        forward_samples_joints = []

        so3_wp_angles = torch.cat([
            reverse_angles, 
            reverse_wp,
            reverse_joints
            ], axis=-2).cpu().numpy() # -1, nsamp, T, K, 3
        nframes = np.prod(so3_wp_angles.shape[:-2])
        reverse_joint_locs = visualizer.articulation_loader.load_files_simple(
                so3_wp_angles.reshape(nframes, -1)
            ).reshape(so3_wp_angles.shape[:-2] + (-1,3))[:,:,::10]
        
        so3_wp_angles_past = torch.cat([
            matrix_to_axis_angle(agent.data.past_angles[i]), 
            agent.data.past[i],
            agent.data.past_joints[i]
            ], axis=-2).cpu().numpy()
        nframes_past = np.prod(so3_wp_angles_past.shape[:-2])
        past_joint_locs = visualizer.articulation_loader.load_files_simple(
            so3_wp_angles_past.reshape(nframes_past, -1)
            ).reshape(so3_wp_angles_past.shape[:-2] + (-1,3))[[0,-1]]

        so3_wp_angles_gt = torch.cat([
            matrix_to_axis_angle(agent.data.x0_angles[i]), 
            agent.data.x0[i],
            agent.data.x0_joints[i]
        ], axis=-2).cpu().numpy()
        nframes_gt = np.prod(so3_wp_angles_gt.shape[:-2])
        gt_joint_locs = visualizer.articulation_loader.load_files_simple(
            so3_wp_angles_gt.reshape(nframes_gt, -1)
        ).reshape(so3_wp_angles_gt.shape[:-2] + (-1,3))[::10]

        visualizer.render_trajectory(
            forward_samples_joints,
            reverse_joint_locs,
            past_joint_locs,
            agent.data.x0_to_world[i],
            prefix=save_prefix,
        )
        visualizer.plot_trajectory_2d(
            forward_samples_joints,
            reverse_joint_locs,
            None,
            gt_joint_locs,
            agent.xyz_grid.cpu().numpy(),
            agent.ysize,
            past_joint_locs,
            agent.data.cam[i],
            prefix=save_prefix,
        )

    visualizer.delete()

    # run the command
    ckpt_path = "logdir/%s/" % config.load_logname
    bash_cmd = f"'python projects/behavior/vis.py --gendir {agent.out_path} --logdir {ckpt_path} --fps 30'"
    bash_cmd = f"/bin/bash -c {bash_cmd}"
    print(bash_cmd)
    os.system(bash_cmd)
    print("results are at %s" % agent.out_path)