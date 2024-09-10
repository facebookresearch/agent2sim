import pdb
import sys, os
import numpy as np
import torch
import trimesh
from utils import get_lab4d_data

sys.path.insert(0, os.getcwd())
from lab4d.utils.geom_utils import rot_angle
from projects.csim.voxelize import BGField

from utils import get_lab4d_data


def eval_ADE(pred, gt, use_rot=False):
    """
    pred: (bs,ntrial,nsamp, T, K, 3)
    gt: (bs, T, K, 3)
    """
    if use_rot:
        diff = rot_angle(
            pred @ gt[:, None, None].transpose(-1, -2)
        )  # bs, ntrail, nsamp, T, K
    else:
        diff = (pred - gt[:, None, None]).norm(2, -1)  # bs, ntrail, nsamp, T, K
    diff = diff.mean(3).mean(3)  # bs, ntrail, nsamp, average over T and K
    mindiff = diff.min(2).values  # bs, ntrial
    mean = mindiff.mean(1)
    std = mindiff.std(1)
    return mean, std


def eval_all(
    pred_goal_all,
    gt_goal_all,
    pred_wp_all,
    gt_wp_all,
    pred_orient_all,
    gt_orient_all,
    pred_joints_all,
    gt_joints_all,
    ntrial=12,
):
    """
    pred_goal_all: (bs, nsamp, T, 1, 3)
    gt_goal_all: (bs, T, 1, 3)
    pred_wp_all: (bs, nsamp, T, K, 3)
    gt_wp_all: (bs, T, K, 3)
    pred_orient_all: (bs, nsamp, T, 1, 3,3)
    gt_orient_all: (bs, T, 1, 3,3)
    pred_joints_all: (bs, nsamp, T, K, 3)
    gt_joints_all: (bs, T, K, 3)
    """
    bs = pred_goal_all.shape[0]
    if pred_wp_all is None:
        pred_wp_all = torch.zeros(bs, ntrial, 1, 1, 3)
        gt_wp_all = torch.zeros(bs, 1, 1, 1, 3)
    if pred_orient_all is None:
        pred_orient_all = torch.zeros(bs, ntrial, 1, 1, 3, 3)
        gt_orient_all = torch.zeros(bs, 1, 1, 1, 3, 3)
    if pred_joints_all is None:
        pred_joints_all = torch.zeros(bs, ntrial, 1, 1, 3)
        gt_joints_all = torch.zeros(bs, 1, 1, 1, 3)
    pred_goal_all = pred_goal_all.view(bs, ntrial, -1, *pred_goal_all.shape[2:])
    pred_wp_all = pred_wp_all.view(bs, ntrial, -1, *pred_wp_all.shape[2:])
    pred_orient_all = pred_orient_all.view(bs, ntrial, -1, *pred_orient_all.shape[2:])
    pred_joints_all = pred_joints_all.view(bs, ntrial, -1, *pred_joints_all.shape[2:])

    minDE_goal, std_goal = eval_ADE(pred_goal_all, gt_goal_all)
    minADE_wp, std_wp = eval_ADE(pred_wp_all, gt_wp_all)
    minADE_orient, std_orient = eval_ADE(pred_orient_all, gt_orient_all, use_rot=True)
    minADE_joints, std_joints = eval_ADE(pred_joints_all, gt_joints_all)

    # Define the header and values with exact alignment by specifying the width of each column
    header = "{:<16} | {:<15} | {:<15}".format(
        "Goal minDE/DE",
        "WP minADE/ADE",
        "Orient minADE/ADE",
        "Body minADE/ADE:",
    )
    print(header)
    for it in range(len(minDE_goal)):
        values = "{:03d} & {:>7.3f} & {:>7.3f} & {:>7.3f} & {:>7.3f} & {:>7.3f} & {:>7.3f}  & {:>7.3f} & {:>7.3f} \\\\".format(
            it,
            minDE_goal[it],
            std_goal[it],
            minADE_wp[it],
            std_wp[it],
            minADE_orient[it],
            std_orient[it],
            minADE_joints[it],
            std_joints[it],
        )
        print(values)
    values = "AVE & {:>7.3f} & {:>7.3f} & {:>7.3f} & {:>7.3f} & {:>7.3f} & {:>7.3f}  & {:>7.3f} & {:>7.3f} \\\\".format(
        minDE_goal.mean(),
        std_goal.mean(),
        minADE_wp.mean(),
        std_wp.mean(),
        minADE_orient.mean(),
        std_orient.mean(),
        minADE_joints.mean(),
        std_joints.mean(),
    )
    print(values)


if __name__ == "__main__":

    # test data
    (
        x0_wp_all,
        past_wp_all,
        cam_all,
        x0_to_world_all,
        x0_joints_all,
        past_joints_all,
        x0_angles_all,
        past_angles_all,
        x0_angles_to_world_all,
    ) = get_lab4d_data("database/motion/home-2023-curated3-test9-L64-S10.pkl")
    x0_goal_test = x0_wp_all[:, -1:]

    # data
    (
        x0_wp_all,
        past_wp_all,
        cam_all,
        x0_to_world_all,
        x0_joints_all,
        past_joints_all,
        x0_angles_all,
        past_angles_all,
        x0_angles_to_world_all,
    ) = get_lab4d_data("database/motion/home-2023-curated3-train-L64-S1.pkl")
    x0_goal_all = x0_wp_all[:, -1:]

    # VISITATION sampling
    # bg_field = BGField()
    # pred_goal = bg_field.voxel_grid.sample_from_voxel(64, mode="root_visitation")

    from projects.csim.voxelize import VoxelGrid
    from denoiser import get_env_grid

    nsamp = 192  # 12*16
    bs = x0_goal_test.shape[0]
    x_dim, y_dim, z_dim = 64, 8, 64
    env_grid = get_env_grid(x_dim, y_dim, z_dim)
    mesh = trimesh.Trimesh(vertices=env_grid, faces=[])
    voxel_grid = VoxelGrid(mesh, res=0.1)
    voxel_grid.count_root_visitation(x0_wp_all[:, -1, 0, :3].cpu().numpy())
    pred_goal = voxel_grid.sample_from_voxel(bs * nsamp, mode="root_visitation")
    # voxel_grid.run_viser()

    pred_goal = torch.tensor(pred_goal, dtype=torch.float32, device="cuda")
    pred_goal = pred_goal.view(bs, nsamp, 1, 1, 3)

    eval_all(pred_goal, x0_goal_test, None, None, None, None, None, None)
