from typing import NamedTuple
from recordclass import RecordClass
import math
import cv2
import os, sys
import pdb
import numpy as np
import glob
from tqdm.auto import tqdm
from einops import rearrange
import trimesh
import argparse
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.spatial.transform import Rotation as R

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.getcwd() + "/../lab4d")
from lab4d.utils.vis_utils import get_pts_traj
from lab4d.utils.pyrender_wrapper import PyRenderWrapper
from lab4d.utils.io import save_vid
from lab4d.utils.quat_transform import matrix_to_axis_angle, axis_angle_to_matrix


class GDMDMMotion(RecordClass):
    """
    explicit motion params for reanimation and transfer
    """
    x0_goal: torch.Tensor
    x0: torch.Tensor
    past: torch.Tensor
    cam: torch.Tensor
    x0_to_world: torch.Tensor
    x0_joints: torch.Tensor
    past_joints: torch.Tensor
    x0_angles: torch.Tensor
    past_angles: torch.Tensor
    x0_angles_to_world: torch.Tensor


def get_data():
    x0_0 = torch.zeros((1000, 2))
    x0_0[:, 0] = torch.linspace(-1, 1, 1000)
    y_0 = torch.zeros((1000, 1))
    x0_1 = torch.zeros((1000, 2))
    x0_1[:, 1] = torch.linspace(-1, 1, 1000)
    y_1 = torch.ones((1000, 1))

    x0 = torch.cat((x0_0, x0_1), dim=0)
    y = torch.cat((y_0, y_1), dim=0)
    return x0, y


def humanact12_to_gdmdm(path):
    path = "../motion-diffusion-model/dataset/HumanAct12Poses/humanact12poses.pkl"
    indata = pkl.load(open(path, "rb"))
    cut_size = 64

    data = {}
    data["poses"] = []
    data["joints3D"] = []
    data["se3"] = []

    for i in range(len(indata["poses"])):
        if len(indata["poses"][i]) < cut_size:
            continue
        poses = indata["poses"][i][:cut_size, 3:]  # 23 joints
        joints3D = indata["joints3D"][i][:cut_size, 3:]

        se3 = np.eye(4).reshape(1, 4, 4).repeat(cut_size, axis=0)
        rotmat = R.from_rotvec(indata["poses"][i][:cut_size, :3]).as_matrix()
        se3[:, :3, :3] = rotmat
        trans = indata["joints3D"][i][:cut_size, 0]
        se3[:, :3, 3] = trans
        se3 = np.linalg.inv(se3)

        data["poses"].append(poses)
        data["joints3D"].append(joints3D)
        data["se3"].append(se3)

    pkl.dump(data, open("database/motion/humanact12.pkl", "wb"))
    print("saved to database/motion/humanact12.pkl")
    print("%d sequences" % len(data["poses"]))
    return data


def humanml3d_to_gdmdm(path):
    path = "/home/gengshay/code/HumanML3D/HumanML3D/new_joint_vecs/"
    # path = "/home/gengshay/code/guided-motion-diffusion/dataset/HumanML3D/new_joint_vecs_abs_3d/"

    data = {}
    data["poses"] = []
    data["joints3D"] = []
    data["se3"] = []

    cut_size = 64
    joints_num = 22
    for npyfile in sorted(glob.glob("%s/*.npy" % path)):
        indata = np.load(npyfile)[:cut_size]  # T,263
        if len(indata) < cut_size:
            continue
        r_rot_quat, r_pos = recover_root_rot_pos(indata)
        start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
        end_indx = start_indx + (joints_num - 1) * 6
        cont6d_params = indata[..., start_indx:end_indx]

        # poses = rotation_6d_to_matrix(cont6d_params.reshape(-1, 6))
        poses = cont6d_to_matrix(cont6d_params.reshape(-1, 6))
        poses = matrix_to_axis_angle(torch.tensor(poses)).numpy()
        poses = poses.reshape(cut_size, -1)
        poses = np.concatenate([poses, np.zeros((cut_size, 6))], axis=1)
        # poses[..., 6:] = poses[..., 6:] * 0
        joints3D = np.zeros_like(poses)

        # convert from
        se3 = np.eye(4)
        se3 = np.tile(se3, (len(indata), 1, 1))
        r_rot_quat = r_rot_quat[..., [1, 2, 3, 0]]
        se3[:, :3, :3] = R.from_quat(r_rot_quat).as_matrix()
        se3[:, :3, 3] = r_pos
        # cv2gl
        # se3[:, [1, 2], 3] *= -1
        se3 = np.linalg.inv(se3)

        data["poses"].append(poses)
        data["joints3D"].append(joints3D)
        data["se3"].append(se3)
    pkl.dump(data, open("database/motion/humanml3d.pkl", "wb"))
    print("saved to database/motion/humanml3d.pkl")
    print("%d sequences" % len(data["poses"]))


def amass_to_gdmdm(amass_dir="../HumanML3D/amass_data/KIT", cut_size=64, stride=10):
    group_path = get_amass_dirs(amass_dir)

    data = {}
    data["poses"] = []
    data["joints3D"] = []
    data["se3"] = []
    for paths in group_path:
        dataset_name = paths[0].split("/")[2]
        pbar = tqdm(paths)
        pbar.set_description("Processing: %s" % dataset_name)
        for path in pbar:
            se3, poses = amass_to_pose(path)
            if se3 is not None and len(se3) > cut_size:
                for i in range(0, len(se3), stride):
                    if i + cut_size > len(se3):
                        break
                    se3_sub = se3[i : i + cut_size]
                    poses_sub = poses[i : i + cut_size]

                    data["se3"].append(se3_sub)
                    data["poses"].append(poses_sub)
                    data["joints3D"].append(np.zeros_like(poses_sub))
    pkl.dump(data, open("database/motion/amass.pkl", "wb"))
    print("saved to database/motion/amass.pkl")
    print("%d sequences" % len(data["poses"]))


def get_amass_dirs(amass_dir):
    paths = []
    folders = []
    dataset_names = []
    for root, dirs, files in os.walk(amass_dir):
        folders.append(root)
        for name in files:
            dataset_name = root.split("/")[2]
            if dataset_name not in dataset_names:
                dataset_names.append(dataset_name)
            paths.append(os.path.join(root, name))

    group_path = [[path for path in paths if name in path] for name in dataset_names]
    return group_path


def amass_to_pose(src_path, ex_fps=10):
    bdata = np.load(src_path, allow_pickle=True)
    try:
        fps = bdata["mocap_framerate"]
        frame_number = bdata["trans"].shape[0]
        # print("fps", fps, "frame_number", frame_number)
    except:
        return None, None

    down_sample = int(fps / ex_fps)
    rots = bdata["poses"][::down_sample, :3]
    trans = bdata["trans"][::down_sample]
    poses = bdata["poses"][::down_sample, 3:66]

    se3s = np.eye(4)
    se3s = np.tile(se3s, (len(rots), 1, 1))
    se3s[:, :3, 3] = trans
    se3s[:, :3, :3] = R.from_rotvec(rots).as_matrix()

    trans_matrix = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    se3s = trans_matrix[None] @ se3s

    se3s = np.linalg.inv(se3s)
    poses = np.concatenate([poses, np.zeros_like(poses[..., :6])], axis=-1)
    return se3s, poses

def rotate_data(roty, x, y, cam, goal_angles, past_angles, e2w_rot):
    bs = x.shape[0]
    dev = x.device
    if not torch.is_tensor(roty):
        roty = torch.tensor([roty], device=dev)[None].repeat(bs,1)

    rotxyz = torch.zeros(bs, 3, device=dev)
    rotxyz[:, 1] = roty[:, 0]
    w2e_rot = axis_angle_to_matrix(rotxyz)[:, None, None]  # N, 1,1,3, 3

    # N, T13 => N, T, 1, 3, 1
    x = (w2e_rot @ x.view(bs, -1, 1, 3, 1)).view(x.shape)
    y = (w2e_rot @ y.view(bs, -1, 1, 3, 1)).view(y.shape)
    cam = (w2e_rot @ cam.view(bs, -1, 1, 3, 1)).view(cam.shape)
    goal_angles = (w2e_rot @ goal_angles.view(bs, -1, 1, 3, 3)).view(goal_angles.shape)
    past_angles = (w2e_rot @ past_angles.view(bs, -1, 1, 3, 3)).view(past_angles.shape)
    e2w_rot = (e2w_rot.view(bs, -1, 1, 3, 3) @ w2e_rot.transpose(4, 3)).reshape(bs, 9)
    return x, y, cam, goal_angles, past_angles, e2w_rot


def get_lab4d_data(pkldatafilepath, full_len=None, idx0=7, use_ego=True, swap_cam_root=False, roty=0):
    if full_len is None:
        full_len = int(pkldatafilepath.split("-L")[1].split("-")[0])
    data = pkl.load(open(pkldatafilepath, "rb"))
    # max_data = 1
    max_data = len(data["poses"])

    pose = [x for x in data["poses"]][:max_data]
    joints = [x for x in data["joints3D"]][:max_data]
    world_to_root = [x for x in data["se3"]][:max_data]
    if "cam_se3" in data:
        world_to_cam = [x for x in data["cam_se3"]][:max_data]
    else:
        seqlen = len(data["se3"][0])
        eye = np.tile(np.eye(4)[None], (seqlen, 1, 1))
        world_to_cam = [eye for _ in data["se3"]][:max_data]
    print("loading dataset of length %d | seq length %d" % (len(pose), len(pose[0])))

    if swap_cam_root:
        tmp = world_to_cam
        world_to_cam = world_to_root
        world_to_root = tmp

    # current frame
    # idx0 = 16
    # convention:
    # past: 0,1,2,3,4,5,6,7 (8 frames)
    # current: 7 (identity SE(3))
    # future: 8,....63 (56 frames)

    goal_idx = list(range(full_len))[idx0 + 1 :]
    print("goal_idx", goal_idx)
    forecast_size = len(goal_idx)

    # load data: N, T, 3
    root_world_se3 = np.linalg.inv(np.stack(world_to_root, axis=0))
    root_world_se3 = torch.tensor(root_world_se3, dtype=torch.float32)
    root_world_trans = root_world_se3[..., :3, 3]  # N,T,3
    root_world_rot = root_world_se3[..., :3, :3]  # N,T,3,3
    cam_world = np.linalg.inv(np.stack(world_to_cam, axis=0))[..., :3, 3]
    cam_world = torch.tensor(cam_world, dtype=torch.float32)
    joints_ego = np.stack(joints, axis=0)  # N,T,K,3
    joints_ego = torch.tensor(joints_ego, dtype=torch.float32)  # [:, :, 0:4]
    angles = torch.tensor(np.stack(pose, axis=0), dtype=torch.float32)  # N,T,K,3

    # transform root to zero centered at t0
    root_to_world_trans = root_world_trans[:, idx0].clone()
    if use_ego:
        cam_world = cam_world - root_world_trans[:, idx0 : idx0 + 1]
        root_world_trans = root_world_trans - root_world_trans[:, idx0 : idx0 + 1]
    else:
        root_to_world_trans = torch.zeros_like(root_to_world_trans)

    # transform root rotation to have identity at t0
    root_to_world_orient = axis_angle_to_matrix(torch.zeros_like(root_to_world_trans))
    root_world_so3 = matrix_to_axis_angle(root_world_rot)

    # get past/goal pairs
    goal_world_trans = root_world_trans[:, goal_idx]  # N, T, 3
    goal_world_so3 = root_world_so3[:, goal_idx]
    goal_joints_ego = joints_ego[:, goal_idx]  # N, T, K, 3
    goal_angles = angles[:, goal_idx]

    past_world_trans = root_world_trans[:, : idx0 + 1]
    past_world_so3 = root_world_so3[:, : idx0 + 1]
    past_joints_ego = joints_ego[:, : idx0 + 1]  # N, T, K, 3
    past_angles = angles[:, : idx0 + 1]
    cam_world = cam_world[:, : idx0 + 1]  # camera position of the past frames

    # reshape
    bs = goal_world_trans.shape[0]
    goal_world_trans = goal_world_trans.view(bs, forecast_size, 1, 3)
    goal_world_so3 = goal_world_so3.view(bs, forecast_size, 1, 3)
    goal_joints_ego = goal_joints_ego.view(bs, forecast_size, -1, 3)
    goal_angles = goal_angles.view(bs, forecast_size, -1, 3)

    past_world_trans = past_world_trans.view(bs, idx0 + 1, 1, 3)
    past_world_so3 = past_world_so3.view(bs, idx0 + 1, 1, 3)
    past_joints_ego = past_joints_ego.view(bs, idx0 + 1, -1, 3)
    past_angles = past_angles.view(bs, idx0 + 1, -1, 3)
    cam_world = cam_world.view(bs, idx0 + 1, 1, 3)

    root_to_world_trans = root_to_world_trans.view(bs, 1, 1, 3)
    root_to_world_orient = root_to_world_orient.view(bs, 1, 1, 3, 3)

    # move to cuda
    goal_world_trans = goal_world_trans.cuda()
    goal_world_so3 = goal_world_so3.cuda()
    goal_joints_ego = goal_joints_ego.cuda()
    goal_angles = goal_angles.cuda()

    past_world_trans = past_world_trans.cuda()
    past_world_so3 = past_world_so3.cuda()
    past_joints_ego = past_joints_ego.cuda()
    past_angles = past_angles.cuda()
    cam_world = cam_world.cuda()
    root_to_world_trans = root_to_world_trans.cuda()
    root_to_world_orient = root_to_world_orient.cuda()

    goal_trans = goal_world_trans
    past_trans = past_world_trans
    goal_orient = goal_world_so3
    past_orient = past_world_so3

    # TODO learn an autoencoder for joints

    past_orient = axis_angle_to_matrix(past_orient)
    goal_orient = axis_angle_to_matrix(goal_orient)

    goal_trans, past_trans, cam_world, goal_orient, past_orient, root_to_world_orient = \
        rotate_data(roty, goal_trans, past_trans, cam_world, goal_orient, past_orient, root_to_world_orient)

    data = GDMDMMotion(
                       x0_goal=goal_trans[:, -1:],
                       x0=goal_trans, 
                       past=past_trans,
                       cam=cam_world,
                       x0_to_world=root_to_world_trans,
                       x0_joints=goal_angles,
                       past_joints=past_angles,
                       x0_angles=goal_orient,
                       past_angles=past_orient,
                       x0_angles_to_world=root_to_world_orient,
                       )
    return data


class TrajDataset(Dataset):
    """Dataset for loading trajectory data."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data.x0)

    def __getitem__(self, idx):
        return (
            self.data.x0[idx],
            self.data.past[idx],
            self.data.cam[idx],
            self.data.x0_to_world[idx],
            self.data.x0_joints[idx],
            self.data.past_joints[idx],
            self.data.x0_angles[idx],
            self.data.past_angles[idx],
            self.data.x0_angles_to_world[idx],
        )


def get_xzbounds(x0):
    xmin, ymin, zmin = x0.view(-1, 3).min(0)[0].cpu().numpy()
    xmax, ymax, zmax = x0.view(-1, 3).max(0)[0].cpu().numpy()
    xzmin = min(xmin, zmin) - 1
    xzmax = max(xmax, zmax) + 1
    return xzmin, xzmax, ymin, ymax


def get_grid_xyz(x0, xsize, ysize, zsize):
    xzmin, xzmax, ymin, ymax = get_xzbounds(x0)
    x = np.linspace(xzmin, xzmax, xsize)
    z = np.linspace(xzmin, xzmax, zsize)
    y = np.linspace(ymin, ymax, ysize)
    xyz = np.stack(np.meshgrid(x, y, z), axis=-1).reshape(-1, 3)
    return xyz, xzmin, xzmax


if __name__ == "__main__":
    path = ""
    amass_to_gdmdm(path)
    # humanml3d_to_gdmdm(path)
    # humanact12_to_gdmdm(path)
