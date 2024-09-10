import os, sys
import cv2
import pdb
import numpy as np
import torch
import shutil
import trimesh
import time
from copy import deepcopy
from absl import app

import ddpm
from utils import get_lab4d_data, get_grid_xyz, get_xzbounds
from denoiser import reverse_diffusion, simulate_forward_diffusion
from visualizer import DiffusionVisualizer, run_sim
from visualizer_3dgs import DiffusionVisualizerGS
from config import get_config

sys.path.insert(0, os.getcwd())
from lab4d.utils.quat_transform import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    symmetric_orthogonalization,
)
from projects.csim.voxelize import BGField
import projects.diffgs.config 
from denoiser import TotalDenoiserThreeStageFull
from trainer import GDMDMTrainer
from utils import GDMDMMotion
import config_comp


def main(_):
    # params
    config = get_config()
    
    rt_dict = GDMDMTrainer.construct_test_model(config)
    data = rt_dict["data"]
    model = rt_dict["model"]
    meta = rt_dict["meta"]

    # visualize
    visualizer = DiffusionVisualizer(
        xzmax=None,
        xzmin=None,
        num_timesteps=None,
        bg_field=model.bg_field,
        logdir="exps/%s-%s/" % (config.load_logname, config.logname_gd),
        lab4d_dir=meta["lab4d_dir"],
        num_seq=len(data.x0)
    )
    visualizer.run_viser()
    agent = run_sim(visualizer, config, rt_dict, save_to_file=True)
    visualizer.delete()

    # run the command
    ckpt_path = "logdir/%s/" % config.load_logname
    bash_cmd = f"'cd ../lab4d/ && python projects/csim/behavior_vis/vis.py --gendir ../gdmdm/{agent.out_path} --logdir {ckpt_path} --fps 30'"
    bash_cmd = f"/bin/bash -c {bash_cmd}"
    print(bash_cmd)
    os.system(bash_cmd)

if __name__ == "__main__":
    app.run(main)