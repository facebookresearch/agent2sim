from scipy.spatial.transform import Rotation as R
import os, sys
import pdb
import trimesh
import time

sys.path.insert(0, os.getcwd())
from utils import get_lab4d_data, get_xzbounds
from config import get_config

from lab4d.utils.quat_transform import matrix_to_axis_angle
from projects.csim.voxelize import BGField
from visualizer import DiffusionVisualizer


if __name__ == "__main__":
    # params
    config = get_config()
    num_timesteps = config.num_timesteps

    # data
    # data
    if config.load_logname == "amass":
        data_path = "database/motion/amass.pkl"
        use_amass = True
    else:
        if config.use_test_data:
            data_path = "database/motion/%s-test-L64-S100.pkl" % config.load_logname
        else:
            data_path = "database/motion/%s-train-L64-S1.pkl" % config.load_logname
        use_amass = False
    data = get_lab4d_data(data_path, full_len=64)

    # visualize
    if use_amass:
        bg_field = None
        lab4d_dir = None
    else:
        bg_field = BGField(load_logname=config.load_logname, use_default_mesh=True)
        lab4d_dir = "logdir/%s/" % config.load_logname

    xzmin, xzmax, _, _ = get_xzbounds(data.x0)
    visualizer = DiffusionVisualizer(
        xzmax=xzmax,
        xzmin=xzmin,
        num_timesteps=num_timesteps,
        bg_field=bg_field,
        logdir="exps/%s-%s/" % (config.load_logname, config.logname_gd),
        lab4d_dir=lab4d_dir,
        num_seq=len(data.x0),
    )
    visualizer.run_viser()

    sample_idx = 0
    while True:
        # sample_idx = sample_idx + 1 if sample_idx < len(x0) - 1 else 0
        sample_idx = visualizer.sequence_slider.value
        # data
        x0_sub = data.x0[sample_idx]
        past_sub = data.past[sample_idx]
        cam_sub = data.cam[sample_idx]
        x0_to_world_sub = data.x0_to_world[sample_idx]
        x0_joints_sub = data.x0_joints[sample_idx]
        x0_angles_sub = data.x0_angles[sample_idx]
        # T, K, 3

        visualizer.render_roottraj_viser(
            past_sub.view(1, -1, 3) + x0_to_world_sub,
            prefix="past",
            cm_name="gray_r",
        )
        visualizer.render_roottraj_viser(
            x0_sub.view(1, -1, 3) + x0_to_world_sub, prefix="waypoints"
        )
        visualizer.render_roottraj_viser(
            cam_sub.view(1, -1, 3) + x0_to_world_sub,
            prefix="observer",
            cm_name="jet",
            point_size=0.05,
        )

        if use_amass:
            # smpl
            visualizer.visualze_smpl(
                x0_joints_sub, x0_angles_sub, x0_sub, x0_to_world_sub
            )
        else:
            # lab4d format
            visualizer.render_fullbody_viser(
                matrix_to_axis_angle(x0_angles_sub).view(-1, 3),
                x0_sub.view(-1, 3) + x0_to_world_sub[0],
                x0_joints_sub,
                render_mesh=True,
            )
            time.sleep(0.02)
