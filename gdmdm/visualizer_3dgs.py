import sys, os
import torch
import numpy as np
import time
import pdb

from lab4d.config import load_flags_from_file
from lab4d.render import get_config
from lab4d.utils.quat_transform import matrix_to_quaternion, quaternion_translation_mul
from visualizer import DiffusionVisualizer
from projects.diffgs.trainer import GSplatTrainer as Trainer
from projects.diffgs.gs_model import GSplatModel

from scipy.spatial.transform import Rotation as R

class DiffusionVisualizerGS(DiffusionVisualizer):
    """
    extend mesh visualizer to 3dgs
    """
    def __init__(
        self,
        xzmax,
        xzmin,
        state_size=3,
        num_timesteps=50,
        bg_field=None,
        logdir="base",
        lab4d_dir=None,
        num_seq=1,
    ):
        super().__init__(
            xzmax,
            xzmin,
            state_size=state_size,
            num_timesteps=num_timesteps,
            bg_field=bg_field,
            logdir=logdir,
            lab4d_dir=lab4d_dir,
            num_seq=num_seq,
        )
        return 
    

    def initialize_server(self):
        # load model/data
        seqname = "cat-pikachu-2024-08-v2"
        logname="diffgs-ft-fg-b32-urdf-quad-r120-const-absgrad-adapt-again"
        opts = load_flags_from_file(f"logdir/{seqname}-{logname}/opts.log")
        opts["load_suffix"] = "latest"
        opts["lab4d_path"] = ""
        opts["use_gui"] = False
        model_fg, data_info, _ = Trainer.construct_test_model(opts, return_refs=False, force_reload=False)

        seqname = "Oct5at10-49AM-poly"
        logname="diffgs-fs-bg-b8-bob-r120-mlp-fixgs-20reset-01th-rgbd001-exp"
        opts = load_flags_from_file(f"logdir/{seqname}-{logname}/opts.log")
        opts["load_suffix"] = "latest"
        opts["lab4d_path"] = ""
        opts["use_gui"] = False
        model_bg, data_info, _ = Trainer.construct_test_model(opts, return_refs=False, force_reload=False)

        # load composed model
        opts["field_type"] = "comp"
        opts["use_gui"] = True
        opts["render_res"] = 512
        model = GSplatModel(opts, data_info)
        model.gaussians.gaussians[0] = model_bg.gaussians
        model.gaussians.gaussians[1] = model_fg.gaussians
        model.cuda()
        model.eval()

        self.server = model.gui.server
        self.update = lambda x: model.gui.update(x)

        import viser.transforms as tf
        self.base_handle = self.server.add_frame(
            "/frames",
            wxyz=tf.SO3.exp(np.array([0.0, 0.0, 0.0])).wxyz,
            position=(0, 0, 0),
            show_axes=False,
        )
        if self.bg_field is not None:
            # self.server.add_mesh_trimesh(
            #     name="/frames/environment",
            #     mesh=self.bg_field.bg_mesh,
            # )

            self.root_visitation_boxes = self.bg_field.voxel_grid.to_boxes(
                mode="root_visitation"
            )
            self.cam_visitation = self.bg_field.voxel_grid.to_boxes(
                mode="cam_visitation"
            )
    # def show_control_points(self, mode="userwp"):
    #     super().show_control_points(mode=mode)
    #     self.update()


    def render_fullbody_viser(self, angles, wp, joints, render_mesh=True, agent_idx=0):
        """
        future: Tx3
        """
        if torch.is_tensor(angles):
            angles = angles.cpu().numpy()
        if torch.is_tensor(wp):
            wp = wp.cpu().numpy()
        if torch.is_tensor(joints):
            joints = joints.cpu().numpy()

        so3_wp_angles = np.concatenate([angles[:, None], wp[:, None], joints], axis=1)
        nframes = so3_wp_angles.shape[0]

        if render_mesh:
            self.articulation_loader.load_files(so3_wp_angles.reshape(nframes, -1))
            for t in range(len(so3_wp_angles)):
                self.current_state = so3_wp_angles[t]
                self.viser_update_func()
                # time.sleep(0.05)
        else:
            raise ValueError

    def viser_update_func(self):
        func_update = lambda x, y: update_gaussians_by_frameid(x, y, self.current_state)
        self.update(func_update)


def update_gaussians_by_frameid(self, batch, so3_wp_angles_t):
    inst_id = self.inst_id_slider.value
    frameid_appr = self.get_frameid(self.frameid_sub_slider_appr.value, inst_id)
    self.renderer.process_frameid(batch) # absolute
    self.renderer.gaussians.update_motion(batch["frameid"])
    self.renderer.gaussians.update_appearance(batch["frameid"] * 0 + frameid_appr)
    self.renderer.gaussians.update_extrinsics(batch["frameid"])

    rmat = R.from_rotvec(so3_wp_angles_t[0]).as_matrix()
    trans = so3_wp_angles_t[1]
    so3 = so3_wp_angles_t[2:].reshape(-1)
    # transform to camera coordinate
    dev = self.renderer.device
    quat_rw = matrix_to_quaternion(torch.tensor(rmat, dtype=torch.float32, device=dev))
    trans_rw = torch.tensor(trans, dtype=torch.float32, device=dev)

    # quat, trans
    self.renderer.gaussians.gaussians[1].quat_cache[-1] = quat_rw
    self.renderer.gaussians.gaussians[1].trans_cache[-1] = trans_rw

    self.renderer.gaussians.gaussians[0].trans_cache[-1] = torch.zeros_like(trans_rw)
    self.renderer.gaussians.gaussians[0].quat_cache[-1] = torch.zeros_like(quat_rw)
    self.renderer.gaussians.gaussians[0].quat_cache[-1][0] = 1

    # N, 7
    gaussians_fg = self.renderer.gaussians.gaussians[1]
    field = gaussians_fg.lab4d_model.fields.field_params["fg"]
    scale_fg = gaussians_fg.scale_field.detach()
    fake_frameid = torch.tensor([0], dtype=torch.long, device="cuda")
    from lab4d.utils.quat_transform import dual_quaternion_to_quaternion_translation

    xyz = gaussians_fg._xyz[None]

    so3 = torch.tensor(so3, dtype=torch.float32, device="cuda").view(1, -1, 3)
    t_articulation = field.warp.articulation.get_vals(
        frame_id=fake_frameid, return_so3=False, override_so3=so3
    )

    # deform mesh
    samples_dict = {"t_articulation": t_articulation,
                    "rest_articulation": field.warp.articulation.get_mean_vals()}
    xyz_t, warp_dict = field.warp(
        xyz[None] * scale_fg,
        None,
        torch.tensor([inst_id], device="cuda"),
        samples_dict=samples_dict,
        return_aux=True,
    )
    xyz_t = xyz_t[:, 0] / scale_fg
    motion = (xyz_t - xyz).transpose(0, 1).contiguous()

    # rotataion and translation of each gaussian
    quat, _ = dual_quaternion_to_quaternion_translation(warp_dict["dual_quat"])
    quat = quat.transpose(0, 1).contiguous()
    trajectory_pred = torch.cat((quat, motion), dim=-1)
    self.renderer.gaussians.gaussians[1].trajectory_cache[-1] = trajectory_pred[:,0] # N, 7