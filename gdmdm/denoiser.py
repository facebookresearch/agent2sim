import pdb
import time
import os, sys
import math
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm.auto import tqdm
from einops import rearrange

sys.path.insert(0, os.getcwd())
from lab4d.nnutils.embedding import PosEmbedding
from lab4d.utils.quat_transform import axis_angle_to_matrix, matrix_to_axis_angle
from lab4d.nnutils.base import BaseMLP
from projects.csim.voxelize import BGField

from arch import TemporalUnet, TemporalUnetCond, TransformerPredictor, zero_module
from ddpm import NoiseScheduler
from flow_matching import NoiseSchedulerFM
from train_utils import remove_ddp_prefix

def simulate_forward_diffusion(x0, noise_scheduler, mean, std):
    forward_samples_goal = []
    forward_samples_goal.append(x0.cpu().numpy())
    for t in range(len(noise_scheduler)):
        timesteps = torch.tensor(np.repeat(t, len(x0)), dtype=torch.long, device="cuda")
        noise_goal = torch.randn_like(x0, device="cuda")
        noise_goal = noise_goal * std + mean
        sample_goal = noise_scheduler.add_noise(x0, noise_goal, timesteps)
        forward_samples_goal.append(sample_goal.cpu().numpy())
    return forward_samples_goal


def reverse_diffusion(
    sample_x0,
    num_timesteps,
    model,
    noise_scheduler,
    past,
    cam,
    x0_to_world,
    x0_angles_to_world,
    feat_volume,
    voxel_grid,
    drop_cam,
    drop_past,
    track_x0=True,
    goal=None,
    visualizer=None,
):
    """
    sample_x0: 1, nsamp,-1
    past: bs, 1,-1

    return: reverse_samples: num_timesteps+1, bs, nsamp, -1
    """
    timesteps = list(range(num_timesteps))[::-1]
    nsamp = sample_x0.shape[1]
    bs = past.shape[0]

    sample_x0 = sample_x0.repeat(bs, 1, 1).view(bs * nsamp, -1)
    std = model.std
    mean = model.mean

    past = past.repeat(1, nsamp, 1).view(bs * nsamp, -1)
    cam = cam.repeat(1, nsamp, 1).view(bs * nsamp, -1)
    if x0_to_world is not None:
        x0_to_world = x0_to_world.repeat(1, nsamp, 1).view(bs * nsamp, 3)
    if x0_angles_to_world is not None:
        x0_angles_to_world = x0_angles_to_world.repeat(1, nsamp, 1, 1).view(
            bs * nsamp, 3, 3
        )
        x0_to_world = (x0_to_world, x0_angles_to_world)
    if goal is not None:
        goal = goal.repeat(1, nsamp, 1).view(bs * nsamp, -1)

    reverse_samples = [sample_x0]
    reverse_grad = []
    for i, t in enumerate(tqdm(timesteps)):
        t = torch.tensor(np.repeat(t, len(sample_x0)), dtype=torch.long, device="cuda")
        with torch.no_grad():
            # # TODO repaint: https://arxiv.org/pdf/2201.09865.pdf
            # # mask: known region
            # # if model.forecast_size > 1 and t[0] / num_timesteps > 0.2:
            # if model.forecast_size > 1:
            #     x0_known = torch.zeros_like(sample_x0)
            #     mask = torch.zeros_like(sample_x0)

            #     # move xyz to origin
            #     past_repaint = past.reshape(
            #         nsamp, -1, model.state_size * model.kp_size
            #     ).clone()
            #     past_repaint[..., :3] = past_repaint[..., :3] - past_repaint[:, :1, :3]
            #     past_repaint = past_repaint[:, 1:]  # remove identity frame
            #     past_repaint = past_repaint.view(nsamp, -1)

            #     past_size = (model.memory_size - 1) * model.state_size * model.kp_size
            #     x0_known[:, :past_size] = past_repaint
            #     mask[:, :past_size] = 1
            #     noise = torch.randn_like(x0_known, device=x0_known.device)
            #     noise = noise * std
            #     sample_x0_known = noise_scheduler.add_noise(x0_known, noise, t[0])
            #     sample_x0 = mask * sample_x0_known + (1 - mask) * sample_x0

            # if bs == 1 and voxel_grid is not None:
            #     # geometry guidance
            #     world_x0 = (
            #         sample_x0.reshape(nsamp, model.forecast_size, -1)[:, :, :3]
            #         + x0_to_world[:, None]
            #     ).reshape(-1, 3)
            #     # Cost guidance
            #     n_guide_steps = 1  # 10
            #     scale = 0  # 0.5
            #     # find the x0 outside boxes, and resample a point in the box

            #     for idx_guide in range(n_guide_steps):
            #         if visualizer is not None:
            #             handle = visualizer.server.add_point_cloud(
            #                 "/frames/pts",
            #                 world_x0.cpu().numpy(),
            #                 colors=(0, int(255 * idx_guide / n_guide_steps), 0),
            #                 point_size=0.02,
            #             )
            #             time.sleep(0.005)
            #         loss, grad = voxel_grid.compute_penetration_gradients(world_x0)
            #         grad = torch.tensor(grad, device=world_x0.device)
            #         # print(loss.sum())
            #         if t[0] / num_timesteps > 0.1:
            #             world_x0 = world_x0 + scale * grad

            #     sample_x0 = sample_x0.reshape(nsamp, model.forecast_size, -1)
            #     sample_x0[:, :, :3] = (
            #         world_x0.reshape(nsamp, -1, 3) - x0_to_world[:, None]
            #     )
            #     sample_x0 = sample_x0.reshape(nsamp, -1)

            ###### goal
            # 3D convs then query, # B1HWD => B3HWD
            if feat_volume is None:
                feat = []
            else:
                if feat_volume.ndim == 4:
                    feat = voxel_grid.readout_in_world(
                        feat_volume, sample_x0, x0_to_world
                    )
                else:
                    feat = feat_volume.repeat(nsamp, 1)
                feat = [feat]

            if visualizer is None:
                cfg_scale = 1
                drop_feat = False
                drop_past = drop_past
                drop_cam = drop_cam
            else:
                cfg_scale = visualizer.cfg_slider.value
                drop_feat = not visualizer.scene_ckbox.value
                drop_past = not visualizer.past_ckbox.value
                drop_cam = not visualizer.observer_ckbox.value
            grad_cond = model(
                sample_x0,
                t[:, None] / num_timesteps,
                past,
                cam,
                feat,
                drop_cam=drop_cam,
                drop_past=drop_past,
                goal=goal,
                drop_feat=drop_feat,
            )
            if cfg_scale == 1:
                grad = grad_cond
            else:
                grad_uncond = model(
                    sample_x0,
                    t[:, None] / num_timesteps,
                    past,
                    cam,
                    feat,
                    drop_cam=True,
                    drop_past=drop_past,
                    drop_feat=drop_feat,
                    goal=None,
                )

                grad = cfg_scale * (grad_cond - grad_uncond) + grad_uncond

        if track_x0:
            sample_x0_norm = (sample_x0 - mean) / std
            sample_x0_norm = torch.nan_to_num(sample_x0_norm, nan=0.0)  # divide by 0
            sample_x0_norm = noise_scheduler.step(grad, t[0], sample_x0_norm)
            sample_x0 = sample_x0_norm * std + mean
        reverse_samples.append(sample_x0)
        reverse_grad.append(grad * std)
    reverse_samples = torch.stack(reverse_samples, 0)
    reverse_grad = torch.stack(reverse_grad, 0)
    reverse_samples = reverse_samples.view(num_timesteps + 1, bs, nsamp, -1)
    reverse_grad = reverse_grad.view(num_timesteps, bs, nsamp, -1)

    if "handle" in locals():
        handle.remove()
    return reverse_samples, reverse_grad


class EnvEncoder(nn.Module):
    # def __init__(self, in_dim=1, feat_dim=64):
    #     super().__init__()
    #     self.unet_3d = Butterfly3D(in_dim, feat_dim)
    def __init__(self, in_dim=1, feat_dim=384):
        super().__init__()
        self.unet_3d = UNet3D(in_dim, feat_dim)
        self.encoder = Encoder3D(in_dim, feat_dim)
        self.feat_dim = feat_dim

    def extract_features(self, occupancy):
        """
        x_world: N,3
        """
        # 3D convs then query B1HWD => B3HWD
        feature_vol = self.unet_3d(occupancy[None])[0]
        return feature_vol


class TrajDenoiser(nn.Module):
    def __init__(
        self,
        mean=None,
        std=None,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        feat_dim: list = [],
        condition_dim: int = 64,
        N_freq: int = 8,
        memory_size: int = 8,
        forecast_size: int = 1,
        state_size: int = 3,
        camera_state_size: int = 3,
        cond_size: int = 0,
        kp_size: int = 1,
        global_env: bool = True,
    ):
        # store mean and std as buffers
        super().__init__()
        if mean is None:
            mean = torch.zeros(state_size * forecast_size * kp_size)
            print("Warning: mean not provided. Make sure to load from ckpts.")
        if std is None:
            std = torch.ones(state_size * forecast_size * kp_size)
            print("Warning: std not provided. Make sure to load from ckpts.")

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.forecast_size = forecast_size
        self.memory_size = memory_size
        self.state_size = state_size
        self.kp_size = kp_size
        self.condition_dim = condition_dim
        self.global_env = global_env

        # state embedding
        time_embed = PosEmbedding(1, N_freq)
        self.time_embed = nn.Sequential(
            time_embed,
            nn.Linear(time_embed.out_channels, condition_dim),
            nn.Mish(),
            zero_module(nn.Linear(condition_dim, condition_dim)),
        )

        # condition embedding
        past_posec = PosEmbedding(state_size * memory_size * kp_size, N_freq)
        past_mlp = nn.Sequential(
            past_posec,
            BaseMLP(D=5, 
                    W=128, 
                    in_channels=past_posec.out_channels, 
                    out_channels=condition_dim,
                    skips=[1,2,3,4]
                    )
        )
        self.past_mlp = nn.Sequential(
            past_mlp,
            nn.Mish(),
            zero_module(nn.Linear(condition_dim, condition_dim)),
        )
        cam_posec = PosEmbedding(camera_state_size * memory_size, N_freq)
        cam_mlp = nn.Sequential(
            cam_posec,
            BaseMLP(
                D=5,
                W=128,
                in_channels=cam_posec.out_channels, 
                out_channels=condition_dim,
                skips=[1,2,3,4] 
            )
        )
        self.cam_mlp = nn.Sequential(
            cam_mlp,
            nn.Mish(),
            zero_module(nn.Linear(condition_dim, condition_dim)),
        )

        if np.sum(feat_dim) == 0:
            feat_dim = []
        if len(feat_dim) > 0:
            feat_proj = []
            for feat_dim_sub in feat_dim:
                if self.global_env:
                    feat_proj.append(
                        # nn.Linear(feat_dim_sub * forecast_size, condition_dim)
                        nn.Linear(feat_dim_sub, condition_dim)
                    )
                else:
                    feat_proj.append(nn.Linear(feat_dim_sub, condition_dim))
            self.feat_proj = nn.ParameterList(feat_proj)

        # goal cond
        if cond_size > 0:
            self.input_projector = nn.Sequential(
                nn.Linear(state_size * kp_size, state_size * kp_size * 4),
                nn.Mish(),
                nn.Linear(state_size * kp_size * 4, state_size * kp_size * 4),
                nn.Mish(),
                nn.Linear(4 * state_size * kp_size, state_size * kp_size),
            )

        if self.global_env:
            in_channels = condition_dim * (3 + 1 + len(feat_dim))  # + env
            env_emb_dim = 0
        else:
            in_channels = condition_dim * (3 + 1)  # past, cam, time, noise
            # in_channels = condition_dim * (2 + 1)  # past, cam, time, noise
            env_emb_dim = len(feat_dim) * condition_dim

        if forecast_size == 1:
            # if True:
            self.pred_head = TransformerPredictor(
                in_channels,
                hidden_size,
                hidden_layers,
                state_size,
                kp_size,
                condition_dim,
                env_emb_dim=env_emb_dim,
                N_freq=N_freq,
            )
        else:
            self.pred_head = TemporalUnetCond(
                input_dim=state_size * kp_size,
                cond_dim=in_channels - condition_dim,
                env_emb_dim=env_emb_dim,
                dim=hidden_size,
            )

        self.drop_rate = 0.1

    def forward(
        self,
        noisy,
        t,
        past,
        cam,
        feat,
        drop_cam=False,
        drop_past=False,
        drop_feat=False,
        goal=None,
    ):
        """
        noisy: N, K*3
        t: N
        past: N, M*3
        cam: N, M*3
        """
        assert noisy.dim() == 2
        bs = noisy.shape[0]
        device = noisy.device
        noisy = noisy.clone()
        past = past.clone()
        cam = cam.clone()
        if goal is not None:
            goal = goal.clone()

        noisy = (noisy - self.mean) / (self.std)
        noisy = torch.nan_to_num(noisy, nan=0.0)  # divide by 0

        # state embedding
        t_goal_emb = self.time_embed(t)

        # condition embedding
        past_emb = self.past_mlp(past.view(past.shape[0], -1))
        cam_emb = self.cam_mlp(cam.view(cam.shape[0], -1))
        if self.training:
            cam_emb = self.drop_embedding(cam_emb)
            past_emb = self.drop_embedding(past_emb)
        if drop_cam:
            cam_emb = cam_emb * 0
        if drop_past:
            past_emb = past_emb * 0

        # merge the embeddings
        emb = torch.cat((t_goal_emb, past_emb, cam_emb), dim=-1)
        # emb = torch.cat((t_goal_emb, cam_emb), dim=-1)

        if hasattr(self, "feat_proj"):
            feat_emb = []
            for i in range(len(self.feat_proj)):
                if self.global_env:
                    feat_sub = feat[i].reshape(feat[i].shape[0], -1)
                else:
                    feat_sub = feat[i].reshape(feat[i].shape[0], self.forecast_size, -1)
                feat_emb_sub = self.feat_proj[i](feat_sub)
                feat_emb.append(feat_emb_sub)
            feat_emb = torch.cat(feat_emb, dim=-1)  # bs, T, F
            feat_emb = feat_emb.reshape(bs, -1)  # bs, TF

            if self.training:
                feat_emb = self.drop_embedding(feat_emb)

            if drop_feat:
                feat_emb = feat_emb * 0
            if self.global_env:
                emb = torch.cat((emb, feat_emb), dim=-1)
                feat_emb = None
        else:
            feat_emb = torch.zeros(bs, 0, device=device)

        guide = torch.zeros_like(noisy)
        if hasattr(self, "input_projector"):
            if goal is not None:
                # TODO make guide the same dim as noisy
                goal = goal.reshape(bs, -1, self.state_size)
                guide = guide.reshape(bs, self.forecast_size, -1)
                guide[:, -goal.shape[1] :, :3] = goal  # for both goal/waypoint
                guide = self.input_projector(guide)

                # avoid confusion between (0,0,0) and no cond
                mask = torch.zeros_like(guide)
                mask[:, -goal.shape[1] :] = 1
                guide = guide * mask

                guide = guide.reshape(bs, -1)

                if self.training:
                    guide = self.drop_embedding(guide)

        noisy = noisy.reshape(bs, self.forecast_size, -1)
        guide = guide.reshape(bs, self.forecast_size, -1)
        if not self.global_env:
            feat_emb = feat_emb.reshape(bs, self.forecast_size, -1)
        # delta = self.pred_head(noisy, emb, feat_emb, guide=guide, past=past_emb)
        delta = self.pred_head(noisy, emb, feat_emb, guide=guide)
        delta = delta.reshape(delta.shape[0], -1)
        return delta

    def simulate_forward_diffusion(self, x0, noise_scheduler):
        x0 = x0.view(x0.shape[0], -1)
        forward_samples_goal = simulate_forward_diffusion(
            x0, noise_scheduler, self.mean, self.std
        )
        return forward_samples_goal

    def drop_embedding(self, emb, drop_rate=None):
        if drop_rate is None:
            drop_rate = self.drop_rate
        bs = emb.shape[0]
        dev = emb.device
        rand_mask = (torch.rand(bs, 1, device=dev) > drop_rate).float()
        emb = emb * rand_mask
        return emb

    def reverse_diffusion(
        self,
        nsamp,
        num_timesteps,
        noise_scheduler,
        past,
        cam,
        x0_to_world,
        x0_angles_to_world,
        feat_volume,
        voxel_grid,
        drop_cam,
        drop_past,
        goal,
        xyz_grid=None,
        visualizer=None,
    ):
        """
        past: bs, T,K,...
        xyz_grid: N, -1
        """
        bs = past.shape[0]
        noisy = torch.randn(
            1, nsamp, self.forecast_size * self.kp_size * self.state_size, device="cuda"
        )
        noisy = noisy * self.std + self.mean
        past = past.view(bs, 1, -1)
        cam = cam.view(bs, 1, -1)
        if x0_to_world is not None:
            x0_to_world = x0_to_world.view(bs, 1, -1)
        if x0_angles_to_world is not None:
            x0_angles_to_world = x0_angles_to_world.view(bs, 1, 3, 3)
        if goal is not None:
            goal = goal.view(bs, 1, -1)

        reverse_samples, _ = reverse_diffusion(
            noisy,
            num_timesteps,
            self,
            noise_scheduler,
            past,
            cam,
            x0_to_world,
            x0_angles_to_world,
            feat_volume,
            voxel_grid,
            drop_cam,
            drop_past,
            goal=goal,
            visualizer=visualizer,
        )
        if xyz_grid is not None:
            # when computing grad, only use the first sample in the batch
            past = past[:1]
            cam = cam[:1]
            if x0_to_world is not None:
                x0_to_world = x0_to_world[:1]
            if goal is not None:
                goal = goal[:1]
            _, reverse_grad_grid = reverse_diffusion(
                xyz_grid[None],
                num_timesteps,
                self,
                noise_scheduler,
                past,
                cam,
                x0_to_world,
                x0_angles_to_world,
                feat_volume,
                voxel_grid,
                drop_cam,
                drop_past,
                track_x0=False,
                goal=goal,
            )
        else:
            reverse_grad_grid = None
        return reverse_samples, reverse_grad_grid


def get_env_grid(x_dim, y_dim, z_dim, unit=0.1):
    # Generate coordinate ranges for each dimension, centered around zero
    x = torch.linspace(-((x_dim - 1) * unit) / 2, ((x_dim - 1) * unit) / 2, x_dim)
    y = torch.linspace(-((y_dim - 1) * unit) / 2, ((y_dim - 1) * unit) / 2, y_dim)
    z = torch.linspace(-((z_dim - 1) * unit) / 2, ((z_dim - 1) * unit) / 2, z_dim)

    # Create the meshgrid
    x_grid, y_grid, z_grid = torch.meshgrid(x, y, z, indexing="ij")
    env_grid = torch.stack([x_grid, y_grid, z_grid], -1).view(-1, 3)
    return env_grid


class TotalDenoiser(nn.Module):
    def __init__(
        self,
        config,
        data,
        in_dim=1,
        env_feat_dim=384,
        use_env=True,
        regress=False,
        global_env=True,
        model=TrajDenoiser,
    ):
        super().__init__()

        # reshape
        x0 = data.x0
        x0_joints = data.x0_joints
        x0_angles = data.x0_angles
        y = data.past

        x0 = x0.view(x0.shape[0], -1)
        x0_joints = x0_joints.view(x0_joints.shape[0], -1)
        x0_angles = x0_angles.view(x0_angles.shape[0], -1)
        y = y.view(y.shape[0], -1)

        # model setup
        state_size = 3
        forecast_size = int(x0.shape[1] / state_size)
        num_kps = int(x0_joints.shape[1] / state_size / forecast_size)
        memory_size = int(y.shape[1] / state_size)
        print(f"state_size: {state_size}")
        print(f"forecast_size: {forecast_size}")
        print(f"num_kps: {num_kps}")
        print(f"memory_size: {memory_size}")
        mean = x0.mean(0)
        std = torch.nan_to_num(x0.std(0), nan=0.0)  # divide by 0
        std = torch.clamp(std, min=1e-3) * 2
        mean_goal = mean[-state_size:]
        std_goal = std[-state_size:]


        self.use_env = use_env
        if use_env:
            bg_field = BGField(load_logname=config.load_logname, use_default_mesh=True)
            voxel_grid = bg_field.voxel_grid
            # C, H, W, D
            env_input = voxel_grid.bg_feature.cuda()
            # env_input = voxel_grid.data[None]
            # env_input = np.concatenate(
            #     [
            #         env_input,
            #         np.transpose(voxel_grid.root_visitation_edt_gradient, [3, 0, 1, 2]),
            #     ],
            #     0,
            # )
            # env_input = torch.tensor(env_input, dtype=torch.float32, device="cuda")
            # -,3 grid
            x_dim, y_dim, z_dim = 64, 8, 64
            self.unit_len = 0.1
            self.env_dim = (x_dim, y_dim, z_dim)
            env_grid = get_env_grid(x_dim, y_dim, z_dim, unit=self.unit_len)
            self.register_buffer("env_grid", env_grid)

            self.voxel_grid = voxel_grid
            self.bg_field = bg_field
            self.env_input = env_input
            self.env_model = EnvEncoder(in_dim, feat_dim=env_feat_dim)
        else:
            self.voxel_grid = None
            self.bg_field = None
            self.env_model = None

        goal_model = model(
            mean_goal,
            std_goal,
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
            feat_dim=[self.env_model.feat_dim] if self.env_model is not None else [],
            forecast_size=1,
            memory_size=memory_size,
            state_size=state_size,
            condition_dim=config.condition_dim,
            global_env=global_env,
        )
        self.goal_model = goal_model

        if regress:
            self.goal_unc = copy.deepcopy(goal_model)

        self.noise_scheduler = NoiseScheduler(num_timesteps=config.num_timesteps)

        self.state_size = state_size
        self.memory_size = memory_size
        self.forecast_size = forecast_size
        self.num_kps = num_kps

    def extract_feature_grid(self):
        feat_volume = self.env_model.extract_features(self.env_input)
        return feat_volume

    def extract_env_feat(self, x0_to_world, do_augment=False):
        # bs, npts=HWD, F
        if isinstance(x0_to_world, tuple):
            bs = x0_to_world[0].shape[0]
        else:
            bs = x0_to_world.shape[0]
        if self.use_env:
            env_grid = self.env_grid[None].repeat(bs, 1, 1)
            if len(x0_to_world) == 2:
                x0_to_world = (x0_to_world[0][:, None], x0_to_world[1][:, None])
            else:
                x0_to_world = x0_to_world[:, None]
            feat_grid = self.bg_field.voxel_grid.readout_in_world(
                self.env_input, env_grid, x0_to_world
            )
            feat_grid = feat_grid.view((bs,) + self.env_dim + (-1,))

            if do_augment:
                mask_prob = 0.5
                min_ratio = 0.1
                max_ratio = 0.5
                if np.random.rand() < mask_prob:
                    # get a 3D bbox
                    x_size = int(
                        np.random.uniform(min_ratio, max_ratio) * self.env_dim[0]
                    )
                    y_size = int(
                        np.random.uniform(min_ratio, max_ratio) * self.env_dim[1]
                    )
                    z_size = int(
                        np.random.uniform(min_ratio, max_ratio) * self.env_dim[2]
                    )

                    x_start = np.random.randint(0, self.env_dim[0] - x_size)
                    y_start = np.random.randint(0, self.env_dim[1] - y_size)
                    z_start = np.random.randint(0, self.env_dim[2] - z_size)

                    feat_grid[
                        x_start : x_start + x_size,
                        y_start : y_start + y_size,
                        z_start : z_start + z_size,
                    ] = 0

            # pdb.set_trace()
            # from skimage import measure
            # import trimesh

            # verts, faces, _, _ = measure.marching_cubes(
            #     1 - feat_grid[0, ..., 0].cpu().numpy(),
            #     level=0.5,
            #     spacing=(self.unit_len,) * 3,
            # )
            # aabb = self.env_grid[[0, -1]].cpu().numpy()
            # verts = verts + aabb[:1]
            # if isinstance(x0_to_world, tuple):
            #     # ego_to_world @ verts
            #     verts = verts @ x0_to_world[1][0, 0].cpu().numpy().T
            # verts = verts + x0_to_world[0][0, 0].cpu().numpy()
            # mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            # mesh.export("tmp/feat_grid.obj")

            # process: bs, F, H, W, D
            feat_grid = feat_grid.permute(0, 4, 1, 2, 3).contiguous()
            # bs, 384
            feat_point = self.env_model.encoder(feat_grid)
        else:
            dev = self.parameters().__next__().device
            feat_point = torch.zeros(bs, self.env_model.feat_dim, device=dev)
        return feat_point

    def forward_goal(
        self, noisy_goal, x0_to_world, t_frac, past, cam, feat_volume, regress=False
    ):
        # get features
        if self.env_model is not None:
            if feat_volume.ndim == 2:
                feat = feat_volume
            else:
                feat = self.voxel_grid.readout_in_world(
                    feat_volume, noisy_goal, x0_to_world
                )
            feat = [feat]
        else:
            feat = []
        # predict noise
        if self.use_env:
            drop_feat = False
        else:
            drop_feat = True
        goal_delta = self.goal_model(
            noisy_goal, t_frac, past, cam, feat, drop_feat=drop_feat
        )
        if regress:
            goal_unc = self.goal_unc(
                noisy_goal, t_frac, past, cam, feat, drop_feat=drop_feat
            )
            return goal_delta, goal_unc
        else:
            return goal_delta

    def load_ckpts(self, config):
        model_path = "exps/%s-%s/ckpt_%s.pth" % (
            config.load_logname,
            config.logname_gd,
            config.suffix,
        )
        model_states = torch.load(model_path)
        if not isinstance(self, torch.nn.parallel.DistributedDataParallel):
            model_states = remove_ddp_prefix(model_states)
        print(f"Loaded ckpts from {model_path}")
        return model_states

    def compute_goal_loss(
        self,
        noise_scheduler,
        log_dict,
        clean,
        x0_to_world,
        past,
        cam,
        feat_volume,
        regress=False,
    ):
        clean_goal = clean[:, -self.state_size :]
        if regress:
            bs = clean.shape[0]
            t_frac = torch.zeros(bs, 1, device=clean.device)
            goal_pred, goal_logvar = self.forward_goal(
                clean_goal * 0,
                x0_to_world,
                t_frac,
                past,
                cam,
                feat_volume,
                regress=True,
            )
            loss_goal = ((goal_pred - clean_goal)).pow(2)
            loss_goal = (loss_goal / (goal_logvar).exp() + goal_logvar).mean()
        else:
            noise_goal, noisy_goal, t_frac = noise_scheduler.sample_noise(
                clean_goal, std=self.goal_model.std, mean=self.goal_model.mean
            )
            goal_delta = self.forward_goal(
                noisy_goal, x0_to_world, t_frac, past, cam, feat_volume
            )
            loss_goal = ((goal_delta - noise_goal)).pow(2).mean()

        log = log_dict["log"]
        global_step = log_dict["global_step"]
        log.add_scalar("loss_goal", loss_goal, global_step)
        return loss_goal

    def compute_waypoint_loss(
        self,
        noise_scheduler,
        log_dict,
        clean,
        x0_to_world,
        past,
        cam,
        feat_volume,
        regress=False,
    ):
        clean_goal = clean[:, -self.state_size :]
        if regress:
            bs = clean.shape[0]
            t_frac = torch.zeros(bs, 1, device=clean.device)
            wp_pred, wp_logvar = self.forward_path(
                clean * 0,
                x0_to_world,
                t_frac,
                past,
                cam,
                feat_volume,
                clean_goal=clean_goal,
                regress=True,
            )
            loss_wp = ((wp_pred - clean)).pow(2)
            loss_wp = (loss_wp / (wp_logvar).exp() + wp_logvar).mean()
        else:
            noise_wp, noisy_wp, t_frac = noise_scheduler.sample_noise(
                clean, std=self.waypoint_model.std, mean=self.waypoint_model.mean
            )
            wp_delta = self.forward_path(
                noisy_wp,
                x0_to_world,
                t_frac,
                past,
                cam,
                feat_volume,
                clean_goal=clean_goal,
            )
            loss_wp = ((wp_delta - noise_wp)).pow(2).mean()

        log = log_dict["log"]
        global_step = log_dict["global_step"]
        log.add_scalar("loss_wp", loss_wp, global_step)
        return loss_wp

    def compute_fullbody_loss(
        self,
        noise_scheduler,
        log_dict,
        clean,
        x0_angles,
        x0_joints,
        past,
        past_joints,
        past_angles,
        cam,
        regress=False,
    ):
        if regress:
            bs = clean.shape[0]
            t_frac = torch.zeros(bs, 1, device=clean.device)
            (
                joints_pred,
                angles_pred,
                wps_final_pred,
                joints_logvar,
                angles_logvar,
                wps_logvar,
            ) = self.forward_fullbody(
                clean * 0,
                x0_joints * 0,
                x0_angles * 0,
                t_frac,
                past,
                past_joints,
                past_angles,
                cam,
                follow_wp=clean,
                regress=True,
            )
            loss_joints = ((joints_pred - x0_joints)).pow(2)
            loss_angles = ((angles_pred - x0_angles)).pow(2)
            loss_wps_final = ((wps_final_pred - clean)).pow(2)

            loss_joints = (loss_joints / (joints_logvar).exp() + joints_logvar).mean()
            loss_angles = (loss_angles / (angles_logvar).exp() + angles_logvar).mean()
            loss_wps_final = (loss_wps_final / (wps_logvar).exp() + wps_logvar).mean()

        else:
            # fullbody
            bs = x0_angles.shape[0]
            tsize = self.forecast_size
            x0_joints = torch.cat(
                [
                    clean.view(bs, tsize, -1),
                    x0_angles.view(bs, tsize, -1),
                    x0_joints.view(bs, tsize, -1),
                ],
                -1,
            ).view(bs, -1)
            noise_joints, noisy_joints, t_frac = noise_scheduler.sample_noise(
                x0_joints,
                std=self.fullbody_model.std,
                mean=self.fullbody_model.mean,
            )
            noise_joints = noise_joints.view(bs, tsize, -1)
            noisy_joints = noisy_joints.view(bs, tsize, -1)
            noise_wps_final = noise_joints[..., :3].reshape(bs, -1)
            noisy_wps_final = noisy_joints[..., :3].reshape(bs, -1)
            noise_angles = noise_joints[..., 3:12].reshape(bs, -1)
            noisy_angles = noisy_joints[..., 3:12].reshape(bs, -1)
            noise_joints = noise_joints[..., 12:].reshape(bs, -1)
            noisy_joints = noisy_joints[..., 12:].reshape(bs, -1)

            joints_delta, angles_delta, wps_final_delta = self.forward_fullbody(
                noisy_wps_final,
                noisy_joints,
                noisy_angles,
                t_frac,
                past,
                past_joints,
                past_angles,
                cam,
                follow_wp=clean,
            )

            loss_joints = ((joints_delta - noise_joints)).pow(2).mean()
            loss_angles = ((angles_delta - noise_angles)).pow(2).mean()
            loss_wps_final = ((wps_final_delta - noise_wps_final)).pow(2).mean()

        log = log_dict["log"]
        global_step = log_dict["global_step"]
        log.add_scalar("loss_joints", loss_joints, global_step)
        log.add_scalar("loss_angles", loss_angles, global_step)
        log.add_scalar("loss_wps_final", loss_wps_final, global_step)
        return loss_joints, loss_angles, loss_wps_final


class TotalDenoiserThreeStageFull(TotalDenoiser):
    def __init__(
        self,
        config,
        data,
        in_dim=16,
        # in_dim=1,
        env_feat_dim=384,
        use_env=True,
        regress=False,
        global_env=False,
        model=TrajDenoiser,
    ):
        super(TotalDenoiserThreeStageFull, self).__init__(
            config,
            data,
            in_dim,
            env_feat_dim,
            use_env,
            regress,
            global_env,
            model,
        )
        x0 = data.x0
        x0_joints = data.x0_joints
        x0_angles = data.x0_angles
        # reshape
        x0 = x0.view(x0.shape[0], -1)
        x0_joints = x0_joints.view(x0_joints.shape[0], -1)
        x0_angles = x0_angles.view(x0_angles.shape[0], -1)

        mean = x0.mean(0)
        std = torch.nan_to_num(x0.std(0), nan=0.0)  # divide by 0
        std = torch.clamp(std, min=1e-3) * 2 # if the std is too small, that means the task is close to regression
        mean_wp = mean
        std_wp = std
        mean_joints = x0_joints.mean(0)
        std_joints = torch.nan_to_num(x0_joints.std(0), nan=0.0)  # divide by 0
        std_joints = torch.clamp(std_joints, min=1e-3) * 2
        mean_angles = x0_angles.mean(0)
        std_angles = torch.nan_to_num(x0_angles.std(0), nan=0.0)  # divide by 0
        std_angles = torch.clamp(x0_angles.std(0), min=1e-3) * 2

        waypoint_model = model(
            mean_wp,
            std_wp,
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
            feat_dim=[self.env_model.feat_dim] if self.env_model is not None else [],
            forecast_size=self.forecast_size,
            memory_size=self.memory_size,
            state_size=self.state_size,
            cond_size=self.state_size,
            condition_dim=config.condition_dim,
            global_env=global_env,
        )

        mean_joints = torch.cat(
            [
                mean_wp.view(self.forecast_size, -1),
                mean_angles.view(self.forecast_size, -1),
                mean_joints.view(self.forecast_size, -1),
            ],
            -1,
        ).view(-1)
        std_joints = torch.cat(
            [
                std_wp.view(self.forecast_size, -1),
                std_angles.view(self.forecast_size, -1),
                std_joints.view(self.forecast_size, -1),
            ],
            -1,
        ).view(-1)
        fullbody_model = model(
            mean_joints,
            std_joints,
            hidden_size=config.hidden_size,
            hidden_layers=config.hidden_layers,
            # feat_dim=[env_model.feat_dim * forecast_size],
            feat_dim=[],
            forecast_size=self.forecast_size,
            memory_size=self.memory_size,
            state_size=self.state_size,
            cond_size=self.state_size * self.forecast_size,
            condition_dim=config.condition_dim,
            kp_size=self.num_kps + 3 + 1,
            global_env=global_env,
        )

        self.waypoint_model = waypoint_model
        self.fullbody_model = fullbody_model

        if regress:
            self.waypoint_unc = copy.deepcopy(waypoint_model)
            self.fullbody_unc = copy.deepcopy(fullbody_model)
            self.regress = True
        else:
            self.regress = False

        self.global_env = global_env

    def forward(self, batch, log_dict):
        # get data
        bs = batch[0].shape[0]
        clean = batch[0].reshape(bs, -1)
        past = batch[1].reshape(bs, -1)
        cam = batch[2].reshape(bs, -1)
        x0_to_world = batch[3].reshape(bs, -1)
        x0_joints = batch[4].reshape(bs, -1)
        past_joints = batch[5].reshape(bs, -1)
        x0_angles = batch[6].reshape(bs, -1)
        past_angles = batch[7].reshape(bs, -1)
        e2w_rot = batch[8].reshape(-1, 3, 3)

        # combine
        x0_to_world = (x0_to_world, e2w_rot)

        # get context
        if self.use_env:
            if self.global_env:
                feat_volume = self.extract_env_feat(x0_to_world, do_augment=False)
            else:
                feat_volume = self.extract_feature_grid()
            # traj = (clean[0].reshape(-1, 3) @ e2w_rot[0].T) + x0_to_world[0][0]
            # trimesh.Trimesh(traj.cpu()).export("tmp/traj.obj")
        else:
            feat_volume = None

        # goal
        loss_goal = self.compute_goal_loss(
            self.noise_scheduler,
            log_dict,
            clean,
            x0_to_world,
            past,
            cam,
            feat_volume,
            regress=self.regress,
        )

        # idx = 100
        # feature_vol = model.bg_field.voxel_grid.data[None]
        # # x_ego = np.random.rand(10000, 3) * 4 - 2
        # # x_ego = torch.tensor(x_ego, dtype=torch.float32, device="cuda")
        # x_ego = noisy_goal
        # values = model.bg_field.voxel_grid.readout_in_world(
        #     feature_vol,
        #     x_ego,
        #     (x0_to_world[0][idx : idx + 1], x0_to_world[1][idx : idx + 1]),
        # )
        # x_ego = x_ego[values[:, 0] > 0]
        # server = model.bg_field.voxel_grid.run_viser()
        # x_ego = x_ego @ x0_to_world[1][idx].T + x0_to_world[0][idx]
        # path_ego = (
        #     clean[idx].view(-1, 3) @ x0_to_world[1][idx].T + x0_to_world[0][idx]
        # )
        # server.add_point_cloud(
        #     "/frames/pts1",
        #     x_ego.cpu().numpy(),
        #     colors=np.zeros((len(x_ego), 3)),
        # )
        # server.add_point_cloud(
        #     "/frames/pts2",
        #     path_ego.cpu().numpy(),
        #     colors=np.ones((56, 3)) * 0.5,
        # )
        # pdb.set_trace()

        # path
        loss_wp = self.compute_waypoint_loss(
            self.noise_scheduler,
            log_dict,
            clean,
            x0_to_world,
            past,
            cam,
            feat_volume,
            regress=self.regress,
        )

        # fullbody
        loss_joints, loss_angles, loss_wps_final = self.compute_fullbody_loss(
            self.noise_scheduler,
            log_dict,
            clean,
            x0_angles,
            x0_joints,
            past,
            past_joints,
            past_angles,
            cam,
            regress=self.regress,
        )

        # sum up
        loss = loss_goal + loss_wp + loss_joints + loss_angles + loss_wps_final
        return loss
    
    def forward_path(
        self,
        noisy_wp,
        x0_to_world,
        t_frac,
        past,
        cam,
        feat_volume,
        clean_goal=None,
        regress=False,
    ):
        ############ waypoint prediction
        # get features
        if self.env_model is not None:
            if feat_volume.ndim == 2:
                feat = feat_volume
            else:
                feat = self.voxel_grid.readout_in_world(
                    feat_volume, noisy_wp, x0_to_world
                )
            feat = [feat]
        else:
            feat = []
        if self.use_env:
            drop_feat = False
        else:
            drop_feat = True
        wp_delta = self.waypoint_model(
            noisy_wp, t_frac, past, cam, feat, goal=clean_goal, drop_feat=drop_feat
        )
        if regress:
            wp_unc = self.waypoint_unc(
                noisy_wp, t_frac, past, cam, feat, goal=clean_goal, drop_feat=drop_feat
            )
            return wp_delta, wp_unc
        else:
            return wp_delta

    def forward_fullbody(
        self,
        noisy_wps,
        noisy_joints,
        noisy_angles,
        t_frac,
        past_wps,
        past_joints,
        past_angles,
        cam,
        follow_wp=None,
        regress=False,
    ):
        bs = noisy_joints.shape[0]
        noisy_joints = torch.cat(
            [
                noisy_wps.view(bs, self.forecast_size, -1),
                noisy_angles.view(bs, self.forecast_size, -1),
                noisy_joints.view(bs, self.forecast_size, -1),
            ],
            dim=-1,
        ).view(bs, -1)
        past_joints = torch.cat(
            [
                past_wps.view(bs, self.memory_size, -1),
                past_angles.view(bs, self.memory_size, -1),
                past_joints.view(bs, self.memory_size, -1),
            ],
            dim=-1,
        ).view(bs, -1)

        if self.use_env:
            drop_feat = False
        else:
            drop_feat = True

        joints_delta = self.fullbody_model(
            noisy_joints,
            t_frac,
            past_joints,
            cam * 0,
            [],
            goal=follow_wp,
            drop_feat=drop_feat,
        )

        joints_delta = joints_delta.view(bs, self.forecast_size, -1)
        wps_delta = joints_delta[..., :3].reshape(bs, -1)
        angles_delta = joints_delta[..., 3:12].reshape(bs, -1)
        joints_delta = joints_delta[..., 12:].reshape(bs, -1)

        if regress:
            joints_unc = self.fullbody_unc(
                noisy_joints,
                t_frac,
                past_joints,
                cam * 0,
                [],
                goal=follow_wp,
                drop_feat=drop_feat,
            )
            joints_unc = joints_unc.view(bs, self.forecast_size, -1)
            wps_unc = joints_unc[..., :3].reshape(bs, -1)
            angles_unc = joints_unc[..., 3:12].reshape(bs, -1)
            joints_unc = joints_unc[..., 12:].reshape(bs, -1)
            return (
                joints_delta,
                angles_delta,
                wps_delta,
                joints_unc,
                angles_unc,
                wps_unc,
            )
        else:
            return joints_delta, angles_delta, wps_delta


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class UNet3D(nn.Module):
    """
    3d U-Net
    """

    def __init__(self, in_planes, out_planes):
        super(UNet3D, self).__init__()
        self.decoder1 = Conv3dBlock(in_planes, 16, stride=(2, 2, 2))  # 2x
        self.decoder2 = Conv3dBlock(16, 128, stride=(2, 2, 2))  # 4x
        self.decoder3 = Conv3dBlock(128, out_planes, stride=(2, 2, 2))  # 8x
        # self.out = Conv3d(256,512,3, (1,1,1),1,bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.out_channels
                )
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if hasattr(m.bias, "data"):
                    m.bias.data.zero_()

    def forward(self, x):
        shape = x.shape
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = F.interpolate(x, size=shape[2:], mode="trilinear", align_corners=False)
        return x


class Encoder3D(nn.Module):
    """
    3d U-Net
    """

    def __init__(self, in_planes, out_planes):
        super(Encoder3D, self).__init__()
        self.decoder1 = Conv3dBlock(in_planes, 16, stride=(2, 2, 2))  # 2x
        self.decoder2 = Conv3dBlock(16, 128, stride=(2, 2, 2))  # 4x
        self.decoder3 = Conv3dBlock(128, out_planes, stride=(2, 2, 2))  # 8x
        # self.out = Conv3d(128, out_planes, 3, (1, 1, 1), 1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.out_channels
                )
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if hasattr(m.bias, "data"):
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        # x = self.out(x)
        x = x.view(x.size(0), x.size(1), -1).max(-1)[0]
        return x


class Butterfly3D(nn.Module):
    """
    3d U-Net
    """

    def __init__(self, in_planes, out_planes):
        super(Butterfly3D, self).__init__()
        self.encoder1 = Conv3dBlock(in_planes, 16, stride=(2, 2, 2))  # 2x
        self.encoder2 = Conv3dBlock(16, 64, stride=(2, 2, 2))  # 4x
        self.encoder3 = Conv3dBlock(64, 128, stride=(2, 2, 2))  # 8x

        self.decoder1 = Conv3dBlock(128 + 64, 64)  # 2x
        self.decoder2 = Conv3dBlock(64 + 16, 16)  # 4x
        self.decoder3 = Conv3d(16 + in_planes, out_planes, 3, (1, 1, 1), 1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.out_channels
                )
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if hasattr(m.bias, "data"):
                    m.bias.data.zero_()

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        x4 = F.interpolate(x3, size=x2.shape[2:], mode="trilinear", align_corners=True)
        x4 = torch.cat([x4, x2], dim=1)
        x4 = self.decoder1(x4)

        x5 = F.interpolate(x4, size=x1.shape[2:], mode="trilinear", align_corners=True)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = self.decoder2(x5)

        x6 = F.interpolate(x5, size=x.shape[2:], mode="trilinear", align_corners=True)
        x6 = torch.cat([x6, x], dim=1)
        x6 = self.decoder3(x6)
        return x6


class Conv3dBlock(nn.Module):
    """
    3d convolution block as 2 convolutions and a projection
    layer
    """

    def __init__(self, in_planes, out_planes, stride=(1, 1, 1)):
        super(Conv3dBlock, self).__init__()
        if in_planes == out_planes and stride == (1, 1, 1):
            self.downsample = None
        else:
            # self.downsample = projfeat3d(in_planes, out_planes,stride)
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, 1, stride, 0),
                nn.BatchNorm3d(out_planes),
            )
        self.conv1 = Conv3d(in_planes, out_planes, 3, stride, 1)
        self.conv2 = Conv3d(out_planes, out_planes, 3, (1, 1, 1), 1)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        if self.downsample:
            x = self.downsample(x)
        out = F.relu(x + self.conv2(out), inplace=True)
        return out


def Conv3d(in_planes, out_planes, kernel_size, stride, pad, bias=False):
    if bias:
        return nn.Sequential(
            nn.Conv3d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                padding=pad,
                stride=stride,
                bias=bias,
            )
        )
    else:
        return nn.Sequential(
            nn.Conv3d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                padding=pad,
                stride=stride,
                bias=bias,
            ),
            nn.BatchNorm3d(out_planes),
        )