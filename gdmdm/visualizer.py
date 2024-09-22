import cv2
import numpy as np
import sys, os
import trimesh
import math
import pdb
import matplotlib.pyplot as plt
from celluloid import Camera
import torch
from torchcubicspline import natural_cubic_spline_coeffs, NaturalCubicSpline
import viser
import viser.transforms as tf
from matplotlib import cm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
import shutil

sys.path.insert(0, os.getcwd())
from lab4d.utils.vis_utils import get_pts_traj, get_camera_mesh, get_user_mesh
from lab4d.utils.geom_utils import align_vector_a_to_b
from lab4d.utils.quat_transform import axis_angle_to_matrix
from lab4d.utils.pyrender_wrapper import PyRenderWrapper
from lab4d.utils.io import save_vid
from lab4d.config import load_flags_from_file
from projects.csim.behavior_vis.articulation_loader import ArticulationLoader

from agent import Agent


def get_img_from_plt(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
        int(height), int(width), 3
    )
    return image


def spline_interp(sample_wp, forecast_size, num_kps=1, state_size=3, interp_size=56):
    if not torch.is_tensor(sample_wp):
        device = "cpu"
        sample_wp = torch.tensor(sample_wp, device=device)
    else:
        device = sample_wp.device

    t = torch.linspace(0, 1, forecast_size, device=device)
    coeffs = natural_cubic_spline_coeffs(
        t, sample_wp.reshape(-1, forecast_size, state_size * num_kps)
    )
    spline = NaturalCubicSpline(coeffs)
    point = torch.linspace(0, 1, interp_size, device=device)
    sample_wp_dense = spline.evaluate(point).reshape(sample_wp.shape[0], -1)
    return sample_wp_dense


def put_text(img, text, pos, color=(255, 0, 0)):
    img = img.astype(np.uint8)
    img = cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )
    return img


def concatenate_points(pointcloud1, pointcloud2):
    pts = np.concatenate([pointcloud1.vertices, pointcloud2.vertices], axis=0)
    colors = np.concatenate(
        [pointcloud1.visual.vertex_colors, pointcloud2.visual.vertex_colors],
        axis=0,
    )
    return trimesh.PointCloud(pts, colors=colors)


class DiffusionVisualizer:
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
        self.state_size = state_size
        self.xzmax = xzmax
        self.xzmin = xzmin
        self.num_timesteps = num_timesteps
        self.bg_field = bg_field
        self.logdir = logdir
        self.num_seq = num_seq

        if xzmax is not None:
            # mesh rendering
            raw_size = [512, 512]
            renderer = PyRenderWrapper(raw_size)
            renderer.set_camera_bev(depth=(xzmax - xzmin) * 1.5)
            # set camera intrinsics
            fl = max(raw_size)
            intr = np.asarray([fl * 2, fl * 2, raw_size[1] / 2, raw_size[0] / 2])
            renderer.set_intrinsics(intr)
            renderer.align_light_to_camera()
            self.renderer = renderer
        else:
            self.renderer = None

        # to store camera position
        self.userwp_list = []
        self.goal_list = []

        # TODO joint parsing
        if lab4d_dir is not None:
            opts = load_flags_from_file("%s/opts.log" % lab4d_dir)
            opts["load_suffix"] = "latest"
            opts["logroot"] = "logdir"
            opts["inst_id"] = 1
            opts["grid_size"] = 128
            opts["level"] = 0
            opts["vis_thresh"] = -20
            opts["extend_aabb"] = True
            self.articulation_loader = ArticulationLoader(opts)
        else:
            self.articulation_loader = None

    def delete(self):
        if self.renderer is not None:
            self.renderer.delete() 

    # rotate the sample
    def render_rotate_sample(self, shape, n_frames=60, pts_traj=None, pts_color=None):
        sample_raw = shape.vertices
        frames = []
        for i in range(n_frames):
            progress = i / n_frames
            progress_1 = np.sin(progress * 4 * np.pi)
            progress_2 = np.sin((0.125 + progress) * 4* np.pi)
            
            # rotate along xz axes
            rot = cv2.Rodrigues(np.array([progress_1 * 0.1 * np.pi, 0, progress_2 * 0.1 * np.pi]))[0]
            sample = np.dot(sample_raw, rot.T)
            shape = trimesh.PointCloud(sample, colors=shape.visual.vertex_colors)
            input_dict = {"shape": shape}
            if pts_traj is not None and pts_color is not None:
                input_dict["pts_traj"] = (pts_traj.reshape(-1, 3) @ rot.T).reshape(
                    pts_traj.shape
                )
                input_dict["pts_color"] = pts_color
            color = self.renderer.render(input_dict)[0]
            frames.append(color)
        return frames

    def render_trajectory(
        self,
        forward_samples,
        reverse_samples,
        past,
        x0_to_world,
        rotate=True,
        prefix="base",
    ):
        """
        forward_samples: list of torch.Tensor

        reverse_samples: M, bs, T,K,3
        x0_to_world: 1,1,3
        past: T,1,3
        """
        num_timesteps = self.num_timesteps
        frames = []
        if len(forward_samples) > 0:
            # forward
            num_wps = int(len(forward_samples[0][0]) / self.state_size)
            colors = np.ones((len(forward_samples[0]), num_wps, 4))  # N, K, 4
            colors[..., 1:3] = 0
            if num_wps > 1:
                colors = colors * np.linspace(0, 1, num_wps)[None, :, None]
            colors = colors.reshape(-1, 4)

            shape_clean = trimesh.PointCloud(
                forward_samples[0].reshape(-1, 3), colors=colors
            )
            if rotate:
                frames += self.render_rotate_sample(shape_clean)

            for i, sample in enumerate(forward_samples):
                shape = trimesh.PointCloud(sample.reshape(-1, 3), colors=colors)
                input_dict = {"shape": shape}
                color = self.renderer.render(input_dict)[0]
                color = put_text(color, f"step {i: 4} / {num_timesteps}", (10, 30))
                color = put_text(color, "Forward process", (10, 60))
                frames.append(color)
        else:
            shape_clean = None

        # reverse
        if torch.is_tensor(reverse_samples):
            reverse_samples = reverse_samples.cpu().numpy()
        num_wps = reverse_samples.shape[2]
        num_kps = reverse_samples.shape[3]
        colors = np.ones((len(reverse_samples[0]), num_wps, num_kps, 4))
        colors[..., 1:3] = 0
        if num_wps > 1:
            colors = colors * np.linspace(0, 1, num_wps)[None, :, None, None]
        colors = colors.reshape(-1, 4)

        # get past location as a sphere
        past = past.reshape(-1, 3)
        if torch.is_tensor(past):
            past = past.cpu().numpy()
        past_shape = []
        for past_idx in range(len(past)):
            shape_sub = trimesh.creation.uv_sphere(radius=0.01, count=[4, 4])
            shape_sub = shape_sub.apply_translation(past[past_idx])
            past_shape.append(shape_sub)
        past_shape = trimesh.util.concatenate(past_shape)
        past_colors = np.array([[0, 1, 0, 1]]).repeat(len(past_shape.vertices), axis=0)

        # get bg mesh
        if self.bg_field is not None:
            bg_pts = self.bg_field.bg_mesh.vertices - x0_to_world[:, 0].cpu().numpy()
            bg_colors = np.ones((len(bg_pts), 4)) * 0.6
        else:
            bg_pts = np.zeros((0, 3))
            bg_colors = np.zeros((0, 4))

        for i, sample in enumerate(reverse_samples):
            sample_vis = np.concatenate(
                [sample.reshape(-1, 3), past_shape.vertices, bg_pts], axis=0
            )
            sample_colors = np.concatenate([colors, past_colors, bg_colors], axis=0)

            shape = trimesh.PointCloud(sample_vis, colors=sample_colors)
            if i == 0:
                shape_noise = trimesh.PointCloud(sample.reshape(-1, 3), colors=colors)
            pts_traj, pts_color = get_pts_traj(
                sample.reshape(num_wps, num_kps, 3),
                num_wps - 1,
                traj_len=num_wps,
            )
            input_dict = {"shape": shape, "pts_traj": pts_traj, "pts_color": pts_color}
            color = self.renderer.render(input_dict)[0]
            color = put_text(
                color, f"step {i: 4} / {num_timesteps}", (10, 30), color=(0, 0, 255)
            )
            color = put_text(color, "Reverse process", (10, 60), color=(0, 0, 255))
            frames.append(color)

        # concat two shapes
        if rotate:
            # shape = concatenate_points(shape_clean, shape)
            frames += self.render_rotate_sample(
                shape, pts_traj=pts_traj, pts_color=pts_color
            )

        filename = "%s/%s-rendering" % (self.logdir, prefix)
        shape = trimesh.PointCloud(sample.reshape(-1, 3), colors=colors)
        shape.vertices = shape.vertices + x0_to_world[:, 0].cpu().numpy()
        shape.export("%s.obj" % filename)
        if shape_clean is not None:
            shape_clean.export("%s-clean.obj" % filename)
        shape_noise.export("%s-noise.obj" % filename)
        save_vid(filename, frames)
        print("Animation saved to %s.mp4" % filename)
        print("Mesh saved to %s.obj" % filename)

    def plot_trajectory_2d(
        self,
        forward_samples,
        reverse_samples,
        reverse_grad,
        gt_goal,
        xyz,
        yshape,
        past,
        cam,
        prefix="base",
    ):
        """
        reverse_samples: M, bs, T,K,3
        reverse_grad: M, bs, TK3
        gt_goal: T,K,3
        xyz: N,3
        past: T,K,3
        cam: T,1,3
        """
        num_wps = reverse_samples.shape[3]
        num_timesteps = self.num_timesteps
        xzmin = self.xzmin
        xzmax = self.xzmax
        past = past.reshape(past.shape[0], -1, 3)
        cam = cam.reshape(cam.shape[0], -1, 3)

        if torch.is_tensor(gt_goal):
            gt_goal = gt_goal.cpu().numpy()
        if torch.is_tensor(past):
            past = past.cpu().numpy()
        if torch.is_tensor(cam):
            cam = cam.cpu().numpy()

        # 2D plot
        frames = []
        # forward
        for i, sample in enumerate(forward_samples):
            fig, ax = plt.subplots()
            sample = sample.reshape(-1, num_wps, 3)
            plt.scatter(
                sample[:, -1, 0],
                sample[:, -1, 2],
                alpha=0.5,
                s=15,
                color="blue",
            )
            plt.scatter(
                sample[0, -1, 0],
                sample[0, -1, 2],
                alpha=0.5,
                s=30,
                color="red",
            )
            ax.text(0.0, 0.95, f"step {i: 4} / {num_timesteps}", transform=ax.transAxes)
            ax.text(0.0, 1.01, "Forward process", transform=ax.transAxes, size=15)
            plt.xlim(xzmin * 0.5, xzmax * 0.5)
            plt.ylim(xzmin * 0.5, xzmax * 0.5)
            plt.axis("off")
            image = get_img_from_plt(fig)
            frames.append(image)
            plt.close(fig)

        # reverse
        if torch.is_tensor(reverse_samples):
            reverse_samples = reverse_samples.cpu().numpy()
        if torch.is_tensor(reverse_grad):
            reverse_grad = reverse_grad.cpu().numpy()
        num_wps = reverse_samples.shape[2]
        num_kps = reverse_samples.shape[3]
        for i, sample in enumerate(reverse_samples):
            fig, ax = plt.subplots()
            sample = sample.reshape(-1, num_wps, num_kps, 3)  # bs,t,k,3
            # draw all the waypoints
            for j in range(num_wps):
                alpha = np.clip(1 - j / num_wps, 0.1, 1)
                s = np.clip(30 + 100 * j / num_wps, 30, 100)
                if num_kps > 1:
                    s = s * 0.1
                plt.scatter(
                    sample[:, j, :, 0].flatten(),
                    sample[:, j, :, 2].flatten(),
                    alpha=alpha,
                    s=s,
                    color="red",
                )
            # draw line passing through the waypoints
            sample = np.transpose(sample.reshape(-1, num_wps, num_kps, 3), [1,0,2,3]).reshape(num_wps,-1,3)  # bs,t,k,3
            plt.plot(sample[:, :, 0], sample[:, :, 2], color="red", alpha=0.1) # T, bsK, 3
            # past
            s = 100
            if num_kps > 1:
                s = s * 0.1
            plt.scatter(
                past[..., 0].flatten(),
                past[..., 2].flatten(),
                s=s,
                color="green",
                marker="x",
            )
            # cam
            plt.scatter(cam[:, 0, 0], cam[:, 0, 2], s=100, color="blue", marker="x")
            # goal
            plt.scatter(
                gt_goal[..., 0].flatten(),
                gt_goal[..., 2].flatten(),
                s=20,
                color="black",
                marker="o",
                alpha=0.1,
            )
            if i < len(reverse_samples) - 1 and reverse_grad is not None:
                grad = reverse_grad[i]
                # aver over height
                grad = grad.reshape((yshape, -1, num_wps * num_kps, 3)).mean(0)
                scale = np.linalg.norm(grad, 2, -1).mean() * 10
                xyz_sliced = xyz.reshape((yshape, -1, 3))[0]
                plt.quiver(
                    xyz_sliced[:, 0],
                    xyz_sliced[:, 2],
                    -grad[:, -1, 0],
                    -grad[:, -1, 2],
                    angles="xy",
                    scale_units="xy",
                    scale=scale,  # inverse scaling
                    color=(0.5, 0.5, 0.5),
                )
            ax.text(0.0, 0.95, f"step {i: 4} / {num_timesteps}", transform=ax.transAxes)
            ax.text(0.0, 1.01, "Reverse process", transform=ax.transAxes, size=15)
            plt.xlim(xzmin * 0.5, xzmax * 0.5)
            plt.ylim(xzmin * 0.5, xzmax * 0.5)
            plt.axis("off")

            image = get_img_from_plt(fig)
            frames.append(image)
            plt.close(fig)

        filename = "%s/%s-animation" % (self.logdir, prefix)
        print("Animation saved to %s.mp4" % filename)
        save_vid(filename, frames)

    def initialize_server(self):
        # visualizations
        server = viser.ViserServer(port=8081)
        # Setup root frame
        self.base_handle = server.add_frame(
            "/frames",
            wxyz=tf.SO3.exp(np.array([-np.pi / 2, 0.0, 0.0])).wxyz,
            position=(0, 0, 0),
            show_axes=False,
        )

        if self.bg_field is not None:
            server.add_mesh_trimesh(
                name="/frames/environment",
                mesh=self.bg_field.bg_mesh,
            )

            self.root_visitation_boxes = self.bg_field.voxel_grid.to_boxes(
                mode="root_visitation"
            )
            self.cam_visitation = self.bg_field.voxel_grid.to_boxes(
                mode="cam_visitation"
            )
            # server.add_mesh_simple(
            #     name="/frames/root_visitation",
            #     vertices=self.root_visitation_boxes.vertices,
            #     vertex_colors=self.root_visitation_boxes.visual.vertex_colors[:, :3],
            #     faces=self.root_visitation_boxes.faces,
            #     color=None,
            #     opacity=0.5,
            # )
            # server.add_mesh_trimesh(
            #     name="/frames/root_visitation",
            #     mesh=self.root_visitation_boxes,
            # )
            # server.add_mesh_trimesh(
            #     name="/frames/cam_visitation",
            #     mesh=self.cam_visitation,
            # )
        self.server = server

    def run_viser(self):
        self.initialize_server()
        self.server.request_share_url()

        with self.server.add_gui_folder("Sim control"):
            self.pause_ckbox = self.server.add_gui_checkbox("Pause sim", initial_value=False)
            self.exit_ckbox = self.server.add_gui_checkbox("Exit sim", initial_value=False)
            self.sequence_slider = self.server.add_gui_slider(
                "Sequence", min=0, max=self.num_seq - 1, step=1, initial_value=0
            )
            self.autoreset_ckbox = self.server.add_gui_checkbox("Reset on", initial_value=True)
            self.add_autoobs_handle = self.server.add_gui_checkbox("Observer on", initial_value=True)

        with self.server.add_gui_folder("Agent setting"):
            self.cfg_slider = self.server.add_gui_slider(
                "Engagement (cfg scale)", min=0, max=1, step=0.05, initial_value=1
            )
            self.scene_ckbox = self.server.add_gui_checkbox(
                "Scene Awareness", initial_value=True
            )
            self.past_ckbox = self.server.add_gui_checkbox("Past Awareness", initial_value=True)
            self.observer_ckbox = self.server.add_gui_checkbox(
                "Observer Awareness", initial_value=True
            )

        with self.server.add_gui_folder("Agent control"):
            self.add_userwp_handle = self.server.add_gui_checkbox("Add observer", initial_value=False)
            self.add_goal_handle = self.server.add_gui_checkbox("Add goal", initial_value=False)
            

        @self.add_userwp_handle.on_update
        def _(_):
            if self.add_userwp_handle.value:
                self.server.on_scene_pointer(event_type="click")(userwp_click_callback)
            else:
                self.server.remove_scene_pointer_callback()

        @self.add_goal_handle.on_update
        def _(_):
            if self.add_goal_handle.value:
                self.server.on_scene_pointer(event_type="click")(goal_click_callback)
            else:
                self.server.remove_scene_pointer_callback()
        
        @self.sequence_slider.on_update
        def _(_):
            self.need_reset = True

        self.need_reset = False

        def get_intersection(message, mesh):
            # Check for intersection with the mesh, using trimesh's ray-mesh intersection.
            # Note that mesh is in the mesh frame, so we need to transform the ray.
            R_world_mesh = tf.SO3(self.base_handle.wxyz)
            R_mesh_world = R_world_mesh.inverse()
            origin = (R_mesh_world @ np.array(message.ray_origin)).reshape(1, 3)
            direction = (R_mesh_world @ np.array(message.ray_direction)).reshape(1, 3)
            # mesh = self.bg_field.bg_mesh
            intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
            hit_pos, _, _ = intersector.intersects_location(origin, direction)

            if len(hit_pos) == 0:
                return []

            # Get the first hit position (based on distance from the ray origin).
            hit_pos = min(hit_pos, key=lambda x: np.linalg.norm(x - origin))
            # print(f"Hit position: {hit_pos}")
            return hit_pos

        # @server.on_scene_click
        def userwp_click_callback(message: viser.ScenePointerEvent) -> None:
            hit_pos = get_intersection(message, self.cam_visitation)
            if len(hit_pos) == 0:
                return
            # always maintain 2 hit points
            if len(self.userwp_list) == 2:
                # self.userwp_list.pop(0)
                self.userwp_list[1] = hit_pos
            else:
                self.userwp_list.append(hit_pos)
            self.show_control_points()

        def goal_click_callback(message: viser.ScenePointerEvent) -> None:
            hit_pos = get_intersection(message, self.root_visitation_boxes)
            if len(hit_pos) == 0:
                return
            # always maintain 1 hit points
            hit_pos[1] = hit_pos[1] + self.bg_field.voxel_grid.res * 0.5
            if len(self.goal_list) == 1:
                self.goal_list.pop(0)
            self.goal_list.append(hit_pos)
            self.show_control_points(mode="goal")

    def show_control_points(self, mode="userwp"):
        if mode == "userwp":
            pts = self.userwp_list
            pts = np.array(pts)
            mesh_raw = get_user_mesh(0.2, [255, 0, 0, 255])

            # compute direction vector
            direction = pts[1] - pts[0]
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            # align mesh such that negative z direction to the new direction
            align_mat = np.eye(4)
            if not (direction == 0).all():
                align_mat[:3, :3] = align_vector_a_to_b(np.array([0, 0, -1]), direction)
            mesh_raw = mesh_raw.apply_transform(align_mat)

            # compute intermediate points
            num_meshes = 5
            meshes = []
            for it in range(0, num_meshes):
                mesh = mesh_raw.copy()
                mesh = mesh.apply_translation(
                    pts[0] + (pts[1] - pts[0]) * it / (num_meshes - 1)
                )
                meshes.append(mesh)

            self.userwp_handles = []
            for it,mesh in enumerate(meshes):
                handle = self.server.add_mesh_simple(
                    f"/frames/control/{mode}/{it}",
                    vertices=mesh.vertices,
                    vertex_colors=mesh.visual.vertex_colors[:, :3],
                    faces=mesh.faces,
                    color=None,
                    opacity=(it + 1) / (num_meshes + 1),
                )
                self.userwp_handles.append(handle)
        elif mode == "goal":
            mesh = trimesh.creation.uv_sphere(radius=0.05)
            mesh.visual.vertex_colors = np.array([[0, 0, 1.0, 1.0]]).repeat(
                len(mesh.vertices), axis=0
            )
            mesh = mesh.apply_translation(self.goal_list[0])
            self.server.add_mesh_trimesh(f"/frames/goal", mesh)
        else:
            raise ValueError("Unknown mode")

    def render_roottraj_viser(self, wp, prefix, cm_name="cool", point_size=0.02):
        """
        future: bsxTx3
        """
        if torch.is_tensor(wp):
            wp = wp.cpu().numpy()

        # add future
        cmap = cm.get_cmap(cm_name)
        bs = len(wp)
        wp = wp.reshape(bs, -1, 3)
        colors = cmap(np.linspace(0, 1, wp.shape[1]))[:, :3]
        for idx in range(bs):
            self.server.add_point_cloud(
                f"/frames/{prefix}/{idx}",
                wp[idx],
                point_size=point_size,
                colors=colors,
            )

    def render_goal_viser(self, goal, color=[1.0, 0.0, 0.0]):
        """
        goal: bsx3
        """
        if torch.is_tensor(goal):
            goal = goal.cpu().numpy()

        goal_meshes = []
        for i in range(len(goal)):
            goal_mesh = trimesh.creation.uv_sphere(radius=0.05)
            goal_mesh = goal_mesh.apply_translation(goal[i])
            goal_mesh.visual.vertex_colors = np.array([color + [0.5]]).repeat(
                len(goal_mesh.vertices), axis=0
            )
            goal_meshes.append(goal_mesh)
        goal_meshes = trimesh.util.concatenate(goal_meshes)
        self.server.add_mesh_trimesh(f"/frames/goal", goal_meshes)

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
                mesh = self.articulation_loader.mesh_dict[t]
                self.server.add_mesh_trimesh(f"/frames/mesh_agent_%d"%agent_idx, mesh)
                # if t % 10 == 0:
                #     self.server.add_mesh_simple(
                #         f"/frames/mesh_%d" % t,
                #         vertices=mesh.vertices,
                #         vertex_colors=mesh.visual.vertex_colors[:, :3],
                #         faces=mesh.faces,
                #         color=None,
                #         opacity=(t + 5) / (nframes + 5),
                #     )
                time.sleep(0.05)
        else:
            # render points
            pts = self.articulation_loader.load_files_simple(
                so3_wp_angles.reshape(nframes, -1)
            )
            self.render_roottraj_viser(
                np.transpose(pts, [1, 0, 2]),
                prefix="joints",
                cm_name="jet",
                point_size=0.01,
            )


    def visualze_smpl(self, x0_joints_sub, x0_angles_sub, x0_sub, x0_to_world_sub):
        if not hasattr(self, "smpl_model"):
            from smpl import SMPL

            self.smpl_model = SMPL().eval().to("cuda")
        smpl_model = self.smpl_model
        output = smpl_model(
            body_pose=axis_angle_to_matrix(x0_joints_sub), global_orient=x0_angles_sub
        )

        vertices = output["vertices"] + x0_to_world_sub + x0_sub
        vertices = vertices.cpu().numpy()
        for i in range(0, x0_joints_sub.shape[0]):
            mesh = trimesh.Trimesh(vertices=vertices[i], faces=smpl_model.faces)
            self.server.add_mesh_trimesh(f"/frames/smpl", mesh)
            time.sleep(0.05)

        self.render_roottraj_viser(
            output["smpl"].view(x0_sub.shape[0], -1, 3) + x0_to_world_sub + x0_sub,
            prefix="smpl",
            cm_name="viridis",
        )

# independent functions
def run_sim(visualizer, config, rt_dict, save_to_file=False):
    data = rt_dict["data"]
    model = rt_dict["model"]
    meta = rt_dict["meta"]

    # agent
    agent = Agent(model, data, config, visualizer, meta, agent_class=meta["agent_class"])

    if config.use_two_agents:
        data_obs = rt_dict["data_obs"]
        model_obs = rt_dict["model_obs"]
        agent_obs = Agent(model_obs, data_obs, config, visualizer, meta, agent_type="root", agent_idx=1, agent_class="smpl")

    if save_to_file:
        agent.out_path = "exps/%s-%s/" % (config.load_logname, config.logname_gd)
        shutil.rmtree("%s/sample/" % agent.out_path, ignore_errors=True)

    if not agent.model.global_env:
        agent.extract_global_feature()
        if config.use_two_agents:
            agent_obs.extract_global_feature()

    seg_idx = 0
    while not visualizer.exit_ckbox.value:
        print("Round %d" % seg_idx)
        # reset
        if visualizer.autoreset_ckbox.value or visualizer.need_reset:
            agent.reset(sample_idx=visualizer.sequence_slider.value) # 1000
            if config.use_two_agents:
                agent_obs.reset(sample_idx=visualizer.sequence_slider.value)
            visualizer.need_reset = False

        # render past trajectory
        visualizer.render_roottraj_viser(
            agent.accumulated_traj.view(1, -1, 3) + agent.data.x0_to_world,
            prefix="past",
            cm_name="gray_r",
        )
        if config.use_two_agents:
            visualizer.render_roottraj_viser(
                agent_obs.accumulated_traj.view(1, -1, 3) + agent_obs.data.x0_to_world,
                prefix="past_obs",
                cm_name="gray_r",
            )

        # pause
        if visualizer.pause_ckbox.value:
            i=0
            i_max = len(visualizer.userwp_handles)
            while visualizer.pause_ckbox.value:
                # play animation 
                time.sleep(0.3)
                for j in range(i_max):
                    if j==i:
                        visualizer.userwp_handles[j].visible = True
                    else:
                        visualizer.userwp_handles[j].visible = False
                i=(i+1)%(i_max)

        # extract feature
        if agent.model.global_env:
            agent.extract_local_feature()
            if config.use_two_agents:
                agent_obs.extract_local_feature()

        # get observer motion before this cycle
        if len(visualizer.userwp_list) > 1:
            # interpret user-specified hit points as camera path
            # interpolate t_-1 and t_-2
            # TODO: remove this approxmation
            cam = torch.tensor(
                visualizer.userwp_list, device="cuda", dtype=torch.float32
            )
            cam = spline_interp(cam.view(1, -1), 2, interp_size=model.memory_size)
            agent.data.cam = cam.view(-1, 1, 3) - agent.data.x0_to_world

        # update observer
        if config.use_two_agents:
            agent_obs.data.past = agent.data.cam + agent.data.x0_to_world - agent_obs.data.x0_to_world

        
        agent.update_goal(visualizer.goal_list)
        visualizer.goal_list = []
        agent.update_waypoint()
        agent.update_fullbody()

        if config.use_two_agents and visualizer.add_autoobs_handle.value:
            agent_obs.update_goal([])
            agent_obs.update_waypoint()
            # add full body motion: not working due to mismatch between observer vs human agent
            # agent_obs.data.past_joints = torch.zeros(agent_obs.model.fullbody_model.memory_size,
            #                                          agent_obs.model.fullbody_model.kp_size-4,
            #                                          3, device=agent_obs.model.parameters().__next__().device)
            # agent_obs.update_fullbody()
            # update influence obs => agent
            cam = agent_obs.reverse_wp_guide[-1].view(agent_obs.nsamp, -1, 3) + agent_obs.data.x0_to_world
            cam = cam[:1,-agent_obs.model.memory_size:]
            visualizer.userwp_list = torch.stack([cam[0,0], cam[0,-1]],0).cpu().numpy().tolist()
            visualizer.show_control_points()
            # update influence agent => obs
            cam_obs = agent.reverse_wp_guide[-1].view(agent.nsamp, -1, 3) + agent.data.x0_to_world
            cam_obs = cam_obs[:1,-agent.model.memory_size:]
            agent_obs.data.cam = cam_obs - agent_obs.data.x0_to_world

        # save
        if save_to_file:
            agent.save_to_file(agent.out_path, seg_idx)
        
        agent.update()
        if config.use_two_agents:
            agent_obs.update()

        seg_idx += 1
    return agent

def run_eval(visualizer, config, rt_dict):
    data = rt_dict["data"]
    model = rt_dict["model"]
    meta = rt_dict["meta"]

    # agent
    agent = Agent(model, data, config, visualizer, meta, agent_class=meta["agent_class"], record_grad=True)

    # extract feature
    if agent.model.global_env:
        agent.extract_local_feature()
    else:
        agent.extract_global_feature()

    agent.update_goal([], replace_goal=False)
    agent.update_waypoint(replace_wp=False)
    agent.update_fullbody()

    # save to file
    agent.out_path = "exps/%s-%s/" % (config.load_logname, config.logname_gd)
    shutil.rmtree("%s/sample/" % agent.out_path, ignore_errors=True)
    for idx, sample_idx in enumerate(agent.sample_idx):
        agent.save_to_file(agent.out_path, sample_idx, batch_idx=idx)
    return agent