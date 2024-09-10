import os, sys
import pdb
import math
import trimesh

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from einops import rearrange

import numpy as np

sys.path.insert(0, os.getcwd())

from utils import get_lab4d_data, TrajDataset
from config import get_config
from denoiser import TotalDenoiserThreeStageFull
from viserviewer import ViserViewer
from train_utils import get_local_rank
from utils import rotate_data

from lab4d.utils.quat_transform import axis_angle_to_matrix


def check_grad(model, log, global_step, thresh):
    """Check if gradients are above a threshold

    Args:
        thresh (float): Gradient clipping threshold
    """
    # # detect large gradients and reload model
    # params_list = []
    # grad_dict = {}
    # for name, p in model.named_parameters():
    #     if p.requires_grad and p.grad is not None:
    #         params_list.append(p)
    #         grad = p.grad.reshape(-1).norm(2, -1)
    #         log.add_scalar("grad/" + name, grad, global_step)
    #         # if p.grad.isnan().any():
    #         #     p.grad.zero_()

    # # check individual parameters
    # grad_norm = torch.nn.utils.clip_grad_norm_(params_list, thresh)

    # grad_th = 1.0
    # # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_th)
    # # if grad_norm > grad_th:
    # #     print("large grad, do not step")
    # # else:
    # #     optimizer.step()
    # nn.utils.clip_grad_norm_(model.parameters(), grad_th)
    # return grad_dict
    torch.nn.utils.clip_grad_norm_(model.parameters(), thresh)
    return None


def augment_batch(batch):
    x = batch[0]
    y = batch[1]
    cam = batch[2]
    x_to_world = batch[3]
    x_joints = batch[4]
    y_joints = batch[5]
    goal_angles = batch[6]
    past_angles = batch[7]
    e2w_rot = batch[8]

    # augment
    bs = x.shape[0]
    dev = x.device
    rand_roty = torch.rand(bs, 1, device=dev) * 2 * np.pi
    x, y, cam, goal_angles, past_angles, e2w_rot = rotate_data(rand_roty, x, y, cam, goal_angles, past_angles, e2w_rot)

    batch = (
        x,
        y,
        cam,
        x_to_world,
        x_joints,
        y_joints,
        goal_angles,
        past_angles,
        e2w_rot,
    )
    return batch

class DataParallelPassthrough(torch.nn.parallel.DistributedDataParallel):
    """For multi-GPU access, forward attributes to the inner module."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def __delattr__(self, name):
        try:
            return super().__delattr__(name)
        except AttributeError:
            return delattr(self.module, name)


class GDMDMTrainer:
    def __init__(self, config):
        self.config = config
        print(config)

        self.define_dataset()
        self.define_model()
        self.move_to_ddp()
        self.optimizer_init()
        
        if get_local_rank()==0:
            rt_dict = {"model": self.model, "data": self.data, "meta": self.meta}
            self.gui = ViserViewer(config, rt_dict)

    @staticmethod
    def parse_config(config):
        meta = {}
        if config.load_logname == "amass":
            meta["datapath"] = "database/motion/amass.pkl"
            meta["use_env"] = False
            meta["target_datapath"] = meta["datapath"]
            meta["lab4d_dir"] = None
            meta["agent_class"] = "smpl"
        else:
            meta["datapath"] = "database/motion/%s-train-L64-S1.pkl" % config.load_logname
            meta["use_env"] = True
            if config.use_test_data:
                meta["target_datapath"] = (
                    "database/motion/%s-test-L64-S100.pkl" % config.load_logname
                )
            else:
                meta["target_datapath"] = meta["datapath"]
            meta["lab4d_dir"] = "logdir/%s/" % config.load_logname
            meta["agent_class"] = "lab4d"
        meta["logdir"] = "exps/%s-%s/" % (config.load_logname, config.logname_gd)

        # regress or not
        if config.pred_type == "regress":
            meta["regress"] = True
        elif config.pred_type == "diffuse":
            meta["regress"] = False
        else:
            raise ValueError("Unknown pred_type")
        return meta

    def rotate_four_ways(self, meta, config):
        data = get_lab4d_data(meta["datapath"], 
                              use_ego=not config.use_world, 
                              full_len=64, 
                              swap_cam_root=config.swap_cam_root,
                              roty=0,
                              )
        for j in range(3):
            dataj = get_lab4d_data(meta["datapath"], 
                                use_ego=not config.use_world, 
                                full_len=64, 
                                swap_cam_root=config.swap_cam_root,
                                roty=np.pi/2*(1+j),
                                )
            for i in range(len(data)):
                data[i] = torch.cat([data[i], dataj[i]])     
        return data       

    def define_dataset(self):
        config = self.config
        meta = self.parse_config(config)

        # # augment dataset with 4-way rotation
        # data = self.rotate_four_ways(meta, config)

        data = get_lab4d_data(meta["datapath"], 
                              use_ego=not config.use_world, 
                              full_len=64, 
                              swap_cam_root=config.swap_cam_root
                              )
        dataset = TrajDataset(data)
        if config.fill_to_size:
            data_size = config.fill_to_size  # 20k samples
            dataset = torch.utils.data.ConcatDataset([dataset] * (data_size // len(dataset)+1))

        # logging
        outdir = f"exps/{config.load_logname}-{config.logname_gd}"
        log = SummaryWriter(outdir, comment=f"{config.load_logname}-{config.logname_gd}")

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=config.ngpu,
            rank=get_local_rank(),
            shuffle=True,
        )

        loader_train = DataLoader(
            dataset, 
            batch_size=config.train_batch_size, 
            drop_last=True, 
            sampler=sampler, 
        )
        loader_eval = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

        self.meta = meta
        self.log = log
        self.outdir = outdir
        self.loader_train = loader_train
        self.total_steps = config.num_epochs * len(loader_train)
        self.data = data

    @staticmethod
    def get_model(config, data, meta):
        model = TotalDenoiserThreeStageFull(
            config,
            data,
            regress=meta["regress"],
            use_env=meta["use_env"],
        )
        return model

    def define_model(self):
        config = self.config
        model = self.get_model(config, self.data, self.meta)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda()
        self.model = model

    def move_to_ddp(self):
        # move model to ddp
        self.model = DataParallelPassthrough(
            self.model,
            device_ids=[get_local_rank()],
            output_device=get_local_rank(),
            find_unused_parameters=True,
        )

    def optimizer_init(self):
        # optimization setup
        config = self.config
        params = list(self.model.parameters())
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
        )
        
        if get_local_rank() == 0:
            print("Total epochs:", config.num_epochs)
            print("Total steps:", self.total_steps)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            config.learning_rate,
            self.total_steps,
            # pct_start=0.1,
            pct_start=1000 / self.total_steps,
            cycle_momentum=False,
            anneal_strategy="linear",
            # div_factor=25,
            # final_div_factor=1,
        )
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer,
        #     0.01 * config.learning_rate,
        #     config.learning_rate,
        #     step_size_up=1000,
        #     step_size_down=19000,
        #     mode="triangular",
        #     gamma=1.0,
        #     scale_mode="cycle",
        #     cycle_momentum=False,
        # )
        self.optimizer = optimizer
        self.scheduler = scheduler

    def print_sum_params(self):
        """Print the sum of parameters"""
        sum = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                sum += p.abs().sum()
        print(f"{sum:.16f}")

    def train(self):
        config = self.config
        model = self.model
        loader_train = self.loader_train
        log = self.log
        optimizer = self.optimizer
        scheduler = self.scheduler
        outdir = self.outdir

        global_step = 0
        frames = []
        losses = []
        for epoch in range(config.num_epochs):
            model.train()
            if get_local_rank() == 0:
                progress_bar = tqdm(total=len(loader_train))
                progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(loader_train):
                if get_local_rank()==0:
                    model.eval()
                    self.gui.update()
                    model.train()

                if not config.norot_aug:
                    batch = augment_batch(batch)

                # logger
                log_dict = {"log": log, "global_step": global_step}
                loss = model(batch, log_dict)
                loss.mean().backward()

                check_grad(model, log, global_step, thresh=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # self.print_sum_params()

                logs = {"loss": loss.detach().item(), "step": global_step}
                log.add_scalar("loss", loss, global_step)
                losses.append(loss.detach().item())
                if get_local_rank() == 0:
                    progress_bar.update(1)
                    progress_bar.set_postfix(**logs)
                
                global_step += 1
            if get_local_rank() == 0:
                progress_bar.close()

            if epoch % config.save_model_epoch == 0 or epoch == config.num_epochs - 1:
                if get_local_rank() == 0:
                    print("Saving model...")
                    os.makedirs(outdir, exist_ok=True)
                    param_path = f"{outdir}/ckpt_%04d.pth" % epoch
                    latest_path = f"{outdir}/ckpt_latest.pth"
                    torch.save(model.state_dict(), param_path)
                    os.system("cp %s %s" % (param_path, latest_path))

    @staticmethod
    def construct_test_model(config):
        """Load a model at test time
        """
        meta = GDMDMTrainer.parse_config(config)
        data = get_lab4d_data(meta["datapath"], 
                              use_ego=not config.use_world, 
                              full_len=64, 
                              swap_cam_root=False)
        model = GDMDMTrainer.get_model(config, data, meta)
        states = model.load_ckpts(config)
        model.load_state_dict(states, strict=True)
        model = model.cuda()
        model.eval()

        # target data
        data = get_lab4d_data(meta["target_datapath"], 
                              use_ego=not config.use_world, 
                              full_len=64, 
                              swap_cam_root=False)
        rt_dict = {"model": model,
                   "data": data,
                   "meta": meta}

        if config.use_two_agents:
            import copy
            config_obs = copy.deepcopy(config)
            config_obs.logname = config_obs.logname + "-swap"

            # load human agent
            # config_obs_body = copy.deepcopy(config)
            # config_obs_body.logname = "b128-local-dino-drop05"
            # config_obs_body.load_logname = "human-2024-05"
            # meta_obs_body = GDMDMTrainer.parse_config(config_obs_body)
            # data_obs_body = get_lab4d_data(meta_obs_body["datapath"], 
            #                     use_ego=not config.use_world, 
            #                     full_len=64, 
            #                     swap_cam_root=False)

            # observer model
            data_obs = get_lab4d_data(meta["datapath"], 
                        use_ego=not config.use_world, 
                        full_len=64, 
                        swap_cam_root=True)
            # load human agent
            # model_obs = GDMDMTrainer.get_model(config_obs_body, data_obs_body, meta_obs_body)
            model_obs = GDMDMTrainer.get_model(config_obs, data_obs, meta)
            states_obs = model_obs.load_ckpts(config_obs)
            # # load body model from another dir
            # states_obs_body = model_obs.load_ckpts(config_obs_body)
            # for k,v in states_obs.items():
            #     if k.startswith("fullbody_model"):
            #         states_obs[k] = states_obs_body[k]

            model_obs.load_state_dict(states_obs, strict=True)
            model_obs = model_obs.cuda()
            model_obs.eval()

            # target
            data_obs = get_lab4d_data(meta["target_datapath"], 
                                      use_ego=not config.use_world, 
                                      full_len=64, 
                                      swap_cam_root=True)
            
            rt_dict["model_obs"] = model_obs
            rt_dict["data_obs"] = data_obs
        
        return rt_dict

if __name__ == "__main__":
    config = get_config()
    trainer = GDMDMTrainer(config)
    trainer.train()