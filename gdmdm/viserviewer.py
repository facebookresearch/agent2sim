import numpy as np
import torch
import viser
from visualizer import DiffusionVisualizer, run_sim

class ViserViewer:
    def __init__(self, config, rt_dict):
        self.config = config
        self.rt_dict = rt_dict
        num_seq = len(rt_dict["data"].x0)

        # initialize the environment
        self.visualizer = DiffusionVisualizer(
            xzmax=None,
            xzmin=None,
            num_timesteps=None,
            bg_field=rt_dict["model"].bg_field,
            logdir=rt_dict["meta"]["logdir"],
            lab4d_dir=rt_dict["meta"]["lab4d_dir"],
            num_seq=num_seq,
        )
        self.visualizer.run_viser()
        self.visualizer.exit_ckbox.value=True

    @torch.no_grad()
    def update(self):
        if not self.visualizer.exit_ckbox.value:
            run_sim(self.visualizer, self.config, self.rt_dict)