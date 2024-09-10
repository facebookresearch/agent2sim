# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
import os
import sys
import pdb
from absl import app
from datetime import timedelta

import torch
import torch.backends.cudnn as cudnn

from trainer import GDMDMTrainer
from config import get_config
from train_utils import get_local_rank

cudnn.benchmark = True


def train_ddp(Trainer):
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)

    opts = get_config()

    torch.distributed.init_process_group(
        "nccl",
        init_method="env://",
        world_size=opts.ngpu,
        rank=local_rank,
        timeout=timedelta(minutes=20),
    )
    print("world size %d | local rank %d" % (opts.ngpu, local_rank))

    trainer = Trainer(opts)
    trainer.train()

if __name__ == "__main__":
    train_ddp(GDMDMTrainer)
