import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from resources.consts import (
    CHECKPOINT_PATH, 
    TRAINING_STATE_SAVE_PATH_FORMAT,
    VAL_SAMPLE_SAVE_PATH,
    MODEL_SAVE_PATH_FORMAT
)
import utils.training_utils as training_utils
from datasets import get_dataset_distributed
from models import define_model

from collections import defaultdict, OrderedDict
import shutil
import click
import glob
import yaml
import logging
import os
import os.path as osp

class Trainer:
    def __init__(self, world_size, rank, cfg_file):
        with open(cfg_file, "r") as f:
            opt = yaml.safe_load(f)
        self.opt = opt

        if opt["exp_name"] is None:
            self.exp_name = opt["dataset"]["dataset_name"] + "_" + opt["model"]["name"]
        else:
            self.exp_name = opt["exp_name"]

        if self.opt["train"]["load_iter"] == "auto":
            exp_path = osp.join(CHECKPOINT_PATH, self.exp_name)
            training_state_paths = sorted(glob.glob(osp.join(exp_path, "training_states/*.state")))
            if len(training_state_paths) == 0:
                self.load_iter = -1
            else:
                self.load_iter = int(osp.basename(training_state_paths[-1]).replace(".state", ""))
            
        else:
            self.load_iter = self.opt["train"]["load_iter"]

        self.total_iters = opt["train"]["total_iters"]
        self.checkpoint_interval = opt["train"]["checkpoint_interval"]
        self.print_interval = opt["train"]["print_interval"]
        self.val_interval = opt["train"]["val_interval"]
        self.rank = rank

        if rank == 0:
            self.initialize_training_folders(self.load_iter == -1)
        
        self.train_batch_size = opt["datasets"]["train"]["batch_size"]
        self.val_batch_size = opt["datasets"]["val"]["batch_size"]
        self.train_dataloader, self.val_dataloader = get_dataset_distributed(
            world_size, rank, opt
        )

        self.model = define_model(opt["model"])
        self.model = DDP(self.model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True)
        self.bare_model = self.model.module if hasattr(self.model, "module") else self.model

        self.optimizer = training_utils.get_optimizer(self.bare_model.get_net_parameters(), opt["train"]["optimizer"])
        self.warmup = opt["train"]["optimizer"]["warmup"]
        self.l_factor = self.opt["train"]["optimizer"]["l_factor"]
        self.cycle_every_n_epoch = self.opt["train"]["optimizer"]["cycle_every"]
    
        self.resume_training()
    
    def initialize_training_folders(self, from_scratch):
        exp_path = osp.join(CHECKPOINT_PATH, self.exp_name)
        model_folder = osp.join(exp_path, "models")
        training_state_folder = osp.join(exp_path, "training_states")

        if from_scratch:
            if osp.isdir(exp_path):
                timestamp = training_utils.get_timestamp()
                empty = (len(glob.glob(osp.join(training_state_folder, "*.state"))) == 0)
                if not empty:
                    os.rename(exp_path, osp.join(osp.dirname(exp_path), self.exp_name + "_archived_" + timestamp))
                else:
                    shutil.rmtree(exp_path)
        
            os.makedirs(exp_path)
            os.makedirs(model_folder, exist_ok=True)
            os.makedirs(training_state_folder, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=exp_path)
        training_utils.setup_logger("base", exp_path, screen=True, tofile=True)
        self.logger = logging.getLogger("base")

    def training_loop(self):

        timer = training_utils.AvgTimer()
        data_timer = training_utils.AvgTimer()
        
        if self.rank == 0:
            self.logger.info(f"Number of params: {self.bare_model.count_parameters()}")

        batches_per_iter = len(self.train_dataloader) / self.train_batch_size

        # self.validation()
        self.bare_model.train()
        for epoch in range(self.current_iter, self.total_iters):
            epoch_logs = defaultdict(float)

            for batch_idx, datapoint in enumerate(self.train_dataloader):
                self.cyclic_steps += 1

                for i, g in enumerate(self.optimizer.param_groups):
                    g["lr"] = self.l_factor * min(1.0, self.cyclic_steps / self.warmup) / max(self.cyclic_steps, self.warmup)

                # record dataloading time
                data_timer.record()
                self.optimizer.zero_grad()

                timer.start()
                loss, tb_logs = self.bare_model(datapoint)

                loss.backward()
                self.optimizer.step()
                timer.record()
                
                if self.rank == 0:
                    for k, v in tb_logs.items():
                        self.writer.add_scalar(k, v, self.global_step)
                        epoch_logs[f"{k}_epoch"] += v

                if self.rank == 0 and self.global_step % self.print_interval == 0:
                    avg_time = timer.get_avg_time()
                    data_avg_time = data_timer.get_avg_time()

                    assert len(self.optimizer.param_groups) == 1
                    curr_lr = self.optimizer.param_groups[0]["lr"]

                    self.logger.info(
                        f"Epoch {epoch}; Step {self.global_step}; Average training time (Net - Data):"
                        f"{avg_time:.2f} - {data_avg_time:.4f} | LR: {curr_lr:.8f}",
                    )

                self.global_step += 1
                if self.global_step % 10 == 0: break

                data_timer.start()
            
            if self.rank == 0:
                for k, _ in epoch_logs.items():
                    assert "_epoch" in k
                    epoch_logs[k] /= batches_per_iter
                    self.writer.add_scalar(k, epoch_logs[k], epoch)
            
            if self.rank == 0 and epoch > 0 and epoch % self.val_interval == 0:
                self.validation()
                self.bare_model.train()

            if self.rank == 0:
                for i, g in enumerate(self.optimizer.param_groups):
                    self.writer.add_scalar(f'param_group{i}/lr', g["lr"], epoch)

            if epoch % self.cycle_every_n_epoch == 0 and epoch > 0:
                self.cyclic_steps = self.warmup - 1

            if self.rank == 0 and epoch > 0 and epoch % self.checkpoint_interval == 0:
                self.logger.info(f"Saving checkpoints {self.current_iter}")
                self.save_training()

            self.current_iter += 1
        self.average_weights()

    def save_training(self):
        self.logger.info(f"Saving iter {self.current_iter}...")
        
        checkpoint = self.bare_model.get_checkpoint()
        pretrained_paths = {
            x: MODEL_SAVE_PATH_FORMAT.format(self.exp_name, self.current_iter, x) for x in checkpoint.keys()
        }

        state = {
            "pretrained_paths": pretrained_paths,
            "optimizer": self.optimizer.state_dict(),
            "current_iter": self.current_iter,
            "cyclic_steps": self.cyclic_steps,
            "global_step": self.global_step,
        }

        training_state_save_path = TRAINING_STATE_SAVE_PATH_FORMAT.format(self.exp_name, self.current_iter)
        torch.save(state, training_state_save_path)
        for ckpt_name, ckpt_state_dict in checkpoint.items():
            torch.save(ckpt_state_dict, MODEL_SAVE_PATH_FORMAT.format(self.exp_name, self.current_iter, ckpt_name))

    def resume_training(self):
        if self.load_iter == -1:
            pretrained_paths = self.opt["train"]["pretrained_paths"]
            if self.rank == 0:
                self.logger.info("Training model from scratch...")
                self.logger.info(f"Loading pretrained models from {pretrained_paths.items()}")
            self.bare_model.load_network(pretrained_paths)
            self.current_iter = 0
            self.cyclic_steps = 0
            self.global_step = 0
        else:
            if self.rank == 0:
                self.logger.info(f"Resuming training from iter {self.load_iter + 1}")
            
            state_path = TRAINING_STATE_SAVE_PATH_FORMAT.format(self.exp_name, self.load_iter)
            if not osp.isfile(state_path):
                raise ValueError(f"Training state for iter {self.load_iter} not found")
            
            state = torch.load(state_path, map_location="cpu")
            self.cyclic_steps = state["cyclic_steps"] # we increment cyclic_step by 1 when batch begins
            self.current_iter = state["current_iter"] + 1
            self.global_step = state["global_step"] + 1

            pretrained_paths = state["pretrained_paths"]
            for path in pretrained_paths.values():
                if not osp.isfile(path):
                    raise ValueError(f"{path} does not exist")

            resume_optimizer = state["optimizer"]
            self.optimizer.load_state_dict(resume_optimizer)
            
            self.bare_model.load_network(pretrained_paths)

    def average_weights(self):
        
        checkpoint = self.bare_model.get_checkpoint()
        weights_to_average = self.opt["model"]["weights_to_average"]
        
        if len(weights_to_average) == 0:
            return

        first_model_path = MODEL_SAVE_PATH_FORMAT.format(self.exp_name, weights_to_average[0], "net")
        avg_state_dict = torch.load(first_model_path, map_location="cpu") 

        for i in weights_to_average[1:]:
            model_path = MODEL_SAVE_PATH_FORMAT.format(self.exp_name, i, "net")
            curr_weights = torch.load(model_path, map_location="cpu")
            for k, v in curr_weights.items():
                avg_state_dict[k] = torch.cat((avg_state_dict[k], curr_weights[k]), dim=0)

        for k, v in avg_state_dict.items():
            avg_state_dict[k] = torch.mean(v, dim=0, keepdim=True)

        avg_s = "".join([str(i) for i in weights_to_average])
        out_path = MODEL_SAVE_PATH_FORMAT.format(self.exp_name, weights_to_average[-1], "net")
        out_path = os.path.join(*out_path.split(os.path.sep)[:-1], f"avg_{avg_s}_net.pth")

        torch.save(avg_state_dict, out_path)

    def validation(self):
        if self.rank != 0:
            return
        
        if self.val_dataloader is None:
            self.logger.warning("No validation dataloader was given. Skipping validation...")
            return
    
        timer = training_utils.AvgTimer()
        val_save_root = VAL_SAMPLE_SAVE_PATH.format(self.exp_name, self.current_iter)
        os.makedirs(val_save_root, exist_ok=True)
        self.logger.info("Evaluating metrics on validation set")

        timer.start()
        self.bare_model.eval()
        tb_logs = self.bare_model.validate(self.val_dataloader, save_root=val_save_root)

        for k, v in tb_logs.items():
            self.writer.add_scalar(k, v, self.current_iter)
        
        timer.record()
        self.logger.info(f"Evaluation time: {timer.get_avg_time()}")


@click.command()
@click.option("--cfg_file", required=True, type=str, help="Config file path")
def main(cfg_file):
    dist.init_process_group(backend="nccl")

    world_size = torch.cuda.device_count()
    gpu_id = int(os.environ["LOCAL_RANK"])

    print("World_size:", world_size)
    print("Rank", gpu_id)
    print(30 * "-")
    
    torch.cuda.set_device(gpu_id)
    Trainer(world_size, gpu_id, cfg_file).training_loop()


if __name__ == "__main__":
    main()
