import torch
import torch.nn as nn
from collections import OrderedDict

class Seq2SeqModel(nn.Module):
    def __init__(self, opt):
        super(Seq2SeqModel, self).__init__()

        self.device = torch.cuda.current_device()
        self.opt = opt
        self.char_to_idx = None
        self.idx_to_char = None

        self.bos = self.opt["tokenizer"]["bos"]
        self.eos = self.opt["tokenizer"]["eos"]
        self.pad_idx = self.opt["tokenizer"]["pad_idx"]

        self.net = define_arch(opt["net"]).to(self.device)

        self.topks = opt["topks"]
        self.beam_size = opt["beam_size"]
        assert self.topks[-1] <= self.beam_size, "max topk cannot be greater than beam size"

        self.criterion = masked_loss()
        self.metric = masked_acc()

    def get_net_parameters(self):
        return list(self.named_parameters())
    
    def count_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)
    
    def get_checkpoint(self):
        checkpoints = {
            "net": self.net.state_dict()
        }
        return checkpoints
    
    def load_network(self, opt):
        pretrained_path_net = opt["net"]
        if pretrained_path_net is not None:
            state_dict = torch.load(pretrained_path_net, map_location="cpu")

            if self.opt["use_torch_compile"]:
                state_dict_clean = OrderedDict()
                for k, v in state_dict.items():
                    if "_orig_mod." not in k:
                        state_dict_clean["_orig_mod." + k] = v
                state_dict = state_dict_clean

            self.net.load_state_dict(state_dict, strict=True)


    def forward(self, data: dict):
        data = {k: v.to(self.device) for k, v in data.items()}

        logits = self.net(**data)
        total_loss = 0

        tb_logs = {}

        masked_loss = self.criterion(labels, logits)

        total_loss += masked_loss
        tb_logs.update({
            "losses/masked_loss": masked_loss.item()
        })

        return total_loss, tb_logs

