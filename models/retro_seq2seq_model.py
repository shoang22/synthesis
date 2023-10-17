import torch
import torch.nn as nn
import os
from collections import OrderedDict
from models.losses import masked_loss
from metrics import masked_acc
from models.archs import define_arch
from generation.utils import gen_beam, tokenizer_from_vocab

class RetroSeq2SeqModel(nn.Module):
    def __init__(self, opt):
        super(RetroSeq2SeqModel, self).__init__()

        self.device = torch.cuda.current_device()
        self.src_char_to_idx = tokenizer_from_vocab(opt["tokenizer"]["src_vocab"])
        self.src_idx_to_char = {idx: char for char, idx in self.src_char_to_idx.items()}
        self.tgt_char_to_idx = tokenizer_from_vocab(opt["tokenizer"]["src_vocab"])
        self.tgt_idx_to_char = {idx: char for char, idx in self.tgt_char_to_idx.items()}
        self.maxlen = opt["max_decoding_len"]

        opt["net"]["opt"]["src_vocab_size"] = len(self.src_char_to_idx)
        opt["net"]["opt"]["tgt_vocab_size"] = len(self.tgt_char_to_idx)

        self.net = define_arch(opt["net"]).to(self.device)
        self.beam_size = opt["beam_size"]
        self.T = opt["temperature"]
        
        self.opt = opt

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

        m_loss = masked_loss(data["gt"], logits)
        m_acc = masked_acc(data["gt"], logits)

        total_loss += m_loss
        tb_logs.update({
            "losses/masked_loss": m_loss.item(),
            "metrics/masked_acc": m_acc.item()
        })

        return total_loss, tb_logs

    @torch.no_grad()
    def validate(self, dataloader, save_root):

        tb_logs = {}

        num_text = 0
        cnt = 0
        ex_1 = 0
        ex_3 = 0
        ex_5 = 0

        for idx, val_data in enumerate(dataloader):
            if num_text > 100:
                break

            cnt += 1
            val_data = {k: v.to(self.device) for k, v in val_data.items()}

            # preprocessing should be in dataset class
            src = val_data["src"]

            for i in range(src.shape[0]):
                answer = []
                beams = []

                try:
                    beams = gen_beam(
                        self.net, 
                        self.T, 
                        src[i],
                        self.maxlen,
                        self.tgt_char_to_idx,
                        self.tgt_idx_to_char,
                        self.src_char_to_idx,
                        self.src_idx_to_char,
                        self.beam_size
                    )
                except KeyboardInterrupt:
                    return
                except Exception as e:
                    pass

                if len (beams) == 0:
                    continue

                answer_s = set(answer)

                ans = []
                for k in range(len(beams)):
                    ans.append([ beams[k][0], beams[k][1] ])

                for step, beam in enumerate(ans):
                    right = answer_s.intersection(set(beam[0]))

                    if len(right) == 0: continue
                    if len(right) == len(answer):
                        if step == 0:
                            ex_1 += 1
                            ex_3 += 1
                            ex_5 += 1
                            print("CNT: ", cnt, ex_1 /cnt *100.0, answer, beam[1], beam[1] / len(".".join(answer)) , 1.0 )
                            break
                        if step < 3:
                            ex_3 += 1
                            ex_5 += 1
                            break
                        if step < 5:
                            ex_5 += 1
                            break
                    break

            tb_logs.update({
                "metrics/top_k/1": ex_1 / cnt * 100.0,
                "metrics/top_k/3": ex_3 / cnt * 100.0,
                "metrics/top_k/5": ex_5 * 100.0 / cnt
            })

            # restructure strings
            lines = []
            
            input_text = "".join([self.src_idx_to_char[c] for c in list(val_data["src"][i].detach().cpu().numpy())])
            gold = "".join([self.src_idx_to_char[c] for c in list(val_data["tgt"][i].detach().cpu().numpy())])

            if len(beams) == 0:
                pred_text = ""
            else:
                pred_text = [(beam, score) for beam, score in beams]
                pred_text = "\n".join(pred_text)

            line = (
                f"INPUT: {input_text}\n\nPRED: \n{pred_text}\n\nGOLD: {gold}\n"
                "{}\n".format(30 * "-")
            )
            lines.append(line)

            with open(os.path.join(save_root, f"{num_text:08d}.txt"), "w", encoding="utf-8") as f:
                f.write("".join(lines))

            num_text += val_data["src"].shape[0]

        return tb_logs


if __name__ == "__main__":
    from datasets.retro_dataset import RetroValDataset, RetroCollator
    from torch.utils.data import DataLoader
    import yaml

    cfg_file = "cfg/augmented.yml"
    with open(cfg_file, "r") as f:
        opt = yaml.safe_load(f)
    
    val_ds = RetroValDataset(opt["datasets"])
    val_ds.debug()

    collate_fn = RetroCollator(opt["model"]["tokenizer"])
    val_dl = DataLoader(val_ds, batch_size=4, collate_fn=collate_fn)

    model = RetroSeq2SeqModel(opt["model"])
    os.makedirs("temp/val", exist_ok=True)
    model.validate(dataloader=val_dl, save_root="temp/val")

    




