import torch
import torch.nn as nn
import os
from collections import OrderedDict

from models.losses import masked_loss
from metrics import masked_acc
from archs import define_arch
from generation.beam_search import gen_beam

class RetroModel(nn.Module):
    def __init__(self, opt):
        super(RetroModel, self).__init__()

        self.device = torch.cuda.current_device()
        self.opt = opt
        self.src_char_to_idx = None
        self.src_idx_to_char = None
        self.tgt_char_to_idx = None
        self.tgt_idx_to_char = None

        # self.bos = self.opt["tokenizer"]["bos"]
        # self.eos = self.opt["tokenizer"]["eos"]
        # self.pad_idx = self.opt["tokenizer"]["pad_idx"]

        self.net = define_arch(opt["net"]).to(self.device)

        self.beam_size = opt["beam_size"]
        self.T = opt["temperature"]
        self.maxlen = opt["maxlen"]
        assert self.topks[-1] <= self.beam_size, "max topk cannot be greater than beam size"

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

        masked_loss = masked_loss(data["tgt"], logits)
        masked_acc = masked_acc(data["tgt"], logits)

        total_loss += masked_loss
        tb_logs.update({
            "losses/masked_loss": masked_loss.item(),
            "metrics/masked_acc": masked_acc.item()
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
            if num_text == 500:
                break

            cnt += 1
            val_data = {k: v.to(self.device) for k, v in val_data.items()}

            # preprocessing should be in dataset class
            src = val_data["src"]
            tgt = val_data["tgt"]
            src_padding_mask = val_data["src_padding_mask"]
            tgt_padding_mask = val_data["tgt_padding_mask"]

            for i in range(src.shape[0]):
                answer = []
                beams = []

                try:
                    beams = gen_beam(
                        self.net, 
                        self.T, 
                        src[i],
                        tgt[i],
                        src_padding_mask[i],
                        tgt_padding_mask[i],
                        self.max_predict,
                        self.tgt_char_to_idx,
                        self.tgt_idx_to_char
                    )
                except KeyboardInterrupt:
                    # print ("\nExact: ", self.T, ex_1 / cnt * 100.0, ex_3 / cnt * 100.0, ex_5 * 100.0 / cnt, cnt)
                    return
                except Exception as e:
                    # print(e)
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

        #     print ("Exact: ", self.T, ex_1 / cnt * 100.0, ex_3 / cnt * 100.0, ex_5 * 100.0 / cnt, cnt)

            tb_logs.update({
                "metrics/top_k/1": ex_1 / cnt * 100.0,
                "metrics/top_k/3": ex_3 / cnt * 100.0,
                "metrics/top_k/5": ex_5 * 100.0 / cnt
            })

            # restructure strings
            lines = []
            
            input_text = "".join([self.src_idx_to_char[c] for c in list(val_data["input_ids"][i].detach().cpu().numpy())])
            gold = "".join([self.src_idx_to_char[c] for c in list(val_data["labels"][i].detach().cpu().numpy())])

            pred_text = [beam for beam, score in beams]
            pred_text = "\n".join(pred_text)

            line = (
                f"INPUT: {input_text}\n\nPRED: \n{pred_text}\n\nGOLD: {gold}\n"
                "{}\n".format(30 * "-")
            )
            lines.append(line)

            with open(os.path.join(save_root, f"{num_text:08d}.txt"), "w", encoding="utf-8") as f:
                f.write("".join(lines))

            num_text += val_data["input_ids"].shape[0]

        return tb_logs


if __name__ == "__main__":
    from datasets.retro_dataset import RetroValDataset, RetroCollator
    from torch.utils.data import DataLoader

    src_char_to_ix = {'#': 4, '(': 5, ')': 6, '+': 7, '-': 8, '.': 9, '/': 10, '1': 11, '2': 12, '3': 13, '4': 14, '5': 15, '6': 16, '7': 17, '=': 18, '@': 19, 'B': 20, 'C': 21, 'F': 22, 'H': 23, 'I': 24, 'L': 25, 'M': 26, 'N': 27, 'O': 28, 'P': 29, 'S': 30, 'Z': 31, '[': 32, '\\': 33, ']': 34, 'c': 35, 'e': 36, 'g': 37, 'i': 38, 'l': 39, 'n': 40, 'o': 41, 'r': 42, 's': 43, 'u': 44, ' ': 0, '?': 1, '^': 2, '$': 3};
    src_vocab_size = len(src_char_to_ix);
    src_ix_to_char = { src_char_to_ix[ch]:ch for ch in src_char_to_ix }

    tgt_char_to_ix = {'#': 4, '(': 5, ')': 6, '+': 7, '-': 8, '.': 9, '/': 10, '1': 11, '2': 12, '3': 13, '4': 14, '5': 15, '6': 16, '7': 17, '=': 18, '@': 19, 'B': 20, 'C': 21, 'F': 22, 'H': 23, 'I': 24, 'L': 25, 'M': 26, 'N': 27, 'O': 28, 'P': 29, 'S': 30, 'Z': 31, '[': 32, '\\': 33, ']': 34, 'c': 35, 'e': 36, 'g': 37, 'i': 38, 'l': 39, 'n': 40, 'o': 41, 'r': 42, 's': 43, 'u': 44, ' ': 0, '?': 1, '^': 2, '$': 3};
    tgt_vocab_size = len(tgt_char_to_ix);
    tgt_ix_to_char = { tgt_char_to_ix[ch]:ch for ch in tgt_char_to_ix }

    opt_d = {"data_path": "uspto-50k/patents40k_x5MSShuf.csv"}
    opt_c = {"src_vocab": "vocab/src.vocab", "tgt_vocab": "vocab/tgt.vocab"}

    
    val_ds = RetroValDataset(opt_d)
    val_ds.debug()

    collate_fn = RetroCollator(opt_c)

    val_dl = DataLoader(val_ds, batch_size=4, collate_fn=collate_fn)

    opt_m = {
        "type": "AugmentedTransformer",
        "opt": {
            "maxlen": 50,
            "embed_dim": 20,
            "key_dim": 20,
            "hidden_dim": 32,
            "src_vocab_size": src_vocab_size, 
            "tgt_vocab_size": tgt_vocab_size, 
            "n_block": 3,
            "n_head": 2
        }
    }
    model = define_arch(opt_m)
    




