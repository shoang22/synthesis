import numpy as np
import math
from rdkit import Chem
import os
import torch
import random

class suppress_stderr(object):
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR)]
        self.save_fds = [os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 2)
    
    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 2)

        for fd in self.null_fds + self.save_fds:
            os.close(fd)


@torch.no_grad()
def gen_greedy(model, T, src, src_padding_mask, 
               max_len, bos_idx, eos_idx, device, **kwargs):

    src_encoded = model.encode(src, src_padding_mask)

    res = torch.ones(1,1).fill_(bos_idx).long().to(device)
    score = 0.0

    for i in range(1, max_len):
        res_padding_mask = torch.ones_like(res).long().to(device)
        p = model.decode(res, res_padding_mask, src_encoded, src_padding_mask, T)
        w = torch.argmax(p, keepdim=True).unsqueeze(0)
        score -= torch.log10(torch.max(p))
        if w == eos_idx:
            break
        
        # try to find the character in the vocab
        res = torch.cat([res, w], dim=-1)

    return res, score[None, :]
        

@torch.no_grad()
def gen_beam(model, T, src, src_padding_mask, 
             max_len, bos_idx, eos_idx, device, beam_size = 1):

    src_encoded = model.encode(src, src_padding_mask)

    if beam_size == 1:
        return [gen_greedy(model, T, src, src_padding_mask, 
                           max_len=max_len, bos_idx=bos_idx, eos_idx=eos_idx, device=device)]

    lines = torch.zeros(beam_size, max_len).long().to(device)
    lengths = torch.zeros(beam_size, ).long().to(device)
    lines[:, 0] = bos_idx
    scores = []

    final_beams = torch.zeros(beam_size, max_len).long().to(device)
    final_scores = torch.zeros(beam_size, ).to(device)
    final_lengths = torch.zeros(beam_size, ).to(device)
    final_row_idx = 0

    for i in range(beam_size):
        scores.append(0.0)

    for step in range(max_len):
        if step == 0:
            res = torch.ones(1,1).fill_(bos_idx).long().to(device)
            res_padding_mask = torch.ones_like(res).long().to(device)
            # during first step, we generate the log prob scores for all the words in the vocab
            p = model.decode(res, res_padding_mask, src_encoded, src_padding_mask, T)
            nr = torch.zeros((p.shape[-1], 2)).to(device) # [prob(word_i), i]
            for i in range(p.shape[-1]):
                nr [i ,0 ] = -torch.log(p[i])
                nr [i ,1 ] = i
            
            lengths += 1
            
        else:
            # cb = beam_size
            # will always be beam size since we prune
            cb = len(lines)
            nr = torch.zeros(( cb * p.shape[-1], 2 )).to(device)
            for i in range(cb):
                cand_padding_mask = torch.ones_like(lines[i]).unsqueeze(0).long().to(device)
                cand_padding_mask[:, step:] = 0
                p = model.decode(lines[i][None, :], cand_padding_mask, src_encoded, src_padding_mask, T)
                for j in range(p.shape[-1]):
                    # add scores for each token for each beam
                    # tokens are indexed by i * tgt_vocab_size + j
                    # to separate them from other beams
                    nr[ i* p.shape[-1] + j, 0] = -torch.log10(p[j]) + scores[i]
                    nr[ i* p.shape[-1] + j, 1] = i * 100 + j
            
            lengths += 1

        # sort across all beams
        # one beam can have multiple candidates if they make the cut
        y = nr[nr[:, 0].argsort()] ; # sorted negative log_probs

        new_beams = torch.zeros(beam_size, max_len).long().to(device)
        new_beams_mask = torch.ones(beam_size, ).bool().to(device)
        new_scores = []

        # taking top n_beams candidates...
        for i in range(beam_size):
            # mod to get remainder (actual token)

            # for debugging
            if random.random() > 0.8:
                c = eos_idx
            else:
                c = y[i, 1] % 100
            beamno = int( y[i, 1] ) // 100 # this how we track the beam no of the current candidate

            if c == eos_idx:
                lines[beamno, step] = c
                if torch.sum(lines[beamno]) != eos_idx: # if not empty string pred, add to final beam
                    # omit adding eos token 
                    # remember, we do not remove candidates from final_beam
                    # as the conditional prob score only decreases as more tokens are added
                    final_beams[final_row_idx] = lines[beamno]
                    final_scores[final_row_idx] = y[i, 0]
                    final_lengths[final_row_idx] = lengths[i]
                    final_row_idx += 1
                beam_size -= 1 # now we have one less beam to decode for
                new_beams_mask[i] = False
                lengths[i] = 0
            else:
                new_beams[i, :] = lines[beamno]
                new_beams[i, step] = c
                new_scores.append(y[i, 0].item())

        # ISSUE: during 2nd > iteration after beam_size has been reduced,
        # len(scores) is < n_beams while len(lines) is still n_beams

        # prune
        new_beams = new_beams[new_beams_mask]
        assert new_beams.shape[0] == beam_size
        lines = new_beams.clone()
        scores = new_scores
        lengths = lengths[new_beams_mask]

        if len(lines) == 0: break

    # remove blank beams
    final_beam_idx  = torch.sum(final_beams, dim=-1) != 0
    final_beams = final_beams[final_beam_idx]
    final_scores = final_scores[final_beam_idx]
    final_lengths = final_lengths[final_beam_idx]

    if len(final_beams) == 0:
        return None, None

    final_scores = final_scores / final_lengths
    final_beams = final_beams[final_scores.argsort()][:5]
    final_scores = final_scores.sort()[0]

    assert len(final_beams) == len(final_scores)
    return final_beams, final_scores


def decode_to_string(tokens_arr, idx_to_char):
    if isinstance(tokens_arr, torch.Tensor):
        tokens_arr = tokens_arr.tolist()
    return "".join([idx_to_char[i] for i in tokens_arr]).strip()


def tokenizer_from_vocab(vocab_path):
    tokenizer = {} 
    with open(vocab_path, "r") as f:
        for line in f:
            idx, char = line.replace("\n", "").split("\t")
            if not isinstance(idx, int):
                idx = int(idx)
            if not tokenizer.get(idx, None):
                tokenizer[char] = idx

    return tokenizer


if __name__ == "__main__":
    import yaml
    import sys
    sys.path.append("./")
    from models.archs.augmentedtransformer import AugmentedTransformer

    path = "cfg/augmented.yml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    mdl_opt = cfg["model"]["net"]["opt"]
    mdl_opt["src_vocab_size"] = 16
    mdl_opt["tgt_vocab_size"] = 16

    model = AugmentedTransformer(**mdl_opt)
    
    bs = 8
    l = 10
    max_len = 150
    bos_idx = 0
    eos_idx = 3

    x = torch.randint(0, 10, (bs, l))
    mx = torch.ones_like(x)
    mx[-2:, -3:] = 0
    for i in range(bs):
        yhat = gen_beam(model, 1.0, x[None, i], mx[None, i], 
                          max_len, bos_idx, eos_idx, beam_size=3)
        print(yhat)


