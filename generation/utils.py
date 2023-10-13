import numpy as np
import math
from rdkit import Chem
import os
import torch


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
def gen_greedy(model, T, src, fp, fp_padding_mask, max_len, tgt_char_to_idx, tgt_idx_to_char,
               src_char_to_idx, src_idx_to_char):

    src_encoded, src_mask = model.encode(src, fp, fp_padding_mask, src_char_to_idx, src_idx_to_char)

    res = ""
    score = 0.0
    for i in range(1, max_len):
        p = model.decode(res, src_encoded, src_mask, 
                         tgt_char_to_idx, tgt_idx_to_char, T).detach().cpu.numpy()
        w = np.argmax(p)
        score -= math.log10( np.max(p))
        if w == tgt_char_to_idx["$"]:
            break
        try:
            res += tgt_idx_to_char[w]
        except:
            res += "?"

    reags = res.split(".")
    sms = set() 
    with suppress_stderr():
        for r in reags:
            r = r.replace("$", "")
            m = Chem.MolFromSmiles(r)
            if m is not None:
                sms.add(Chem.MolToSmiles(m))
        if len(sms):
            return [sorted(list(sms)), score ]

    return ["", 0.0]


@torch.no_grad()
def gen_beam(
    model, T, src, fp, fp_padding_mask, max_len, tgt_char_to_idx, tgt_idx_to_char, 
    src_char_to_idx, src_idx_to_char, beam_size = 1
):
    tgt_vocab_size = len(tgt_char_to_idx)

    src_encoded, src_mask = model.encode(src, fp, fp_padding_mask, src_char_to_idx, src_idx_to_char)

    if beam_size == 1:
        return [gen_greedy(model, T, src, max_len=max_len, fp=fp, fp_padding_mask=fp_padding_mask, 
                           tgt_char_to_idx=tgt_char_to_idx, tgt_idx_to_char=tgt_idx_to_char,
                           src_char_to_idx=src_char_to_idx, src_idx_to_char=src_idx_to_char)]

    lines = []
    scores = []
    final_beams = []

    for i in range(beam_size):
        lines.append("")
        scores.append(0.0)

    for step in range(max_len):
        if step == 0:
            # during first step, top num_beam preds are used as first beam tokens
            p = model.decode("", src_encoded, src_mask, tgt_char_to_idx, tgt_idx_to_char, T)
            p = p.detach().cpu().numpy()
            nr = np.zeros((tgt_vocab_size, 2)) # [prob(word_i), i]
            for i in range(tgt_vocab_size):
                nr [i ,0 ] = -math.log(p[i])
                nr [i ,1 ] = i
        else:
            # cb = beam_size
            cb = len(lines)
            nr = np.zeros(( cb * tgt_vocab_size, 2))
            for i in range(cb):
                p = model.decode(lines[i], src_encoded, src_mask, tgt_char_to_idx, tgt_idx_to_char, T)
                p = p.detach().cpu().numpy()
            for j in range(tgt_vocab_size):
                nr[ i* tgt_vocab_size + j, 0] = -math.log10(p[j]) + scores[i]
                nr[ i* tgt_vocab_size + j, 1] = i * 100 + j

        y = nr [ nr[:, 0].argsort() ] ; # sorted negative log_probs

        new_beams = []
        new_scores = []

        for i in range(beam_size):

            c = tgt_idx_to_char[ y[i, 1] % 100 ]
            beamno = int( y[i, 1] ) // 100

            if c == '$':
                added = lines[beamno] + c
                if added != "$": # if not empty string pred, add
                    final_beams.append( [ lines[beamno] + c, y[i,0]])
                beam_size -= 1
            else:
                new_beams.append( lines[beamno] + c )
                new_scores.append( y[i, 0])

        lines = new_beams
        scores = new_scores

        if len(lines) == 0: break

    for i in range(len(final_beams)):
        # score divided by length (longer = better)
        final_beams[i][1] = final_beams[i][1] / len(final_beams[i][0])

    final_beams = list(sorted(final_beams, key=lambda x:x[1]))[:5]
    answer = []

    for k in range(5):
        reags = set(final_beams[k][0].split("."))
        sms = set()

        with suppress_stderr():
            for r in reags:
                r = r.replace("$", "")
                m = Chem.MolFromSmiles(r)
            if m is not None:
                sms.add(Chem.MolToSmiles(m))
            if len(sms):
                answer.append([sorted(list(sms)), final_beams[k][1] ])

    return answer


def decode_to_string(tokens_arr, idx_to_char):
    if isinstance(tokens_arr, torch.Tensor):
        tokens_arr = tokens_arr.tolist()
    return "".join([idx_to_char[i] for i in tokens_arr])


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