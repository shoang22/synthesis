import numpy as np
import math
import rdkit as Chem
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


def GetPosEncodingMatrix(max_len, d_emb):
	pos_enc = np.array([
		[pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
		    for pos in range(max_len)
			])
	pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
	pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])
	return pos_enc


@torch.no_grad()
def gen_greedy(
    model,
    T,
    src,
    tgt,
    src_padding_mask,
    tgt_padding_mask,
    max_predict,
    tgt_char_to_idx,
    tgt_ix_to_char
):
    product_encoded = model.encode(src, src_padding_mask)
    res = tgt_char_to_idx[""]
    score = 0.0

    for i in range(1, max_predict):
        p = model.decode(
            res, 
            product_encoded, 
            src_padding_mask, 
            tgt_padding_mask, 
            T,
        )
        w = np.argmax(p)
        score -= math.log10(np.max(p))
        if w == tgt_char_to_idx["$"]:
            break
        try:
            res += tgt_ix_to_char[w]
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
                return [sorted(list(sms)), score]
            
    return ["", 0.0]


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
        