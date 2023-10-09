import torch
from rdkit import Chem
import math
import numpy as np
# import sys
# sys.path.append("./")
from generation.utils import gen_greedy, encode_position, decode_to_string, suppress_stderr


@torch.no_grad()
def gen_beam(
    model, T, src, max_len, tgt_char_to_idx, tgt_idx_to_char, 
    src_idx_to_char, src_char_to_idx, beam_size = 1
):

    src_encoded, src_mask = model.encode(src, src_char_to_idx, src_idx_to_char)

    if beam_size == 1:
        return [gen_greedy(model, T, src, max_len=max_len, 
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

    print("Final beams:", len(final_beams))

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
            #print(sms);
            if len(sms):
                answer.append([sorted(list(sms)), final_beams[k][1] ])

    return answer


if __name__ == "__main__":
    from archs import define_arch

    src_char_to_idx = {'#': 4, '(': 5, ')': 6, '+': 7, '-': 8, '.': 9, '/': 10, '1': 11, '2': 12, '3': 13, '4': 14, '5': 15, '6': 16, '7': 17, '=': 18, '@': 19, 'B': 20, 'C': 21, 'F': 22, 'H': 23, 'I': 24, 'L': 25, 'M': 26, 'N': 27, 'O': 28, 'P': 29, 'S': 30, 'Z': 31, '[': 32, '\\': 33, ']': 34, 'c': 35, 'e': 36, 'g': 37, 'i': 38, 'l': 39, 'n': 40, 'o': 41, 'r': 42, 's': 43, 'u': 44, ' ': 0, '?': 1, '^': 2, '$': 3}
    src_vocab_size = len(src_char_to_idx)
    src_idx_to_char = { src_char_to_idx[ch]:ch for ch in src_char_to_idx }

    tgt_char_to_idx = {'#': 4, '(': 5, ')': 6, '+': 7, '-': 8, '.': 9, '/': 10, '1': 11, '2': 12, '3': 13, '4': 14, '5': 15, '6': 16, '7': 17, '=': 18, '@': 19, 'B': 20, 'C': 21, 'F': 22, 'H': 23, 'I': 24, 'L': 25, 'M': 26, 'N': 27, 'O': 28, 'P': 29, 'S': 30, 'Z': 31, '[': 32, '\\': 33, ']': 34, 'c': 35, 'e': 36, 'g': 37, 'i': 38, 'l': 39, 'n': 40, 'o': 41, 'r': 42, 's': 43, 'u': 44, ' ': 0, '?': 1, '^': 2, '$': 3}
    tgt_vocab_size = len(tgt_char_to_idx)
    tgt_idx_to_char = { tgt_char_to_idx[ch]:ch for ch in tgt_char_to_idx }

    bs = 1
    maxlen = 20
    src = torch.randint(1, src_vocab_size, (bs, maxlen - 1))
    src_mask = torch.ones(bs, maxlen - 1)
    src_mask[-1, -2:] = 0

    tgt = torch.randint(1, tgt_vocab_size, (bs, maxlen - 1))
    tgt_mask = torch.ones(bs, maxlen - 1)
    tgt_mask[-1, -2:] = 0

    embed_dim = 32

    opt_m = {
        "type": "AugmentedTransformer",
        "opt": {
            "maxlen": 50,
            "embed_dim": embed_dim,
            "key_dim": embed_dim,
            "hidden_dim": 32,
            "src_vocab_size": src_vocab_size, 
            "tgt_vocab_size": tgt_vocab_size, 
            "n_block": 3,
            "n_head": 2
        }
    }
    model = define_arch(opt_m)

    model.eval()
    answer = gen_beam(
        model,
        1.0,
        src,
        max_len=maxlen,
        tgt_char_to_idx=tgt_char_to_idx,
        tgt_idx_to_char=tgt_idx_to_char,
        src_char_to_idx=src_char_to_idx,
        src_idx_to_char=src_idx_to_char,
        beam_size=2
    )
    print("test completed")