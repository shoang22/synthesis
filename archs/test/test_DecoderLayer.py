import torch
import sys
sys.path.append('./')
from archs.augmentedtransformer import (
    DecoderLayer, 
    MaskLayerRight, 
    MaskLayerTriangular, 
    MaskLayerLeft, 
    EncoderLayer
)

if __name__ == '__main__':
    bs = 3
    l = 5
    ed = 10
    hd = 15
    n_head = 3

    src = torch.randn(bs, l, ed)
    src_mask = torch.ones(bs, l)
    src_mask[-1, -2:] = 0

    encoder = EncoderLayer(ed, ed, n_head, n_head)
    left_mask = MaskLayerLeft(l + 3)

    sm = left_mask(src_mask)
    mem = encoder(src, sm) 

    tgt = torch.randn(bs, l-1, ed)
    tgt_mask = torch.ones(bs, l-1)
    tgt_mask[-1, -2:] = 0

    decoder = DecoderLayer(ed, ed, n_head, n_head)
    mem_mask = MaskLayerRight(l + 3)
    tril_mask = MaskLayerTriangular(l + 3)

    mm = mem_mask([tgt_mask, src_mask])
    rm = tril_mask(tgt_mask)
    out = decoder(tgt, mem, rm, mm) 

    print(out)

