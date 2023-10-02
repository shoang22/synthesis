import torch
import sys
sys.path.append('./')
from archs.augmentedtransformer import EncoderLayer, MaskLayerLeft

if __name__ == '__main__':
    bs = 3
    l = 5
    ed = 10
    hd = 15
    n_head = 3
    x = torch.randn(bs, l, ed)
    m = torch.ones(bs, l)

    encoder = EncoderLayer(ed, ed, n_head, n_head)
    left_mask = MaskLayerLeft(l + 3)

    m = left_mask(m)
    mem = encoder(x, m) 
    print(mem)

