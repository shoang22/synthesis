import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from generation.utils import decode_to_string


class EmbeddingLayer(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embed_dim)


class PositionLayer(nn.Module):
    def __init__(self, embed_dim, maxlen):
        super(PositionLayer, self).__init__()
        mask = torch.arange(0, maxlen, dtype=torch.float)[:, None]
        bins = torch.arange(0, embed_dim, 2, dtype=torch.float)[None, :]
        
        evens = torch.matmul(mask, 1.0 / torch.pow(10000.0, bins / embed_dim))
        odds = torch.clone(evens)

        evens = torch.sin(evens)
        odds = torch.cos(odds)
    
        pos_embed = torch.stack([evens, odds], axis=2).reshape(-1, embed_dim)
        pos_embed = pos_embed[None, ...]
        self.register_buffer("pos_embed", pos_embed)

    def forward(self, token_embed):
        return self.pos_embed[: ,:token_embed.shape[1], :]


class MaskLayerLeft(nn.Module):
    def __init__(self, maxlen):
        super(MaskLayerLeft, self).__init__()
        rank = torch.ones((1, maxlen), dtype=torch.float)
        self.register_buffer("r_left", rank)

    def forward(self, x):
        x = x.float().unsqueeze(-1)
        mask = torch.matmul(x, self.r_left[:, :x.shape[1]])
        return mask.transpose(1, 2)


class MaskLayerRight(nn.Module):
    def __init__(self, maxlen):
        super(MaskLayerRight, self).__init__()
        rank = torch.ones((1, maxlen), dtype=torch.float)
        self.register_buffer("r_right", rank)

    def forward(self, x: torch.Tensor):
        right = x[0].float()
        left = x[1].float()

        left = left.unsqueeze(-1)
        mask = torch.matmul(left, self.r_right[:, :right.shape[1]])
        return mask.transpose(1, 2)


class MaskLayerTriangular(nn.Module):
    def __init__(self, maxlen):
        super(MaskLayerTriangular, self).__init__()
        t = torch.ones((maxlen, maxlen), dtype=torch.float)
        tri = t.tril(diagonal=0)
        self.register_buffer("tril", tri)
        
        rank = torch.ones(1, maxlen, dtype=torch.float)
        self.register_buffer("r_tril", rank)

    def forward(self, x):
        x = x.unsqueeze(-1).float()
        mask = torch.matmul(x, self.r_tril[:, :x.shape[1]])
        return self.tril[:x.shape[1], :x.shape[1]] * mask.transpose(1, 2)


class LayerNormalization(nn.Module):
    def __init__(self, embed_dim, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        num = self.gamma * (x - mean)
        denom = (std + self.eps) + self.beta
        return num / denom


class SelfLayer(nn.Module):
    def __init__(self, embed_dim, key_dim, n_head):
        super(SelfLayer, self).__init__()
        self.embed_dim = embed_dim
        self.key_dim = key_dim
        self.n_head = n_head

        self.denom = math.sqrt(key_dim)

        self.linear_qkvs = nn.ModuleList([])

        for _ in range(n_head):
            qkv_head = nn.ModuleList([
                nn.Linear(embed_dim, key_dim, bias=False), 
                nn.Linear(embed_dim, key_dim, bias=False), 
                nn.Linear(embed_dim, key_dim, bias=False)
            ])
            qkv_head.apply(weights_init)
            self.linear_qkvs.append(qkv_head) 

    def forward(self, inputs):
        out = []

        for linear_Q, linear_K, linear_V in self.linear_qkvs:

            Q = linear_Q(inputs[0])
            K = linear_K(inputs[1])
            V = linear_V(inputs[2])

            A = torch.bmm(Q, K.transpose(1, 2)) / self.denom
            A = A * inputs[3] + ((inputs[3] - 1.0) * (1e9))
            A = A - torch.max(A, dim=2, keepdim=True)[0]

            A = torch.exp(A)
            A = A / torch.sum(A, dim=2, keepdim=True)
            A = F.dropout(A, p=0.1)

            out.append(torch.bmm(A, V))

        return torch.cat(out, dim=-1)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        key_dim,
        n_head,
        hidden_dim
    ):
        super(EncoderLayer, self).__init__()
        self.norm1 = LayerNormalization(embed_dim)
        self.dropout1 = nn.Dropout(p=0.1)
        self.self_attn = SelfLayer(embed_dim, key_dim, n_head)

        # TimeDistributed is dependent on key_size * n_head
        self.dense1 = TimeDistributed(
            nn.Linear(key_dim * n_head, embed_dim, bias=False)
        )

        self.norm2 = LayerNormalization(embed_dim)

        self.c1 = TimeDistributed(nn.Linear(embed_dim, hidden_dim))
        self.relu = nn.ReLU()
        self.c2 = TimeDistributed(nn.Linear(hidden_dim, embed_dim))
        self.dropout2 = nn.Dropout(p=0.1)
    
    def _sa_block(self, x, attn_mask):
        x = self.self_attn([x, x, x, attn_mask])
        return self.dense1(x)
    
    def _mma_block(self, x, x_mm, attn_mask):
        x = self.self_attn([x_mm, x, x, attn_mask])
        return self.dense1(x)
    
    def _ff_block(self, x):
        x = self.c2(self.relu(self.c1(x)))
        return self.dropout2(x)

    def forward(self, x, x_mm, mask):
        # remember to call dropout after attn
        x_norm = self.norm1(x)
        x_mm_norm = self.norm1(x_mm)
        out = x_mm + self._mma_block(x_norm, x_mm_norm, mask)
        out = out + self._ff_block(self.norm2(self.dropout1(out)))
        return out


class DecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        key_dim,
        n_head,
        hidden_dim
    ):
        super(DecoderLayer, self).__init__()
        self.norm1 = LayerNormalization(embed_dim)
        self.dropout1 = nn.Dropout(p=0.1)
        self.self_attn = SelfLayer(embed_dim, key_dim, n_head)
        self.dense1 = TimeDistributed(
            nn.Linear(key_dim * n_head, embed_dim, bias=False)
        )
        self.dropout2 = nn.Dropout(p=0.1)

        self.norm2 = LayerNormalization(embed_dim)
        self.multihead_attn = SelfLayer(embed_dim, key_dim, n_head)
        self.dense2 = TimeDistributed(
            nn.Linear(key_dim * n_head, embed_dim, bias=False)
        )
        self.dropout3 = nn.Dropout(p=0.1)
        self.norm3 = LayerNormalization(embed_dim)

        self.c1 = TimeDistributed(nn.Linear(embed_dim, hidden_dim))
        self.relu = nn.ReLU()
        self.c2 = TimeDistributed(nn.Linear(hidden_dim, embed_dim))
        self.dropout4 = nn.Dropout(p=0.1)
    
    def _sa_block(self, x, attn_mask):
        x = self.self_attn([x, x, x, attn_mask])
        return self.dropout2(self.dense1(x))

    def _mha_block(self, x, mem, attn_mask):
        x = self.multihead_attn([x, mem, mem, attn_mask])
        return self.dropout3(self.dense2(x))
    
    def _ff_block(self, x):
        x = self.c2(self.relu(self.c1(x)))
        return self.dropout4(x)
    
    def forward(self, x, memory, tgt_mask, memory_mask):
        x = x + self._sa_block(self.dropout1(self.norm1(x)), tgt_mask)
        x = x + self._mha_block(self.norm2(x), memory, memory_mask)
        x = x + self._ff_block(self.norm3(x))
        return x


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        t, n = x.size(0), x.size(1) 
        # merge batch and seq dimensions
        x_reshape = x.contiguous().view(t * n, x.size(2))
        y = self.module(x_reshape)
        # We have to reshape Y
        y = y.contiguous().view(t, n, y.size()[1])
        
        return y


class AugmentedTransformer(nn.Module):
    def __init__(
        self, 
        maxlen, 
        embed_dim, 
        key_dim,
        hidden_dim,
        src_vocab_size, 
        tgt_vocab_size, 
        fingerprint_size,
        n_block,
        n_head
    ):
        super(AugmentedTransformer, self).__init__()
        self.embed_dim = embed_dim

        self.fingerprint_embed = EmbeddingLayer(embed_dim, fingerprint_size)
        self.positional_encoding = PositionLayer(embed_dim, max(maxlen, fingerprint_size))
        self.src_mask = MaskLayerLeft(maxlen)
        self.src_vocab_embed = EmbeddingLayer(embed_dim, src_vocab_size)
        self.maxlen = maxlen

        self.encoder = nn.ModuleList([
            EncoderLayer(embed_dim, key_dim, n_head, hidden_dim) for _ in range(n_block)
        ])
        self.encoder_norm = LayerNormalization(embed_dim)

        self.tgt_vocab_embed = EmbeddingLayer(embed_dim, tgt_vocab_size)
        self.memory_mask = MaskLayerRight(maxlen)
        self.tgt_mask = MaskLayerTriangular(maxlen)
        self.dropout = nn.Dropout(p=0.1)

        self.decoder = nn.ModuleList([
            DecoderLayer(embed_dim, key_dim, n_head, hidden_dim) for _ in range(n_block)
        ])
        self.decoder_norm = LayerNormalization(embed_dim)

        self.generator = TimeDistributed(nn.Linear(embed_dim, tgt_vocab_size))

    def forward(
        self, 
        fp, 
        src, 
        tgt, 
        src_padding_mask, 
        fp_padding_mask,
        tgt_padding_mask, 
        **kwargs
    ):
        fp_embed = self.fingerprint_embed(fp)
        fp_pos = self.positional_encoding(fp_padding_mask)
        
        src_pos = self.positional_encoding(src_padding_mask)
        src_mask = self.src_mask(src_padding_mask)
        src_embed = self.src_vocab_embed(src)

        fp_embed += fp_pos
        src_embed += src_pos

        mm_embed = torch.cat([src_embed, fp_embed], dim=1)
        memory = mm_embed

        for block in self.encoder:
            memory = block(src_embed, memory, src_mask)
        
        memory = self.encoder_norm(memory)
        memory_mask = self.memory_mask([tgt_padding_mask, src_padding_mask])

        tgt_pos = self.positional_encoding(tgt_padding_mask)
        tgt_mask = self.tgt_mask(tgt_padding_mask)
        tgt_embed = self.tgt_vocab_embed(tgt)
        tgt_embed = self.dropout(tgt_embed + tgt_pos)

        output = tgt_embed

        for block in self.decoder:
            output = block(output, memory, tgt_mask, memory_mask)

        output = self.decoder_norm(output)

        return self.generator(output)
    
    def encode(self, x, x_fp, fp_padding_mask, src_char_to_ix, src_idx_to_char):
        
        # src input will be a tensor
        src, src_padding_mask, src_pos = gen_left(
            [decode_to_string(x, src_idx_to_char)],
            embed_dim=self.embed_dim,
            src_char_to_ix=src_char_to_ix,
            max_len=self.maxlen
        )

        src = src.to(x.device)
        src_padding_mask = src_padding_mask.to(x.device)
        src_pos = src_pos.to(x.device)

        # positional encoding needs to be pre-computed
        src_mask = self.src_mask(src_padding_mask)
        src_embed = self.src_vocab_embed(src)
        src_embed += src_pos
        
        fp_embed = self.fingerprint_embed(x_fp)
        fp_pos = self.positional_encoding(fp_padding_mask)
        fp_embed += fp_pos

        mm_embed = torch.cat([src_embed, fp_embed], dim=1)
        memory = mm_embed

        for block in self.encoder:
            memory = block(src_embed, memory, src_mask)
        
        return self.encoder_norm(memory), src_padding_mask
    
    def decode(self, y, memory, src_padding_mask, 
               tgt_char_to_idx, tgt_idx_to_char, T=1.0):

        tgt, tgt_padding_mask, tgt_pos = gen_right(
            [y],
            embed_dim=self.embed_dim,
            tgt_char_to_ix=tgt_char_to_idx,
            max_len=self.maxlen
        )
        
        tgt = tgt.to(memory.device)
        tgt_padding_mask = tgt_padding_mask.to(memory.device)
        tgt_pos = tgt_pos.to(memory.device)

        memory_mask = self.memory_mask([tgt_padding_mask, src_padding_mask])

        tgt_mask = self.tgt_mask(tgt_padding_mask)
        tgt_embed = self.tgt_vocab_embed(tgt)
        tgt_embed = self.dropout(tgt_embed + tgt_pos)

        output = tgt_embed

        for block in self.decoder:
            output = block(output, memory, tgt_mask, memory_mask)

        output = self.generator(self.decoder_norm(output))
        # taking the final token
        prob = output[0, len(y), :] / T
        # softmax
        prob = torch.exp(prob) / torch.sum(torch.exp(prob))
        return prob


def GetPosEncodingMatrix(max_len, embed_dim):
    pos_enc = torch.Tensor([
        [pos / math.pow(10000, 2 * (j // 2) / embed_dim) for j in range(embed_dim)]
        for pos in range(max_len)
	])
    pos_enc[1:, 0::2] = torch.sin(pos_enc[1:, 0::2])
    pos_enc[1:, 1::2] = torch.cos(pos_enc[1:, 1::2])
        
    return pos_enc


def gen_left(data, embed_dim, src_char_to_ix, max_len):
    geo = GetPosEncodingMatrix(max_len, embed_dim)
    batch_size = len(data)
    nl = len(data[0]) + 1

    x = torch.zeros((batch_size, nl)).type(torch.LongTensor)
    mx = torch.zeros((batch_size, nl)).type(torch.LongTensor)
    px = torch.zeros((batch_size, nl, embed_dim), dtype=torch.float32)

    for cnt in range(batch_size):
        product = data[cnt] + "$"
        for i, p in enumerate(product):

           try: 
              x[cnt, i] = src_char_to_ix[ p] 
           except:
              x[cnt, i] = 1

           px[cnt, i] = geo[i, :embed_dim]
        mx[cnt, :i+1] = 1        
    return x, mx, px


def gen_right(data, embed_dim, tgt_char_to_ix, max_len):
    geo = GetPosEncodingMatrix(max_len, embed_dim)
    batch_size = len(data)
    # +1 for start token
    nr = len(data[0]) + 1

    y = torch.zeros((batch_size, nr)).type(torch.LongTensor)
    my = torch.zeros((batch_size, nr)).type(torch.LongTensor)
    py = torch.zeros((batch_size, nr, embed_dim), dtype=torch.float32)

    for cnt in range(batch_size):
        # we should take care of this in validation dataloader
        reactants = "^" + data[cnt]
        for i, p in enumerate(reactants):
           try: 
              y[cnt, i] = tgt_char_to_ix[p]
           except:
              # for oov tokens
              y[cnt, i] = 1

           py[cnt, i] = geo[i, :embed_dim ]
        my[cnt, :i+1] =1
    return y, my, py


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    # else:
        # print(f"weights_init is not supported for {type(m)}. Skipping...")


if __name__ == "__main__":
    bs = 3
    l = 5
    ed = 10
    hd = 15
    n_head = 3
    vocab_size = l * 3

    src = torch.randint(1, vocab_size, (bs, l))
    src_mask = torch.ones(bs, l)
    src_mask[-1, -2:] = 0

    tgt = torch.randint(1, vocab_size, (bs, l-1))
    tgt_mask = torch.ones(bs, l-1)
    tgt_mask[-1, -2:] = 0

    model = AugmentedTransformer(
        maxlen=l + 3,
        embed_dim=ed,
        key_dim=ed,
        hidden_dim=hd,
        src_vocab_size=vocab_size,
        tgt_vocab_size=l*3,
        n_block=3,
        n_head=n_head
    )

    out = model(src, tgt, src_mask, tgt_mask)
    assert list(out.shape) == [bs, l-1, vocab_size]
    print(out)