import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    

if __name__ == "__main__":
    bs = 3
    l = 5
    d = 10
    n_head = 3
    x = torch.randn(bs, l, d)
    self_attn = SelfLayer(d, d, n_head)


    for i in range(len(self_attn.linear_qkvs)):
        curr_head = self_attn.linear_qkvs[i]
        for j in range(i + 1, len(self_attn.linear_qkvs)):
            next_head = self_attn.linear_qkvs[j]
            assert not (curr_head[0].weight == next_head[0].weight).all(), "weights should be different"

    print("test completed")
