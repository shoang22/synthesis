import torch

def masked_acc(y_true, y_pred):
    mask = torch.ne(torch.sum(y_true, dim=-1), 0).float()
    eq = torch.eq(torch.argmax(y_true, dim=-1), torch.argmax(y_pred, dim=-1)).float()
    eq = torch.sum(eq * mask, dim=-1) / torch.sum(mask, dim=-1)
    eq = torch.mean(eq)
    return eq