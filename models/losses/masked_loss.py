import torch

def masked_loss(y_true, y_pred):
    loss = torch.nn.functional.cross_entropy(input=y_pred, target=y_true)
    mask = torch.ne(torch.sum(y_true, dim=-1), 0).float()
    loss = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    loss = torch.mean(loss)
    return loss
