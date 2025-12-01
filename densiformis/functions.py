import torch
from torch.nn.functional import softplus

def log_cosh(y_true:torch.Tensor,y_pred:torch.Tensor):
    return (y_pred-y_true).cosh().log()

def rand_soft_label(size, dim=1, device="cpu"):
    p = torch.rand(size, dtype=torch.float32,device=device)
    p = torch.cumprod(p,dim) + 1e-4
    p = p/p.sum(1,keepdim=True)
    p = torch.gather(p, dim=dim, index=torch.argsort(torch.rand_like(p), dim=dim))
    return p

def symmetric_sigmoid(x:torch.Tensor):
    return x.sigmoid()*2-1

def symmetric_softplus_loss(y_true:torch.Tensor,y_pred:torch.Tensor):
    return 0.5 * (softplus(y_pred) + softplus(-y_pred) - y_true * y_pred) - (2-y_true.square()).log()

def symmetric_softmax(x:torch.Tensor,dim=1):    
    return x.softmax(dim=dim) - (-x).softmax(dim=dim)

def symmetric_logsumexp_loss(y_true: torch.Tensor, y_pred: torch.Tensor, dim=1):
    return 0.5 * (y_pred.logsumexp(dim=dim,keepdim=True) + (-y_pred).logsumexp(dim=dim,keepdim=True) - (y_true * y_pred).sum(dim=dim,keepdim=True)) - (2-y_true.square()).log().mean(dim=dim,keepdim=True)