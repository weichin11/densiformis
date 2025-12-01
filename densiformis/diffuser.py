import os
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Iterator, Optional
import tqdm
from .distributions import get_distribution_behavior, BaseDistributionBehavior

def fit(
    model:Module,
    distribution_types:List[str], 
    train_dataset:Dataset,
    valid_dataset:Dataset,
    optimizer:Optimizer,
    epochs:int,
    batch_size:int,
    checkpoint_save_root:Optional[str] = None,
    save_period:int = 1,
    device:str = "cpu",
    verbose:bool = True,
) -> None:
    
    distribution_behaviors = [get_distribution_behavior(dt) for dt in distribution_types]
    epoch_train_loss = []
    epoch_valid_loss = []

    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=False)

    epoch_bar = tqdm.trange(epochs, ncols=0, disable = not verbose)
    for e in epoch_bar:
        model.train()
        batch_train_loss = []
        train_bar = tqdm.tqdm(train_loader, total = len(train_loader), ncols = 0, leave = False, disable = not verbose)
        for batch_x in train_bar:
            batch_x, batch_y, batch_t = _random_interpolation(batch_x, distribution_behaviors, device)
            pred_y = model(batch_x,batch_t)
            loss = _get_batch_loss(batch_y,pred_y,distribution_behaviors)
            
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            batch_train_loss.extend(list(loss.detach().cpu().numpy().flatten()))
            train_bar.set_description(f"Traning train_loss={sum(batch_train_loss)/len(batch_train_loss):.4f}")

        epoch_train_loss.append(sum(batch_train_loss)/len(batch_train_loss))

        model.eval()
        batch_valid_loss = []
        valid_bar = tqdm.tqdm(valid_loader, total = len(valid_loader), ncols = 0, leave = False, disable = not verbose)
        for batch_x in valid_bar:
            batch_x, batch_y, batch_t = _random_interpolation(batch_x, distribution_behaviors, device)
            pred_y = model(batch_x,batch_t)
            loss = _get_batch_loss(batch_y,pred_y,distribution_behaviors)
            batch_valid_loss.extend(list(loss.detach().cpu().numpy().flatten()))
            valid_bar.set_description(f"Validating: valid_loss={sum(batch_valid_loss)/len(batch_valid_loss):.4f}")

        epoch_valid_loss.append(sum(batch_valid_loss)/len(batch_valid_loss))
        epoch_bar.set_description(f"Epoch: train_loss={epoch_train_loss[-1]:.4f}, valid_loss={epoch_valid_loss[-1]:.4f}")

        if e % save_period == save_period - 1 and checkpoint_save_root is not None:
            if not os.path.exists(checkpoint_save_root):
                os.mkdir(checkpoint_save_root)
            torch.save(model.state_dict(),os.path.join(checkpoint_save_root,"model.pt"))
            torch.save(optimizer.state_dict(),os.path.join(checkpoint_save_root,"optimizer.pt"))

def sample(
    model:Module,
    distribution_types:List[str], 
    inputs:List[torch.Tensor],
    t_init:List[float],
    steps:int,
    direction: str,
    device:str = "cpu"
) -> Iterator[List[torch.Tensor]]:

    assert direction in ['forward', 'backward'], "direction must be either 'forward' or 'backward'"
    batch_size = inputs[0].size()[0]
    assert all([batch_size == i.size()[0] for i in inputs]), "All inputs must have the same batch size"

    distribution_behaviors = [get_distribution_behavior(dt) for dt in distribution_types]

    model.eval()
    xt_list = [i.to(device) for i in inputs]
    with torch.no_grad():
        for s in range(steps):
            t_list = (s / steps) * torch.ones((len(inputs), batch_size), device=device, dtype=torch.float32)
            t_list = list(torch.split(t_list,1,dim=0))

            if direction == 'forward':
                t_list = [ti + (1.0 - ti) * t for ti, t in zip(t_init, t_list)]
                pred_diff = model(xt_list, t_list)
                xt_list = [xt + (1.0 - ti) * (1.0 / steps) * behavior.output_activation(y) for xt, y, behavior, ti in zip(xt_list, pred_diff, distribution_behaviors, t_init)]
            else:  # direction == 'backward'
                t_list = [ti - ti * t for ti, t in zip(t_init, t_list)]
                pred_diff = model(xt_list, t_list)
                xt_list = [xt - ti * (1.0 / steps) * behavior.output_activation(y) for xt, y, behavior, ti in zip(xt_list, pred_diff, distribution_behaviors, t_init)]
            
            yield xt_list

def _random_interpolation(x0: List[torch.Tensor], distribution_behaviors:List[BaseDistributionBehavior], device) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    batch_size = x0[0].size()[0]
    assert all([batch_size == i.size()[0] for i in x0]), "All inputs must have the same batch size"
    
    t_list = torch.rand(size=(len(x0),batch_size), device=device)
    t_list = [t.squeeze(0) for t in torch.split(t_list,1,dim=0)]

    xt = []
    diff = []
    for i, t, behavior in zip(x0, t_list, distribution_behaviors):
        i = i.to(device)
        x1 = behavior.generate_noise(i.size(), device=device)
        t_reshape = t.reshape([-1] + [1] * (i.dim() - 1))
        xt.append((1 - t_reshape) * i + t_reshape * x1)
        diff.append(x1 - i)

    return xt, diff, t_list

def _get_batch_loss(y_true:List[torch.Tensor], y_pred:List[torch.Tensor], distribution_behaviors:List[BaseDistributionBehavior]) -> torch.Tensor:
    losses = [behavior.loss(y_t, y_p).mean(dim=list(range(1, len(y_t.size())))) for y_t, y_p, behavior in zip(y_true, y_pred, distribution_behaviors)]
    return torch.sum(torch.stack(losses,dim=1),dim=1)