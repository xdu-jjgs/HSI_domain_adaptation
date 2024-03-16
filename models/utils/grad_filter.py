import torch


def cal_grad_scores(features, out, task_ind: int, use_abs: bool):
    assert task_ind in [1, 2]  # 1 for source domain, 2 for target domain
    out_class = out[:, task_ind - 1].sum()
    grads = torch.autograd.grad(out_class, features, retain_graph=True)[0]
    scores = torch.mul(grads, features)
    scores = scores.squeeze()
    if use_abs:
        scores = torch.abs(scores)
    # print(grads)
    return scores


def mask_channels(features, scores, filter_num: int, largest=True):
    mask_ds = torch.ones_like(features).to(features.device)
    row = torch.arange(mask_ds.size()[0]).unsqueeze(1).to(mask_ds.device)
    _, index_ds_channels = scores.topk(filter_num, largest=largest)
    mask_ds[row, index_ds_channels] = 0.
    masked_ds_shared_features = features * mask_ds
    return masked_ds_shared_features
