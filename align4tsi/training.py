import math

import torch
from torch.nn import functional as F


def binary_mask(mask, shape, name):
    if not isinstance(mask, torch.Tensor) or mask.shape != shape:
        raise ValueError(f"{name} must be a tensor with shape {tuple(shape)}")
    if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError(f"{name} must be binary")
    return mask.bool()


def mask_observations(values, observation_mask, target_mask):
    """Remove artificial targets from conditioning, without changing the truth."""
    observed = binary_mask(observation_mask, values.shape, "observation_mask").to(values.device)
    target = binary_mask(target_mask, values.shape, "target_mask").to(values.device)
    if bool((target & ~observed).any()):
        raise ValueError("Targets must be a subset of originally observed positions")
    if not bool(torch.isfinite(values[observed]).all()):
        raise ValueError("Originally observed values must be finite")
    condition = observed & ~target
    return torch.where(condition, values, torch.zeros_like(values)), condition


def masked_huber_loss(prediction, truth, target_mask, valid_mask=None, delta=1.0):
    """Average Huber loss only over originally observed artificial targets."""
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must have the same shape")
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    target = binary_mask(target_mask, prediction.shape, "target_mask").to(prediction.device)
    if valid_mask is not None:
        valid = binary_mask(valid_mask, prediction.shape, "valid_mask").to(prediction.device)
        if bool((target & ~valid).any()):
            raise ValueError("Targets cannot include padding positions")
    if not bool(target.any()):
        raise ValueError("At least one valid artificial target is required")
    predicted, actual = prediction[target], truth[target]
    if not bool(torch.isfinite(predicted).all()) or not bool(torch.isfinite(actual).all()):
        raise ValueError("Target predictions and target truth must be finite")
    return F.huber_loss(predicted, actual, delta=delta, reduction="mean")


def cosine_learning_rate(epoch, initial_lr=1e-3, final_lr=1e-4, decay_epochs=400):
    """Decay from epoch1 to epoch400, then retain the minimum learning rate."""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("epoch must be a positive one-based integer")
    if isinstance(decay_epochs, bool) or not isinstance(decay_epochs, int) or decay_epochs < 1:
        raise ValueError("decay_epochs must be a positive integer")
    if not all(math.isfinite(x) for x in (initial_lr, final_lr)) or not 0 < final_lr <= initial_lr:
        raise ValueError("Require finite learning rates with 0 < final_lr <= initial_lr")
    progress = min((epoch - 1) / max(decay_epochs - 1, 1), 1.)
    if decay_epochs == 1:
        return final_lr
    return final_lr + .5 * (initial_lr - final_lr) * (1 + math.cos(math.pi * progress))
