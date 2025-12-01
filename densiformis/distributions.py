import torch

from .functions import (
    log_cosh,
    symmetric_softplus_loss,
    symmetric_sigmoid,
    symmetric_softmax,
    rand_soft_label,
    symmetric_logsumexp_loss,
)

DISTRIBUTION_BEHAVIORS = {}

def register_distribution_behavior(key: str):
    def decorator(cls):
        DISTRIBUTION_BEHAVIORS[key] = cls()
        return cls
    return decorator

class BaseDistributionBehavior:
    def generate_noise(self, size, device = "cpu"):
        raise NotImplementedError

    def output_activation(self, x):
        raise NotImplementedError

    def loss(self, y_true, y_pred):
        raise NotImplementedError

@register_distribution_behavior("numerical")
class NumericalDistribution(BaseDistributionBehavior):
    def generate_noise(self, size, device = "cpu"):
        return torch.randn(size, device=device)

    def output_activation(self, x):
        return x

    def loss(self, y_true, y_pred):
        return log_cosh(y_true, y_pred)

@register_distribution_behavior("binary")
class BinaryDistribution(BaseDistributionBehavior):
    def generate_noise(self, size, device = "cpu"):
        return torch.rand(size, device=device)

    def output_activation(self, x):
        return symmetric_sigmoid(x)

    def loss(self, y_true, y_pred):
        return symmetric_softplus_loss(y_true, y_pred)

@register_distribution_behavior("categorical")
class CategoricalDistribution(BaseDistributionBehavior):
    def generate_noise(self, size, device = "cpu"):
        return rand_soft_label(size, device=device)

    def output_activation(self, x):
        return symmetric_softmax(x)

    def loss(self, y_true, y_pred):
        return symmetric_logsumexp_loss(y_true, y_pred)

def get_distribution_behavior(key: str) -> BaseDistributionBehavior:
    behavior = DISTRIBUTION_BEHAVIORS.get(key)
    if behavior is None:
        raise ValueError(f"Unknown distribution type: {key}, available types: {list(DISTRIBUTION_BEHAVIORS.keys())}")
    return behavior
