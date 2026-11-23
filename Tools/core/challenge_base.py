"""Local CPU fixture factory for the small interface imported by the pinned catalog."""
import torch


class ChallengeBase:
    device = torch.device("cpu")


def _dtype(dtype):
    return getattr(torch, dtype) if isinstance(dtype, str) else dtype


def RandTensor(shape, low=0, high=1, dtype="float32"):
    return torch.empty(shape, dtype=_dtype(dtype)).uniform_(low, high)


def RandnTensor(shape, mean=0.0, std=1.0, dtype="float32"):
    return torch.empty(shape, dtype=_dtype(dtype)).normal_(mean, std)


def RandIntTensor(shape, low, high, dtype="int32"):
    return torch.randint(low, high, shape, dtype=_dtype(dtype))


def OutTensor(shape, dtype="float32"):
    return torch.empty(shape, dtype=_dtype(dtype))


def FullTensor(shape, value, dtype="float32"):
    return torch.full(shape, value, dtype=_dtype(dtype))
