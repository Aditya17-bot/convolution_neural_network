import torch
from model import Net
import pytest


def test_parameter_count():
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20_000, f"Model has {total_params} parameters, should be less than 20k"

def test_batch_norm_usage():
    model = Net()
    has_batch_norm = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_batch_norm, "Model should use Batch Normalization"

def test_dropout_usage():
    model = Net()
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should use Dropout"

def test_gap_usage():
    model = Net()
    has_gap = any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules())
    assert has_gap, "Model should use Global Average Pooling"

def test_no_linear_layer():
    model = Net()
    has_linear = any(isinstance(m, torch.nn.Linear) for m in model.modules())
    assert not has_linear, "Model should not use Fully Connected (Linear) layers" 