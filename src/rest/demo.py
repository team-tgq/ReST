"""A synthetic forward/backward example; its errors are not scientific results."""

import json

import torch

from . import ObservedHistory, ReST, ReSTConfig, ReSTInput
from .train import masked_mae


def synthetic_inputs(batch_size=4):
    histories = []
    for length in (30, 30, 75):
        raw = torch.randn(batch_size, length, 15)
        raw[..., 0] = (raw[..., 0] * 30 + 150).clamp_min(0)
        mask = (torch.rand_like(raw) > 0.1).float()
        raw = torch.where(mask > 0, raw, torch.zeros_like(raw))
        histories.append(ObservedHistory(raw / 200, raw, mask))
    static = torch.rand(batch_size, 5)
    prior = torch.softmax(torch.randn(batch_size, 5), dim=-1)
    return ReSTInput(*histories, static, prior)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(52)
    model = ReST(15, 5, ReSTConfig())
    inputs = synthetic_inputs()
    output = model(inputs)
    target = torch.full((4, 10), 150.0)
    loss = masked_mae(output.raw_prediction, target, torch.ones_like(target))
    loss = loss + 0.01 * output.router_load_balance_loss + output.auxiliary_loss
    loss.backward()
    model.eval()
    with torch.no_grad():
        prediction = model(inputs).prediction
    assert torch.isfinite(prediction).all() and (prediction >= 0).all()
    print(
        json.dumps(
            {
                "synthetic_demo": True,
                "prediction_shape": list(prediction.shape),
                "parameters": sum(p.numel() for p in model.parameters()),
                "backward_finite": all(
                    p.grad is None or torch.isfinite(p.grad).all().item()
                    for p in model.parameters()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
