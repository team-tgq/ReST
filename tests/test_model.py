from dataclasses import replace

import pytest
import torch

from rest import ObservedHistory, ReST, ReSTConfig
from rest.demo import synthetic_inputs
from rest.train import masked_mae


@pytest.mark.parametrize("case", ["observed", "tail", "no_swe", "all_missing"])
def test_forecast_finite_for_missing_inputs(case):
    torch.manual_seed(52)
    model = ReST(15, 5, ReSTConfig()).eval()
    inputs = synthetic_inputs(3)
    histories = []
    for history in (inputs.short, inputs.long, inputs.analog):
        mask = history.observed.clone()
        if case == "tail":
            mask[:, -7:, :2] = 0
        elif case == "no_swe":
            mask[:, :, :2] = 0
        elif case == "all_missing":
            mask[:] = 0
        histories.append(ObservedHistory(history.normalized, history.raw, mask))
    inputs = replace(inputs, short=histories[0], long=histories[1], analog=histories[2])
    with torch.no_grad():
        output = model(inputs)
    assert output.prediction.shape == (3, 10)
    assert torch.isfinite(output.prediction).all()
    assert (output.prediction >= 0).all()
    assert ((output.router_weights > 0).sum(-1) == 2).all()
    torch.testing.assert_close(output.router_weights.sum(-1), torch.ones(3))
    torch.testing.assert_close(
        output.raw_prediction, output.base + output.trend + output.residual
    )


def test_masked_values_cannot_change_predictions():
    model = ReST(15, 5, ReSTConfig()).eval()
    inputs = synthetic_inputs(2)
    changed = []
    for h in (inputs.short, inputs.long, inputs.analog):
        raw = torch.where(h.observed > 0, h.raw, torch.full_like(h.raw, float("nan")))
        normalized = torch.where(
            h.observed > 0, h.normalized, torch.full_like(h.normalized, 1e12)
        )
        changed.append(ObservedHistory(normalized, raw, h.observed))
    with torch.no_grad():
        a = model(inputs).prediction
        b = model(
            replace(inputs, short=changed[0], long=changed[1], analog=changed[2])
        ).prediction
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_training_gradients_and_inference_batch_independence():
    torch.manual_seed(101)
    model = ReST(15, 5, ReSTConfig())
    inputs = synthetic_inputs(2)
    output = model(inputs)
    loss = masked_mae(
        output.raw_prediction, torch.full((2, 10), 100.0), torch.ones(2, 10)
    )
    (loss + output.auxiliary_loss + 0.01 * output.router_load_balance_loss).backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    model.eval()
    single = replace(
        inputs,
        short=ObservedHistory(
            *(
                t[:1]
                for t in (
                    inputs.short.normalized,
                    inputs.short.raw,
                    inputs.short.observed,
                )
            )
        ),
        long=ObservedHistory(
            *(
                t[:1]
                for t in (inputs.long.normalized, inputs.long.raw, inputs.long.observed)
            )
        ),
        analog=ObservedHistory(
            *(
                t[:1]
                for t in (
                    inputs.analog.normalized,
                    inputs.analog.raw,
                    inputs.analog.observed,
                )
            )
        ),
        static_topography=inputs.static_topography[:1],
        mechanism_prior=inputs.mechanism_prior[:1],
    )
    with torch.no_grad():
        torch.testing.assert_close(
            model(inputs).prediction[:1], model(single).prediction, rtol=1e-5, atol=1e-4
        )


def test_invalid_input_and_empty_supervision_rejected():
    model = ReST(15, 5, ReSTConfig())
    inputs = synthetic_inputs(2)
    with pytest.raises(ValueError):
        model(replace(inputs, mechanism_prior=torch.zeros(2, 4)))
    with pytest.raises(ValueError):
        ReSTConfig(feature_names=("swe_prev",))
    with pytest.raises(ValueError):
        masked_mae(torch.ones(1, 10), torch.ones(1, 10), torch.zeros(1, 10))
