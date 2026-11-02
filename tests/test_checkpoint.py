import torch

from rest import ReST, ReSTConfig
from rest.demo import synthetic_inputs
from rest.predict import load_checkpoint


def test_checkpoint_roundtrip_preserves_forecast(tmp_path):
    torch.manual_seed(211)
    config = ReSTConfig()
    model = ReST(15, 5, config).eval()
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "normalization": {"minimum": [0.0] * 20, "maximum": [1.0] * 20},
        },
        path,
    )
    restored, normalizer = load_checkpoint(path)
    inputs = synthetic_inputs(2)
    with torch.no_grad():
        torch.testing.assert_close(
            model(inputs).prediction, restored(inputs).prediction, rtol=0, atol=0
        )
    assert normalizer.state["maximum"] == [1.0] * 20
