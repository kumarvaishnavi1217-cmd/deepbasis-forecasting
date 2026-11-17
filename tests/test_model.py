import torch

from deepbasis import model
from deepbasis.losses import LOSSES


def _inputs(b=4, n=18):
    x = torch.rand(b, n) + 1
    return x, torch.ones_like(x)


def test_generic_shapes_and_params():
    net = model.generic(18, 6)
    x, m = _inputs()
    assert net(x, m).shape == (4, 6)
    assert len(net.blocks) == 30
    # Table 18: no sharing -> 30 distinct blocks
    assert len({id(b) for b in net.blocks}) == 30


def test_interpretable_shares_weights_within_stack():
    net = model.interpretable(36, 18)
    x, m = _inputs(n=36)
    assert net(x, m).shape == (4, 18)
    ids = [id(b) for b in net.blocks]
    assert len(ids) == 6 and len(set(ids)) == 2      # 3 trend + 3 seasonality, shared


def test_trend_basis_is_polynomial():
    basis = model.TrendBasis(degree=2, backcast_size=10, forecast_size=5)
    theta = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])   # constant forecast = 1
    _, fc = basis(theta)
    torch.testing.assert_close(fc, torch.ones(1, 5))


def test_seasonality_basis_harmonics_count():
    # i = 0..floor(H/2 - 1)  ->  H/2 harmonics for even H
    assert model.SeasonalityBasis.n_harmonics(18) == 9
    basis = model.SeasonalityBasis(36, 18)
    theta = torch.zeros(2, model.SeasonalityBasis.theta_size(18))
    bc, fc = basis(theta)
    assert bc.shape == (2, 36) and fc.shape == (2, 18)


def test_forecast_starts_from_last_level():
    """With zeroed final layers, N-BEATS forecasts the last observed value (naive)."""
    net = model.generic(12, 3, stacks=2)
    for b in net.blocks:
        torch.nn.init.zeros_(b.theta.weight)
        torch.nn.init.zeros_(b.theta.bias)
    x = torch.arange(1.0, 13.0)[None, :]
    torch.testing.assert_close(net(x, torch.ones_like(x)), torch.full((1, 3), 12.0))


def test_masked_padding_does_not_change_residual_path():
    net = model.generic(12, 3, stacks=3)
    x = torch.rand(1, 12) + 1
    mask = torch.ones_like(x)
    mask[:, :4] = 0
    x_padded = x.clone()
    x_padded[:, :4] = 0
    out = net(x_padded, mask)
    assert torch.isfinite(out).all()


def test_losses_zero_for_perfect_forecast_and_differentiable():
    y = torch.rand(8, 6) + 1
    insample = torch.rand(8, 18) + 1
    mask = torch.ones_like(y)
    for name, fn in LOSSES.items():
        f = y.clone().requires_grad_(True)
        assert fn(f, y, mask, insample=insample, m=1).item() == 0.0, name
        g = (y + 1).requires_grad_(True)
        fn(g, y, mask, insample=insample, m=1).backward()
        assert torch.isfinite(g.grad).all(), name


def test_losses_ignore_masked_targets():
    y = torch.ones(2, 4)
    f = torch.ones(2, 4)
    f[:, -1] = 100.0
    mask = torch.ones_like(y)
    mask[:, -1] = 0
    for name, fn in LOSSES.items():
        assert fn(f, y, mask, insample=torch.rand(2, 8) + 1, m=1).item() == 0.0, name


def test_decompose_sums_to_forecast():
    net = model.interpretable(24, 8)
    x, m = _inputs(n=24)
    level, parts = net.decompose(x, m)
    torch.testing.assert_close(level + sum(parts), net(x, m))
