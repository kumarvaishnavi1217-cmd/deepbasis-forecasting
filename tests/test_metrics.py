import numpy as np
import pytest

from deepbasis import metrics


def test_smape_perfect_forecast_is_zero():
    y = np.array([[1.0, 2.0, 3.0]])
    assert metrics.smape(y, y)[0] == 0.0


def test_smape_known_value():
    # |100-110| / (100+110) * 200 = 9.5238...
    assert metrics.smape(np.array([[100.0]]), np.array([[110.0]]))[0] == pytest.approx(200 * 10 / 210)


def test_smape_bounded_by_200():
    assert metrics.smape(np.array([[1.0]]), np.array([[-1.0]]))[0] == pytest.approx(200.0)


def test_smape_zero_zero_is_zero_not_nan():
    assert metrics.smape(np.array([[0.0]]), np.array([[0.0]]))[0] == 0.0


def test_mase_uses_seasonal_insample_scale():
    insample = np.array([1.0, 2.0, 3.0, 4.0, 5.0])   # seasonal naive (m=2) error = 2
    y = np.array([[10.0, 10.0]])
    yhat = np.array([[12.0, 8.0]])                     # MAE = 2
    assert metrics.mase([insample], y, yhat, m=2)[0] == pytest.approx(1.0)


def test_mase_short_series_falls_back_to_m1():
    assert metrics.mase_scale(np.array([1.0, 3.0]), m=12) == pytest.approx(2.0)


def test_owa_of_naive2_is_one():
    assert metrics.owa(14.2, 1.9, 14.2, 1.9) == pytest.approx(1.0)
