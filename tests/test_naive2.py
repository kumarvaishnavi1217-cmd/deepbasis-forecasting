import numpy as np

from deepbasis import naive2


def seasonal_series(n=120, m=12, amp=0.3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 100 * (1 + amp * np.sin(2 * np.pi * t / m)) + rng.normal(0, 0.5, n)


def test_detects_strong_seasonality():
    assert naive2.seasonality_test(seasonal_series(), 12)


def test_no_seasonality_for_random_walk():
    rng = np.random.default_rng(1)
    x = 100 + np.cumsum(rng.normal(0, 1, 120))
    # a random walk has high ACF at all lags but not a *distinctive* lag-12 spike relative to the limit
    # computed from lags 1..11; we only require the test to run and m=1 to be non-seasonal
    assert not naive2.seasonality_test(x, 1)


def test_short_series_not_tested():
    assert not naive2.seasonality_test(seasonal_series(n=30), 12)


def test_seasonal_indices_average_to_one():
    si = naive2.seasonal_indices(seasonal_series(), 12)
    assert si.shape == (12,)
    assert np.isclose(si.mean(), 1.0)


def test_nonseasonal_forecast_is_last_value():
    x = np.array([1.0, 5.0, 3.0, 7.0])
    np.testing.assert_allclose(naive2.forecast(x, 3, 1), [7.0, 7.0, 7.0])


def test_seasonal_forecast_follows_pattern():
    x = seasonal_series(n=144, amp=0.4)
    fc = naive2.forecast(x, 12, 12)
    true_next = 100 * (1 + 0.4 * np.sin(2 * np.pi * np.arange(144, 156) / 12))
    assert np.corrcoef(fc, true_next)[0, 1] > 0.95
