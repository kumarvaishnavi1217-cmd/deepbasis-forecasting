import numpy as np

from deepbasis.sampler import WindowSampler, last_window


def test_windows_come_from_recent_history_and_are_masked():
    series = [np.arange(1, 101, dtype=np.float32), np.arange(1, 6, dtype=np.float32)]
    s = WindowSampler(series, input_size=12, horizon=6, history_size=1.5, batch_size=256, seed=0)
    x, xm, y, ym = s.sample()
    assert x.shape == (256, 12) and y.shape == (256, 6)
    # zeros exactly where masked
    assert np.all(x[xm == 0] == 0) and np.all(y[ym == 0] == 0)
    # anchors lie within the last L_H*H = 9 points of the long series: target start >= 100-9+1
    long_rows = xm.sum(1) == 12
    first_target = y[long_rows, 0]
    assert first_target.min() >= 92 and first_target.max() <= 100
    # input is contiguous and immediately precedes the target
    assert np.all(x[long_rows, -1] + 1 == y[long_rows, 0])


def test_last_window_left_pads_short_series():
    x, m = last_window([np.array([5.0, 6.0], dtype=np.float32)], 4)
    np.testing.assert_array_equal(x, [[0, 0, 5, 6]])
    np.testing.assert_array_equal(m, [[0, 0, 1, 1]])
