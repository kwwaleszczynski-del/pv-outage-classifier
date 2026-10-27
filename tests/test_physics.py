import numpy as np
import pytest

from pv_outage import physics


@pytest.fixture(scope="module")
def module_table():
    m = physics.Module.from_cec()
    return m, physics.MppTable(m)


def test_detailed_path_matches_lookup_for_uniform_strings(module_table):
    m, tab = module_table
    rng = np.random.default_rng(0)
    g, tc = rng.uniform(100, 1100, 500), rng.uniform(-5, 60, 500)
    i_mp, v_mp, _ = tab.lookup(g, tc)
    cur, volt = physics.string_iv(m, g, g * 0.2, tc, 78, np.zeros(500), np.zeros(500))
    v_op, i_op, p_mpp = physics.operating_points([(cur, volt), (cur, volt)], np.full(500, np.nan), np.ones(500))
    analytic = 2 * 78 * i_mp * v_mp
    assert np.max(np.abs(p_mpp / analytic - 1)) < 1e-3
    assert np.max(np.abs(v_op / (78 * v_mp) - 1)) < 5e-3


def test_fault_signatures(module_table):
    m, _ = module_table
    n = 50
    g, tc = np.full(n, 900.0), np.full(n, 40.0)
    ok = physics.string_iv(m, g, g * 0.2, tc, 78, np.zeros(n), np.zeros(n))
    byp = physics.string_iv(m, g, g, tc, 78, np.zeros(n), np.full(n, 3))
    shade = physics.string_iv(m, g, g * 0.2, tc, 78, np.full(n, 9), np.zeros(n))
    nan, one = np.full(n, np.nan), np.ones(n)
    _, _, p_ok = physics.operating_points([ok, ok], nan, one)
    _, _, p_byp = physics.operating_points([byp, ok], nan, one)
    _, _, p_sh = physics.operating_points([shade, ok], nan, one)
    assert 0.005 < 1 - p_byp[0] / p_ok[0] < 0.04  # 3 of 78 substrings shorted, string paired with a healthy one
    assert 1 - p_sh[0] / p_ok[0] > 0.04  # nine shaded substrings cost more than their share
    # a tracker stuck at 75 % of the MPP voltage: more current, less power
    v, i, p = physics.operating_points([ok, ok], np.full(n, 0.75), one)
    v0, i0, _ = physics.operating_points([ok, ok], nan, one)
    assert i[0][0] > i0[0][0]
    assert v[0] * (i[0][0] + i[1][0]) < 0.9 * p_ok[0]
    # shedding to 70 %: voltage above the MPP, power on target
    v7, i7, _ = physics.operating_points([ok, ok], nan, np.full(n, 0.7))
    assert np.allclose(v7 * (i7[0] + i7[1]) / p_ok, 0.7, atol=0.01)
    assert np.all(v7 > v0)


def test_inverter_efficiency_inverse():
    pdc = np.array([10.0, 50.0, 90.0])
    pac = physics.pvwatts_ac(pdc, 100.0)
    assert np.allclose(physics.dc_for_ac(pac, 100.0), pdc, rtol=1e-3)
    assert physics.pvwatts_ac([500.0], 100.0)[0] == pytest.approx(100.0)
