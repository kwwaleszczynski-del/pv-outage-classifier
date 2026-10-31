"""Outage chronology of the PVDAQ site facts on a small hand-made inverter."""
import numpy as np
import pandas as pd

from pv_outage.site_facts import chronology


def test_chronology_stop_logged_zeros_gap_and_return():
    """Stop at 06:00 on day 1, zeros with code 0 until a code on day 3, inverter data missing from day 4 noon,
    output back on day 6 at 07:00."""
    t = pd.date_range("2016-06-01", periods=7 * 96, freq="15min")
    hour = t.hour + t.minute / 60
    poa = pd.Series(np.clip(1000 * np.sin(np.pi * (hour - 5) / 14), 0, None), index=t)
    p = 0.12 * poa
    code = pd.Series(0.0, index=t)
    stop, gap, back = pd.Timestamp("2016-06-02 06:00"), pd.Timestamp("2016-06-05 12:00"), pd.Timestamp("2016-06-07 07:00")
    p[(t >= stop) & (t < back)] = 0.0
    p[(t >= gap) & (t < back)] = np.nan
    code[(t >= pd.Timestamp("2016-06-04 07:30")) & (t < back)] = 139.0
    c = chronology(p, poa, code, pdc_kw=146.6)
    assert c["last_output"] == "2016-06-02 05:45:00"
    assert c["stop_in_sun"] == "2016-06-02 06:00:00"
    assert c["inverter_data_missing_from"] == "2016-06-05 12:00:00"
    assert c["zero_output_logged_days"] == 3.25
    assert c["first_error_code"] == {"time": "2016-06-04 07:30:00", "code": 139}
    assert c["first_error_code_after_stop_days"] == 2.06
    assert c["output_back"] == "2016-06-07 07:00:00"
    assert c["error_codes"] == [139]


def test_chronology_gap_without_logged_stop():
    """Inverter data missing from the night on, no zero in sun: the stop time stays unknown."""
    t = pd.date_range("2017-08-20", periods=5 * 96, freq="15min")
    hour = t.hour + t.minute / 60
    poa = pd.Series(np.clip(1000 * np.sin(np.pi * (hour - 5) / 14), 0, None), index=t)
    p = 0.05 * poa
    code = pd.Series(0.0, index=t)
    gap, back = pd.Timestamp("2017-08-22 02:00"), pd.Timestamp("2017-08-24 08:00")
    p[(t >= gap) & (t < back)] = np.nan
    code[(t >= gap) & (t < back)] = 139.0
    c = chronology(p, poa, code, pdc_kw=68.5)
    assert c["stop_in_sun"] is None and c["zero_output_logged_days"] is None
    assert c["inverter_data_missing_from"] == "2017-08-22 02:00:00"
    assert c["output_back"] == "2017-08-24 08:00:00"
