# PVDAQ Las Vegas campus: site notes

Several repositories use the NREL PVDAQ rooftop systems of one school campus in Las Vegas
([PVDAQ](https://doi.org/10.25984/1846021), CC BY 4.0): building A (system 34), B (1276), D (1278, inverters D1 and
D2) and the gymnasium G (35). This note is the one place for the facts they share. Every number comes from
`python -m pv_outage site-facts` (`src/pv_outage/site_facts.py`), which reads the raw daily files and writes
`results/pvdaq_site_facts.json`. Times are local standard time (UTC-8), as logged.

Used by [pv-outage-classifier](README.md),
[pv-budget-bridge](https://github.com/kwwaleszczynski-del/pv-budget-bridge),
[pv-anomaly-autoencoder](https://github.com/kwwaleszczynski-del/pv-anomaly-autoencoder),
[pv-thermal-check](https://github.com/kwwaleszczynski-del/pv-thermal-check),
[pv-shade-fingerprint](https://github.com/kwwaleszczynski-del/pv-shade-fingerprint),
[grafana-fleet-as-code](https://github.com/kwwaleszczynski-del/grafana-fleet-as-code) and
[res-uplift-verification](https://github.com/kwwaleszczynski-del/res-uplift-verification).

## The inverter outages

The two logged outages follow one pattern. The inverter stops, the logger keeps writing zero power and the
open-circuit DC voltage with error code 0, a code arrives days later, and then the inverter channels (AC and DC
power, current, voltage, inverter temperature) go missing while the met channels and an error code 139 keep
logging, until output returns. Building B in 2017 shows only the second half.

| | Building A, June-July 2016 | Inverter D1, October-December 2017 | Building B, August-September 2017 |
|---|---|---|---|
| Last output | 4 Jun 05:45 | 16 Oct 16:45 | 23 Aug 17:30 |
| Stop with sun on the array | 4 Jun 06:00 | 17 Oct 07:00 | none logged |
| Zero output logged | 12.24 days | 23.16 days | - |
| First error code | 160 on 14 Jun 07:28, 10.06 days after the stop | 160 on 30 Oct 09:02, 13.09 days after the stop | 139 on 24 Aug 05:15, with the gap |
| Inverter data missing | from 16 Jun 11:45, 21.97 days | from 9 Nov 10:45, 26.00 days | from 24 Aug 05:15, 32.09 days |
| Error codes seen | 160, 139 | 160, 139, 141, 142 | 139, 160 |
| Output back | 8 Jul 11:00 | 5 Dec 10:45 | 25 Sep 07:30 |
| Days without output | 34.22 | 49.75 | 32.58 |

How the repositories describe them:

- **Building A** is one 34-day outage. The 12 days of logged zeros are the open-circuit window that
  pv-thermal-check uses as a thermometer, the 10 days are the delay of the first error code that
  pv-anomaly-autoencoder and grafana-fleet-as-code compare their detectors with, and the 34 days run to the return
  of output (res-uplift-verification). Detectors that count from the first step above a fixed irradiance (for
  example 300 W/m2) start at 06:30.
- **Inverter D1** is one 49-day outage: about three weeks of logged zeros, which the detectors see, then 26 days
  without inverter data.
- **Building B in 2017 is a gap, not a logged stop.** The inverter data disappear before sunrise and come back
  32 days later; nothing says whether the roof produced in between. The codes are the ones that accompany missing
  inverter data in the two outages above, so an outage is plausible but not shown. pv-outage-classifier counts the
  period as a communication loss, pv-budget-bridge leaves missing data out of availability, and
  res-uplift-verification labels it a data gap.
- **"Not logged".** PVDAQ has no maintenance or outage log. The inverter error code is the only status signal, and
  it came 10 and 13 days after the two logged stops. Where a repository calls an outage "unlogged", it means
  that no outage record exists and the error code is not read.

## Timestamps: the channels are not aligned the same way

On clear days (236 days of 2016-2017), each channel of system 34 was fitted to a clear-sky plane-of-array curve
(azimuth 180 deg) shifted in time. The shift is the minutes to add to the stamp to get the instant the value
represents: a 15-minute mean stamped at the end of its interval gives -7.5, at the start +7.5, a centred mean or
an instantaneous sample 0.

| Channel | Plane | Shift, median (interquartile range) |
|---|---|---|
| Met-station POA irradiance | 5 deg tilt | -5.4 min (-6.2 to -4.6) |
| AC power of system 34 | 11.2 deg tilt | +3.8 min (+1.5 to +5.2) |
| AC power against the measured POA, no model | - | +9.2 min (+7.8 to +10.7) |

- **The irradiance channel behaves like a mean stamped at the end of its interval.** pv-budget-bridge reaches the
  same conclusion with its own clock check on this channel: with `data.label: end` the residual is -0.9 min on
  all three systems, against -0.7 min on its synthetic plant, whose labels are known.
- **AC power sits within a few minutes of its stamp**: +3.8 min here, -1.2 min with the sun-path fit of
  pv-shade-fingerprint, which works on AC power alone. It is not stamped like the irradiance.
- **The two channels are about 9 minutes apart** (the model-free row). An orientation error of a few degrees moves
  each absolute shift by minutes, so this offset between two channels logged side by side is the firmest number.
  It would vanish only if the sensor faced well east of the array.
- **What it changes.** Monthly and daily energy, PR and availability: nothing measurable. Quarter-hour comparisons
  of power with irradiance (a PR per interval, an expected-power residual): a lag of about 9 minutes, which shows
  as a morning-evening asymmetry.

## Other quirks

- The power channels are in hectowatts until 4 August 2018 and in watts afterwards (see `src/pv_outage/real.py`).
- The irradiance sensor's level, the stuck ambient temperature sensor and the anemometer are described in
  pv-budget-bridge and pv-thermal-check.
