*2 code files for checking packet loss percentage in sensors.*

*Single-file analysis (FIRMAWARE.py)*
Point the function at one F4D battery CSV and it returns five interactive Plotly charts:

1.Hourly packet-loss heatmap (per sensor) — color map of % packet loss by hour × sensor.

2.Hourly packet-loss time series (per sensor) — line chart of % packet loss over time for each sensor.

3.Packet loss vs. battery level (per hour) — scatter of % packet loss against the hourly average battery voltage (mV) for each sensor–hour.

4.Per-sensor packet-loss box plot — distribution of hourly % packet loss for each sensor (with outliers).

5.Overall packet-loss distribution — histogram of each sensor’s overall % packet loss across the run.


Config options:
1.Trim incomplete edge hours: If the first/last hour has fewer than the expected readings (default 20 for a 3-minute cadence), you can drop them via drop_edge_hours=True.
2.Sensor filtering: Exclude known-bad sensors entirely (sensors_to_drop=[...]) or keep them but ignore their values (sensors_to_nan=[...]).



*Folder-level firmware summary (Firmware_Summary.py)*

Scan a folder of F4D battery CSVs (one file = one experiment) and compare experiments with 3 interactive plots plus compact stats tables. Call: figs, dfs = build_all(folder=...).

Plots:
1.Distributions per experiment — stacked histograms of overall packet-loss % per sensor; subplot titles include sensor count and run hours; consistent colors per experiment.
2.Box plot by experiment — same colors; optional point overlay (points='outliers' | 'all' | 'suspectedoutliers' | False).
3.Sensors count vs. loss (scatter) — X = number of sensors in the experiment (all sensors of that experiment share the same X); Y = packet-loss % per sensor; black diamond = experiment mean; optional linear fit y = ax + b with R² (by default the fit uses experiment means; set fit_on="all" to fit all points).

Statistics:
1.describe — per-experiment summary (mean, SD, quartiles, median).
2.group — means, SD, CLD letters (A/B/…: groups sharing a letter are not significantly different), average |Hedges g|; method auto-selects Games–Howell (unequal variances) or Tukey–Kramer.
3.pairwise — experiment-pair differences with adjusted p-values and Hedges g magnitude.
4.welch — Welch ANOVA across experiments.

Config options:
1.Trim incomplete edge hours: If the first/last hour has fewer than the expected readings (default 20 for a 3-minute cadence), you can drop them via drop_edge_hours=True.
2.Sensor filtering: Exclude known-bad sensors entirely (sensors_to_drop=[...]) or keep them but ignore their values (sensors_to_nan=[...]).


