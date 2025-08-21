# Firmware experiments summary (minimal, self-contained).
# Scans a folder of CSVs (one experiment per CSV), computes per-sensor packet-loss,
# plots: stacked histograms (per experiment), box plot, and scatter (X = sensors count, Y = loss) with AVG and linear fit,
# and returns data tables: describe per experiment, group summary (CLD letters, avg |g|), pairwise effects, Welch ANOVA.

import os, re, glob
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import itertools

# Optional: pingouin for Games–Howell + Welch ANOVA
try:
    import pingouin as pg
    _HAS_PINGOUIN = True
except Exception:
    _HAS_PINGOUIN = False


# ----------------------------- I/O + per-sensor loss -----------------------------

def parse_experiment_name(path: str) -> str:
    """exp_5_Firmware_4_battery_2025-08-19.csv -> exp_5_Firmware_4"""
    base = os.path.basename(path)
    m = re.search(r'^(.*?)_battery', base, flags=re.IGNORECASE)
    return m.group(1) if m else os.path.splitext(base)[0]


def overall_packet_loss_per_sensor(
    df: pd.DataFrame,
    *,
    freq: str = "3T",
    expected_per_hour: int = 20,
    drop_edge_hours: bool = True,
    sensors_to_drop: list[str] | None = None,
    sensors_to_nan:  list[str] | None = None,
) -> pd.DataFrame:
    """Compute overall packet loss (%) per sensor over the run window."""
    if "Timestamp" not in df.columns:
        raise ValueError("CSV must contain a 'Timestamp' column")

    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").set_index("Timestamp")

    # Optionally drop incomplete first/last hour
    if drop_edge_hours and not df.empty:
        _h = df.index.floor("H")
        hour_counts = pd.Series(1, index=_h).groupby(level=0).size().sort_index()
        first_h, last_h = hour_counts.index.min(), hour_counts.index.max()
        to_drop = [h for h in (first_h, last_h) if hour_counts.get(h, 0) < expected_per_hour]
        if to_drop:
            df = df[~df.index.floor("H").isin(to_drop)]

    sensor_cols = [c for c in df.columns if c.lower() != "timestamp"]
    if not sensor_cols or df.empty:
        return pd.DataFrame(columns=["sensor", "overall_packet_loss_pct"])

    # Apply sensor rules (case-insensitive)
    norm = lambda s: str(s).strip().lower()
    look = {norm(c): c for c in sensor_cols}

    if sensors_to_drop:
        drop_actual = [look.get(norm(s)) for s in sensors_to_drop if norm(s) in look]
        df = df.drop(columns=[c for c in drop_actual if c], errors="ignore")

    sensor_cols = [c for c in df.columns if c.lower() != "timestamp"]
    look = {norm(c): c for c in sensor_cols}

    if sensors_to_nan:
        for c in [look.get(norm(s)) for s in sensors_to_nan if norm(s) in look]:
            df[c] = np.nan

    if not sensor_cols or df.empty:
        return pd.DataFrame(columns=["sensor", "overall_packet_loss_pct"])

    df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors="coerce")

    # Expected vs actual packets on regular slots
    start = df.index.min().floor(freq)
    end   = df.index.max().ceil(freq)
    slots = pd.date_range(start, end, freq=freq)
    expected_total = len(slots)

    actual_total = df[sensor_cols].resample(freq).count().reindex(slots).fillna(0).sum(axis=0)

    out = pd.DataFrame({"sensor": actual_total.index.astype(str), "actual": actual_total.values})
    out["expected"] = expected_total
    out["overall_packet_loss_pct"] = (1 - out["actual"].clip(upper=expected_total) / out["expected"]) * 100
    return out[["sensor", "overall_packet_loss_pct"]]


def _resolve_sensor_rules_for_exp(
    experiment_name: str,
    *,
    sensors_to_drop: list[str] | None,
    sensors_to_nan:  list[str] | None,
    sensors_to_drop_by_experiment: dict[str, list[str]] | None,
    sensors_to_nan_by_experiment:  dict[str, list[str]] | None,
) -> tuple[list[str], list[str]]:
    """Merge global + per-experiment rules (substring or 're:<pattern>')."""
    exp = experiment_name or ""
    exp_l = exp.lower()

    drop = set(sensors_to_drop or [])
    nan  = set(sensors_to_nan  or [])

    def apply(rules, target):
        if not rules:
            return
        for key, sensors in rules.items():
            if key.startswith("re:"):
                if re.search(key[3:], exp, flags=re.IGNORECASE):
                    target.update(sensors)
            else:
                if key.lower() in exp_l:
                    target.update(sensors)

    apply(sensors_to_drop_by_experiment, drop)
    apply(sensors_to_nan_by_experiment,  nan)
    return sorted(drop), sorted(nan)


def _run_hours_excluding_edges(raw_df: pd.DataFrame) -> int:
    if "Timestamp" not in raw_df.columns:
        return 0
    t = pd.to_datetime(raw_df["Timestamp"], errors="coerce").dropna()
    if t.empty:
        return 0
    hours = t.dt.floor("H").unique()
    return int(max(0, len(hours) - 2))


# ----------------------------- Histograms (stacked) -----------------------------

def summarize_folder_distributions_px_vertical(
    folder: str,
    *,
    file_glob: str = "*.csv",
    freq: str = "3T",
    expected_per_hour: int = 20,
    drop_edge_hours: bool = True,
    sensors_to_drop: list[str] | None = None,
    sensors_to_nan:  list[str] | None = None,
    sensors_to_drop_by_experiment: dict[str, list[str]] | None = None,
    sensors_to_nan_by_experiment:  dict[str, list[str]] | None = None,
    bin_size: float = 1.0,
    x_tick_step: float | None = 5.0,
    colors: list[str] | None = None,
    show_legend: bool = False
):
    """Build vertical stack of histograms (one subplot per experiment)."""
    files = sorted(glob.glob(os.path.join(folder, file_glob)))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder} (pattern: {file_glob})")

    all_rows, meta_rows = [], []
    for fp in files:
        try:
            exp = parse_experiment_name(fp)
            raw = pd.read_csv(fp)
            run_hours = _run_hours_excluding_edges(raw)
            drop_for_this, nan_for_this = _resolve_sensor_rules_for_exp(
                exp,
                sensors_to_drop=sensors_to_drop, sensors_to_nan=sensors_to_nan,
                sensors_to_drop_by_experiment=sensors_to_drop_by_experiment,
                sensors_to_nan_by_experiment=sensors_to_nan_by_experiment,
            )
            per_sensor = overall_packet_loss_per_sensor(
                raw, freq=freq, expected_per_hour=expected_per_hour,
                drop_edge_hours=drop_edge_hours,
                sensors_to_drop=drop_for_this, sensors_to_nan=nan_for_this,
            )
            if per_sensor.empty:
                continue

            n_sensors = int(per_sensor.shape[0])
            per_sensor["experiment"] = exp
            per_sensor["file"] = os.path.basename(fp)
            per_sensor["sensors_count"] = n_sensors
            per_sensor["run_hours_excl_edges"] = run_hours
            all_rows.append(per_sensor)

            meta_rows.append({
                "experiment": exp,
                "file": os.path.basename(fp),
                "sensors_count": n_sensors,
                "run_hours_excl_edges": run_hours,
            })
        except Exception:
            pass

    if not all_rows:
        raise RuntimeError("No data produced from any file.")

    combined = pd.concat(all_rows, ignore_index=True)
    meta_df = pd.DataFrame(meta_rows).drop_duplicates().reset_index(drop=True)

    experiments = sorted(combined["experiment"].unique(), key=str)

    meta_indexed = meta_df.set_index("experiment")
    label_map = {
        e: f"{e} — {int(meta_indexed.loc[e, 'sensors_count'])} sensor"
           f"{'' if int(meta_indexed.loc[e, 'sensors_count'])==1 else 's'} • "
           f"{int(meta_indexed.loc[e, 'run_hours_excl_edges'])} h"
        for e in experiments if e in meta_indexed.index
    }

    palette = colors or px.colors.qualitative.Set2
    color_map = {exp: palette[i % len(palette)] for i, exp in enumerate(experiments)}

    fig = make_subplots(
        rows=len(experiments), cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=[label_map.get(e, e) for e in experiments]
    )

    for i, exp in enumerate(experiments, start=1):
        df_exp = combined[combined["experiment"] == exp]
        h = px.histogram(df_exp, x="overall_packet_loss_pct", nbins=None)
        trace = h.data[0]
        trace.name = exp
        trace.marker.color = color_map[exp]
        trace.opacity = 0.7
        trace.xbins = dict(start=0, end=100.0001, size=bin_size)
        fig.add_trace(trace, row=i, col=1)
        fig.update_yaxes(title_text="Count", row=i, col=1)

    xaxis_kwargs = dict(autorange=True, ticks="outside")
    if x_tick_step is not None:
        xaxis_kwargs["dtick"] = x_tick_step
    fig.update_xaxes(**xaxis_kwargs)

    fig.update_layout(
        template="plotly_white",
        title="Distribution of Overall Packet Loss per Sensor — by Experiment",
        bargap=0.05,
        showlegend=show_legend,
        height=220 * len(experiments) + 120,
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return combined, meta_df, fig


# ----------------------------- Stats (letters, effects) -----------------------------

def _hedges_g_pair(m1, s1, n1, m2, s2, n2, equal_var=False):
    if n1 < 2 or n2 < 2 or np.isnan(s1) or np.isnan(s2):
        return np.nan
    if equal_var:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * (s1**2) + (n2 - 1) * (s2**2)) / df
        sp = np.sqrt(sp2)
        if sp == 0: return np.nan
        d = (m1 - m2) / sp
    else:
        s_av = np.sqrt((s1**2 + s2**2) / 2.0)
        if s_av == 0: return np.nan
        d = (m1 - m2) / s_av
        df_num = (s1**2 / n1 + s2**2 / n2) ** 2
        df_den = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
        df = df_num / df_den if df_den > 0 else (n1 + n2 - 2)
    J = 1 - 3 / (4 * df - 1) if df and df > 3 else 1.0
    return J * d


def _label_magnitude(g_abs):
    if np.isnan(g_abs): return "NA"
    if g_abs < 0.2:  return "negligible"
    if g_abs < 0.5:  return "small"
    if g_abs < 0.8:  return "medium"
    if g_abs < 1.2:  return "large"
    return "very large"


def _cld_letters(pairs_df, group1_col="group1", group2_col="group2",
                 p_col=None, reject_col=None, alpha=0.05):
    """Compact Letter Display (A/B/C...)."""
    groups = sorted(set(pairs_df[group1_col]) | set(pairs_df[group2_col]))
    sig = {(min(a,b), max(a,b)): False for a,b in itertools.combinations(groups, 2)}
    for _, r in pairs_df.iterrows():
        a, b = r[group1_col], r[group2_col]
        if reject_col is not None:
            is_sig = bool(r[reject_col])
        else:
            is_sig = (float(r[p_col]) < alpha)
        sig[(min(a,b), max(a,b))] = is_sig
    letter_map, remaining, letter_ord = {g: set() for g in groups}, set(groups), ord('A')
    while remaining:
        bucket = []
        for g in list(remaining):
            if all(sig.get((min(g,h), max(g,h)), False) is False for h in bucket):
                bucket.append(g)
        for g in bucket:
            letter_map[g].add(chr(letter_ord))
            remaining.discard(g)
        letter_ord += 1
    return {g: "".join(sorted(v)) for g, v in letter_map.items()}


def make_summary_tables(
    combined_df: pd.DataFrame,
    *,
    value_col: str = "overall_packet_loss_pct",
    group_col: str = "experiment",
    alpha: float = 0.05,
):
    """Return (group_summary_df, pairwise_df)."""
    df = combined_df[[group_col, value_col]].dropna().copy()
    gstats = df.groupby(group_col)[value_col].agg(['count', 'mean', 'std']).reset_index().rename(columns={'count': 'n'})
    groups_vals = [g[value_col].values for _, g in df.groupby(group_col)]
    levene_p = np.nan
    if len(groups_vals) >= 2 and all(len(g) > 1 for g in groups_vals):
        _, levene_p = stats.levene(*groups_vals, center="median")
    use_GH = _HAS_PINGOUIN and (not np.isnan(levene_p)) and (levene_p < alpha)

    if use_GH:
        gh = pg.pairwise_gameshowell(dv=value_col, between=group_col, data=df)
        gh = gh.rename(columns={"A":"group1","B":"group2","pval":"p_value","hedges":"g"})
        gh["abs_g"] = gh["g"].abs()
        gh["magnitude"] = gh["abs_g"].apply(_label_magnitude)
        gh["method"] = "Games-Howell"
        letters = _cld_letters(gh, p_col="p_value", alpha=alpha)
        pairwise_df = gh[["group1","group2","mean(A)","mean(B)","diff","p_value","g","abs_g","magnitude","method"]]
        pairwise_df = pairwise_df.rename(columns={"mean(A)":"mean1","mean(B)":"mean2","diff":"mean_diff"})
        method_str = "Games-Howell"
    else:
        tk = pairwise_tukeyhsd(endog=df[value_col], groups=df[group_col], alpha=alpha)
        tk_df = pd.DataFrame(
            tk._results_table.data[1:],
            columns=[c.lower().replace(" ", "_") for c in tk._results_table.data[0]]
        ).rename(columns={"p-adj":"p_value","meandiff":"mean_diff"})
        stats_map = gstats.set_index(group_col).to_dict(orient="index")
        g_list, abs_g_list, mag_list, m1_list, m2_list = [], [], [], [], []
        for _, r in tk_df.iterrows():
            a, b = r["group1"], r["group2"]
            s1, s2 = stats_map[a], stats_map[b]
            g_val = _hedges_g_pair(s1["mean"], s1["std"], int(s1["n"]),
                                   s2["mean"], s2["std"], int(s2["n"]),
                                   equal_var=True)
            g_list.append(g_val)
            abs_g_list.append(abs(g_val) if not np.isnan(g_val) else np.nan)
            mag_list.append(_label_magnitude(abs_g_list[-1]))
            m1_list.append(s1["mean"]); m2_list.append(s2["mean"])
        tk_df["mean1"], tk_df["mean2"] = m1_list, m2_list
        tk_df["g"], tk_df["abs_g"], tk_df["magnitude"] = g_list, abs_g_list, mag_list
        tk_df["method"] = "Tukey-Kramer"
        letters = _cld_letters(tk_df, reject_col="reject", alpha=alpha)
        pairwise_df = tk_df[["group1","group2","mean1","mean2","mean_diff","p_value","g","abs_g","magnitude","method"]]
        method_str = "Tukey-Kramer"

    # average |g| per group
    av_abs_g = {}
    for gname in gstats[group_col]:
        mask = (pairwise_df["group1"] == gname) | (pairwise_df["group2"] == gname)
        vals = pairwise_df.loc[mask, "abs_g"].dropna().values
        av_abs_g[gname] = float(np.mean(vals)) if len(vals) else np.nan

    group_summary_df = gstats.copy()
    group_summary_df["letters"] = group_summary_df[group_col].map(letters)
    group_summary_df["avg_|g|"] = group_summary_df[group_col].map(av_abs_g)
    group_summary_df["method"] = method_str
    group_summary_df["levene_p"] = levene_p
    group_summary_df = group_summary_df.sort_values("mean", ascending=True).reset_index(drop=True)
    return group_summary_df, pairwise_df.reset_index(drop=True)


def welch_anova_df(
    combined_df: pd.DataFrame,
    *,
    value_col: str = "overall_packet_loss_pct",
    group_col: str = "experiment",
) -> pd.DataFrame:
    """Welch ANOVA summary (one row)."""
    if _HAS_PINGOUIN:
        res = pg.welch_anova(dv=value_col, between=group_col, data=combined_df.dropna(subset=[value_col, group_col]))
        return pd.DataFrame({
            "F": [float(res.loc[0, "F"])],
            "df_num": [float(res.loc[0, "ddof1"])],
            "df_denom": [float(res.loc[0, "ddof2"])],
            "p_value": [float(res.loc[0, "p-unc"])],
            "method": ["Welch ANOVA (pingouin)"],
        })
    else:
        try:
            import statsmodels.stats.oneway as sm_oneway
            groups = [g[value_col].dropna().values for _, g in combined_df.groupby(group_col)]
            res = sm_oneway.anova_oneway(groups, use_var='unequal', welch_correction=True)
            return pd.DataFrame({
                "F": [float(res.statistic)],
                "df_num": [float(res.df_num)],
                "df_denom": [float(res.df_denom)],
                "p_value": [float(res.pvalue)],
                "method": ["Welch ANOVA (statsmodels)"],
            })
        except Exception:
            return pd.DataFrame({"F":[np.nan],"df_num":[np.nan],"df_denom":[np.nan],"p_value":[np.nan],"method":["Welch ANOVA (unavailable)"]})


def describe_per_experiment_df(
    combined: pd.DataFrame,
    *,
    value_col: str = "overall_packet_loss_pct",
    group_col: str = "experiment"
) -> pd.DataFrame:
    """Tidy describe() per experiment with a 'median' column."""
    return (
        combined.groupby(group_col)[value_col]
        .describe()
        .rename(columns={"50%": "median"})
        .reset_index()
    )


# ----------------------------- Box & Scatter (with fit) -----------------------------

def make_boxplot_by_experiment(
    combined_df: pd.DataFrame,
    *,
    value_col: str = "overall_packet_loss_pct",
    group_col: str = "experiment",
    order: list[str] | None = None,
    points: str | bool = "outliers",
    color_map: dict | None = None,
    tick_labels: dict[str, str] | None = None,
):
    df = combined_df[[group_col, value_col]].dropna().copy()
    if order is None:
        order = sorted(df[group_col].unique(), key=str)
    fig = px.box(
        df, x=group_col, y=value_col, color=group_col,
        category_orders={group_col: order},
        points=points, color_discrete_map=color_map or {},
    )
    fig.update_traces(boxmean=True)
    fig.update_yaxes(title="Overall packet loss (%)", autorange=True, dtick=5)
    fig.update_xaxes(title="Experiment", ticks="outside")
    fig.update_layout(template="plotly_white", showlegend=False,
                      title="Packet Loss per Sensor — Box Plot by Experiment",
                      margin=dict(l=60, r=30, t=60, b=50))
    if tick_labels:
        ticktext = [tick_labels.get(x, x) for x in order]
        fig.update_xaxes(tickvals=order, ticktext=ticktext)
    return fig


def _linear_fit(x, y):
    """Return slope a, intercept b, and R² for y ~ a*x + b."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return np.nan, np.nan, np.nan
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return a, b, r2


def make_scatter_sensor_vs_loss(
    combined_df: pd.DataFrame,
    *,
    value_col: str = "overall_packet_loss_pct",
    group_col: str = "experiment",
    sensor_col: str = "sensor",
    order: list[str] | None = None,
    color_map: dict | None = None,
    x_jitter: float = 0.0,
    show_fit: bool = True,
    fit_on: str = "means",   # "means" (recommended) or "all"
):
    """Scatter: X = sensors count in experiment, Y = per-sensor loss, with mean diamonds and optional y=ax+b + R²."""
    df = combined_df[[group_col, sensor_col, value_col]].dropna().copy()
    if order is None:
        order = sorted(df[group_col].unique(), key=str)

    n_map = df.groupby(group_col)[sensor_col].size().reindex(order).astype(int).to_dict()

    # optional offsets when several experiments share the same n
    counts_to_groups, offsets = {}, {g: 0.0 for g in order}
    for g in order:
        counts_to_groups.setdefault(n_map[g], []).append(g)
    if x_jitter > 0:
        for n, groups_at_n in counts_to_groups.items():
            k = len(groups_at_n)
            if k > 1:
                for i, g in enumerate(groups_at_n):
                    offsets[g] = -x_jitter + i * (2 * x_jitter / (k - 1))

    df["x_n"] = df[group_col].map(lambda g: n_map[g] + offsets[g])
    df["n_sensors"] = df[group_col].map(n_map)

    fig = px.scatter(
        df, x="x_n", y=value_col, color=group_col,
        category_orders={group_col: order},
        color_discrete_map=color_map or {},
        hover_data={group_col: True, sensor_col: True, "n_sensors": True},
    )
    fig.update_traces(marker=dict(size=8, opacity=0.85), showlegend=True)

    # mean diamonds
    means = df.groupby(group_col)[value_col].mean().reindex(order)
    mean_x = [n_map[g] + offsets[g] for g in order]
    mean_y = [means.loc[g] for g in order]
    fig.add_trace(go.Scatter(
        x=mean_x, y=mean_y, mode="markers", name="Mean (AVG)",
        marker=dict(symbol="diamond", size=14, line=dict(width=1), color="black"),
        hovertemplate="Mean: %{y:.3f}<extra>Mean (AVG)</extra>", showlegend=True,
    ))

    vals = sorted(set(n_map.values()))
    fig.update_xaxes(title="Number of sensors (n)", tickmode="array", tickvals=vals,
                     range=[vals[0]-0.6, vals[-1]+0.6], ticks="outside",
                     ticklen=3, tickwidth=1, showline=True, linewidth=1)
    fig.update_yaxes(title="Overall packet loss (%)", rangemode="tozero")
    fig.update_layout(template="plotly_white",
                      title="Per-Sensor Packet Loss — Sensors share X=n, AVG shown per experiment",
                      margin=dict(l=30, r=30, t=60, b=50))

    # add linear fit + R²
    if show_fit and len(vals) >= 2:
        if fit_on == "all":
            x_fit, y_fit = df["n_sensors"].values, df[value_col].values
        else:
            x_fit = [n_map[g] for g in order]
            y_fit = [means.loc[g] for g in order]
        a, b, r2 = _linear_fit(x_fit, y_fit)
        if np.isfinite(a) and np.isfinite(b):
            x_line = np.linspace(min(vals), max(vals), 100)
            y_line = a * x_line + b
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines", name="Fit (y=ax+b)",
                line=dict(color="black", dash="dash")
            ))
            fig.add_annotation(
                xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False,
                align="left", bgcolor="white",
                text=f"y = {a:.3f}x + {b:.3f}<br>R² = {r2:.3f}"
            )
    return fig


# ----------------------------- Small utilities -----------------------------

def _round_numeric_columns(df: pd.DataFrame, cols: list[str], ndigits: int = 3) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(ndigits)
    return out


def _apply_cld_to_hist_titles(fig, experiments_order: list[str], meta_df: pd.DataFrame, letters: dict[str, str]):
    meta_idx = meta_df.set_index("experiment")
    new_titles = []
    for e in experiments_order:
        if e in meta_idx.index:
            sensors = int(meta_idx.loc[e, "sensors_count"])
            hours = int(meta_idx.loc[e, "run_hours_excl_edges"])
            letter = letters.get(e, "")
            new_titles.append(f"{e} — {sensors} sensor{'' if sensors==1 else 's'} • {hours} h • [{letter}]")
        else:
            new_titles.append(f"{e} • [{letters.get(e,'')}]")
    for i, anno in enumerate(fig.layout.annotations):
        if i < len(new_titles):
            anno.update(text=new_titles[i])


def _tick_labels_with_letters(order: list[str], letters: dict[str, str]) -> dict[str, str]:
    return {e: f"{e} [{letters.get(e,'')}]" for e in order}


# ----------------------------- One-stop builder -----------------------------

def build_all(
    folder: str,
    *,
    file_glob: str = "*.csv",
    freq: str = "3T",
    expected_per_hour: int = 20,
    drop_edge_hours: bool = True,
    sensors_to_drop: list[str] | None = None,
    sensors_to_nan:  list[str] | None = None,
    sensors_to_drop_by_experiment: dict[str, list[str]] | None = None,
    sensors_to_nan_by_experiment:  dict[str, list[str]] | None = None,
    bin_size: float = 0.5,
    colors: list[str] | None = None,
    show_legend: bool = False,
    points: str | bool = "outliers",
    round_numeric: bool = True,
    ndigits: int = 3,
    x_tick_step: float | None = 5.0,
):
    """
    Returns:
      figs: {'hist': fig_hist, 'box': fig_box, 'scatter': fig_scatter}
      dfs : {'describe': desc_df, 'group': group_df, 'pairwise': pairwise_df, 'welch': welch_df}
    """
    # 1) Combined + meta + histogram
    combined, meta, fig_hist = summarize_folder_distributions_px_vertical(
        folder,
        file_glob=file_glob, freq=freq, expected_per_hour=expected_per_hour,
        drop_edge_hours=drop_edge_hours,
        sensors_to_drop=sensors_to_drop, sensors_to_nan=sensors_to_nan,
        sensors_to_drop_by_experiment=sensors_to_drop_by_experiment,
        sensors_to_nan_by_experiment=sensors_to_nan_by_experiment,
        bin_size=bin_size, x_tick_step=x_tick_step, colors=colors, show_legend=show_legend,
    )

    # 2) Tables
    desc_df = describe_per_experiment_df(combined)
    group_df, pairwise_df = make_summary_tables(combined_df=combined, value_col="overall_packet_loss_pct", group_col="experiment", alpha=0.05)
    welch_df = welch_anova_df(combined_df=combined, value_col="overall_packet_loss_pct", group_col="experiment")

    if round_numeric:
        desc_df = _round_numeric_columns(desc_df, ["mean", "std", "min", "25%", "median", "75%", "max"], ndigits)
        group_df = _round_numeric_columns(group_df, ["mean", "std", "avg_|g|", "levene_p"], ndigits)
        pairwise_df = _round_numeric_columns(pairwise_df, ["mean1", "mean2", "mean_diff", "p_value", "g", "abs_g"], ndigits)
        welch_df = _round_numeric_columns(welch_df, ["F", "df_num", "df_denom", "p_value"], ndigits)

    # 3) Consistent colors across plots
    experiments_order = sorted(combined["experiment"].unique(), key=str)
    palette = colors or px.colors.qualitative.Set2
    color_map = {exp: palette[i % len(palette)] for i, exp in enumerate(experiments_order)}

    # annotate histogram titles with CLD letters
    letters_map = dict(zip(group_df["experiment"], group_df["letters"]))
    _apply_cld_to_hist_titles(fig_hist, experiments_order, meta, letters_map)
    tick_labels = _tick_labels_with_letters(experiments_order, letters_map)

    # 4) Box (order by ascending mean) & Scatter
    order_by_mean = list(group_df["experiment"])
    fig_box = make_boxplot_by_experiment(
        combined, order=order_by_mean, points=points, color_map=color_map, tick_labels=tick_labels
    )
    fig_scatter = make_scatter_sensor_vs_loss(combined_df=combined, order=order_by_mean, color_map=color_map)

    figs = {"hist": fig_hist, "box": fig_box, "scatter": fig_scatter}
    dfs  = {"describe": desc_df, "group": group_df, "pairwise": pairwise_df, "welch": welch_df}
    return figs, dfs





# figs, dfs = build_all(
#     folder=r"C:\Users\...",
#     sensors_to_drop_by_experiment={
#         "Firmware_2": ["D_1_1", "C_2_1"],
#         "Firmware_4": ["C_3_3"],
#     },
#     bin_size=0.5,          # histogram bin width in % (e.g., 0.5%)
#     x_tick_step=5.0,       # x-axis ticks every 5% on histograms
#     points="outliers",     # boxplot points overlay: 'all' | 'outliers' | 'suspectedoutliers' | False
# )

# # Show the figures
# figs["hist"].show()
# figs["box"].show()
# figs["scatter"].show()

# # See the data tables
# desc     = dfs["describe"]   # describe per experiment
# group    = dfs["group"]      # group summary (means, SD, CLD letters, avg |g|)
# pairwise = dfs["pairwise"]   # pairwise comparisons (mean diff, p, Hedges g)
# welch    = dfs["welch"]      # Welch ANOVA summary

# print(desc.head())
# print(group)
# print(pairwise)
# print(welch)
