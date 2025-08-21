# Single-file analysis of packet loss from an F4D battery CSV.
# Produces interactive Plotly figures + tidy tables.
# Includes cumulative missing-packets line chart per sensor.

import os
import re
import numpy as np
import pandas as pd
import plotly.express as px


def analyze_battery_packet_loss(
    file_path: str,
    *,
    freq: str = "3T",
    expected_per_hour: int = 20,
    drop_edge_hours: bool = True,
    heatmap_max_pct: float | str | None = "auto",  # "auto" => dynamic max from data; or pass a number
    coords_df: pd.DataFrame | None = None,         # optional: columns sensor,x,y
    sensors_to_drop: list[str] | None = None,      # drop these sensors entirely
    sensors_to_nan:  list[str] | None = None       # set these sensors' values to NaN
):
    """
    Analyze packet loss from an F4D battery data CSV and return Plotly figures + tables.

    Returns
    -------
    dict
        {'figs': {<name>: plotly.Figure, ...},
         'tables': {'hourly': DataFrame, 'sensor_overall': DataFrame,
                    'sensor_stats': DataFrame, 'sensor_grid': DataFrame}}
    """
    # ---------- Load & trim ----------
    df = pd.read_csv(file_path)
    if 'Timestamp' not in df.columns:
        raise ValueError("CSV must contain a 'Timestamp' column")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp']).sort_values('Timestamp')

    if drop_edge_hours:
        df['_hour'] = df['Timestamp'].dt.floor('H')
        hour_counts = df.groupby('_hour').size().sort_index()
        first_hour, last_hour = hour_counts.index.min(), hour_counts.index.max()
        to_drop = [h for h in (first_hour, last_hour) if hour_counts.get(h, 0) < expected_per_hour]
        df = df[~df['_hour'].isin(to_drop)].drop(columns=['_hour']).reset_index(drop=True)
        if to_drop:
            print("Dropped edge hours:", [h.strftime('%Y-%m-%d %H:00') for h in to_drop])

    # numeric sensor matrix
    df = df.set_index('Timestamp')

    # --- Sensor filtering before aggregation ---
    all_cols = [c for c in df.columns if c.lower() != 'timestamp']
    norm = lambda s: str(s).strip().lower()
    name_map = {norm(c): c for c in all_cols}

    to_drop_cols = []
    if sensors_to_drop:
        for s in sensors_to_drop:
            key = norm(s)
            if key in name_map:
                to_drop_cols.append(name_map[key])
    if to_drop_cols:
        df = df.drop(columns=to_drop_cols, errors='ignore')
        print("Dropped sensors (removed from analysis):", to_drop_cols)

    to_nan_cols = []
    if sensors_to_nan:
        remaining = [c for c in df.columns if c.lower() != 'timestamp']
        name_map_nan = {norm(c): c for c in remaining}
        for s in sensors_to_nan:
            key = norm(s)
            if key in name_map_nan:
                to_nan_cols.append(name_map_nan[key])
    if to_nan_cols:
        for c in to_nan_cols:
            df[c] = np.nan
        print("Converted sensors to NaN (kept structurally):", to_nan_cols)

    sensor_cols = [c for c in df.columns if c.lower() != 'timestamp']
    df[sensor_cols] = df[sensor_cols].apply(pd.to_numeric, errors='coerce')

    # ---------- Hourly stats ----------
    hourly_mean  = df[sensor_cols].resample('H').mean()   # mV
    hourly_count = df[sensor_cols].resample('H').count()  # non-NaN

    hourly = (
        pd.concat({'avg_batt_mV': hourly_mean.stack(), 'n_non_nan': hourly_count.stack()}, axis=1)
          .reset_index()
          .rename(columns={'level_1': 'sensor', 'Timestamp': 'hour'})
    )
    hourly['coverage_pct']    = (hourly['n_non_nan'].clip(upper=expected_per_hour) / expected_per_hour) * 100
    hourly['packet_loss_pct'] = (100 - hourly['coverage_pct']).clip(lower=0, upper=100)
    hourly['hour_str'] = pd.to_datetime(hourly['hour']).dt.strftime('%Y-%m-%d %H:00')

    # per-sensor Avg ± SE battery (from all readings, not hourly means)
    sensor_mean = df[sensor_cols].mean()
    sensor_n    = df[sensor_cols].count()
    sensor_sd   = df[sensor_cols].std(ddof=1)
    sensor_se   = sensor_sd / np.sqrt(sensor_n)

    hourly['avg_batt_mV_sensor'] = hourly['sensor'].map(sensor_mean)
    hourly['se_batt_mV_sensor']  = hourly['sensor'].map(sensor_se)
    hourly['n_readings_sensor']  = hourly['sensor'].map(sensor_n)

    # natural sort sensors (F1, F2, ..., F10)
    def natural_key(s):
        return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(s))]
    sensor_order = sorted(hourly['sensor'].astype(str).unique(), key=natural_key)

    # ---------- Missing packets per hour + cumulative by sensor ----------
    hourly["missing_in_hour"] = (
        expected_per_hour - hourly["n_non_nan"].clip(upper=expected_per_hour)
    ).clip(lower=0)

    hourly = hourly.sort_values(["sensor", "hour"])
    hourly["cum_missing"] = hourly.groupby("sensor")["missing_in_hour"].cumsum()

    fig_cum_missing = px.line(
        hourly.sort_values("hour"),
        x="hour", y="cum_missing", color="sensor",
        category_orders={"sensor": sensor_order},
        labels={"hour": "Hour", "cum_missing": "Cumulative missing packets", "sensor": "Sensor"},
        title="Cumulative Missing Packets per Sensor (hourly)"
    )
    fig_cum_missing.update_layout(template="plotly_white")

    # ---------- Plot A: Heatmap (ALL sensors shown, dynamic colorbar) ----------
    hours   = pd.date_range(hourly['hour'].min().floor('H'), hourly['hour'].max().ceil('H'), freq='H')
    loss_wide = (
        hourly.pivot(index='sensor', columns='hour', values='packet_loss_pct')
              .reindex(index=sensor_order)
              .reindex(columns=hours)
    )
    vals = loss_wide.values.astype(float)
    x_labels = [h.strftime('%Y-%m-%d %H:%M') for h in loss_wide.columns]
    y_labels = loss_wide.index.tolist()

    # decide zmax (dynamic by default)
    if heatmap_max_pct == "auto" or heatmap_max_pct is None:
        vmax = float(np.nanmax(vals)) if np.isfinite(np.nanmax(vals)) else 1.0
        vmax = float(np.clip(np.ceil(vmax), 1, 100))
    else:
        vmax = float(heatmap_max_pct)

    fig_heat = px.imshow(
        vals, origin='lower', aspect='auto',
        zmin=0, zmax=vmax,
        color_continuous_scale='Viridis',
        labels={'x': 'Hour', 'y': 'Sensor', 'color': 'Packet Loss (%)'},
        title='Hourly Packet Loss by Sensor (Heatmap — all sensors)'
    )
    tick_idx = list(range(0, len(x_labels), 4))
    if len(x_labels) > 0 and (len(x_labels) - 1) not in tick_idx:
        tick_idx.append(len(x_labels) - 1)
    fig_heat.update_xaxes(tickmode='array', tickvals=tick_idx, ticktext=[x_labels[i] for i in tick_idx], tickangle=-45)
    fig_heat.update_yaxes(tickmode='array', tickvals=list(range(len(y_labels))), ticktext=y_labels)
    fig_heat.update_layout(template='plotly_white', height=max(500, 30*len(y_labels)), margin=dict(l=120, r=40, t=60, b=80))
    fig_heat.update_coloraxes(colorbar=dict(ticksuffix='%'))

    # ---------- Plot B: Time series ----------
    fig_lines = px.line(
        hourly.sort_values('hour'),
        x='hour', y='packet_loss_pct', color='sensor',
        category_orders={'sensor': sensor_order},
        labels={'hour': 'Hour', 'packet_loss_pct': 'Packet Loss (%)'},
        title='Hourly Packet Loss by Sensor (Time Series)'
    )
    fig_lines.update_layout(template='plotly_white', yaxis=dict(range=[0, 100]))

    # ---------- Plot C: Scatter ----------
    colors16 = px.colors.qualitative.Dark24[:16]
    color_map = {s: colors16[i % 16] for i, s in enumerate(sensor_order)}
    fig_scatter = px.scatter(
        hourly,
        x='avg_batt_mV', y='packet_loss_pct', color='sensor',
        color_discrete_map=color_map, category_orders={'sensor': sensor_order},
        hover_data={'sensor': True, 'hour_str': True, 'avg_batt_mV': ':.0f',
                    'n_non_nan': True, 'packet_loss_pct': ':.2f'},
        labels={'avg_batt_mV': 'Avg Battery (mV)', 'packet_loss_pct': 'Packet Loss (%)', 'hour_str': 'Hour'},
        title='Hourly Avg Battery vs Packet Loss (per Sensor-Hour)'
    )
    fig_scatter.update_traces(marker=dict(size=8, opacity=0.85))
    fig_scatter.update_layout(template='plotly_white', yaxis=dict(range=[0, 100]))

    # ---------- Plot D: Box plot ----------
    label_map = {s: f"{s} — {sensor_mean.get(s, np.nan):.0f} ± {sensor_se.get(s, np.nan):.1f} mV" for s in sensor_order}
    hourly['sensor_label'] = hourly['sensor'].map(label_map)
    label_order = [label_map[s] for s in sensor_order]
    fig_box = px.box(
        hourly,
        y='sensor_label', x='packet_loss_pct', points='outliers',
        category_orders={'sensor_label': label_order},
        hover_data={'sensor': True, 'avg_batt_mV_sensor': ':.0f', 'se_batt_mV_sensor': ':.1f',
                    'n_readings_sensor': True, 'packet_loss_pct': ':.2f', 'hour': True},
        labels={'sensor_label': 'Sensor (Avg ± SE mV)', 'packet_loss_pct': 'Packet Loss per Hour (%)'},
        title='Packet Loss per Hour — Box Plot by Sensor (Avg ± SE battery shown)'
    )
    fig_box.update_layout(template='plotly_white', height=max(650, 35*len(label_order)),
                          width=1000, margin=dict(l=220, r=40, t=70, b=40))
    fig_box.update_xaxes(range=[0, 100])

    # ---------- Plot E: Overall packet loss per sensor ----------
    start = df.index.min().floor(freq)
    end   = df.index.max().ceil(freq)
    slots = pd.date_range(start, end, freq=freq)
    expected_total = len(slots)
    counts_freq = df[sensor_cols].resample(freq).count()
    actual_total = counts_freq.reindex(slots).fillna(0).sum(axis=0)
    sensor_overall = pd.DataFrame({'sensor': actual_total.index.astype(str), 'actual': actual_total.values})
    sensor_overall['expected'] = expected_total
    sensor_overall['overall_packet_loss_pct'] = (1 - sensor_overall['actual'].clip(upper=expected_total) / expected_total) * 100
    sensor_overall = sensor_overall.sort_values('sensor', key=lambda s: s.map(natural_key))

    fig_overall = px.histogram(
        sensor_overall, x='overall_packet_loss_pct', nbins=100,
        labels={'overall_packet_loss_pct': 'Overall Packet Loss per Sensor (%)', 'count': 'Sensor Count'},
        title='Distribution of Overall Packet Loss — Sensors'
    )
    fig_overall.update_traces(xbins=dict(start=0, end=100.0001, size=1))
    fig_overall.update_xaxes(autorange=True, dtick=10, ticks="outside")

    # ---------- GRID PARSING (A_1_2 names) ----------
    so = sensor_overall.copy()
    so['sensor_norm'] = (
        so['sensor'].str.strip().str.upper().str.replace(r'[\s\-]+', '_', regex=True)
    )
    pat = re.compile(r'([A-Z]+)_(\d+)(?:_(\d+))?', re.IGNORECASE)

    def letters_to_number(letters: str) -> int:
        n = 0
        for ch in letters:
            n = n * 26 + (ord(ch.upper()) - 64)
        return n

    parsed = []
    for raw, normed in zip(so['sensor'], so['sensor_norm']):
        m = pat.search(normed)
        if not m:
            parsed.append({'sensor': raw, 'col_letters': None, 'col_idx': None,
                           'row': None, 'unit': None, 'cell': None})
            continue
        col_letters, row_str, unit_str = m.groups()
        col_idx = letters_to_number(col_letters)
        row = int(row_str)
        unit = int(unit_str) if unit_str else None
        cell = f"{col_letters.upper()}_{row}"
        parsed.append({'sensor': raw, 'col_letters': col_letters.upper(), 'col_idx': col_idx,
                       'row': row, 'unit': unit, 'cell': cell})

    sensor_grid = pd.DataFrame(parsed).merge(
        so[['sensor','overall_packet_loss_pct']], on='sensor', how='left'
    )

    # ---------- Figures dict ----------
    figs = {
        'heatmap_hourly_loss': fig_heat,
        'lines_hourly_loss': fig_lines,
        'scatter_batt_vs_loss': fig_scatter,
        'box_by_sensor': fig_box,
        'hist_overall_loss': fig_overall,
        'cum_missing_by_sensor': fig_cum_missing,
    }

    # ---------- Per-sensor GRID heatmap (dynamic color scale) ----------
    valid = sensor_grid.dropna(subset=['col_idx', 'row']).copy()
    if not valid.empty:
        cols_letters = sorted(valid['col_letters'].unique(), key=lambda L: letters_to_number(L))
        row_vals     = sorted(valid['row'].unique())
        rows_display = row_vals[::-1]

        def natural_key_local(s): return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(s))]
        def unit_num(name: str) -> int:
            m = re.search(r'(\d+)$', str(name))
            return int(m.group(1)) if m else -1

        cell_lists = (
            valid.sort_values('sensor', key=lambda s: s.map(natural_key_local))
                 .groupby(['col_letters', 'col_idx', 'row'])['sensor']
                 .apply(list)
                 .reset_index(name='names')
        )
        cell_map2 = {}
        for _, r in cell_lists.iterrows():
            names_sorted = sorted(r['names'], key=unit_num, reverse=True)  # [_2, _1]
            cell_map2[(r['col_idx'], int(r['row']))] = names_sorted

        max_per_cell2 = max((len(v) for v in cell_map2.values()), default=1)

        z, text = [], []
        col_vals = sorted(valid['col_idx'].unique())
        loss_map = dict(zip(valid['sensor'], valid['overall_packet_loss_pct']))

        for r in rows_display:
            names_here_by_col = {c: cell_map2.get((c, r), []) for c in col_vals}
            for k in range(max_per_cell2):  # k=0 top
                row_z, row_t = [], []
                for c in col_vals:
                    names = names_here_by_col[c]
                    if k < len(names):
                        nm = names[k]
                        row_z.append(float(loss_map.get(nm, np.nan)))
                        row_t.append(nm)
                    else:
                        row_z.append(np.nan)
                        row_t.append("")
                z.append(row_z)
                text.append(row_t)

        # Dynamic color range based on actual values
        losses_series = valid['overall_packet_loss_pct'].astype(float)
        if losses_series.notna().any():
            vmin = float(losses_series.min())
            vmax_grid = float(losses_series.max())
            if vmin == vmax_grid:
                vmin = max(0.0, vmin - 0.5)
                vmax_grid = vmin + 1.0
        else:
            vmin, vmax_grid = 0.0, 1.0

        fig_grid_sensors_heat = px.imshow(
            z,
            x=col_vals,
            y=list(range(len(z))),
            color_continuous_scale="Viridis",
            aspect="auto",
            zmin=vmin,
            zmax=vmax_grid
        )
        fig_grid_sensors_heat.update_coloraxes(colorbar=dict(title='Overall loss', ticksuffix='%'))

        # Annotate numeric cells
        span = (vmax_grid - vmin) if (vmax_grid > vmin) else 1.0
        for i in range(len(z)):
            for j in range(len(z[0])):
                val = z[i][j]
                if isinstance(val, (int, float)) and not np.isnan(val):
                    lbl = text[i][j]
                    if lbl:
                        frac = (val - vmin) / span
                        fig_grid_sensors_heat.add_annotation(
                            x=col_vals[j], y=i, text=lbl, showarrow=False,
                            xanchor='center', yanchor='middle',
                            font=dict(size=12, color=('white' if frac > 0.5 else 'black'))
                        )

        def number_to_letters(n: int) -> str:
            s = ""
            while n > 0:
                n, r = divmod(n-1, 26)
                s = chr(65 + r) + s
            return s

        fig_grid_sensors_heat.update_xaxes(
            tickmode='array', tickvals=col_vals,
            ticktext=[number_to_letters(n) for n in col_vals],
            showgrid=False, zeroline=False
        )
        fig_grid_sensors_heat.update_yaxes(
            autorange='reversed',
            tickmode='array',
            tickvals=[i*max_per_cell2 for i in range(len(rows_display))],
            ticktext=[str(r) for r in rows_display],
            showgrid=False, zeroline=False
        )
        fig_grid_sensors_heat.update_layout(
            title='Per-Sensor Grid Heatmap (color = each sensor’s overall packet loss)',
            template='plotly_white', plot_bgcolor='#f5f6f8',
            xaxis_title='Column', yaxis_title='Row', yaxis_scaleanchor='x',
            margin=dict(l=40, r=40, t=60, b=40),
            height=max(450, 120 * len(rows_display)),
            width=max(500, 120 * len(col_vals)),
        )
        figs['grid_sensors_heat'] = fig_grid_sensors_heat

    # ---------- Optional spatial plot ----------
    if coords_df is not None and {'sensor','x','y'}.issubset(coords_df.columns):
        merged = coords_df.merge(sensor_overall[['sensor','overall_packet_loss_pct']], on='sensor', how='left')
        fig_spatial = px.scatter(
            merged, x='x', y='y',
            color='overall_packet_loss_pct',
            color_continuous_scale='Viridis',
            range_color=[0, vmax],   # use hourly heatmap cap for consistency
            hover_data={'sensor': True, 'overall_packet_loss_pct': ':.2f', 'x': True, 'y': True},
            labels={'x': 'X', 'y': 'Y', 'color': 'Overall Packet Loss (%)'},
            title='Overall Packet Loss per Sensor (Scatter Heat Map)'
        )
        fig_spatial.update_traces(mode='markers', marker=dict(size=28, symbol='square'))
        fig_spatial.update_layout(template='plotly_white', yaxis_scaleanchor='x')
        fig_spatial.update_coloraxes(colorbar=dict(ticksuffix='%'))
        figs['spatial_overall_loss'] = fig_spatial

    # ---------- Title branding ----------
    base_name = os.path.basename(file_path)
    m_exp = re.search(r'^(.*?)_battery', base_name, flags=re.IGNORECASE)
    exp_name = m_exp.group(1) if m_exp else os.path.splitext(base_name)[0]

    # Count only sensors with at least one reading (excludes all-NaN)
    num_sensors = int((sensor_n > 0).sum())
    sensor_tag = f"{num_sensors} sensor" if num_sensors == 1 else f"{num_sensors} sensors"

    def brand(title_core: str) -> str:
        return f"{exp_name} • {title_core} • {sensor_tag}"

    fig_heat.update_layout(title=brand('Hourly Packet Loss by Sensor (Heatmap — all sensors)'))
    fig_lines.update_layout(title=brand('Hourly Packet Loss by Sensor (Time Series)'))
    fig_scatter.update_layout(title=brand('Hourly Avg Battery vs Packet Loss (per Sensor–Hour)'))
    fig_box.update_layout(title=brand('Packet Loss per Hour — Box Plot by Sensor (Avg ± SE battery shown)'))
    fig_overall.update_layout(title=brand('Distribution of Overall Packet Loss — Sensors'))
    if 'grid_sensors_heat' in figs:
        figs['grid_sensors_heat'].update_layout(
            title=brand('Per-Sensor Grid Heatmap (color = each sensor’s overall packet loss)')
        )
    if 'spatial_overall_loss' in figs:
        figs['spatial_overall_loss'].update_layout(
            title=brand('Overall Packet Loss per Sensor (Scatter Heat Map)')
        )
    if 'cum_missing_by_sensor' in figs:
        figs['cum_missing_by_sensor'].update_layout(
            title=brand('Cumulative Missing Packets per Sensor (hourly)')
        )

    # ---------- Tables ----------
    sensor_stats = pd.DataFrame({
        'sensor': sensor_mean.index,
        'avg_batt_mV': sensor_mean.values,
        'sd_mV': sensor_sd.values,
        'se_mV': sensor_se.values,
        'n': sensor_n.values
    }).sort_values('sensor', key=lambda s: s.map(natural_key))

    return {
        'figs': figs,
        'tables': {
            'hourly': hourly,
            'sensor_overall': sensor_overall,
            'sensor_stats': sensor_stats,
            'sensor_grid': sensor_grid
        }
    }

results = analyze_battery_packet_loss(
#     r"C:\Users\...",
#     sensors_to_drop=["C_3_3"],
#     drop_edge_hours=True
# )

# # Display the results
# for name, fig in results['figs'].items():
#     print(f"Displaying figure: {name}")
#     fig.show()
