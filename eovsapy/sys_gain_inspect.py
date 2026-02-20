#!/usr/bin/env python3
"""
Inspect EOVSA IDB SOLPNTCAL system-gain scans.

Default behavior (no --gui): run in headless batch mode for cron.
  - Select the UTC date (default: today UTC)
  - Find IDB scans near 18:30 and 21:30 UT in /data1/eovsa/fits/IDB/YYYYMMDD
  - Save T/F JPG plots under /common/webplots/solpntcal (daily_track-style subdirs)

Optional behavior (--gui): interactive Tk viewer for a single IDB dataset.
"""

import argparse
import datetime
import os
import re
import shutil
import sys
from glob import glob

import matplotlib

# Keep headless mode safe by default; GUI mode uses Tk canvas explicitly.
matplotlib.use("Agg")

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from eovsapy import read_idb as ri

try:
    from tkinter import *
    from tkinter import ttk
    from tkinter.filedialog import askdirectory
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False


DEFAULT_IDB_ROOT = "/data1/eovsa/fits/IDB"
DEFAULT_OUTDIR = "/common/webplots/solpntcal"
DEFAULT_FIDXS = (120, 320)
DEFAULT_TIDXS = (405, 415)
TARGET_SOLPNT_HHMMS = ("1830", "2130")


def _utc_yyyymmdd():
    return datetime.datetime.utcnow().strftime("%Y%m%d")


def _normalize_date_tag(date_text):
    if date_text is None:
        return _utc_yyyymmdd()
    s = str(date_text).strip()
    if re.match(r"^\d{8}$", s):
        return s
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d").strftime("%Y%m%d")
    except Exception:
        raise ValueError("Invalid --date '{}'. Use YYYYMMDD or YYYY-MM-DD.".format(date_text))


def default_idb_dir(date_tag=None, idb_root=DEFAULT_IDB_ROOT):
    tag = _normalize_date_tag(date_tag) if date_tag is not None else _utc_yyyymmdd()
    d = os.path.join(idb_root, tag)
    if os.path.isdir(d):
        return d
    return idb_root


def _get_power_array(out):
    if isinstance(out, dict) and "p" in out:
        return out["p"]
    raise KeyError("read_idb output does not contain key 'p'")


def _make_axes_grid(fig, nrows, ncols, sharex=True, sharey=False):
    axes = []
    ref = None
    for i in range(1, nrows * ncols + 1):
        if ref is None:
            ax = fig.add_subplot(nrows, ncols, i)
            ref = ax
        else:
            kwargs = {}
            if sharex:
                kwargs["sharex"] = ref
            if sharey:
                kwargs["sharey"] = ref
            ax = fig.add_subplot(nrows, ncols, i, **kwargs)
        axes.append(ax)
    return axes


def _log_ylim_for_values(y, floor=None):
    """Return a robust (lo, hi) for log-scale plots based on y values."""
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    y = y[y > 0]
    if y.size == 0:
        return 1e-3, 1e-2
    lo = np.nanpercentile(y, 1)
    hi = np.nanpercentile(y, 99)
    if not np.isfinite(lo) or lo <= 0:
        lo = np.nanmin(y)
    if not np.isfinite(hi) or hi <= lo:
        hi = np.nanmax(y)
    if floor is None:
        floor = max(1e-6, lo * 0.1)
    lo = max(floor, lo * 0.8)
    hi = hi * 1.2
    if not np.isfinite(hi) or hi <= lo:
        hi = lo * 10.0
    return lo, hi


def _normalize_pols(pols, npol):
    if pols is None:
        requested = [0, 1]
    elif isinstance(pols, (int, np.integer)):
        requested = [int(pols)]
    else:
        requested = []
        for p in pols:
            try:
                requested.append(int(p))
            except Exception:
                continue
    valid = []
    for p in requested:
        if 0 <= p < npol and p not in valid:
            valid.append(p)
    if not valid:
        valid = [0] if npol > 0 else []
    return valid


def _good_ants_for_ylim(p, pols, threshold=10.0, ant_count=None):
    """Return mask of antennas to include for y-lim determination."""
    nant = p.shape[0]
    if ant_count is None:
        ant_count = nant
    n = min(int(ant_count), nant)
    pols = _normalize_pols(pols, p.shape[1])
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        y = p[i, pols, :, :].ravel()
        y = y[np.isfinite(y)]
        if y.size == 0:
            continue
        med = np.nanmedian(y)
        if np.isfinite(med) and med >= threshold:
            mask[i] = True
    return mask


def _draw_low_arrow(ax, label=None, color="crimson"):
    ax.annotate(
        "",
        xy=(0.92, 0.04),
        xytext=(0.92, 0.22),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
    )
    if label:
        ax.text(0.70, 0.24, label, transform=ax.transAxes, color=color, fontsize=9)


def _parse_idx_list(text, max_inclusive=None):
    """Parse '1,2', '1 2', '1-3', '1:3' into sorted unique integer list."""
    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []
    parts = s.replace(",", " ").split()
    idxs = []
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                a = int(a)
                b = int(b)
            except Exception:
                continue
            idxs.extend(range(min(a, b), max(a, b) + 1))
            continue
        if ":" in part:
            a, b = part.split(":", 1)
            try:
                a = int(a)
                b = int(b)
            except Exception:
                continue
            idxs.extend(range(min(a, b), max(a, b) + 1))
            continue
        try:
            idxs.append(int(part))
        except Exception:
            continue
    idxs = sorted(set(idxs))
    if max_inclusive is not None:
        m = int(max_inclusive)
        idxs = [i for i in idxs if 0 <= i <= m]
    return idxs


def _parse_tidx_pair(text, defaults=DEFAULT_TIDXS):
    vals = _parse_idx_list(text)
    if len(vals) >= 2:
        return int(vals[0]), int(vals[1])
    if len(vals) == 1:
        return int(vals[0]), int(defaults[1])
    return int(defaults[0]), int(defaults[1])


def _remove_figure_legends(fig):
    try:
        for leg in list(fig.legends):
            leg.remove()
    except Exception:
        pass


def _linestyle_for_pol_index(idx):
    if idx == 0:
        return "-"
    if idx == 1:
        return ":"
    styles = ("--", "-.")
    return styles[(idx - 2) % len(styles)]


def _pol_label(pol):
    if pol == 0:
        return "pol 0 (X)"
    if pol == 1:
        return "pol 1 (Y)"
    return "pol {}".format(pol)


def _scan_label_from_idb_name(idb_name):
    ts = _parse_idb_timestamp(idb_name)
    if ts is None:
        return None
    return ts.strftime("%Y-%m-%d %H:%M:%S UT")


def _scan_file_tag_from_idb_name(idb_name):
    """
    Return filename-safe scan tag without the 'IDB' prefix.
    Preferred format: YYYYMMDDHHMMSS from parsed timestamp.
    """
    ts = _parse_idb_timestamp(idb_name)
    if ts is not None:
        return ts.strftime("%Y%m%d%H%M%S")
    # Fallback for unexpected names: strip leading IDB if present
    return re.sub(r"^IDB", "", str(idb_name))


def _extract_scan_datetime_from_text(text):
    """
    Extract scan datetime from any text containing YYYYMMDDHHMMSS.
    """
    m = re.search(r"(20\d{12})", str(text))
    if not m:
        return None
    try:
        return datetime.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except Exception:
        return None


def _newest_saved_scan_datetime(save_dir, kind):
    """
    Return newest timestamp from dated sys_gain files of one kind ('t' or 'f').
    Excludes *_latest.jpg.
    """
    newest = None
    pattern = os.path.join(save_dir, "sys_gain_{}_*".format(kind))
    for pth in glob(pattern):
        base = os.path.basename(pth)
        if base.endswith("_latest.jpg"):
            continue
        dt = _extract_scan_datetime_from_text(base)
        if dt is None:
            continue
        if newest is None or dt > newest:
            newest = dt
    return newest


def _should_update_latest(latest_path, scan_dt, existing_latest_dt):
    """
    Update latest only when current scan is not older than the current latest date.
    """
    if not os.path.exists(latest_path):
        return True
    if scan_dt is None:
        return False
    if existing_latest_dt is not None:
        return scan_dt >= existing_latest_dt
    # Fallback: use file mtime if latest exists but no dated files can be parsed.
    try:
        latest_mtime_dt = datetime.datetime.utcfromtimestamp(os.path.getmtime(latest_path))
        return scan_dt >= latest_mtime_dt
    except Exception:
        return False


def plot_sys_gain_t(
        fig,
        axes,
        out,
        pols=(0, 1),
        fidxs=DEFAULT_FIDXS,
        tidx1=DEFAULT_TIDXS[0],
        tidx2=DEFAULT_TIDXS[1],
        ant_count=15,
        ylog=True,
        scan_label=None):
    """
    Time-series plot: p[ant, pol, freq, time] at selected frequency indices.
    """
    p = _get_power_array(out)
    nants = min(ant_count, p.shape[0], len(axes))
    npol = p.shape[1]
    nf = p.shape[2]
    nt = p.shape[3]

    pols = _normalize_pols(pols, npol)

    def _clamp_tidx(v):
        try:
            v = int(v)
        except Exception:
            return None
        if v < 0 or v >= nt:
            return None
        return v

    tidx1 = _clamp_tidx(tidx1)
    tidx2 = _clamp_tidx(tidx2)
    tidxs = [t for t in (tidx1, tidx2) if t is not None]

    try:
        fidxs = list(fidxs)
    except Exception:
        fidxs = [DEFAULT_FIDXS[0]]
    cleaned = []
    for v in fidxs:
        try:
            v = int(v)
        except Exception:
            continue
        if 0 <= v < nf:
            cleaned.append(v)
    fidxs = sorted(set(cleaned))
    if not fidxs:
        fidxs = [0]

    for ax in axes:
        ax.cla()

    if ylog:
        try:
            good = _good_ants_for_ylim(p, pols, threshold=10.0, ant_count=nants)
            if np.any(good):
                all_y = p[:nants][:, pols][:, :, fidxs, :][good].ravel()
            else:
                all_y = p[:nants][:, pols][:, :, fidxs, :].ravel()
        except Exception:
            all_y = np.array([])
        ylo, yhi = _log_ylim_for_values(all_y, floor=1.0)
    else:
        good = _good_ants_for_ylim(p, pols, threshold=10.0, ant_count=nants)

    colors = ["tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    for i in range(nants):
        ax = axes[i]
        x = np.arange(nt)
        for k, fi in enumerate(fidxs):
            c = colors[k % len(colors)]
            for ip, pol in enumerate(pols):
                ls = _linestyle_for_pol_index(ip)
                ax.plot(x, p[i, pol, fi, :], color=c, linestyle=ls, alpha=0.9, lw=1.1)
        if tidx1 is not None:
            ax.axvline(tidx1, color="tab:orange", linestyle="--", linewidth=1, alpha=0.7)
        if tidx2 is not None:
            ax.axvline(tidx2, color="tab:blue", linestyle="--", linewidth=1, alpha=0.7)
        if ylog:
            ax.set_yscale("log")
            ax.set_ylim(ylo, yhi)
            if i < len(good) and not good[i]:
                _draw_low_arrow(ax, label="<10")
        ax.text(0.02, 0.92, "Ant {0}".format(i + 1), transform=ax.transAxes)
        if i >= (len(axes) - 4):
            ax.set_xlabel("Time index")
        if i % 4 == 0:
            ax.set_ylabel("Power")

    for j in range(nants, len(axes)):
        axes[j].axis("off")

    _remove_figure_legends(fig)
    handles = []
    for k, fi in enumerate(fidxs):
        handles.append(Line2D([0], [0], color=colors[k % len(colors)], linestyle="-", lw=1.8, label="fidx {}".format(fi)))
    for ip, pol in enumerate(pols):
        handles.append(Line2D([0], [0], color="k", linestyle=_linestyle_for_pol_index(ip), lw=1.8, label=_pol_label(pol)))
    if tidx1 is not None:
        handles.append(Line2D([0], [0], color="tab:orange", linestyle="--", lw=1.2, label="tidx1 {}".format(tidx1)))
    if tidx2 is not None:
        handles.append(Line2D([0], [0], color="tab:blue", linestyle="--", lw=1.2, label="tidx2 {}".format(tidx2)))
    if handles:
        fig.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.995, 0.985),
            ncol=2,
            framealpha=0.85,
            fontsize=8,
            borderaxespad=0.2,
        )

    pmark = ",".join(str(pol) for pol in pols)
    fmark = ",".join(str(f) for f in fidxs)
    if tidxs:
        tmark = ",".join(str(t) for t in tidxs)
        title = "T plot (pols={}) fidx={} tidx={}".format(pmark, fmark, tmark)
    else:
        title = "T plot (pols={}) fidx={}".format(pmark, fmark)
    if scan_label:
        title = "{} | {}".format(title, scan_label)
    fig.suptitle(title, y=0.988)
    fig.tight_layout(rect=[0, 0.03, 1, 0.955])
    return {"nants": nants, "nt": nt, "nf": nf, "pols": tuple(pols), "fidxs": tuple(fidxs), "tidxs": tuple(tidxs)}


def plot_sys_gain_f(
        fig,
        axes,
        out,
        pols=(0, 1),
        tidx1=DEFAULT_TIDXS[0],
        tidx2=DEFAULT_TIDXS[1],
        fidxs=DEFAULT_FIDXS,
        ant_count=15,
        ylog=True,
        scan_label=None):
    """
    Frequency plot: p[ant, pol, freq, time] at selected time index/indices.
    """
    p = _get_power_array(out)
    nants = min(ant_count, p.shape[0], len(axes))
    npol = p.shape[1]
    nf = p.shape[2]
    nt = p.shape[3]

    pols = _normalize_pols(pols, npol)

    def _clamp_tidx(v):
        try:
            v = int(v)
        except Exception:
            v = 0
        if v < 0:
            v = 0
        if v >= nt:
            v = nt - 1
        return v

    tidx1 = _clamp_tidx(tidx1)
    tidx2 = _clamp_tidx(tidx2)

    try:
        fidxs = list(fidxs)
    except Exception:
        fidxs = [DEFAULT_FIDXS[0]]
    cleaned = []
    for v in fidxs:
        try:
            v = int(v)
        except Exception:
            continue
        if 0 <= v < nf:
            cleaned.append(v)
    fidxs = sorted(set(cleaned))

    x = np.arange(nf)

    for ax in axes:
        ax.cla()

    if ylog:
        try:
            good = _good_ants_for_ylim(p, pols, threshold=10.0, ant_count=nants)
            all_chunks = []
            for pol in pols:
                y1 = p[:nants, pol, :, tidx1]
                y2 = p[:nants, pol, :, tidx2]
                if np.any(good):
                    y1 = y1[good]
                    y2 = y2[good]
                all_chunks.extend([y1.ravel(), y2.ravel()])
            all_y = np.concatenate(all_chunks) if all_chunks else np.array([])
        except Exception:
            all_y = np.array([])
        ylo, yhi = _log_ylim_for_values(all_y, floor=None)
    else:
        good = _good_ants_for_ylim(p, pols, threshold=10.0, ant_count=nants)

    for i in range(nants):
        ax = axes[i]
        for ip, pol in enumerate(pols):
            ls = _linestyle_for_pol_index(ip)
            ax.plot(x, p[i, pol, :, tidx1], color="tab:orange", linestyle=ls, lw=1.1)
            ax.plot(x, p[i, pol, :, tidx2], color="tab:blue", linestyle=ls, lw=1.1)
        for fi in fidxs:
            ax.axvline(fi, color="0.3", linestyle="--", linewidth=1, alpha=0.4)
        if ylog:
            ax.set_yscale("log")
            ax.set_ylim(ylo, yhi)
            if i < len(good) and not good[i]:
                _draw_low_arrow(ax, label="<10")
        ax.text(0.02, 0.92, "Ant {0}".format(i + 1), transform=ax.transAxes)
        if i >= (len(axes) - 4):
            ax.set_xlabel("Freq index")
        if i % 4 == 0:
            ax.set_ylabel("Power")

    for j in range(nants, len(axes)):
        axes[j].axis("off")

    _remove_figure_legends(fig)
    handles = []
    for ip, pol in enumerate(pols):
        handles.append(Line2D([0], [0], color="k", linestyle=_linestyle_for_pol_index(ip), lw=1.8, label=_pol_label(pol)))
    handles.append(Line2D([0], [0], color="tab:orange", linestyle="-", lw=1.5, label="tidx1 {}".format(tidx1)))
    handles.append(Line2D([0], [0], color="tab:blue", linestyle="-", lw=1.5, label="tidx2 {}".format(tidx2)))
    if fidxs:
        handles.append(Line2D([0], [0], color="0.35", linestyle="--", lw=1.2, label="selected fidx"))
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.985),
        ncol=2,
        framealpha=0.85,
        fontsize=8,
        borderaxespad=0.2,
    )

    pmark = ",".join(str(pol) for pol in pols)
    if fidxs:
        title = "F plot (pols={}) tidx={},{} fidx={}".format(
            pmark, tidx1, tidx2, ",".join(str(fi) for fi in fidxs)
        )
    else:
        title = "F plot (pols={}) tidx={},{}".format(pmark, tidx1, tidx2)
    if scan_label:
        title = "{} | {}".format(title, scan_label)
    fig.suptitle(title, y=0.988)
    fig.tight_layout(rect=[0, 0.03, 1, 0.955])
    return {"nants": nants, "nt": nt, "nf": nf, "pols": tuple(pols), "tidxs": (tidx1, tidx2), "fidxs": tuple(fidxs)}


def _parse_idb_timestamp(name):
    m = re.match(r"^IDB(\d{14})", str(name))
    if not m:
        return None
    try:
        return datetime.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except Exception:
        return None


def _pick_scan_for_hhmm(day_dir, date_tag, hhmm):
    pattern = os.path.join(day_dir, "IDB{}{}*".format(date_tag, hhmm))
    candidates = sorted([p for p in glob(pattern) if os.path.isdir(p)])
    if not candidates:
        return None
    target = datetime.datetime.strptime("{}{}00".format(date_tag, hhmm), "%Y%m%d%H%M%S")

    def _score(path):
        ts = _parse_idb_timestamp(os.path.basename(path.rstrip("/")))
        if ts is None:
            return (10**9, os.path.basename(path))
        return (abs((ts - target).total_seconds()), os.path.basename(path))

    return sorted(candidates, key=_score)[0]


def find_solpntcal_idbs_for_date(date_tag, idb_root=DEFAULT_IDB_ROOT, hhmm_targets=TARGET_SOLPNT_HHMMS):
    day_dir = os.path.join(idb_root, date_tag)
    if not os.path.isdir(day_dir):
        raise IOError("IDB day directory not found: {}".format(day_dir))
    picked = []
    for hhmm in hhmm_targets:
        path = _pick_scan_for_hhmm(day_dir, date_tag, hhmm)
        if path is not None:
            picked.append(path)
    # Keep deterministic order
    picked = sorted(set(picked))
    return picked


def _daily_track_style_outdir(base_outdir, date_tag):
    save_dir = base_outdir
    try:
        year = int(date_tag[:4])
    except Exception:
        year = datetime.datetime.utcnow().year
    if year < datetime.datetime.utcnow().year:
        save_dir = os.path.join(base_outdir, "{:04d}".format(year))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def _cleanup_legacy_latest_files(save_dir):
    """
    Remove old per-slot latest files from previous versions.
    Keep only:
      - sys_gain_t_latest.jpg
      - sys_gain_f_latest.jpg
    """
    patterns = [
        os.path.join(save_dir, "sys_gain_t_????_latest.jpg"),
        os.path.join(save_dir, "sys_gain_f_????_latest.jpg"),
    ]
    for pat in patterns:
        for pth in glob(pat):
            try:
                os.remove(pth)
            except Exception:
                pass


def _save_scan_figures(idb_path, date_tag, outdir, fidxs, tidxs, ant_count=15, ylog=True):
    out = ri.read_idb([idb_path])
    p = _get_power_array(out)
    if p.ndim != 4:
        raise RuntimeError("Unexpected p array shape: {}".format(p.shape))

    tidx1, tidx2 = int(tidxs[0]), int(tidxs[1])
    save_dir = _daily_track_style_outdir(outdir, date_tag)
    _cleanup_legacy_latest_files(save_dir)
    scan_name = os.path.basename(idb_path.rstrip("/"))
    scan_label = _scan_label_from_idb_name(scan_name)
    scan_tag = _scan_file_tag_from_idb_name(scan_name)
    scan_dt = _extract_scan_datetime_from_text(scan_name) or _extract_scan_datetime_from_text(scan_tag)

    prev_latest_t_dt = _newest_saved_scan_datetime(save_dir, "t")
    prev_latest_f_dt = _newest_saved_scan_datetime(save_dir, "f")

    fig_t = Figure(figsize=(12, 8))
    axes_t = _make_axes_grid(fig_t, 3, 5, sharex=True, sharey=True)
    plot_sys_gain_t(
        fig_t,
        axes_t,
        out,
        pols=(0, 1),
        fidxs=fidxs,
        tidx1=tidx1,
        tidx2=tidx2,
        ant_count=ant_count,
        ylog=ylog,
        scan_label=scan_label,
    )
    t_path = os.path.join(save_dir, "sys_gain_t_{}.jpg".format(scan_tag))
    fig_t.savefig(t_path, dpi=110, bbox_inches="tight")

    fig_f = Figure(figsize=(12, 8))
    axes_f = _make_axes_grid(fig_f, 3, 5, sharex=True, sharey=True)
    plot_sys_gain_f(
        fig_f,
        axes_f,
        out,
        pols=(0, 1),
        tidx1=tidx1,
        tidx2=tidx2,
        fidxs=fidxs,
        ant_count=ant_count,
        ylog=ylog,
        scan_label=scan_label,
    )
    f_path = os.path.join(save_dir, "sys_gain_f_{}.jpg".format(scan_tag))
    fig_f.savefig(f_path, dpi=110, bbox_inches="tight")

    written = [t_path, f_path]

    t_latest = os.path.join(save_dir, "sys_gain_t_latest.jpg")
    f_latest = os.path.join(save_dir, "sys_gain_f_latest.jpg")
    if _should_update_latest(t_latest, scan_dt, prev_latest_t_dt):
        shutil.copy(t_path, t_latest)
        written.append(t_latest)
    if _should_update_latest(f_latest, scan_dt, prev_latest_f_dt):
        shutil.copy(f_path, f_latest)
        written.append(f_latest)

    return written


def run_batch(date_text=None, outdir=DEFAULT_OUTDIR, fidxs=DEFAULT_FIDXS, tidxs=DEFAULT_TIDXS, idb_root=DEFAULT_IDB_ROOT):
    date_tag = _normalize_date_tag(date_text)
    try:
        scan_paths = find_solpntcal_idbs_for_date(date_tag, idb_root=idb_root)
    except Exception as exc:
        print("Failed to scan IDB directory for {}: {}".format(date_tag, exc), file=sys.stderr)
        return []
    if not scan_paths:
        print("No SOLPNTCAL-style IDB scans found for {} in {}.".format(date_tag, os.path.join(idb_root, date_tag)))
        return []

    written = []
    for scan_path in scan_paths:
        try:
            written.extend(_save_scan_figures(scan_path, date_tag, outdir, fidxs, tidxs))
        except Exception as exc:
            print("Failed to process {}: {}".format(scan_path, exc), file=sys.stderr)
    return written


if TK_AVAILABLE:
    class App(object):
        def __init__(self):
            self.root = Tk()
            self.root.wm_title("Sys Gain Inspect (IDB)")

            self.out = None
            self.p_shape = None
            self.idb_path = StringVar()
            self.status = StringVar()
            self.status.set("Open an IDB file to begin.")

            self._build_ui()

        def _build_ui(self):
            top = Frame(self.root)
            top.pack(side=TOP, fill=X)

            Label(top, text="IDB file:").pack(side=LEFT, padx=4)
            e = Entry(top, textvariable=self.idb_path, width=70)
            e.pack(side=LEFT, padx=4, fill=X, expand=1)
            Button(top, text="Browse", command=self.browse).pack(side=LEFT, padx=4)
            Button(top, text="Load", command=self.load).pack(side=LEFT, padx=4)

            self.nb = ttk.Notebook(self.root)
            self.nb.pack(side=TOP, fill=BOTH, expand=1)

            self.tab_t = Frame(self.nb)
            self.tab_f = Frame(self.nb)
            self.nb.add(self.tab_t, text="T plot")
            self.nb.add(self.tab_f, text="F plot")

            self._build_t_tab(self.tab_t)
            self._build_f_tab(self.tab_f)

            status_frame = Frame(self.root)
            status_frame.pack(side=TOP, fill=X)
            Label(status_frame, textvariable=self.status, anchor=W).pack(side=LEFT, fill=X, expand=1, padx=4, pady=2)
            Button(status_frame, text="Exit", command=self.root.quit).pack(side=RIGHT, padx=4, pady=2)

        def _build_t_tab(self, parent):
            controls = Frame(parent)
            controls.pack(side=TOP, fill=X)

            self.t_pol = IntVar()
            self.t_fidx_text = StringVar()
            self.t_info = StringVar()
            self.t_info.set("pol=?, fidxs=?")

            Label(controls, text="pol").pack(side=LEFT, padx=3)
            self.t_pol_sb = Spinbox(controls, from_=0, to=1, width=4, textvariable=self.t_pol, command=self.update_t)
            self.t_pol_sb.pack(side=LEFT)

            Label(controls, text="fidxs").pack(side=LEFT, padx=3)
            self.t_f_entry = Entry(controls, textvariable=self.t_fidx_text, width=18)
            self.t_f_entry.pack(side=LEFT)
            self.t_f_entry.bind("<Return>", lambda e: self.update_all())
            self.t_f_entry.bind("<FocusOut>", lambda e: self.update_all())

            Button(controls, text="Update", command=self.update_all).pack(side=LEFT, padx=6)
            Label(controls, textvariable=self.t_info).pack(side=LEFT, padx=8)

            plot_frame = Frame(parent)
            plot_frame.pack(side=TOP, fill=BOTH, expand=1)

            self.fig_t = Figure(figsize=(10, 7))
            self.axes_t = _make_axes_grid(self.fig_t, 3, 5, sharex=True, sharey=True)
            self.canvas_t = FigureCanvasTkAgg(self.fig_t, plot_frame)
            self.canvas_t.draw()
            self.canvas_t.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
            toolbar = NavigationToolbar2Tk(self.canvas_t, plot_frame)
            toolbar.update()

        def _build_f_tab(self, parent):
            controls = Frame(parent)
            controls.pack(side=TOP, fill=X)

            self.f_pol = IntVar()
            self.f_tidx1 = IntVar()
            self.f_tidx2 = IntVar()
            self.f_info = StringVar()
            self.f_info.set("pol=?, tidx1=?, tidx2=?")

            Label(controls, text="pol").pack(side=LEFT, padx=3)
            self.f_pol_sb = Spinbox(controls, from_=0, to=1, width=4, textvariable=self.f_pol, command=self.update_f)
            self.f_pol_sb.pack(side=LEFT)

            Label(controls, text="tidx1").pack(side=LEFT, padx=3)
            self.f_t1_sb = Spinbox(controls, from_=0, to=1, width=8, textvariable=self.f_tidx1, command=self.update_f)
            self.f_t1_sb.pack(side=LEFT)

            Label(controls, text="tidx2").pack(side=LEFT, padx=3)
            self.f_t2_sb = Spinbox(controls, from_=0, to=1, width=8, textvariable=self.f_tidx2, command=self.update_f)
            self.f_t2_sb.pack(side=LEFT)

            Button(controls, text="Update", command=self.update_all).pack(side=LEFT, padx=6)
            Label(controls, textvariable=self.f_info).pack(side=LEFT, padx=8)

            plot_frame = Frame(parent)
            plot_frame.pack(side=TOP, fill=BOTH, expand=1)

            self.fig_f = Figure(figsize=(10, 7))
            self.axes_f = _make_axes_grid(self.fig_f, 3, 5, sharex=True, sharey=True)
            self.canvas_f = FigureCanvasTkAgg(self.fig_f, plot_frame)
            self.canvas_f.draw()
            self.canvas_f.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
            toolbar = NavigationToolbar2Tk(self.canvas_f, plot_frame)
            toolbar.update()

        def browse(self):
            initialdir = default_idb_dir()
            path = askdirectory(initialdir=initialdir, title="Select IDB dataset directory", mustexist=True)
            if path:
                self.idb_path.set(path)

        def load(self):
            path = self.idb_path.get().strip()
            if not path:
                self.status.set("No IDB file selected.")
                return
            if not os.path.exists(path):
                self.status.set("File not found: {0}".format(path))
                return
            if not os.path.isdir(path):
                self.status.set("Not an IDB dataset directory: {0}".format(path))
                return

            try:
                self.status.set("Loading IDB: {0}".format(path))
                self.root.update_idletasks()
                self.out = ri.read_idb([path])
                p = _get_power_array(self.out)
                self.p_shape = p.shape
            except Exception as e:
                self.out = None
                self.p_shape = None
                self.status.set("Load failed: {0}".format(e))
                return

            _, npol, nf, nt = self.p_shape
            default_fidxs = [f for f in DEFAULT_FIDXS if f < nf]
            if not default_fidxs:
                default_fidxs = [max(0, nf - 1)]

            self.t_pol.set(0)
            self.f_pol.set(0)
            self.t_fidx_text.set(",".join(str(v) for v in default_fidxs))
            self.f_tidx1.set(min(DEFAULT_TIDXS[0], nt - 1))
            self.f_tidx2.set(min(DEFAULT_TIDXS[1], nt - 1))

            self.t_pol_sb.config(to=max(0, npol - 1))
            self.f_pol_sb.config(to=max(0, npol - 1))
            self.f_t1_sb.config(to=max(0, nt - 1))
            self.f_t2_sb.config(to=max(0, nt - 1))

            self.status.set("Loaded: {0}  p.shape={1}".format(os.path.basename(path), self.p_shape))
            self.update_t()
            self.update_f()

        def update_t(self):
            if self.out is None:
                return
            pol = int(self.t_pol.get())
            fidxs = _parse_idx_list(self.t_fidx_text.get(), max_inclusive=self.p_shape[2] - 1)
            if len(fidxs) == 0:
                fidxs = [min(DEFAULT_FIDXS[0], self.p_shape[2] - 1)]
            tidx1 = int(self.f_tidx1.get())
            tidx2 = int(self.f_tidx2.get())
            info = plot_sys_gain_t(
                self.fig_t, self.axes_t, self.out, pols=[pol], fidxs=fidxs, tidx1=tidx1, tidx2=tidx2, ant_count=15
            )
            fmark = ",".join([str(f) for f in info["fidxs"]])
            if len(info.get("tidxs", ())) > 0:
                tmark = ",".join([str(t) for t in info["tidxs"]])
                self.t_info.set("pol={0}, fidx={1}, tidx={2} (nt={3}, nf={4})".format(pol, fmark, tmark, info["nt"], info["nf"]))
            else:
                self.t_info.set("pol={0}, fidx={1} (nt={2}, nf={3})".format(pol, fmark, info["nt"], info["nf"]))
            self.canvas_t.draw()

        def update_f(self):
            if self.out is None:
                return
            pol = int(self.f_pol.get())
            tidx1 = int(self.f_tidx1.get())
            tidx2 = int(self.f_tidx2.get())
            fidxs = _parse_idx_list(self.t_fidx_text.get(), max_inclusive=self.p_shape[2] - 1)
            info = plot_sys_gain_f(
                self.fig_f, self.axes_f, self.out, pols=[pol], tidx1=tidx1, tidx2=tidx2, fidxs=fidxs, ant_count=15
            )
            if len(info.get("fidxs", ())) > 0:
                fmark = ",".join([str(f) for f in info["fidxs"]])
                self.f_info.set(
                    "pol={0}, tidx={1},{2}, fidx={3} (nt={4}, nf={5})".format(
                        pol, info["tidxs"][0], info["tidxs"][1], fmark, info["nt"], info["nf"]
                    )
                )
            else:
                self.f_info.set("pol={0}, tidx={1},{2} (nt={3}, nf={4})".format(pol, info["tidxs"][0], info["tidxs"][1], info["nt"], info["nf"]))
            self.canvas_f.draw()

        def update_all(self):
            if self.out is None:
                return
            self.update_t()
            self.update_f()

        def run(self):
            self.root.mainloop()
else:
    class App(object):
        def __init__(self):
            raise RuntimeError("Tkinter is not available in this Python environment.")


def _build_parser():
    examples = """Examples:
  # Batch mode (default): UT today, default outdir/fidxs/tidxs
  python /common/python/eovsapy-src/eovsapy/sys_gain_inspect.py

  # Batch mode for a specific UTC date
  python /common/python/eovsapy-src/eovsapy/sys_gain_inspect.py --date 20260218

  # Batch mode with custom output directory and indices
  python /common/python/eovsapy-src/eovsapy/sys_gain_inspect.py --date 2026-02-18 --outdir /common/webplots/solpntcal --fidxs 120,320 --tidxs 405,415

  # GUI mode with a specific IDB dataset
  python /common/python/eovsapy-src/eovsapy/sys_gain_inspect.py --gui /data1/eovsa/fits/IDB/20260218/IDB20260218183022
"""
    parser = argparse.ArgumentParser(
        description="Plot EOVSA SOLPNTCAL system-gain diagnostics.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("idb_path", nargs="?", help="Optional IDB dataset path. If provided, GUI mode is used.")
    parser.add_argument("--gui", action="store_true", help="Launch interactive Tk GUI mode.")
    parser.add_argument("--date", type=str, default=None, help="UTC date (YYYYMMDD or YYYY-MM-DD). Default: today UTC.")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="Output directory for batch-mode JPG files.")
    parser.add_argument("--fidxs", type=str, default="120,320", help="Frequency indices, e.g. '120,320'.")
    parser.add_argument("--tidxs", type=str, default="405,415", help="Time indices, e.g. '405,415'.")
    parser.add_argument("--idb-root", type=str, default=DEFAULT_IDB_ROOT, help="IDB root directory (contains YYYYMMDD subdirs).")
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.gui or args.idb_path:
        if not TK_AVAILABLE:
            raise RuntimeError("GUI mode requested but Tkinter is unavailable.")
        app = App()
        if args.idb_path:
            app.idb_path.set(args.idb_path)
            app.load()
        app.run()
        return 0

    fidxs = _parse_idx_list(args.fidxs)
    if len(fidxs) == 0:
        fidxs = list(DEFAULT_FIDXS)
    tidx_pair = _parse_tidx_pair(args.tidxs, defaults=DEFAULT_TIDXS)

    written = run_batch(date_text=args.date, outdir=args.outdir, fidxs=fidxs, tidxs=tidx_pair, idb_root=args.idb_root)
    for p in written:
        print("Wrote {}".format(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
