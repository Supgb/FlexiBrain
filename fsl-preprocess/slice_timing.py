"""General slice-timing resolution and FSL slicetimer --tcustom conversion utilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import yaml

SUPPORTED_SCHEMES = (
    "seq_asc",
    "seq_desc",
    "siemens_interleaved_fh",
    "siemens_interleaved_hf",
    "ge_interleaved_asc",
    "skip",
)


def load_protocol_yaml(path: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """Load a site slice-timing protocol YAML mapping."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Protocol YAML not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Protocol YAML must be a mapping, got {type(data).__name__}")
    return data


def resolve_site_scheme(site: str, protocol: Optional[Dict[str, Any]]) -> Optional[str]:
    """Resolve a slice-timing scheme name for a site from a protocol mapping."""
    if not protocol:
        return None
    sites = protocol.get("sites") or {}
    if site in sites:
        return sites[site]

    site_lower = site.lower()
    for key, val in sites.items():
        if str(key).lower() == site_lower:
            return val

    # Fallback: match base site prefix (e.g., NYU_1 -> NYU)
    prefix = site.split("_")[0].split("-")[0]
    for key, val in sites.items():
        if str(key).lower() == prefix.lower():
            return val

    return protocol.get("default_scheme")


def resolve_drop_initial_trs(
    site: str,
    protocol: Optional[Dict[str, Any]],
    cli_override: Optional[int] = None,
) -> int:
    """Resolve number of initial dummy TRs to drop for a site."""
    if cli_override is not None:
        return max(0, int(cli_override))
    if not protocol:
        return 0
    per_site = protocol.get("drop_initial_trs_by_site") or {}
    if site in per_site:
        return max(0, int(per_site[site]))

    site_lower = site.lower()
    for key, val in per_site.items():
        if str(key).lower() == site_lower:
            return max(0, int(val))

    prefix = site.split("_")[0].split("-")[0]
    for key, val in per_site.items():
        if str(key).lower() == prefix.lower():
            return max(0, int(val))

    default_val = protocol.get("default_drop_initial_trs", 0)
    return max(0, int(default_val or 0))


def compute_acquisition_times_sec(
    num_slices: int,
    tr: float,
    scheme: str,
    z_axis_code: str = "S",
) -> List[float]:
    """Compute per-slice acquisition times in seconds indexed along NIfTI Z axis.

    Parameters
    ----------
    num_slices:
        Number of slices along NIfTI axis 2 (Z).
    tr:
        Repetition time in seconds.
    scheme:
        One of 'seq_asc', 'seq_desc', 'siemens_interleaved_fh',
        'siemens_interleaved_hf', or 'ge_interleaved_asc'.
    z_axis_code:
        Anatomical direction of increasing NIfTI Z index from nibabel.aff2axcodes:
        'S' (Inferior -> Superior / Feet -> Head) or 'I' (Superior -> Inferior / Head -> Feet).

    Returns
    -------
    List of acquisition times (seconds) of length `num_slices` indexed by NIfTI slice index z.
    """
    if num_slices <= 0:
        raise ValueError(f"num_slices must be positive, got {num_slices}")
    if not math.isfinite(tr) or tr <= 0:
        raise ValueError(f"TR must be positive and finite, got {tr}")
    if z_axis_code not in {"S", "I"}:
        raise ValueError(f"Unsupported NIfTI z_axis_code {z_axis_code!r}; expected 'S' or 'I'")

    n = int(num_slices)
    # 0-based acquisition order in anatomical Feet-to-Head ('S') coordinates
    if scheme == "seq_asc":
        order_fh = list(range(n))
    elif scheme == "seq_desc":
        order_fh = list(range(n - 1, -1, -1))
    elif scheme == "siemens_interleaved_fh":
        if n % 2 == 1:
            order_fh = list(range(0, n, 2)) + list(range(1, n, 2))
        else:
            order_fh = list(range(1, n, 2)) + list(range(0, n, 2))
    elif scheme == "siemens_interleaved_hf":
        # Scanner acquires H >> F interleaved (even/odd parity in H >> F slice numbering)
        if n % 2 == 1:
            order_hf = list(range(0, n, 2)) + list(range(1, n, 2))
        else:
            order_hf = list(range(1, n, 2)) + list(range(0, n, 2))
        order_fh = [(n - 1) - s for s in order_hf]
    elif scheme == "ge_interleaved_asc":
        # GE bottom-up interleaved always acquires 1st, 3rd, 5th... (0-based even) then 2nd, 4th...
        order_fh = list(range(0, n, 2)) + list(range(1, n, 2))
    else:
        raise ValueError(f"Unknown slice acquisition scheme: {scheme!r}")

    if z_axis_code == "I":
        order_nifti = [(n - 1) - z for z in order_fh]
    else:
        order_nifti = order_fh

    if len(order_nifti) != n or set(order_nifti) != set(range(n)):
        raise RuntimeError(f"Invalid slice permutation produced for scheme={scheme}, n={n}")

    dt = float(tr) / float(n)
    acq_times = [0.0] * n
    for rank, z_idx in enumerate(order_nifti):
        acq_times[z_idx] = rank * dt
    return acq_times


def load_bids_slice_timing(json_path: Optional[Union[str, Path]]) -> Tuple[Optional[List[float]], Optional[float]]:
    """Read SliceTiming (seconds) and RepetitionTime (seconds) from a BIDS JSON sidecar."""
    if json_path is None:
        return None, None
    p = Path(json_path)
    if not p.exists():
        return None, None
    data = json.loads(p.read_text(encoding="utf-8"))
    st = data.get("SliceTiming")
    tr = data.get("RepetitionTime")
    st_list = [float(v) for v in st] if isinstance(st, list) and len(st) > 0 else None
    tr_val = float(tr) if isinstance(tr, (int, float)) and tr > 0 else None
    return st_list, tr_val


def acq_times_to_fsl_tcustom(acq_times_sec: Sequence[float], tr: float) -> List[float]:
    """Convert acquisition times (seconds from TR start) to FSL slicetimer --tcustom shifts.

    Per FSL slicetimer.cc, positive values in --tcustom shift slices forward in time.
    To align all slices to the temporal midpoint of the volume acquisition:
        ref_frac = 0.5 * (min(t) + max(t)) / TR
        offset[z] = ref_frac - (acq_times_sec[z] / TR)
    For uniform single-band acquisition (t_k = k * TR / N), ref_frac = (N - 1) / (2 * N),
    which matches FSL's internal `recenter` calculation bit-for-bit.
    """
    if len(acq_times_sec) == 0:
        raise ValueError("acq_times_sec cannot be empty")
    if not math.isfinite(tr) or tr <= 0:
        raise ValueError(f"TR must be positive and finite, got {tr}")

    times = [float(t) for t in acq_times_sec]
    ref_frac = 0.5 * (min(times) + max(times)) / float(tr)
    offsets = [ref_frac - (t / float(tr)) for t in times]
    if any((not math.isfinite(o)) or o <= -0.55 or o >= 0.55 for o in offsets):
        raise ValueError(f"Computed FSL tcustom offsets out of expected bounds (-0.55, 0.55): {offsets}")
    return offsets


def write_tcustom_file(offsets: Sequence[float], path: Union[str, Path]) -> Path:
    """Write single-column FSL slicetimer --tcustom timing file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(f"{float(v):.8f}\n" for v in offsets)
    out_path.write_text(content, encoding="utf-8")
    return out_path
