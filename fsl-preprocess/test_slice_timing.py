"""Unit tests for slice_timing.py and ABIDE1 protocol resolution."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slice_timing import (
    acq_times_to_fsl_tcustom,
    compute_acquisition_times_sec,
    load_bids_slice_timing,
    load_protocol_yaml,
    resolve_drop_initial_trs,
    resolve_site_scheme,
)


def test_abide1_protocol_yaml() -> None:
    proto = load_protocol_yaml(Path(__file__).resolve().parent / "protocols" / "abide1.yaml")
    # Stanford must match UM_1 / UM_2 (seq_asc)
    assert resolve_site_scheme("UM_1", proto) == "seq_asc"
    assert resolve_site_scheme("UM_2", proto) == "seq_asc"
    assert resolve_site_scheme("Stanford", proto) == "seq_asc"
    assert resolve_site_scheme("SBL", proto) == "seq_desc"
    assert resolve_site_scheme("Olin", proto) == "seq_asc"
    assert resolve_site_scheme("SDSU", proto) == "ge_interleaved_asc"
    assert resolve_site_scheme("MaxMun_a", proto) == "siemens_interleaved_hf"
    assert resolve_site_scheme("NYU_a", proto) == "siemens_interleaved_fh"
    assert resolve_site_scheme("Caltech", proto) == "siemens_interleaved_fh"
    assert resolve_drop_initial_trs("NYU", proto) == 5
    assert resolve_drop_initial_trs("Olin", proto) == 5
    assert resolve_drop_initial_trs("NYU", proto, cli_override=3) == 3


def test_adhd200_protocol_yaml() -> None:
    proto = load_protocol_yaml(Path(__file__).resolve().parent / "protocols" / "adhd200.yaml")
    assert resolve_site_scheme("KKI", proto) == "seq_asc"
    for siemens_site in ("NYU", "NeuroIMAGE", "OHSU", "Peking_1", "Peking_2", "Peking_3", "Pittsburgh", "WashU", "Brown"):
        assert resolve_site_scheme(siemens_site, proto) == "siemens_interleaved_fh", siemens_site
    assert resolve_drop_initial_trs("OHSU", proto) == 0


def test_acquisition_orders_and_tcustom() -> None:
    tr = 2.0

    # 1. Sequential ascending (e.g. Stanford / UM_1 with N=40, TR=2.0)
    t_asc = compute_acquisition_times_sec(40, tr, "seq_asc", z_axis_code="S")
    assert abs(t_asc[0] - 0.0) < 1e-9
    assert abs(t_asc[39] - 1.95) < 1e-9
    off_asc = acq_times_to_fsl_tcustom(t_asc, tr)
    # ref_frac = (0 + 1.95)/(2*2.0) = 0.4875 -> slice 0 shifted +0.4875, slice 39 shifted -0.4875
    assert abs(off_asc[0] - 0.4875) < 1e-9
    assert abs(off_asc[39] - (-0.4875)) < 1e-9

    # 2. Sequential descending (SBL, N=4)
    t_desc = compute_acquisition_times_sec(4, tr, "seq_desc", z_axis_code="S")
    assert [round(v, 4) for v in t_desc] == [1.5, 1.0, 0.5, 0.0]

    # 3. Siemens interleaved F>>H: odd N vs even N
    # odd N=5 -> order [0, 2, 4, 1, 3]
    t_si_odd = compute_acquisition_times_sec(5, 2.5, "siemens_interleaved_fh", z_axis_code="S")
    assert [round(v, 4) for v in t_si_odd] == [0.0, 1.5, 0.5, 2.0, 1.0]
    # even N=4 -> order [1, 3, 0, 2]
    t_si_even = compute_acquisition_times_sec(4, tr, "siemens_interleaved_fh", z_axis_code="S")
    assert [round(v, 4) for v in t_si_even] == [1.0, 0.0, 1.5, 0.5]

    # 4. Siemens interleaved H>>F (MaxMun):
    # even N=4 -> scanner H>>F acquires s=[1,3,0,2] -> z=3-s=[2,0,3,1]
    t_hf_even = compute_acquisition_times_sec(4, tr, "siemens_interleaved_hf", z_axis_code="S")
    assert [round(v, 4) for v in t_hf_even] == [0.5, 1.5, 0.0, 1.0]
    # odd N=5 -> scanner H>>F acquires s=[0,2,4,1,3] -> z=4-s=[4,2,0,3,1]
    t_hf_odd = compute_acquisition_times_sec(5, 2.5, "siemens_interleaved_hf", z_axis_code="S")
    assert [round(v, 4) for v in t_hf_odd] == [1.0, 2.0, 0.5, 1.5, 0.0]

    # 5. GE interleaved bottom-up (SDSU): even parity still starts at z=0
    t_ge_even = compute_acquisition_times_sec(4, tr, "ge_interleaved_asc", z_axis_code="S")
    assert [round(v, 4) for v in t_ge_even] == [0.0, 1.0, 0.5, 1.5]

    # 6. Z-axis flip ('I') reverses spatial slice indices
    t_asc_i = compute_acquisition_times_sec(4, tr, "seq_asc", z_axis_code="I")
    assert [round(v, 4) for v in t_asc_i] == [1.5, 1.0, 0.5, 0.0]


def test_bids_multiband_sidecar() -> None:
    with tempfile.TemporaryDirectory() as td:
        jp = Path(td) / "test_bold.json"
        # Multiband=2, 4 slices, TR=1.0 -> slices (0,2) at 0.0s, (1,3) at 0.5s
        jp.write_text(json.dumps({"RepetitionTime": 1.0, "SliceTiming": [0.0, 0.5, 0.0, 0.5]}), encoding="utf-8")
        st, tr = load_bids_slice_timing(jp)
        assert tr == 1.0 and st == [0.0, 0.5, 0.0, 0.5]
        off = acq_times_to_fsl_tcustom(st, tr)
        assert [round(v, 4) for v in off] == [0.25, -0.25, 0.25, -0.25]


if __name__ == "__main__":
    test_abide1_protocol_yaml()
    test_adhd200_protocol_yaml()
    test_acquisition_orders_and_tcustom()
    test_bids_multiband_sidecar()
    print("ALL SLICE TIMING TESTS PASSED")
