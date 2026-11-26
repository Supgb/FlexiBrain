#!/usr/bin/env python3
"""Single-run FSL native-space fMRI preprocessing worker (STC + HMC + BET + masking)."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for _p in (str(THIS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_process import center_affine_origin  # noqa: E402
from slice_timing import (  # noqa: E402
    acq_times_to_fsl_tcustom,
    compute_acquisition_times_sec,
    load_bids_slice_timing,
    write_tcustom_file,
)


def build_fsl_env(fsl_dir: Path, tmp_dir: Optional[Path] = None) -> Dict[str, str]:
    """Construct subprocess environment for single-threaded FSL execution."""
    env = os.environ.copy()
    fsl_bin = str(fsl_dir / "bin")
    env["FSLDIR"] = str(fsl_dir)
    env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    env["PATH"] = f"{fsl_bin}:{env.get('PATH', '')}"
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    if tmp_dir is not None:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(tmp_dir)
        env["TEMP"] = str(tmp_dir)
        env["TMP"] = str(tmp_dir)
    return env


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.tmp.{os.getpid()}"
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _atomic_save_nifti(img: nib.Nifti1Image, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # nibabel infers .nii.gz from filename extension, so keep .nii.gz suffix on temp path
    tmp = dst.parent / f".tmp.{os.getpid()}.{dst.name}"
    try:
        nib.save(img, str(tmp))
        os.replace(tmp, dst)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _run_cmd(cmd: List[str], env: Dict[str, str], cwd: Path) -> None:
    res = subprocess.run(
        cmd,
        env=env,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={res.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{res.stdout[-1500:]}\nSTDERR:\n{res.stderr[-1500:]}"
        )


def _compute_fd_stats(par_path: Path, radius_mm: float = 50.0) -> Tuple[float, float]:
    """Compute Power framewise displacement summary from mcflirt .par (rot_rad[3], trans_mm[3])."""
    if not par_path.exists():
        return 0.0, 0.0
    arr = np.loadtxt(par_path)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[0] < 2 or arr.shape[1] < 6:
        return 0.0, 0.0
    diffs = np.abs(np.diff(arr[:, :6], axis=0))
    fd = diffs[:, :3].sum(axis=1) * radius_mm + diffs[:, 3:6].sum(axis=1)
    return float(fd.mean()), float(fd.max())


def _detect_nonsteady_prefix(arr4d: np.ndarray, max_check: int = 10, z_thresh: float = 4.0) -> int:
    """Detect contiguous initial non-steady-state volumes using robust MAD over post-prefix baseline."""
    t_len = int(arr4d.shape[3])
    k_limit = min(int(max_check), max(0, t_len - 15))
    if k_limit <= 0 or t_len < 20:
        return 0
    frame_means = np.asarray(arr4d, dtype=np.float64).mean(axis=(0, 1, 2))
    steady = frame_means[k_limit:]
    med = float(np.median(steady))
    mad = float(np.median(np.abs(steady - med)))
    scale = max(1.4826 * mad, float(steady.std()) * 0.5, 1e-6)
    z_scores = np.abs(frame_means[:k_limit] - med) / scale
    n_drop = 0
    for t in range(k_limit):
        if z_scores[t] > z_thresh:
            n_drop = t + 1
        else:
            break
    return n_drop


def _extract_native_meta(nifti_path: Path) -> Tuple[Tuple[int, ...], Tuple[float, float, float], float, np.ndarray, nib.Nifti1Header, str]:
    img = nib.load(str(nifti_path))
    shape = img.shape
    if len(shape) != 4:
        raise ValueError(f"Expected 4D fMRI NIfTI, got shape {shape} for {nifti_path}")
    if shape[3] < 10:
        raise ValueError(f"Too few timepoints (T={shape[3]}) in {nifti_path}")

    header = img.header.copy()
    zooms = header.get_zooms()
    space_u, time_u = header.get_xyzt_units()

    vx, vy, vz = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    if space_u == "m":
        vx, vy, vz = vx * 1000.0, vy * 1000.0, vz * 1000.0
    if any((not np.isfinite(v)) or v <= 0 for v in (vx, vy, vz)):
        raise ValueError(f"Invalid voxel spacing {(vx, vy, vz)} in {nifti_path}")

    tr = float(zooms[3]) if len(zooms) >= 4 else 0.0
    if time_u == "msec":
        tr = tr / 1000.0
    elif time_u in ("usec", "microsec"):
        tr = tr / 1e6

    affine = img.affine.copy()
    axcodes = nib.aff2axcodes(affine)
    z_axis_code = axcodes[2]
    return shape, (vx, vy, vz), tr, affine, header, z_axis_code


def _run_bet_with_fallback(
    mean_bold: Path,
    work_dir: Path,
    env: Dict[str, str],
    fsl_dir: Path,
    bet_f: float = 0.3,
) -> Tuple[Path, float, str]:
    """Run BET (preferring `bet`, falling back to `bet2`) and check foreground mask fraction."""
    bet_bin = shutil.which("bet", path=env["PATH"])
    bet2_bin = shutil.which("bet2", path=env["PATH"])
    if bet_bin is None and bet2_bin is None:
        raise FileNotFoundError(f"Neither 'bet' nor 'bet2' found under {fsl_dir / 'bin'}")

    out_prefix = work_dir / "brain"
    mask_path = work_dir / "brain_mask.nii.gz"

    def invoke(f_val: float, use_robust: bool) -> float:
        for cand in (mask_path, work_dir / "brain.nii.gz"):
            if cand.exists():
                cand.unlink()
        if bet_bin is not None:
            cmd = [bet_bin, str(mean_bold), str(out_prefix), "-f", f"{f_val:.2f}", "-m", "-n"]
            if use_robust:
                cmd.append("-R")
        else:
            cmd = [bet2_bin, str(mean_bold), str(out_prefix), "-f", f"{f_val:.2f}", "-m"]
        _run_cmd(cmd, env=env, cwd=work_dir)
        if not mask_path.exists():
            raise FileNotFoundError(f"BET did not produce mask at {mask_path}")
        mask_arr = np.asarray(nib.load(str(mask_path)).dataobj) > 0
        return float(mask_arr.mean())

    fg_frac = invoke(bet_f, use_robust=False)
    note = f"bet_f={bet_f:.2f}"
    if fg_frac < 0.03:
        fg_frac = invoke(0.25, use_robust=(bet_bin is not None))
        note = f"bet_fallback_low(f=0.25,R={bet_bin is not None})"
    elif fg_frac > 0.85:
        fg_frac = invoke(0.40, use_robust=(bet_bin is not None))
        note = f"bet_fallback_high(f=0.40,R={bet_bin is not None})"
    return mask_path, fg_frac, note


def process_single_run(
    run_id: str,
    out_stem: str,
    site: str,
    raw_path: Path,
    output_root: Path,
    fsl_dir: Path,
    slice_scheme: Optional[str] = None,
    sidecar_json: Optional[Path] = None,
    bet_f: float = 0.3,
    keep_work: bool = False,
    cleanup_staged_raw: bool = False,
    staged_dir: Optional[Path] = None,
    drop_initial_trs: int = 0,
    auto_drop_nonsteady: bool = False,
    skip_stc: bool = False,
    center_origin: str = "foreground",
) -> Dict[str, Any]:
    """Run TR trimming -> optional STC -> HMC -> BET -> native masking & origin centering."""
    t0 = time.perf_counter()
    work_dir = output_root / "work" / "runs" / run_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    env = build_fsl_env(fsl_dir, tmp_dir=work_dir / "tmp")

    out_nifti = output_root / "fsl_native" / "nativespace" / f"{out_stem}.nii.gz"
    out_mask = output_root / "masks" / f"{out_stem}_brain_mask.nii.gz"
    out_par = output_root / "qc" / "motion" / f"{out_stem}.par"

    try:
        orig_shape, voxel_mm, tr_sec, orig_affine, orig_header, z_axis_code = _extract_native_meta(raw_path)
        bids_st, bids_tr = load_bids_slice_timing(sidecar_json)
        if (not np.isfinite(tr_sec)) or tr_sec <= 0:
            if bids_tr is not None and bids_tr > 0:
                tr_sec = float(bids_tr)
            else:
                raise ValueError(f"Missing/invalid TR ({tr_sec}) in both NIfTI header and sidecar for {raw_path}")

        # 0. Drop initial non-steady-state / T1 equilibration dummy TRs
        base_drop = max(0, int(drop_initial_trs))
        orig_t = int(orig_shape[3])
        if auto_drop_nonsteady and orig_t - base_drop >= 20:
            raw_img = nib.load(str(raw_path))
            raw_arr = np.asarray(raw_img.dataobj)
            extra_drop = _detect_nonsteady_prefix(raw_arr[..., base_drop:])
            drop_n = base_drop + extra_drop
        else:
            raw_img = None
            raw_arr = None
            drop_n = base_drop
        drop_n = min(drop_n, max(0, orig_t - 15))

        if drop_n > 0:
            if raw_arr is None:
                raw_img = nib.load(str(raw_path))
                raw_arr = np.asarray(raw_img.dataobj)
            trimmed_arr = np.ascontiguousarray(raw_arr[..., drop_n:], dtype=np.float32)
            trim_header = orig_header.copy()
            trim_header.set_data_dtype(np.float32)
            trim_header.set_zooms((voxel_mm[0], voxel_mm[1], voxel_mm[2], tr_sec))
            trim_header.set_xyzt_units(xyz="mm", t="sec")
            trim_img = nib.Nifti1Image(trimmed_arr, orig_affine, header=trim_header)
            trim_img.set_data_dtype(np.float32)
            trimmed_path = work_dir / "trimmed.nii.gz"
            _atomic_save_nifti(trim_img, trimmed_path)
            del raw_arr, trimmed_arr
            stc_input = trimmed_path
            target_shape = (orig_shape[0], orig_shape[1], orig_shape[2], orig_t - drop_n)
        else:
            del raw_arr
            stc_input = raw_path
            target_shape = orig_shape

        # 1. Slice-Timing Correction (explicit skip takes precedence over BIDS JSON and protocol YAML)
        stc_source = stc_input
        stc_method = "skipped"
        num_slices = target_shape[2]
        if skip_stc or slice_scheme == "skip":
            offsets = None
            stc_method = "skipped"
        elif bids_st is not None and len(bids_st) == num_slices:
            offsets = acq_times_to_fsl_tcustom(bids_st, tr_sec)
            stc_method = "bids_json"
        elif slice_scheme:
            acq_times = compute_acquisition_times_sec(num_slices, tr_sec, slice_scheme, z_axis_code=z_axis_code)
            offsets = acq_times_to_fsl_tcustom(acq_times, tr_sec)
            stc_method = f"protocol:{slice_scheme}"
        else:
            offsets = None

        if offsets is not None:
            tcustom_path = work_dir / "tcustom.txt"
            write_tcustom_file(offsets, tcustom_path)
            stc_out = work_dir / "stc.nii.gz"
            slicetimer_bin = shutil.which("slicetimer", path=env["PATH"]) or str(fsl_dir / "bin" / "slicetimer")
            _run_cmd(
                [
                    slicetimer_bin,
                    "-i", str(stc_input),
                    "-o", str(stc_out),
                    "-r", f"{tr_sec:.6f}",
                    f"--tcustom={tcustom_path}",
                ],
                env=env,
                cwd=work_dir,
            )
            stc_source = stc_out

        # 2. Head-Motion Correction (native space rigid 6-DOF)
        mcf_prefix = work_dir / "mcf"
        mcflirt_bin = shutil.which("mcflirt", path=env["PATH"]) or str(fsl_dir / "bin" / "mcflirt")
        _run_cmd(
            [mcflirt_bin, "-in", str(stc_source), "-out", str(mcf_prefix), "-meanvol", "-plots"],
            env=env,
            cwd=work_dir,
        )
        mcf_bold = work_dir / "mcf.nii.gz"
        mcf_mean = work_dir / "mcf_mean_reg.nii.gz"
        mcf_par = work_dir / "mcf.par"
        if not mcf_bold.exists() or not mcf_mean.exists():
            raise FileNotFoundError(f"mcflirt missing expected outputs in {work_dir}")
        mean_fd, max_fd = _compute_fd_stats(mcf_par)
        if mcf_par.exists():
            _atomic_copy(mcf_par, out_par)

        # 3. Brain Extraction on motion-corrected mean volume
        mask_path, fg_frac, bet_note = _run_bet_with_fallback(mcf_mean, work_dir, env, fsl_dir, bet_f=bet_f)

        # 4. Mask 4D BOLD via fslmaths & guarantee exact 0.0 background + native header metadata
        masked_tmp = work_dir / "masked.nii.gz"
        fslmaths_bin = shutil.which("fslmaths", path=env["PATH"]) or str(fsl_dir / "bin" / "fslmaths")
        _run_cmd(
            [fslmaths_bin, str(mcf_bold), "-mas", str(mask_path), str(masked_tmp), "-odt", "float"],
            env=env,
            cwd=work_dir,
        )

        masked_img = nib.load(str(masked_tmp))
        if masked_img.shape != target_shape:
            raise ValueError(f"Output shape mismatch for {run_id}: target={target_shape}, masked={masked_img.shape}")

        mask_img_tmp = nib.load(str(mask_path))
        mask_arr = np.asarray(mask_img_tmp.dataobj) > 0
        arr = np.asarray(masked_img.dataobj, dtype=np.float32).copy()
        if not np.isfinite(arr).all():
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr[~mask_arr, :] = 0.0

        # Preserve native voxel spacing, orientation, and obliquity; optionally center world origin
        out_affine, raw_origin_mm, centered_origin_mm = center_affine_origin(
            orig_affine,
            target_shape[:3],
            foreground_mask=mask_arr,
            mode=center_origin,
        )
        out_header = orig_header.copy()
        out_header.set_data_dtype(np.float32)
        out_header.set_zooms((voxel_mm[0], voxel_mm[1], voxel_mm[2], tr_sec))
        out_header.set_xyzt_units(xyz="mm", t="sec")
        qcode = int(out_header.get("qform_code", 1)) or 1
        scode = int(out_header.get("sform_code", 1)) or 1
        out_img = nib.Nifti1Image(arr, out_affine, header=out_header)
        out_img.set_data_dtype(np.float32)
        out_img.set_qform(out_affine, code=qcode)
        out_img.set_sform(out_affine, code=scode)
        _atomic_save_nifti(out_img, out_nifti)

        # Save brain mask with identical affine/qform/sform so mask and BOLD stay world-aligned
        mask_header = mask_img_tmp.header.copy()
        mask_header.set_data_dtype(np.uint8)
        mask_header.set_zooms((voxel_mm[0], voxel_mm[1], voxel_mm[2]))
        mask_header.set_xyzt_units(xyz="mm", t="unknown")
        mask_out_img = nib.Nifti1Image(mask_arr.astype(np.uint8), out_affine, header=mask_header)
        mask_out_img.set_data_dtype(np.uint8)
        mask_out_img.set_qform(out_affine, code=qcode)
        mask_out_img.set_sform(out_affine, code=scode)
        _atomic_save_nifti(mask_out_img, out_mask)

        if cleanup_staged_raw and staged_dir is not None:
            try:
                if raw_path.resolve().is_relative_to(staged_dir.resolve()):
                    raw_path.unlink(missing_ok=True)
            except Exception:
                pass

        elapsed = time.perf_counter() - t0
        return {
            "run_id": run_id,
            "out_stem": out_stem,
            "site": site,
            "status": "ok",
            "elapsed_sec": round(elapsed, 2),
            "orig_t": orig_t,
            "dropped_trs": int(drop_n),
            "shape": list(target_shape),
            "voxel_mm": [round(v, 5) for v in voxel_mm],
            "tr_sec": round(tr_sec, 6),
            "z_axis": z_axis_code,
            "stc_method": stc_method,
            "origin_mode": center_origin,
            "raw_origin_mm": [round(v, 3) for v in raw_origin_mm],
            "centered_origin_mm": [round(v, 4) for v in centered_origin_mm],
            "bet_note": bet_note,
            "fg_frac": round(fg_frac, 5),
            "mean_fd_mm": round(mean_fd, 5),
            "max_fd_mm": round(max_fd, 5),
            "fsl_native_path": str(out_nifti),
            "mask_path": str(out_mask),
            "error": "",
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "run_id": run_id,
            "out_stem": out_stem,
            "site": site,
            "status": "failed",
            "elapsed_sec": round(elapsed, 2),
            "orig_t": 0,
            "dropped_trs": 0,
            "shape": [],
            "voxel_mm": [],
            "tr_sec": 0.0,
            "z_axis": "",
            "stc_method": "",
            "origin_mode": center_origin,
            "raw_origin_mm": [],
            "centered_origin_mm": [],
            "bet_note": "",
            "fg_frac": 0.0,
            "mean_fd_mm": 0.0,
            "max_fd_mm": 0.0,
            "fsl_native_path": str(out_nifti),
            "mask_path": str(out_mask),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if not keep_work and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run native-space FSL preprocessing on a single 4D BOLD file.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-stem", required=True)
    parser.add_argument("--site", default="unknown")
    parser.add_argument("--raw-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fsl-dir", type=Path, default=Path(os.environ.get("FSLDIR", "/mnt/ironwolf/conda-envs/fsl-tools")))
    parser.add_argument("--slice-scheme", default=None)
    parser.add_argument("--sidecar-json", type=Path, default=None)
    parser.add_argument("--bet-f", type=float, default=0.3)
    parser.add_argument("--drop-initial-trs", type=int, default=0)
    parser.add_argument("--auto-drop-nonsteady", action="store_true")
    parser.add_argument("--skip-stc", action="store_true")
    parser.add_argument("--center-origin", choices=["foreground", "grid", "none"], default="foreground")
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()

    res = process_single_run(
        run_id=args.run_id,
        out_stem=args.out_stem,
        site=args.site,
        raw_path=args.raw_path,
        output_root=args.output_root,
        fsl_dir=args.fsl_dir,
        slice_scheme=args.slice_scheme,
        sidecar_json=args.sidecar_json,
        bet_f=args.bet_f,
        keep_work=args.keep_work,
        drop_initial_trs=args.drop_initial_trs,
        auto_drop_nonsteady=args.auto_drop_nonsteady,
        skip_stc=args.skip_stc,
        center_origin=args.center_origin,
    )
    print(res)
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
