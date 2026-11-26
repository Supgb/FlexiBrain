#!/usr/bin/env python3
"""General parallel FSL native-space fMRI preprocessing + FlexiBrain 96x96x96 conversion orchestrator."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for p in (str(THIS_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import data_process  # noqa: E402
from datasets import ADAPTERS, RunRecord  # noqa: E402
from fsl_native_worker import build_fsl_env, process_single_run  # noqa: E402
from slice_timing import load_protocol_yaml, resolve_drop_initial_trs  # noqa: E402


MANIFEST_FIELDS = [
    "run_id",
    "out_stem",
    "site",
    "status",
    "elapsed_sec",
    "orig_t",
    "dropped_trs",
    "shape",
    "voxel_mm",
    "tr_sec",
    "z_axis",
    "stc_method",
    "origin_mode",
    "raw_origin_mm",
    "centered_origin_mm",
    "bet_note",
    "fg_frac",
    "mean_fd_mm",
    "max_fd_mm",
    "fsl_native_path",
    "mask_path",
    "error",
]


def _filter_records(
    records: List[RunRecord],
    subjects: Optional[List[str]],
    limit: Optional[int],
) -> List[RunRecord]:
    if subjects:
        wanted = {s.strip().lstrip("0") or "0" for s in subjects}
        wanted_raw = {s.strip() for s in subjects}
        records = [
            r for r in records
            if (r.subject_id.lstrip("0") or "0") in wanted or r.subject_id in wanted_raw
        ]
    if limit is not None and limit > 0:
        records = records[:limit]
    return records


def _run_fsl_worker_task(
    task: Tuple[RunRecord, Path, Path, Path, float, bool, bool, Path, int, bool, bool, str]
) -> Dict[str, Any]:
    (
        rec,
        raw_path,
        output_root,
        fsl_dir,
        bet_f,
        keep_work,
        cleanup_staged,
        stage_dir,
        drop_initial_trs,
        auto_drop_nonsteady,
        skip_stc,
        center_origin,
    ) = task
    return process_single_run(
        run_id=rec.run_id,
        out_stem=rec.out_stem,
        site=rec.site,
        raw_path=raw_path,
        output_root=output_root,
        fsl_dir=fsl_dir,
        slice_scheme=rec.slice_scheme,
        sidecar_json=Path(rec.sidecar_json) if rec.sidecar_json else None,
        bet_f=bet_f,
        keep_work=keep_work,
        cleanup_staged_raw=cleanup_staged,
        staged_dir=stage_dir,
        drop_initial_trs=drop_initial_trs,
        auto_drop_nonsteady=auto_drop_nonsteady,
        skip_stc=skip_stc,
        center_origin=center_origin,
    )


def _run_data_process_task(task: Tuple[Path, Path, Path, Path, bool]) -> Tuple[str, str]:
    in_file, out_file, input_root, output_root, overwrite = task
    dp_args = data_process.build_arg_parser().parse_args(
        [
            "--input-root", str(input_root),
            "--output-root", str(output_root),
            "--spaces", "native",
            "--target-shape", "96,96,96",
        ]
        + (["--overwrite"] if overwrite else [])
    )
    if out_file.exists() and not overwrite:
        return "skipped_existing", str(out_file)
    status = data_process.process_one_file(in_file, out_file, "native", None, dp_args)
    return status, str(out_file)


def _qc_one_flexibrain_file(task: Tuple[Path, str]) -> Dict[str, Any]:
    path, expected_center_origin = task
    try:
        img = nib.load(str(path))
        shape = img.shape
        zooms = tuple(float(z) for z in img.header.get_zooms())
        units = img.header.get_xyzt_units()
        arr = np.asarray(img.dataobj, dtype=np.float32)

        ok = True
        reasons: List[str] = []
        if len(shape) != 4 or shape[:3] != (96, 96, 96) or shape[3] < 10:
            ok = False
            reasons.append(f"bad_shape={shape}")
        if any((not np.isfinite(z)) or z <= 0 for z in zooms[:4]):
            ok = False
            reasons.append(f"bad_zooms={zooms}")
        if units != ("mm", "sec"):
            ok = False
            reasons.append(f"bad_units={units}")
        if not np.isfinite(arr).all():
            ok = False
            reasons.append("nonfinite_values")

        bg_mask = (arr == 0).all(axis=3)
        bg_zero_frac = float(bg_mask.mean())
        fg_mask = ~bg_mask
        fg_vals = arr[fg_mask, :]
        if fg_vals.size == 0:
            ok = False
            reasons.append("empty_foreground")
            fg_mean, fg_std = 0.0, 0.0
            origin_mm = [0.0, 0.0, 0.0]
            origin_norm_mm = 0.0
        else:
            fg_mean = float(fg_vals.mean())
            fg_std = float(fg_vals.std())
            if abs(fg_mean) > 1e-2 or not (0.95 <= fg_std <= 1.05):
                ok = False
                reasons.append(f"bad_zscore(mean={fg_mean:.4f},std={fg_std:.4f})")

            if expected_center_origin == "grid":
                c_vox = (np.asarray(shape[:3], dtype=np.float64) - 1.0) / 2.0
            else:
                c_vox = np.argwhere(fg_mask).mean(axis=0).astype(np.float64)
            origin_vec = img.affine[:3, :3] @ c_vox + img.affine[:3, 3]
            origin_mm = [round(float(v), 4) for v in origin_vec]
            origin_norm_mm = float(np.linalg.norm(origin_vec))
            if expected_center_origin in ("foreground", "grid") and origin_norm_mm > 1e-2:
                ok = False
                reasons.append(f"uncentered_origin(mode={expected_center_origin},norm_mm={origin_norm_mm:.4f})")

        qform_sform_max_diff = float(np.abs(img.get_qform() - img.get_sform()).max())
        if not np.isfinite(qform_sform_max_diff) or qform_sform_max_diff > 1e-2:
            ok = False
            reasons.append(f"qform_sform_mismatch(max_diff={qform_sform_max_diff:.5f})")

        if not (0.50 <= bg_zero_frac <= 0.992):
            ok = False
            reasons.append(f"suspect_bg_zero_frac={bg_zero_frac:.4f}")

        return {
            "path": str(path),
            "name": path.name,
            "ok": ok,
            "reasons": reasons,
            "shape": list(shape),
            "zooms": [round(z, 5) for z in zooms],
            "units": list(units),
            "bg_zero_frac": round(bg_zero_frac, 5),
            "fg_mean": round(fg_mean, 6),
            "fg_std": round(fg_std, 6),
            "origin_mm": origin_mm,
            "origin_norm_mm": round(origin_norm_mm, 5),
            "qform_sform_max_diff": round(qform_sform_max_diff, 6),
        }
    except Exception as exc:
        return {
            "path": str(path),
            "name": path.name,
            "ok": False,
            "reasons": [f"{type(exc).__name__}: {exc}"],
            "shape": [],
            "zooms": [],
            "units": [],
            "bg_zero_frac": 0.0,
            "fg_mean": 0.0,
            "fg_std": 0.0,
            "origin_mm": [],
            "origin_norm_mm": 0.0,
            "qform_sform_max_diff": 0.0,
        }


def _check_fsl_binaries(fsl_dir: Path, require_slicetimer: bool = True) -> None:
    env = build_fsl_env(fsl_dir)
    required = ["mcflirt", "fslmaths"]
    if require_slicetimer:
        required.insert(0, "slicetimer")

    missing = []
    for req in required:
        if shutil.which(req, path=env["PATH"]) is None:
            missing.append(req)
    if shutil.which("bet", path=env["PATH"]) is None and shutil.which("bet2", path=env["PATH"]) is None:
        missing.append("bet/bet2")
    if missing:
        raise FileNotFoundError(f"Missing required FSL tools under {fsl_dir / 'bin'}: {missing}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parallel native-space FSL + FlexiBrain preprocessing orchestrator.")
    p.add_argument("--dataset", choices=sorted(ADAPTERS.keys()), required=True)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument(
        "--fsl-dir",
        type=Path,
        default=Path(os.environ.get("FSLDIR", "/mnt/ironwolf/conda-envs/fsl-tools")),
    )
    p.add_argument("--protocol-yaml", type=Path, default=None)
    p.add_argument("--index-json", type=Path, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--stage-workers", type=int, default=6)
    p.add_argument("--bet-f", type=float, default=0.3)
    p.add_argument("--drop-initial-trs", type=int, default=None, help="Override initial dummy TRs to drop per run before STC/HMC.")
    p.add_argument("--auto-drop-nonsteady", action="store_true", help="Extend TR trimming past contiguous initial non-steady outlier volumes.")
    p.add_argument("--skip-stc", action="store_true", help="Disable slice-timing correction regardless of BIDS JSON or protocol YAML.")
    p.add_argument(
        "--center-origin",
        choices=["foreground", "grid", "none"],
        default="foreground",
        help="Shift native affine translation so brain foreground centroid ('foreground') or grid center ('grid') sits at [0, 0, 0] mm.",
    )
    p.add_argument("--subjects", default=None, help="Comma-separated subject IDs to process (optional).")
    p.add_argument("--limit", type=int, default=None, help="Process at most N runs (optional).")
    p.add_argument("--skip-fsl", action="store_true")
    p.add_argument("--skip-flexibrain", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--keep-work", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    t_start = time.perf_counter()
    args = build_arg_parser().parse_args(argv)

    protocol = load_protocol_yaml(args.protocol_yaml) if args.protocol_yaml else {}
    adapter = ADAPTERS[args.dataset]()
    all_records = adapter.discover(args.input_root, protocol=protocol, index_json=args.index_json)
    subjects_filter = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else None
    records = _filter_records(all_records, subjects_filter, args.limit)
    print(f"[DISCOVER] dataset={args.dataset}, total_discovered={len(all_records)}, selected={len(records)}")
    if not records:
        print("[ERROR] No runs matched selection.", file=sys.stderr)
        return 2

    args.output_root.mkdir(parents=True, exist_ok=True)
    fsl_native_root = args.output_root / "fsl_native"
    fsl_native_dir = fsl_native_root / "nativespace"
    flexi_root = args.output_root / "flexibrain_ready"
    flexi_dir = flexi_root / "nativespace"
    qc_dir = args.output_root / "qc"
    stage_dir = args.output_root / "work" / "raw_stage"
    fsl_native_dir.mkdir(parents=True, exist_ok=True)
    flexi_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    fsl_failed = 0
    manifest_path = qc_dir / "pipeline_manifest.csv"
    existing_rows: Dict[str, Dict[str, Any]] = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                existing_rows[row["run_id"]] = row

    if not args.skip_fsl:
        _check_fsl_binaries(args.fsl_dir, require_slicetimer=not args.skip_stc)
        pending: List[RunRecord] = []
        skipped_existing = 0
        for rec in records:
            target = fsl_native_dir / f"{rec.out_stem}.nii.gz"
            if target.exists() and not args.overwrite:
                skipped_existing += 1
            else:
                pending.append(rec)
        print(f"[FSL] pending={len(pending)}, skipped_existing={skipped_existing}")

        if pending:
            t_stage = time.perf_counter()
            staged_paths = adapter.stage_runs(pending, stage_dir, workers=args.stage_workers, overwrite=args.overwrite)
            print(f"[STAGE] Materialized {len(staged_paths)} input(s) in {time.perf_counter() - t_stage:.1f}s")

            tasks = [
                (
                    rec,
                    staged_paths[rec.run_id],
                    args.output_root,
                    args.fsl_dir,
                    args.bet_f,
                    args.keep_work,
                    (not args.keep_work),
                    stage_dir,
                    resolve_drop_initial_trs(rec.site, protocol, cli_override=args.drop_initial_trs),
                    args.auto_drop_nonsteady,
                    args.skip_stc,
                    args.center_origin,
                )
                for rec in pending
            ]

            new_rows: List[Dict[str, Any]] = []
            max_w = max(1, min(args.workers, len(tasks)))
            with ProcessPoolExecutor(max_workers=max_w) as ex:
                futs = [ex.submit(_run_fsl_worker_task, t) for t in tasks]
                iterator = as_completed(futs)
                if tqdm is not None:
                    iterator = tqdm(iterator, total=len(futs), desc="FSL native")
                for fut in iterator:
                    res = fut.result()
                    new_rows.append(res)
                    existing_rows[res["run_id"]] = res
                    if res["status"] != "ok":
                        fsl_failed += 1
                        print(f"[ERROR] FSL failed {res['run_id']}: {res['error']}", file=sys.stderr)

            with open(manifest_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
                writer.writeheader()
                for rid in sorted(existing_rows.keys()):
                    writer.writerow({k: existing_rows[rid].get(k, "") for k in MANIFEST_FIELDS})
            print(f"[FSL DONE] processed={len(new_rows)}, failed={fsl_failed}")

        # Cleanup empty work dirs if requested
        if not args.keep_work:
            for d in (stage_dir, args.output_root / "work" / "runs", args.output_root / "work"):
                if d.exists():
                    try:
                        d.rmdir()
                    except OSError:
                        pass

    # Stage 2: FlexiBrain data_process.py conversion (96x96x96 + global foreground z-score)
    dp_failed = 0
    if not args.skip_flexibrain:
        fsl_inputs = [
            fsl_native_dir / f"{rec.out_stem}.nii.gz"
            for rec in records
            if (fsl_native_dir / f"{rec.out_stem}.nii.gz").exists()
        ]
        dp_tasks = [
            (
                in_p,
                flexi_dir / data_process.clean_output_name(in_p),
                fsl_native_root,
                flexi_root,
                args.overwrite,
            )
            for in_p in fsl_inputs
        ]
        pending_dp = [t for t in dp_tasks if args.overwrite or not t[1].exists()]
        print(f"[FLEXIBRAIN data_process] total={len(dp_tasks)}, pending={len(pending_dp)}")
        if pending_dp:
            dp_counts: Dict[str, int] = {}
            max_w = max(1, min(args.workers, len(pending_dp)))
            with ProcessPoolExecutor(max_workers=max_w) as ex:
                futs = [ex.submit(_run_data_process_task, t) for t in pending_dp]
                iterator = as_completed(futs)
                if tqdm is not None:
                    iterator = tqdm(iterator, total=len(futs), desc="data_process.py")
                for fut in iterator:
                    status, path_str = fut.result()
                    dp_counts[status] = dp_counts.get(status, 0) + 1
                    if status != "processed" and status != "skipped_existing":
                        dp_failed += 1
                        print(f"[WARN] data_process status={status} for {path_str}", file=sys.stderr)
            print(f"[FLEXIBRAIN DONE] counts={dp_counts}")

    # Stage 3: Post-conversion QC audit on selected FlexiBrain outputs
    qc_targets = [
        flexi_dir / f"{rec.out_stem}_global_zscore.nii.gz"
        for rec in records
        if (flexi_dir / f"{rec.out_stem}_global_zscore.nii.gz").exists()
    ]
    missing_final = [
        rec.run_id for rec in records
        if not (flexi_dir / f"{rec.out_stem}_global_zscore.nii.gz").exists()
    ]
    qc_results: List[Dict[str, Any]] = []
    if qc_targets:
        qc_tasks = [(p, args.center_origin) for p in qc_targets]
        max_w = max(1, min(args.workers, len(qc_tasks)))
        with ProcessPoolExecutor(max_workers=max_w) as ex:
            futs = [ex.submit(_qc_one_flexibrain_file, t) for t in qc_tasks]
            iterator = as_completed(futs)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(futs), desc="QC audit")
            for fut in iterator:
                qc_results.append(fut.result())

    selected_rids = {rec.run_id for rec in records}
    stc_method_counts: Dict[str, int] = {}
    origin_mode_counts: Dict[str, int] = {}
    for rid in selected_rids:
        row = existing_rows.get(rid)
        if not row:
            continue
        sm = str(row.get("stc_method", "")) or "unknown"
        stc_method_counts[sm] = stc_method_counts.get(sm, 0) + 1
        om = str(row.get("origin_mode", "")) or "unknown"
        origin_mode_counts[om] = origin_mode_counts.get(om, 0) + 1

    qc_failed_items = [r for r in qc_results if not r["ok"]]
    max_origin_norm = max((r.get("origin_norm_mm", 0.0) for r in qc_results), default=0.0)
    max_qform_sform_diff = max((r.get("qform_sform_max_diff", 0.0) for r in qc_results), default=0.0)
    qc_summary = {
        "dataset": args.dataset,
        "selected_runs": len(records),
        "skip_stc": bool(args.skip_stc),
        "center_origin": args.center_origin,
        "auto_drop_nonsteady": bool(args.auto_drop_nonsteady),
        "drop_initial_trs_override": args.drop_initial_trs,
        "stc_method_counts": stc_method_counts,
        "origin_mode_counts": origin_mode_counts,
        "fsl_failed": fsl_failed,
        "data_process_failed": dp_failed,
        "final_files_present": len(qc_targets),
        "missing_final_count": len(missing_final),
        "missing_final_sample": missing_final[:20],
        "qc_passed": len(qc_results) - len(qc_failed_items),
        "qc_failed": len(qc_failed_items),
        "max_origin_norm_mm": round(float(max_origin_norm), 5),
        "max_qform_sform_diff_mm": round(float(max_qform_sform_diff), 6),
        "qc_failed_items": qc_failed_items[:20],
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
    }
    (qc_dir / "qc_summary.json").write_text(json.dumps(qc_summary, indent=2), encoding="utf-8")
    print(f"[QC SUMMARY] {json.dumps({k: v for k, v in qc_summary.items() if k not in {'qc_failed_items', 'missing_final_sample'}})}")

    if fsl_failed or dp_failed or missing_final or qc_failed_items:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
