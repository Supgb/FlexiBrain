#!/usr/bin/env python3
"""Generate FlexiBrain train/val/test path lists, labels.csv, and QC/dedup audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


_SUB_DIGITS_RE = re.compile(r"sub-0*([0-9]+)", re.IGNORECASE)
_RUN_RE = re.compile(r"run-([0-9]+)", re.IGNORECASE)


def extract_subject_key(filename: str) -> str:
    m = _SUB_DIGITS_RE.search(filename)
    if m:
        return str(int(m.group(1)))
    digits = re.findall(r"(\d{4,8})", filename)
    if not digits:
        raise ValueError(f"Cannot extract subject ID from {filename}")
    return str(int(digits[-1]))


def extract_run_number(filename: str) -> int:
    m = _RUN_RE.search(filename)
    return int(m.group(1)) if m else 1


def discover_processed_niftis(nifti_dir: Path) -> List[Path]:
    return sorted(
        p for p in nifti_dir.iterdir()
        if p.is_file() and not p.name.startswith("._") and (p.name.endswith(".nii.gz") or p.name.endswith(".nii"))
    )


def _strip_nifti_stem(name: str) -> str:
    for suffix in ("_global_zscore.nii.gz", "_global_zscore.nii", ".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _hash_nifti_voxels(path_str: str) -> Tuple[str, Tuple[int, ...], str]:
    img = nib.load(path_str)
    arr = np.ascontiguousarray(np.asarray(img.dataobj, dtype=np.float32))
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    return path_str, tuple(int(s) for s in arr.shape), digest


def _load_manifest(manifest_csv: Path) -> Dict[str, Dict[str, str]]:
    rows_by_stem: Dict[str, Dict[str, str]] = {}
    with open(manifest_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out_stem = (row.get("out_stem") or "").strip()
            if out_stem:
                rows_by_stem[out_stem] = row
            fsl_p = (row.get("fsl_native_path") or "").strip()
            if fsl_p:
                rows_by_stem.setdefault(_strip_nifti_stem(Path(fsl_p).name), row)
    return rows_by_stem


def _safe_parse_sub_key(val: Any) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return None


def _map_label(val: Any, preset: str) -> Optional[int]:
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        iv = int(float(s))
    except (ValueError, TypeError):
        return None
    if preset == "abide_dx":
        if iv == 2:
            return 0  # Control
        if iv == 1:
            return 1  # ASD
        return None
    if preset == "adhd200_dx":
        if iv == 0:
            return 0  # TDC
        if iv in (1, 2, 3):
            return 1  # ADHD (combined, hyperactive/impulsive, inattentive)
        return None
    return iv if iv >= 0 else None


def stratified_7_1_2_split(
    subjects: List[str],
    strata_keys: List[str],
    fallback_keys: List[int],
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split subject IDs into 70% train / 10% val / 20% test with stratification."""
    counts: Dict[str, int] = {}
    for k in strata_keys:
        counts[k] = counts.get(k, 0) + 1

    # Merge strata with fewer than 5 subjects into class-only fallback so 7:1:2 splits succeed cleanly
    merged_strata = [
        sk if counts[sk] >= 5 else f"fallback_{fk}"
        for sk, fk in zip(strata_keys, fallback_keys)
    ]

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(subjects, merged_strata))

    tv_subjects = [subjects[i] for i in trainval_idx]
    tv_strata = [merged_strata[i] for i in trainval_idx]
    tv_fallback = [fallback_keys[i] for i in trainval_idx]
    tv_counts: Dict[str, int] = {}
    for k in tv_strata:
        tv_counts[k] = tv_counts.get(k, 0) + 1
    tv_merged = [
        sk if tv_counts[sk] >= 3 else f"fallback_{fk}"
        for sk, fk in zip(tv_strata, tv_fallback)
    ]

    # 0.125 of 0.80 == 0.10 of total -> 0.70 train / 0.10 val / 0.20 test
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=seed)
    tr_rel_idx, va_rel_idx = next(sss2.split(tv_subjects, tv_merged))

    train_subs = [tv_subjects[i] for i in tr_rel_idx]
    val_subs = [tv_subjects[i] for i in va_rel_idx]
    test_subs = [subjects[i] for i in test_idx]
    return train_subs, val_subs, test_subs


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate FlexiBrain split lists and labels.csv with QC and zero-leakage guarantees.")
    p.add_argument(
        "--nifti-dir",
        type=Path,
        required=True,
        help="Directory containing FlexiBrain-ready 96x96x96 NIfTIs (e.g. /mnt/ironwolf/abide1-fsl/flexibrain_ready/nativespace).",
    )
    p.add_argument("--phenotypic-csv", type=Path, required=True, help="Phenotypic CSV or TSV with subject IDs and labels.")
    p.add_argument("--manifest-csv", type=Path, default=None, help="Optional pipeline_manifest.csv for FD motion QC and site names.")
    p.add_argument("--max-mean-fd", type=float, default=0.5, help="Drop runs with mean_fd_mm > threshold (<= 0 disables).")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to write labels.csv, lists/*.txt, and split_audit.json.")
    p.add_argument("--id-column", default="SUB_ID", help="Subject ID column in phenotypic CSV/TSV.")
    p.add_argument("--label-column", default="DX_GROUP", help="Label column in phenotypic CSV/TSV.")
    p.add_argument("--site-column", default="SITE_ID", help="Optional site column for stratification.")
    p.add_argument(
        "--label-preset",
        choices=["abide_dx", "adhd200_dx", "raw"],
        default="abide_dx",
        help="'abide_dx' (2->0, 1->1), 'adhd200_dx' (0->0, 1/2/3->1, pending ignored), or 'raw' (>=0 int).",
    )
    p.add_argument("--no-dedup", action="store_true", help="Disable exact voxel-level duplicate scan removal.")
    p.add_argument(
        "--include-unlabeled-in-pretrain",
        action="store_true",
        help="Include QC-passing unlabeled runs (e.g. ADHD200 Brown pending) in pretrain_train.txt.",
    )
    p.add_argument("--workers", type=int, default=8, help="Worker processes for duplicate voxel hashing.")
    p.add_argument("--seed", type=int, default=42)
    return p


def _subject_sort_key(sk: str) -> Tuple[int, str]:
    return (int(sk), sk) if sk.isdigit() else (10**12, sk)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    niftis = discover_processed_niftis(args.nifti_dir)
    if not niftis:
        raise FileNotFoundError(f"No NIfTI files found in {args.nifti_dir}")

    sep = "\t" if args.phenotypic_csv.suffix.lower() in (".tsv", ".txt") else ","
    ph = pd.read_csv(args.phenotypic_csv, sep=sep)
    ph = ph.copy()
    ph["_sub_key"] = ph[args.id_column].apply(_safe_parse_sub_key)
    ph = ph[ph["_sub_key"].notna()].copy()
    ph["_group_idx"] = ph[args.label_column].apply(lambda v: _map_label(v, args.label_preset))

    manifest_rows: Dict[str, Dict[str, str]] = _load_manifest(args.manifest_csv) if args.manifest_csv else {}

    # Track labeled metadata (keep last non-null label per subject if CSV has per-run rows)
    sub_meta: Dict[str, Tuple[int, str]] = {}
    csv_site_by_sub: Dict[str, str] = {}
    for _, row in ph.iterrows():
        sk = row["_sub_key"]
        raw_site = row[args.site_column] if args.site_column in ph.columns and pd.notna(row[args.site_column]) else None
        if raw_site is not None:
            site_str = str(raw_site).strip()
            if site_str:
                csv_site_by_sub[sk] = site_str
        if pd.notna(row["_group_idx"]):
            sub_meta[sk] = (int(row["_group_idx"]), csv_site_by_sub.get(sk, "unknown"))

    # Filter NIfTIs through manifest QC (status == ok and mean_fd_mm <= max_mean_fd)
    qc_passed_niftis: List[Path] = []
    manifest_site_by_sub: Dict[str, str] = {}
    excluded_status: List[Dict[str, Any]] = []
    excluded_motion: List[Dict[str, Any]] = []
    manifest_unmatched = 0

    for p in niftis:
        sk = extract_subject_key(p.name)
        stem = _strip_nifti_stem(p.name)
        mrow = manifest_rows.get(stem)
        if mrow is not None:
            msite = (mrow.get("site") or "").strip()
            if msite:
                manifest_site_by_sub.setdefault(sk, msite)
            status = (mrow.get("status") or "").strip().lower()
            if status != "ok":
                excluded_status.append({"path": str(p), "subject": sk, "status": status})
                continue
            if args.max_mean_fd is not None and args.max_mean_fd > 0:
                fd_raw = mrow.get("mean_fd_mm", "")
                try:
                    fd_val = float(fd_raw)
                except (ValueError, TypeError):
                    fd_val = float("nan")
                if not math.isfinite(fd_val) or fd_val > args.max_mean_fd:
                    excluded_motion.append({"path": str(p), "subject": sk, "mean_fd_mm": fd_raw})
                    continue
        elif manifest_rows:
            manifest_unmatched += 1
        qc_passed_niftis.append(p)

    # Prefer manifest site name when CSV site is missing, 'unknown', or numeric code (e.g. ADHD200 Site 1..8)
    for sk in list(sub_meta.keys()):
        lbl, csv_site = sub_meta[sk]
        is_numeric_site = csv_site.replace(".", "", 1).isdigit()
        if (csv_site == "unknown" or is_numeric_site) and sk in manifest_site_by_sub:
            sub_meta[sk] = (lbl, manifest_site_by_sub[sk])

    # Group candidate runs into labeled vs unlabeled before deduplication
    candidate_paths: List[Path] = []
    unmatched_excluded: List[Path] = []
    for p in qc_passed_niftis:
        sk = extract_subject_key(p.name)
        if sk in sub_meta or args.include_unlabeled_in_pretrain:
            candidate_paths.append(p)
        else:
            unmatched_excluded.append(p)

    # Exact voxel-level duplicate detection across candidate scans
    dropped_paths: Set[Path] = set()
    duplicate_audit: List[Dict[str, Any]] = []
    if not args.no_dedup and len(candidate_paths) > 1:
        path_strs = [str(p) for p in candidate_paths]
        if args.workers > 1 and len(path_strs) >= 16:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                hashed = list(ex.map(_hash_nifti_voxels, path_strs))
        else:
            hashed = [_hash_nifti_voxels(ps) for ps in path_strs]

        groups: Dict[Tuple[Tuple[int, ...], str], List[Path]] = {}
        for ps, shape, digest in hashed:
            groups.setdefault((shape, digest), []).append(Path(ps))

        for (shape, digest), group_paths in groups.items():
            if len(group_paths) <= 1:
                continue
            group_subs = sorted({extract_subject_key(gp.name) for gp in group_paths}, key=_subject_sort_key)
            distinct_labels = sorted({sub_meta[s][0] for s in group_subs if s in sub_meta})

            if len(group_subs) > 1 and len(distinct_labels) > 1:
                for gp in group_paths:
                    dropped_paths.add(gp)
                duplicate_audit.append(
                    {
                        "action": "drop_all_conflicting_labels",
                        "shape": list(shape),
                        "sha256": digest,
                        "subjects": group_subs,
                        "labels": distinct_labels,
                        "dropped_paths": [gp.name for gp in group_paths],
                    }
                )
            elif len(group_subs) > 1:
                keep_sub = group_subs[0]
                keep_candidates = [gp for gp in group_paths if extract_subject_key(gp.name) == keep_sub]
                keep_path = min(keep_candidates, key=lambda gp: (extract_run_number(gp.name), gp.name))
                dropped_here = [gp for gp in group_paths if gp != keep_path]
                for gp in dropped_here:
                    dropped_paths.add(gp)
                duplicate_audit.append(
                    {
                        "action": "keep_lowest_subject_drop_duplicates",
                        "shape": list(shape),
                        "sha256": digest,
                        "kept_subject": keep_sub,
                        "kept_path": keep_path.name,
                        "subjects": group_subs,
                        "labels": distinct_labels,
                        "dropped_paths": [gp.name for gp in dropped_here],
                    }
                )
            else:
                keep_path = min(group_paths, key=lambda gp: (extract_run_number(gp.name), gp.name))
                dropped_here = [gp for gp in group_paths if gp != keep_path]
                for gp in dropped_here:
                    dropped_paths.add(gp)
                duplicate_audit.append(
                    {
                        "action": "dedup_same_subject_runs",
                        "shape": list(shape),
                        "sha256": digest,
                        "kept_subject": group_subs[0],
                        "kept_path": keep_path.name,
                        "dropped_paths": [gp.name for gp in dropped_here],
                    }
                )

    runs_by_sub: Dict[str, List[Path]] = {}
    unlabeled_runs_by_sub: Dict[str, List[Path]] = {}
    for p in candidate_paths:
        if p in dropped_paths:
            continue
        sk = extract_subject_key(p.name)
        if sk in sub_meta:
            runs_by_sub.setdefault(sk, []).append(p)
        else:
            unlabeled_runs_by_sub.setdefault(sk, []).append(p)

    if unmatched_excluded and not args.include_unlabeled_in_pretrain:
        print(f"[WARN] {len(unmatched_excluded)} QC-passed NIfTIs had no valid label in {args.phenotypic_csv}; example: {unmatched_excluded[0].name}")

    # Pick lowest run number per subject for downstream 1-scan-per-subject lists
    primary_run_by_sub: Dict[str, Path] = {
        sk: min(paths, key=lambda p: (extract_run_number(p.name), p.name))
        for sk, paths in sorted(runs_by_sub.items(), key=lambda kv: _subject_sort_key(kv[0]))
    }

    subjects = sorted(primary_run_by_sub.keys(), key=_subject_sort_key)
    strata_keys = [f"{sub_meta[s][1]}__{sub_meta[s][0]}" for s in subjects]
    fallback_keys = [sub_meta[s][0] for s in subjects]

    train_subs, val_subs, test_subs = stratified_7_1_2_split(subjects, strata_keys, fallback_keys, seed=args.seed)

    # Zero-leakage pretraining splits derived strictly from downstream train/val (+ unlabeled if requested)
    train_sub_set = set(train_subs)
    val_sub_set = set(val_subs)
    test_sub_set = set(test_subs)

    pre_tr_runs = sorted((p for s in train_subs for p in runs_by_sub[s]), key=lambda p: p.name)
    pre_va_runs = sorted((p for s in val_subs for p in runs_by_sub[s]), key=lambda p: p.name)

    extra_unlabeled_subs: List[str] = []
    if args.include_unlabeled_in_pretrain and unlabeled_runs_by_sub:
        extra_unlabeled_subs = sorted(
            (s for s in unlabeled_runs_by_sub if s not in test_sub_set and s not in val_sub_set and s not in train_sub_set),
            key=_subject_sort_key,
        )
        unlabeled_runs = sorted((p for s in extra_unlabeled_subs for p in unlabeled_runs_by_sub[s]), key=lambda p: p.name)
        pre_tr_runs = sorted(pre_tr_runs + unlabeled_runs, key=lambda p: p.name)

    # Explicit zero-leakage assertions across subjects and file paths
    test_runs_all = {p.resolve() for s in test_sub_set for p in runs_by_sub.get(s, [])}
    pre_tr_resolved = {p.resolve() for p in pre_tr_runs}
    pre_va_resolved = {p.resolve() for p in pre_va_runs}
    pre_tr_sub_set = {extract_subject_key(p.name) for p in pre_tr_runs}
    pre_va_sub_set = {extract_subject_key(p.name) for p in pre_va_runs}

    assert not (train_sub_set & val_sub_set)
    assert not (train_sub_set & test_sub_set)
    assert not (val_sub_set & test_sub_set)
    assert not (pre_tr_sub_set & test_sub_set)
    assert not (pre_va_sub_set & test_sub_set)
    assert not (pre_tr_sub_set & pre_va_sub_set)
    assert not (pre_tr_resolved & test_runs_all)
    assert not (pre_va_resolved & test_runs_all)
    assert not (pre_tr_resolved & pre_va_resolved)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lists_dir = args.output_dir / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.DataFrame(
        [{"Subject": s, "Group_idx": sub_meta[s][0], "Site": sub_meta[s][1]} for s in subjects]
    )
    labels_csv_path = args.output_dir / "labels.csv"
    labels_df.to_csv(labels_csv_path, index=False)

    def write_list(path: Path, items: List[Path]) -> None:
        path.write_text("".join(f"{p.resolve()}\n" for p in items), encoding="utf-8")

    retained_all_runs = sorted(
        {p for paths in runs_by_sub.values() for p in paths}
        | ({p for paths in unlabeled_runs_by_sub.values() for p in paths} if args.include_unlabeled_in_pretrain else set()),
        key=lambda p: p.name,
    )
    write_list(lists_dir / "all_runs.txt", retained_all_runs)
    write_list(lists_dir / "train.txt", [primary_run_by_sub[s] for s in sorted(train_subs, key=_subject_sort_key)])
    write_list(lists_dir / "val.txt", [primary_run_by_sub[s] for s in sorted(val_subs, key=_subject_sort_key)])
    write_list(lists_dir / "test.txt", [primary_run_by_sub[s] for s in sorted(test_subs, key=_subject_sort_key)])
    write_list(lists_dir / "pretrain_train.txt", pre_tr_runs)
    write_list(lists_dir / "pretrain_val.txt", pre_va_runs)

    audit = {
        "nifti_dir": str(args.nifti_dir),
        "phenotypic_csv": str(args.phenotypic_csv),
        "manifest_csv": str(args.manifest_csv) if args.manifest_csv else None,
        "max_mean_fd": args.max_mean_fd,
        "label_preset": args.label_preset,
        "dedup_enabled": not args.no_dedup,
        "include_unlabeled_in_pretrain": bool(args.include_unlabeled_in_pretrain),
        "total_discovered_niftis": len(niftis),
        "excluded_manifest_status_count": len(excluded_status),
        "excluded_high_motion_count": len(excluded_motion),
        "manifest_unmatched_count": manifest_unmatched,
        "unmatched_excluded_count": len(unmatched_excluded),
        "duplicate_groups_count": len(duplicate_audit),
        "duplicate_dropped_files_count": len(dropped_paths),
        "retained_labeled_subjects": len(subjects),
        "retained_unlabeled_pretrain_subjects": len(extra_unlabeled_subs),
        "downstream_subjects": {"train": len(train_subs), "val": len(val_subs), "test": len(test_subs)},
        "pretrain_runs": {"train": len(pre_tr_runs), "val": len(pre_va_runs)},
        "zero_leakage_verified": True,
        "test_overlap_in_pretrain": 0,
        "excluded_manifest_status": excluded_status,
        "excluded_high_motion": excluded_motion[:50],
        "duplicates": duplicate_audit,
    }
    audit_path = args.output_dir / "split_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(
        f"[SPLITS] discovered={len(niftis)}, motion_excluded={len(excluded_motion)}, "
        f"dedup_dropped={len(dropped_paths)}, labeled_subjects={len(subjects)}, "
        f"unlabeled_pretrain_subs={len(extra_unlabeled_subs)}, "
        f"downstream(train/val/test)={len(train_subs)}/{len(val_subs)}/{len(test_subs)}, "
        f"pretrain_runs(train/val)={len(pre_tr_runs)}/{len(pre_va_runs)}"
    )
    print(f"[SAVED] {labels_csv_path}, {lists_dir}, and {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
