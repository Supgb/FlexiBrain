"""Dataset discovery and archive staging adapters for ABIDE1, BIDS (ADHD200), and HCP-YA."""

from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from slice_timing import resolve_site_scheme


@dataclass(frozen=True)
class RunRecord:
    dataset: str
    subject_id: str
    site: str
    session: str
    run_label: str
    run_id: str
    out_stem: str
    source_type: str  # "file" | "tar" | "zip"
    source_path: str
    member_name: Optional[str] = None
    sidecar_json: Optional[str] = None
    slice_scheme: Optional[str] = None


def _is_appledouble(path_str: str) -> bool:
    return any(part.startswith("._") for part in Path(path_str).parts)


def _atomic_copy_stream(src_fileobj, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.parent / f".{dst_path.name}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "wb") as out_f:
            shutil.copyfileobj(src_fileobj, out_f, length=1024 * 1024)
        os.replace(tmp_path, dst_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _extract_tar_batch(task: Tuple[str, List[Tuple[str, str]]]) -> List[str]:
    """Open a .tar/.tgz archive once and extract all requested members."""
    archive_path, member_dst_pairs = task
    mode = "r:gz" if archive_path.endswith((".tgz", ".tar.gz")) else "r:"
    needed = {member: Path(dst) for member, dst in member_dst_pairs}
    extracted: List[str] = []
    with tarfile.open(archive_path, mode) as tf:
        for m in tf:
            if not m.isfile():
                continue
            dst_path = needed.get(m.name)
            if dst_path is None:
                continue
            src_f = tf.extractfile(m)
            if src_f is None:
                raise RuntimeError(f"Failed to read tar member {m.name} from {archive_path}")
            with src_f:
                _atomic_copy_stream(src_f, dst_path)
            extracted.append(str(dst_path))
            if len(extracted) == len(needed):
                break
    if len(extracted) != len(needed):
        missing = sorted(set(needed.keys()) - {m for m, d in member_dst_pairs if str(d) in extracted})
        raise FileNotFoundError(f"Missing members in {archive_path}: {missing[:5]}")
    return extracted


def _extract_zip_batch(task: Tuple[str, List[Tuple[str, str]]]) -> List[str]:
    """Open a .zip archive once and extract all requested members."""
    archive_path, member_dst_pairs = task
    extracted: List[str] = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for member, dst in member_dst_pairs:
            dst_path = Path(dst)
            with zf.open(member, "r") as src_f:
                _atomic_copy_stream(src_f, dst_path)
            extracted.append(str(dst_path))
    return extracted


class BaseAdapter:
    name: str = "base"

    def discover(
        self,
        input_root: Path,
        protocol: Optional[Dict[str, Any]] = None,
        index_json: Optional[Path] = None,
    ) -> List[RunRecord]:
        raise NotImplementedError

    def stage_runs(
        self,
        records: Sequence[RunRecord],
        stage_dir: Path,
        workers: int = 4,
        overwrite: bool = False,
    ) -> Dict[str, Path]:
        """Materialize NIfTI inputs on disk and return {run_id: local_path}."""
        stage_dir.mkdir(parents=True, exist_ok=True)
        staged: Dict[str, Path] = {}
        tar_groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        zip_groups: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for rec in records:
            if rec.source_type == "file":
                staged[rec.run_id] = Path(rec.source_path)
                continue
            dst = stage_dir / f"{rec.run_id}.nii.gz"
            staged[rec.run_id] = dst
            if dst.exists() and not overwrite:
                continue
            if not rec.member_name:
                raise ValueError(f"Archive run {rec.run_id} missing member_name")
            if rec.source_type == "tar":
                tar_groups[rec.source_path].append((rec.member_name, str(dst)))
            elif rec.source_type == "zip":
                zip_groups[rec.source_path].append((rec.member_name, str(dst)))
            else:
                raise ValueError(f"Unsupported source_type: {rec.source_type!r}")

        tasks_tar = list(tar_groups.items())
        tasks_zip = list(zip_groups.items())
        if tasks_tar or tasks_zip:
            max_w = max(1, min(workers, len(tasks_tar) + len(tasks_zip)))
            with ProcessPoolExecutor(max_workers=max_w) as ex:
                futures = [ex.submit(_extract_tar_batch, t) for t in tasks_tar]
                futures.extend(ex.submit(_extract_zip_batch, z) for z in tasks_zip)
                for fut in as_completed(futures):
                    fut.result()

        return staged


class ABIDE1Adapter(BaseAdapter):
    name = "abide1"
    _member_re = re.compile(r"^([^/]+)/(\d+)/session_(\d+)/rest_(\d+)/rest\.nii\.gz$")

    def discover(
        self,
        input_root: Path,
        protocol: Optional[Dict[str, Any]] = None,
        index_json: Optional[Path] = None,
    ) -> List[RunRecord]:
        records: List[RunRecord] = []
        if index_json is not None and index_json.exists():
            data = json.loads(index_json.read_text(encoding="utf-8"))
            subjects = data.get("subjects", data)
            for sub_id, info in sorted(subjects.items()):
                archive_rel = info["archive"]
                archive_path = input_root / archive_rel
                if not archive_path.exists():
                    raise FileNotFoundError(f"Archive listed in index not found: {archive_path}")
                site = info["site"]
                scheme = resolve_site_scheme(site, protocol)
                rest_members = info.get("rest_members", {})
                for run_key, member_name in sorted(rest_members.items(), key=lambda kv: int(kv[0])):
                    run_num = int(run_key)
                    run_label = f"run-{run_num:02d}"
                    out_stem = f"sub-{sub_id}_task-rest_{run_label}_bold"
                    run_id = f"sub-{sub_id}_ses-1_task-rest_{run_label}"
                    records.append(
                        RunRecord(
                            dataset="abide1",
                            subject_id=str(sub_id),
                            site=site,
                            session="1",
                            run_label=run_label,
                            run_id=run_id,
                            out_stem=out_stem,
                            source_type="tar",
                            source_path=str(archive_path),
                            member_name=member_name,
                            sidecar_json=None,
                            slice_scheme=scheme,
                        )
                    )
            return sorted(records, key=lambda r: r.run_id)

        # Fallback: scan .tar and .tgz archives directly
        archives = sorted(
            p for p in input_root.iterdir()
            if p.is_file() and not p.name.startswith("._") and p.name.endswith((".tgz", ".tar.gz", ".tar"))
        )
        for arc in archives:
            site = arc.name[:-7] if arc.name.endswith(".tar.gz") else arc.stem
            scheme = resolve_site_scheme(site, protocol)
            mode = "r:gz" if arc.name.endswith((".tgz", ".tar.gz")) else "r:"
            with tarfile.open(arc, mode) as tf:
                for m in tf.getmembers():
                    if not m.isfile() or _is_appledouble(m.name):
                        continue
                    match = self._member_re.match(m.name)
                    if not match:
                        continue
                    _, sub_id, ses_id, run_str = match.groups()
                    run_num = int(run_str)
                    run_label = f"run-{run_num:02d}"
                    out_stem = f"sub-{sub_id}_task-rest_{run_label}_bold"
                    run_id = f"sub-{sub_id}_ses-{ses_id}_task-rest_{run_label}"
                    records.append(
                        RunRecord(
                            dataset="abide1",
                            subject_id=sub_id,
                            site=site,
                            session=ses_id,
                            run_label=run_label,
                            run_id=run_id,
                            out_stem=out_stem,
                            source_type="tar",
                            source_path=str(arc),
                            member_name=m.name,
                            sidecar_json=None,
                            slice_scheme=scheme,
                        )
                    )
        return sorted(records, key=lambda r: r.run_id)


class BIDSAdapter(BaseAdapter):
    """Generic BIDS discovery adapter (ready for ADHD200 and similar BIDS trees)."""

    name = "bids"
    _sub_re = re.compile(r"(sub-[A-Za-z0-9]+)")
    _ses_re = re.compile(r"(ses-[A-Za-z0-9]+)")
    _run_re = re.compile(r"(run-\d+)")

    @staticmethod
    def _strip_nifti_ext(name: str) -> str:
        if name.endswith(".nii.gz"):
            return name[:-7]
        if name.endswith(".nii"):
            return name[:-4]
        return Path(name).stem

    def discover(
        self,
        input_root: Path,
        protocol: Optional[Dict[str, Any]] = None,
        index_json: Optional[Path] = None,
    ) -> List[RunRecord]:
        records: List[RunRecord] = []
        bold_files = sorted(
            p for p in input_root.rglob("*_bold.nii*")
            if p.is_file() and not _is_appledouble(str(p)) and (p.name.endswith(".nii.gz") or p.name.endswith(".nii"))
        )
        for p in bold_files:
            rel_parts = p.relative_to(input_root).parts
            site = rel_parts[0] if len(rel_parts) >= 2 and not rel_parts[0].startswith("sub-") else "unknown"
            sub_m = self._sub_re.search(p.name)
            sub_tag = sub_m.group(1) if sub_m else next((part for part in rel_parts if part.startswith("sub-")), "sub-unknown")
            subject_id = sub_tag[4:] if sub_tag.startswith("sub-") else sub_tag
            ses_m = self._ses_re.search(p.name) or self._ses_re.search(str(p))
            session = ses_m.group(1)[4:] if ses_m else "1"
            run_m = self._run_re.search(p.name)
            run_label = run_m.group(1) if run_m else "run-01"

            stem = self._strip_nifti_ext(p.name)
            json_cand: Optional[Path] = None
            direct_json = p.parent / f"{stem}.json"
            if direct_json.exists():
                json_cand = direct_json
            else:
                # BIDS inheritance: walk up to input_root looking for matching *_bold.json sidecars
                # whose BIDS key-value entities (e.g. task-rest, acq-1) are a subset of stem entities.
                stem_entities = set(stem.split("_")[:-1])
                cur = p.parent
                while True:
                    cands = sorted(
                        c for c in cur.glob("*_bold.json")
                        if not c.name.startswith("._")
                    )
                    best_match: Optional[Path] = None
                    best_score = -1
                    for c in cands:
                        c_entities = set(c.stem.split("_")[:-1])
                        if c_entities.issubset(stem_entities) and len(c_entities) > best_score:
                            best_match = c
                            best_score = len(c_entities)
                    if best_match is not None:
                        json_cand = best_match
                        break
                    if cur == input_root or cur.parent == cur:
                        break
                    cur = cur.parent

            scheme = resolve_site_scheme(site, protocol)
            run_id = stem if site == "unknown" else f"{site}_{stem}"
            records.append(
                RunRecord(
                    dataset="bids",
                    subject_id=subject_id,
                    site=site,
                    session=session,
                    run_label=run_label,
                    run_id=run_id,
                    out_stem=stem,
                    source_type="file",
                    source_path=str(p),
                    member_name=None,
                    sidecar_json=str(json_cand) if json_cand is not None else None,
                    slice_scheme=scheme,
                )
            )
        return sorted(records, key=lambda r: r.run_id)


class HCPYAAdapter(BaseAdapter):
    """HCP-YA unprocessed rfMRI ZIP archive adapter."""

    name = "hcp_ya"
    _zip_member_re = re.compile(r"^(\d+)/unprocessed/3T/(rfMRI_[^/]+)/\1_3T_\2\.nii\.gz$")

    def discover(
        self,
        input_root: Path,
        protocol: Optional[Dict[str, Any]] = None,
        index_json: Optional[Path] = None,
    ) -> List[RunRecord]:
        records: List[RunRecord] = []
        zips = sorted(
            p for p in input_root.rglob("*_rfMRI_*_unproc.zip")
            if p.is_file() and not _is_appledouble(str(p))
        )
        scheme = resolve_site_scheme("HCP_YA", protocol) or "skip"
        for zp in zips:
            with zipfile.ZipFile(zp, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir() or _is_appledouble(info.filename):
                        continue
                    match = self._zip_member_re.match(info.filename)
                    if not match:
                        continue
                    sub_id, acq = match.groups()
                    acq_short = acq.replace("rfMRI_", "")
                    out_stem = f"sub-{sub_id}_task-rfMRI_acq-{acq_short}_bold"
                    run_id = out_stem
                    records.append(
                        RunRecord(
                            dataset="hcp_ya",
                            subject_id=sub_id,
                            site="HCP_YA",
                            session="3T",
                            run_label=acq_short,
                            run_id=run_id,
                            out_stem=out_stem,
                            source_type="zip",
                            source_path=str(zp),
                            member_name=info.filename,
                            sidecar_json=None,
                            slice_scheme=scheme,
                        )
                    )
        return sorted(records, key=lambda r: r.run_id)


ADAPTERS = {
    "abide1": ABIDE1Adapter,
    "bids": BIDSAdapter,
    "hcp_ya": HCPYAAdapter,
}
