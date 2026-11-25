import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FSL_PREPROCESS_DIR = REPO_ROOT / "fsl-preprocess"
if str(FSL_PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(FSL_PREPROCESS_DIR))

from fsl_native_worker import _detect_nonsteady_prefix  # noqa: E402
import make_splits  # noqa: E402


def _write_tiny_nifti(path: Path, data: np.ndarray, tr: float = 2.0) -> None:
    img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    img.header.set_zooms((3.0, 3.0, 3.0, tr))
    img.header.set_xyzt_units(xyz="mm", t="sec")
    nib.save(img, str(path))


class PreprocessAndSplitsTests(unittest.TestCase):
    def test_detect_nonsteady_prefix_spikes_and_clean(self):
        rng = np.random.default_rng(0)
        clean = rng.normal(loc=100.0, scale=1.0, size=(4, 4, 4, 40)).astype(np.float32)
        self.assertEqual(_detect_nonsteady_prefix(clean), 0)

        spiky = clean.copy()
        spiky[..., 0] += 500.0
        spiky[..., 1] += 150.0
        spiky[..., 2] += 40.0
        self.assertEqual(_detect_nonsteady_prefix(spiky), 3)

        short = clean[..., :12]
        self.assertEqual(_detect_nonsteady_prefix(short), 0)

    def test_make_splits_motion_qc_dedup_unlabeled_and_zero_leakage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            nifti_dir = td / "niftis"
            nifti_dir.mkdir()
            out_dir = td / "out"

            rng = np.random.default_rng(42)
            base_shape = (6, 6, 6, 12)

            # 12 normal ADHD200 subjects (100..111) across two sites, alternating TDC (0) and ADHD subtypes (1,2,3)
            # 112: high motion (mean_fd_mm=0.82) -> dropped by motion QC
            # 113 & 114: identical scan, matching label -> keep 113, drop 114
            # 115 & 116: identical scan, conflicting labels (0 vs 1) -> drop BOTH 115 & 116
            # 117: Site Brown with DX='pending' -> included ONLY in pretrain_train.txt when --include-unlabeled-in-pretrain
            # 101 also has run-2 -> secondary run stays in pretrain split alongside primary run (no test leakage)
            manifest_rows = []
            pheno_rows = []

            def add_run(sub_id: int, run_num: int, site: str, dx_str: str, mean_fd: float, data: np.ndarray, add_pheno: bool = True):
                stem = f"sub-{sub_id:07d}_ses-1_task-rest_run-{run_num}_bold"
                p = nifti_dir / f"{stem}_global_zscore.nii.gz"
                _write_tiny_nifti(p, data)
                manifest_rows.append(
                    {
                        "run_id": stem,
                        "out_stem": stem,
                        "site": site,
                        "status": "ok",
                        "mean_fd_mm": str(mean_fd),
                        "fsl_native_path": f"/tmp/{stem}.nii.gz",
                    }
                )
                if add_pheno:
                    pheno_rows.append({"ScanDir ID": f"{sub_id:07d}", "Site": "5" if site == "NYU" else "3", "DX": dx_str})

            for idx, sid in enumerate(range(100, 112)):
                dx = "0" if idx % 2 == 0 else str((idx // 2 % 3) + 1)
                site = "NYU" if idx < 6 else "KKI"
                add_run(sid, 1, site, dx, 0.15, rng.normal(0, 1, size=base_shape))
            add_run(101, 2, "NYU", "1", 0.18, rng.normal(0, 1, size=base_shape), add_pheno=False)

            add_run(112, 1, "NYU", "0", 0.82, rng.normal(0, 1, size=base_shape))

            dup_same = rng.normal(0, 1, size=base_shape)
            add_run(113, 1, "KKI", "0", 0.10, dup_same)
            add_run(114, 1, "KKI", "0", 0.10, dup_same)

            dup_conflict = rng.normal(0, 1, size=base_shape)
            add_run(115, 1, "NYU", "0", 0.12, dup_conflict)
            add_run(116, 1, "NYU", "2", 0.12, dup_conflict)

            add_run(117, 1, "Brown", "pending", 0.20, rng.normal(0, 1, size=base_shape))

            pheno_path = td / "pheno.tsv"
            with open(pheno_path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["ScanDir ID", "Site", "DX"], delimiter="\t")
                w.writeheader()
                w.writerows(pheno_rows)

            manifest_path = td / "pipeline_manifest.csv"
            with open(manifest_path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["run_id", "out_stem", "site", "status", "mean_fd_mm", "fsl_native_path"])
                w.writeheader()
                w.writerows(manifest_rows)

            rc = make_splits.main(
                [
                    "--nifti-dir", str(nifti_dir),
                    "--phenotypic-csv", str(pheno_path),
                    "--manifest-csv", str(manifest_path),
                    "--max-mean-fd", "0.5",
                    "--label-preset", "adhd200_dx",
                    "--id-column", "ScanDir ID",
                    "--label-column", "DX",
                    "--site-column", "Site",
                    "--include-unlabeled-in-pretrain",
                    "--workers", "1",
                    "--seed", "42",
                    "--output-dir", str(out_dir),
                ]
            )
            self.assertEqual(rc, 0)

            audit = json.loads((out_dir / "split_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(audit["zero_leakage_verified"])
            self.assertEqual(audit["test_overlap_in_pretrain"], 0)
            self.assertEqual(audit["excluded_high_motion_count"], 1)
            self.assertEqual(audit["duplicate_dropped_files_count"], 3)  # 114 + 115 + 116
            self.assertEqual(audit["retained_unlabeled_pretrain_subjects"], 1)  # 117

            labels_text = (out_dir / "labels.csv").read_text(encoding="utf-8")
            self.assertIn("NYU", labels_text)
            self.assertIn("KKI", labels_text)
            retained_subs = {line.split(",")[0] for line in labels_text.splitlines()[1:] if line}
            self.assertIn("113", retained_subs)
            self.assertNotIn("112", retained_subs)
            self.assertNotIn("114", retained_subs)
            self.assertNotIn("115", retained_subs)
            self.assertNotIn("116", retained_subs)
            self.assertNotIn("117", retained_subs)

            def read_subs(fname: str) -> set[str]:
                lines = [ln.strip() for ln in (out_dir / "lists" / fname).read_text(encoding="utf-8").splitlines() if ln.strip()]
                return {make_splits.extract_subject_key(Path(ln).name) for ln in lines}

            tr_s = read_subs("train.txt")
            va_s = read_subs("val.txt")
            te_s = read_subs("test.txt")
            pre_tr_s = read_subs("pretrain_train.txt")
            pre_va_s = read_subs("pretrain_val.txt")

            self.assertFalse(tr_s & te_s)
            self.assertFalse(va_s & te_s)
            self.assertFalse(pre_tr_s & te_s)
            self.assertFalse(pre_va_s & te_s)
            self.assertFalse(pre_tr_s & pre_va_s)
            self.assertIn("117", pre_tr_s)
            self.assertNotIn("117", pre_va_s)


if __name__ == "__main__":
    unittest.main()
