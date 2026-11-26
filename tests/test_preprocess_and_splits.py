import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
FSL_PREPROCESS_DIR = REPO_ROOT / "fsl-preprocess"
for _p in (str(REPO_ROOT), str(FSL_PREPROCESS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import data_process  # noqa: E402
import fsl_native_worker  # noqa: E402
import make_splits  # noqa: E402
import run_pipeline  # noqa: E402
from fsl_native_worker import _detect_nonsteady_prefix  # noqa: E402
from slice_timing import load_protocol_yaml, resolve_drop_initial_trs  # noqa: E402


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

    def test_skip_stc_and_skip_scheme_override_bids_json_while_preserving_tr(self):
        """Both skip_stc=True and slice_scheme='skip' must bypass slicetimer even with BIDS SliceTiming, while keeping bids_tr fallback."""
        for label, slice_scheme, skip_stc in [
            ("cli_skip_stc", "siemens_interleaved_fh", True),
            ("scheme_skip", "skip", False),
        ]:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmpdir:
                td = Path(tmpdir)
                raw_path = td / "sub-0001_bold.nii.gz"
                sidecar_path = td / "sub-0001_bold.json"
                out_root = td / "out"

                # Missing TR in NIfTI header (0.0) so worker must still read RepetitionTime from BIDS JSON
                rng = np.random.default_rng(7)
                raw_arr = rng.normal(loc=100.0, scale=5.0, size=(8, 8, 6, 24)).astype(np.float32)
                theta = np.deg2rad(15.0)
                rot = np.array(
                    [
                        [np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                oblique_aff = np.eye(4, dtype=np.float64)
                oblique_aff[:3, :3] = rot @ np.diag([-3.0, 3.0, 4.0])
                oblique_aff[:3, 3] = np.array([-250.0, 380.0, -140.0])

                img = nib.Nifti1Image(raw_arr, oblique_aff)
                img.header.set_zooms((3.0, 3.0, 4.0, 0.0))
                img.header.set_xyzt_units(xyz="mm", t="sec")
                img.set_qform(oblique_aff, code=1)
                img.set_sform(oblique_aff, code=1)
                nib.save(img, str(raw_path))

                # Matching 6-slice BIDS SliceTiming + RepetitionTime=2.5
                sidecar_path.write_text(
                    json.dumps(
                        {
                            "RepetitionTime": 2.5,
                            "SliceTiming": [0.0, 0.4, 0.8, 1.2, 1.6, 2.0],
                        }
                    ),
                    encoding="utf-8",
                )

                called_cmds = []

                def fake_run_cmd(cmd, env, cwd):
                    called_cmds.append(cmd)
                    exe = Path(cmd[0]).name
                    if exe == "slicetimer":
                        raise AssertionError("slicetimer should never be invoked when STC is skipped")
                    if exe == "mcflirt":
                        in_p = Path(cmd[cmd.index("-in") + 1])
                        out_prefix = Path(cmd[cmd.index("-out") + 1])
                        src = nib.load(str(in_p))
                        nib.save(src, str(out_prefix.with_suffix(".nii.gz")))
                        mean_arr = np.asarray(src.dataobj, dtype=np.float32).mean(axis=3)
                        nib.save(nib.Nifti1Image(mean_arr, src.affine), str(out_prefix.parent / f"{out_prefix.name}_mean_reg.nii.gz"))
                        np.savetxt(out_prefix.with_suffix(".par"), np.zeros((src.shape[3], 6), dtype=np.float64))
                    elif exe in ("bet", "bet2"):
                        in_p = Path(cmd[1])
                        out_prefix = Path(cmd[2])
                        src = nib.load(str(in_p))
                        mask = np.zeros(src.shape[:3], dtype=np.uint8)
                        mask[2:6, 2:7, 1:5] = 1
                        nib.save(nib.Nifti1Image(mask, src.affine), str(out_prefix.parent / f"{out_prefix.name}_mask.nii.gz"))
                    elif exe == "fslmaths":
                        in_p = Path(cmd[1])
                        mask_p = Path(cmd[cmd.index("-mas") + 1])
                        out_p = Path(cmd[cmd.index("-mas") + 2])
                        src = nib.load(str(in_p))
                        arr = np.asarray(src.dataobj, dtype=np.float32)
                        mask = np.asarray(nib.load(str(mask_p)).dataobj) > 0
                        arr = arr.copy()
                        arr[~mask, :] = 0.0
                        nib.save(nib.Nifti1Image(arr, src.affine, header=src.header), str(out_p))

                with (
                    mock.patch.object(fsl_native_worker, "_run_cmd", side_effect=fake_run_cmd),
                    mock.patch.object(fsl_native_worker.shutil, "which", side_effect=lambda name, path=None: f"/fake/bin/{name}"),
                ):
                    res = fsl_native_worker.process_single_run(
                        run_id="sub-0001_run-01",
                        out_stem="sub-0001_task-rest_run-01_bold",
                        site="KKI",
                        raw_path=raw_path,
                        output_root=out_root,
                        fsl_dir=td / "fake_fsl",
                        slice_scheme=slice_scheme,
                        sidecar_json=sidecar_path,
                        drop_initial_trs=5,
                        auto_drop_nonsteady=False,
                        skip_stc=skip_stc,
                        center_origin="foreground",
                    )

                self.assertEqual(res["status"], "ok", msg=res.get("error"))
                self.assertEqual(res["stc_method"], "skipped")
                self.assertAlmostEqual(res["tr_sec"], 2.5, places=5)
                self.assertEqual(res["dropped_trs"], 5)
                self.assertEqual(res["shape"], [8, 8, 6, 19])
                self.assertEqual(res["origin_mode"], "foreground")
                self.assertFalse(any(Path(c[0]).name == "slicetimer" for c in called_cmds))

                # Verify saved native BOLD and brain mask share centered oblique affine & synced qform/sform
                out_img = nib.load(res["fsl_native_path"])
                mask_img = nib.load(res["mask_path"])
                np.testing.assert_allclose(out_img.affine[:3, :3], oblique_aff[:3, :3], atol=1e-5)
                np.testing.assert_allclose(out_img.affine, mask_img.affine, atol=1e-5)
                np.testing.assert_allclose(out_img.get_qform(), out_img.get_sform(), atol=1e-4)
                np.testing.assert_allclose(mask_img.get_qform(), mask_img.get_sform(), atol=1e-4)

                fg_mask = (np.asarray(out_img.dataobj) != 0).any(axis=3)
                c_vox = np.argwhere(fg_mask).mean(axis=0)
                c_world = out_img.affine[:3, :3] @ c_vox + out_img.affine[:3, 3]
                np.testing.assert_allclose(c_world, [0.0, 0.0, 0.0], atol=1e-4)

    def test_affine_origin_centering_oblique_qform_sform_and_data_process(self):
        """center_affine_origin preserves oblique A[:3,:3], centers centroid to 0, syncs qform/sform, and survives pad_crop_keep_world."""
        theta_x, theta_y, theta_z = np.deg2rad([12.0, -8.0, 21.0])
        rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
        ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
        rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
        linear = (rz @ ry @ rx) @ np.diag([-3.125, 3.125, 4.0])
        raw_aff = np.eye(4, dtype=np.float64)
        raw_aff[:3, :3] = linear
        raw_aff[:3, 3] = np.array([-294.1, 419.7, -169.2])

        shape = (64, 64, 34, 20)
        fg = np.zeros(shape[:3], dtype=bool)
        fg[18:50, 12:46, 6:28] = True  # Intentionally off-center inside FOV

        centered_fg, raw_orig, new_orig = data_process.center_affine_origin(raw_aff, shape[:3], foreground_mask=fg, mode="foreground")
        np.testing.assert_allclose(centered_fg[:3, :3], raw_aff[:3, :3], atol=1e-12)
        np.testing.assert_allclose(new_orig, [0.0, 0.0, 0.0], atol=1e-10)
        self.assertGreater(np.linalg.norm(raw_orig), 100.0)

        centered_grid, _, grid_orig = data_process.center_affine_origin(raw_aff, shape[:3], mode="grid")
        np.testing.assert_allclose(centered_grid[:3, :3], raw_aff[:3, :3], atol=1e-12)
        np.testing.assert_allclose(grid_orig, [0.0, 0.0, 0.0], atol=1e-10)
        # Foreground centroid and grid center differ because fg is off-center in FOV
        self.assertGreater(np.linalg.norm(centered_fg[:3, 3] - centered_grid[:3, 3]), 1.0)

        unchanged, _, _ = data_process.center_affine_origin(raw_aff, shape[:3], foreground_mask=fg, mode="none")
        np.testing.assert_allclose(unchanged, raw_aff, atol=1e-12)

        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            rng = np.random.default_rng(123)
            vol = np.zeros(shape, dtype=np.float32)
            vol[fg, :] = rng.normal(loc=500.0, scale=20.0, size=(int(fg.sum()), shape[3])).astype(np.float32)

            # Case 1: Pre-centered input -> data_process pad_crop_keep_world preserves [0,0,0] foreground centroid
            in_centered = td / "in_centered" / "nativespace" / "sub-01_bold.nii.gz"
            in_centered.parent.mkdir(parents=True)
            img1 = nib.Nifti1Image(vol, centered_fg)
            img1.header.set_zooms((3.125, 3.125, 4.0, 2.0))
            img1.header.set_xyzt_units(xyz="mm", t="sec")
            img1.set_qform(centered_fg, code=1)
            img1.set_sform(centered_fg, code=1)
            nib.save(img1, str(in_centered))

            out1 = td / "out1" / "nativespace" / "sub-01_bold_global_zscore.nii.gz"
            dp_args1 = data_process.build_arg_parser().parse_args(
                ["--input-root", str(td / "in_centered"), "--output-root", str(td / "out1"), "--spaces", "native"]
            )
            self.assertEqual(data_process.process_one_file(in_centered, out1, "native", None, dp_args1), "processed")

            qc1 = run_pipeline._qc_one_flexibrain_file((out1, "foreground"))
            self.assertTrue(qc1["ok"], msg=qc1["reasons"])
            self.assertLess(qc1["origin_norm_mm"], 1e-3)
            self.assertLess(qc1["qform_sform_max_diff"], 1e-3)

            # Case 2: Standalone data_process --center-origin foreground on raw uncentered oblique input
            in_raw = td / "in_raw" / "nativespace" / "sub-02_bold.nii.gz"
            in_raw.parent.mkdir(parents=True)
            img2 = nib.Nifti1Image(vol, raw_aff)
            img2.header.set_zooms((3.125, 3.125, 4.0, 2.0))
            img2.header.set_xyzt_units(xyz="mm", t="sec")
            img2.set_qform(raw_aff, code=1)
            img2.set_sform(raw_aff, code=1)
            nib.save(img2, str(in_raw))

            out2 = td / "out2" / "nativespace" / "sub-02_bold_global_zscore.nii.gz"
            dp_args2 = data_process.build_arg_parser().parse_args(
                [
                    "--input-root", str(td / "in_raw"),
                    "--output-root", str(td / "out2"),
                    "--spaces", "native",
                    "--center-origin", "foreground",
                ]
            )
            self.assertEqual(data_process.process_one_file(in_raw, out2, "native", None, dp_args2), "processed")
            qc2 = run_pipeline._qc_one_flexibrain_file((out2, "foreground"))
            self.assertTrue(qc2["ok"], msg=qc2["reasons"])
            self.assertLess(qc2["origin_norm_mm"], 1e-3)
            self.assertLess(qc2["qform_sform_max_diff"], 1e-3)

            saved1 = nib.load(str(out1))
            saved2 = nib.load(str(out2))
            np.testing.assert_allclose(saved1.affine, saved2.affine, atol=1e-4)

    def test_unified_dummy_tr_policy_across_protocols(self):
        """Both ABIDE1 and ADHD-200 protocols enforce unified default_drop_initial_trs == 5 with CLI override."""
        abide_proto = load_protocol_yaml(FSL_PREPROCESS_DIR / "protocols" / "abide1.yaml")
        adhd_proto = load_protocol_yaml(FSL_PREPROCESS_DIR / "protocols" / "adhd200.yaml")
        for site in ("CMU_a", "NYU_a", "OHSU", "UM_1"):
            self.assertEqual(resolve_drop_initial_trs(site, abide_proto), 5)
        for site in ("Brown", "KKI", "NYU", "OHSU", "Peking_2", "Pittsburgh", "WashU"):
            self.assertEqual(resolve_drop_initial_trs(site, adhd_proto), 5)
        self.assertEqual(resolve_drop_initial_trs("OHSU", adhd_proto, cli_override=0), 0)
        self.assertEqual(resolve_drop_initial_trs("NYU", abide_proto, cli_override=3), 3)


if __name__ == "__main__":
    unittest.main()
