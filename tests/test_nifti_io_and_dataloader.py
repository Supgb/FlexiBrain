import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import torch

from flexibrain.config import DataConfig, TrainingConfig
from flexibrain.data import nifti as nifti_mod
from flexibrain.data.builders import build_downstream_dataloaders, build_pretrain_dataloaders
from flexibrain.data.collate import prepare_batch_data as pretrain_prepare_batch_data
from flexibrain.data.classification import prepare_batch_data as downstream_prepare_batch_data
from flexibrain.data.nifti import NiftiTxtDataset, _fold_4d_array, _load_nifti


REAL_NIFTI_LIST = Path("/mnt/ironwolf/multi-cohort-fsl-nostc/lists/pretrain_train.txt")


def _write_synthetic_nifti(
    path: Path,
    data: np.ndarray,
    *,
    tr: float = 2.0,
    dtype=np.float32,
    nifti2: bool = False,
) -> np.ndarray:
    affine = np.array(
        [
            [-2.5, 0.1, 0.0, 12.0],
            [0.0, 2.5, -0.2, -18.0],
            [0.05, 0.0, 3.0, -5.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cls = nib.Nifti2Image if nifti2 else nib.Nifti1Image
    img = cls(data.astype(np.float32), affine)
    img.set_data_dtype(dtype)
    img.header.set_zooms((2.5, 2.5, 3.0, tr))
    img.header.set_xyzt_units(xyz="mm", t="sec")
    nib.save(img, str(path))
    return affine


class NiftiIoAndDataLoaderTests(unittest.TestCase):
    def test_slice_view_releases_full_array_and_matches_reference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            rng = np.random.default_rng(7)
            raw = rng.normal(0.0, 1.0, size=(8, 8, 6, 30)).astype(np.float32)

            cases = [
                ("unscaled_gz", td / "unscaled.nii.gz", np.float32, False),
                ("unscaled_nii", td / "unscaled.nii", np.float32, False),
                ("scaled_int16_gz", td / "scaled.nii.gz", np.int16, False),
                ("nifti2_gz", td / "nifti2.nii.gz", np.float32, True),
            ]
            for label, path, dtype, is_nifti2 in cases:
                with self.subTest(case=label):
                    _write_synthetic_nifti(path, raw, tr=2.0, dtype=dtype, nifti2=is_nifti2)
                    list_file = td / f"{label}.txt"
                    list_file.write_text(f"{path}\n", encoding="utf-8")

                    ref_img = nib.load(str(path), mmap=True)
                    ref_slice = ref_img.get_fdata(dtype=np.float32)[..., :9]

                    # T_prime=3, tau=6.0, TR=2.0 -> kt=3, T_selected=9 (< 30)
                    ds = NiftiTxtDataset(
                        list_file,
                        return_torch=False,
                        memory_map=True,
                        T_prime=3,
                        tau_seconds=6.0,
                    )
                    sample = ds[0]
                    arr = sample["data"]
                    self.assertEqual(arr.shape, (8, 8, 6, 9))
                    self.assertEqual(arr.dtype, np.float32)
                    self.assertIsNone(arr.base)
                    self.assertTrue(arr.flags.writeable)
                    np.testing.assert_array_equal(arr, ref_slice)
                    np.testing.assert_array_equal(sample["affine"], ref_img.affine)
                    self.assertEqual(sample["header"].get_zooms(), ref_img.header.get_zooms())
                    self.assertEqual(sample["header"].get_xyzt_units(), ref_img.header.get_xyzt_units())

    def test_stdlib_fallback_when_isal_unavailable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            rng = np.random.default_rng(11)
            raw = rng.normal(0.0, 1.0, size=(6, 6, 6, 20)).astype(np.float32)
            path = td / "fallback.nii.gz"
            _write_synthetic_nifti(path, raw, tr=2.0, dtype=np.float32)
            list_file = td / "list.txt"
            list_file.write_text(f"{path}\n", encoding="utf-8")

            ref_img = nib.load(str(path), mmap=True)
            ref_slice = ref_img.get_fdata(dtype=np.float32)[..., :6]

            with mock.patch.object(nifti_mod, "_igzip", None):
                ds = NiftiTxtDataset(list_file, return_torch=False, T_prime=2, tau_seconds=6.0)
                sample = ds[0]
                arr = sample["data"]
                self.assertIsNone(arr.base)
                self.assertTrue(arr.flags.writeable)
                np.testing.assert_array_equal(arr, ref_slice)

    def test_isal_igzip_bit_exact_against_nib_load_on_real_nifti_gz(self):
        if nifti_mod._igzip is None:
            self.skipTest("python-isal is not installed")
        if not REAL_NIFTI_LIST.exists():
            self.skipTest(f"Real dataset list not present: {REAL_NIFTI_LIST}")

        first_line = next(
            (ln.strip() for ln in REAL_NIFTI_LIST.read_text(encoding="utf-8").splitlines() if ln.strip()),
            None,
        )
        self.assertIsNotNone(first_line)
        real_path = Path(first_line)
        if not real_path.exists():
            self.skipTest(f"Real NIfTI file not present: {real_path}")

        ref_img = nib.load(str(real_path), mmap=True)
        ref_full = ref_img.get_fdata(dtype=np.float32)

        # Full load via _load_nifti (igzip path)
        data_full, aff_full, hdr_full = _load_nifti(real_path, mmap=True)
        self.assertIsNotNone(data_full)
        self.assertIsNone(data_full.base)
        self.assertTrue(data_full.flags.writeable)
        np.testing.assert_array_equal(data_full, ref_full)
        np.testing.assert_array_equal(aff_full, ref_img.affine)
        self.assertEqual(hdr_full.get_zooms(), ref_img.header.get_zooms())
        self.assertEqual(hdr_full.get_xyzt_units(), ref_img.header.get_xyzt_units())

        # Partial load via NiftiTxtDataset (igzip + early-stop slice path)
        with tempfile.TemporaryDirectory() as tmpdir:
            list_file = Path(tmpdir) / "one.txt"
            list_file.write_text(f"{real_path}\n", encoding="utf-8")
            ds = NiftiTxtDataset(list_file, return_torch=False, T_prime=30, tau_seconds=6.0)
            sample = ds[0]
            t_sel = sample["T_selected"]
            arr = sample["data"]
            self.assertIsNone(arr.base)
            self.assertTrue(arr.flags.writeable)
            np.testing.assert_array_equal(arr, ref_full[..., :t_sel])
            np.testing.assert_array_equal(sample["affine"], ref_img.affine)
            self.assertEqual(sample["header"].get_zooms(), ref_img.header.get_zooms())
            self.assertEqual(sample["header"].get_xyzt_units(), ref_img.header.get_xyzt_units())

    def test_dataloader_builders_worker_config_zero_and_positive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            rng = np.random.default_rng(19)
            paths = []
            for idx in range(4):
                p = td / f"sub-{1000 + idx:07d}_task-rest_run-01_bold.nii.gz"
                _write_synthetic_nifti(p, rng.normal(size=(4, 4, 4, 12)).astype(np.float32), tr=2.0)
                paths.append(p)

            train_txt = td / "train.txt"
            val_txt = td / "val.txt"
            test_txt = td / "test.txt"
            train_txt.write_text("\n".join(str(p) for p in paths[:2]) + "\n", encoding="utf-8")
            val_txt.write_text(f"{paths[2]}\n", encoding="utf-8")
            test_txt.write_text(f"{paths[3]}\n", encoding="utf-8")

            csv_path = td / "labels.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["Subject", "Group_idx"])
                w.writeheader()
                for idx in range(4):
                    w.writerow({"Subject": f"{1000 + idx:07d}", "Group_idx": idx % 2})

            train_cfg = TrainingConfig(seed=42)

            # num_workers == 0 must not pass prefetch_factor or persistent_workers
            data_zero_raw = DataConfig(
                train_list=str(train_txt),
                val_list=str(val_txt),
                test_list=str(test_txt),
                csv=str(csv_path),
                batch_size=2,
                num_workers=0,
                prefetch_factor=1,
                persistent_workers=True,
                temporal_fold=False,
                T_prime=2,
                tau_seconds=6.0,
            )
            pt_train_0, pt_val_0 = build_pretrain_dataloaders(data_zero_raw, train_cfg)
            self.assertEqual(pt_train_0.num_workers, 0)
            self.assertIsNone(pt_train_0.prefetch_factor)
            self.assertFalse(pt_train_0.persistent_workers)
            batch_0 = next(iter(pt_train_0))
            self.assertIsInstance(batch_0["data"], torch.Tensor)
            self.assertEqual(tuple(batch_0["data"].shape), (2, 4, 4, 4, 6))
            self.assertEqual(pt_val_0.num_workers, 0)

            # Default temporal_fold=True produces (B, kt, Z, Y, X) and nonzero_mask
            data_zero_folded = DataConfig(
                train_list=str(train_txt),
                val_list=str(val_txt),
                test_list=str(test_txt),
                csv=str(csv_path),
                batch_size=2,
                num_workers=0,
                prefetch_factor=1,
                persistent_workers=True,
                temporal_fold=True,
                T_prime=2,
                tau_seconds=6.0,
            )
            pt_train_f, _ = build_pretrain_dataloaders(data_zero_folded, train_cfg)
            batch_f = next(iter(pt_train_f))
            self.assertEqual(tuple(batch_f["data"].shape), (2, 3, 4, 4, 4))
            self.assertEqual(tuple(batch_f["nonzero_mask"].shape), (2, 4, 4, 4))
            self.assertEqual(batch_f["folded_kt"].tolist(), [3, 3])
            _, meta_f, orig_ts_f, _ = pretrain_prepare_batch_data(batch_f, torch.device("cpu"))
            self.assertIn("_stape_folded", meta_f)
            self.assertEqual(orig_ts_f.tolist(), [6, 6])

            ds_train_0, ds_val_0, ds_test_0 = build_downstream_dataloaders(data_zero_folded, train_cfg)
            self.assertEqual(ds_train_0.num_workers, 0)
            self.assertFalse(ds_train_0.persistent_workers)
            self.assertIsNone(ds_val_0.prefetch_factor)
            self.assertIsNotNone(ds_test_0)
            self.assertEqual(ds_test_0.num_workers, 0)
            ds_batch_f = next(iter(ds_train_0))
            self.assertEqual(tuple(ds_batch_f["data"].shape), (2, 3, 4, 4, 4))
            self.assertEqual(tuple(ds_batch_f["nonzero_mask"].shape), (2, 4, 4, 4))
            _, ds_meta_f, _, _, _ = downstream_prepare_batch_data(ds_batch_f, torch.device("cpu"))
            self.assertIn("_stape_folded", ds_meta_f)

            # num_workers > 0 must forward prefetch_factor and persistent_workers
            data_pos = DataConfig(
                train_list=str(train_txt),
                val_list=str(val_txt),
                test_list=str(test_txt),
                csv=str(csv_path),
                batch_size=2,
                num_workers=2,
                prefetch_factor=1,
                persistent_workers=True,
                T_prime=2,
                tau_seconds=6.0,
            )
            pt_train_2, pt_val_2 = build_pretrain_dataloaders(data_pos, train_cfg)
            self.assertEqual(pt_train_2.num_workers, 2)
            self.assertEqual(pt_train_2.prefetch_factor, 1)
            self.assertTrue(pt_train_2.persistent_workers)
            self.assertEqual(pt_val_2.prefetch_factor, 1)
            self.assertFalse(pt_val_2.persistent_workers)

            ds_train_2, ds_val_2, ds_test_2 = build_downstream_dataloaders(data_pos, train_cfg)
            self.assertEqual(ds_train_2.num_workers, 2)
            self.assertEqual(ds_train_2.prefetch_factor, 1)
            self.assertTrue(ds_train_2.persistent_workers)
            for loader in (ds_val_2, ds_test_2):
                self.assertEqual(loader.num_workers, 2)
                self.assertEqual(loader.prefetch_factor, 1)
                self.assertFalse(loader.persistent_workers)

    def test_temporal_fold_matches_raw_nifti_across_formats_and_partial_windows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            rng = np.random.default_rng(23)
            raw = rng.normal(0.0, 1.0, size=(6, 5, 4, 14)).astype(np.float32)
            raw[0, 0, 0, :] = 0.0  # ensure some all-zero spatial locations

            cases = [
                ("unscaled_gz_partial", td / "unscaled.nii.gz", np.float32, False, None),
                ("unscaled_nii_partial", td / "unscaled.nii", np.float32, False, None),
                ("scaled_int16_gz", td / "scaled.nii.gz", np.int16, False, None),
                ("nifti2_gz", td / "nifti2.nii.gz", np.float32, True, None),
                ("stdlib_gz_fallback", td / "unscaled.nii.gz", np.float32, False, None),
            ]
            for label, path, dtype, is_nifti2, _ in cases:
                with self.subTest(case=label):
                    if not path.exists():
                        _write_synthetic_nifti(path, raw, tr=2.0, dtype=dtype, nifti2=is_nifti2)
                    list_file = td / f"{label}.txt"
                    list_file.write_text(f"{path}\n", encoding="utf-8")

                    # T_prime=6, tau=6.0, TR=2.0 -> kt=3, T_selected=min(18, 14)=14 (14 % 3 == 2 partial window)
                    ctx = mock.patch.object(nifti_mod, "_igzip", None) if label == "stdlib_gz_fallback" else mock.MagicMock()
                    with ctx:
                        ds_raw = NiftiTxtDataset(list_file, return_torch=False, T_prime=6, tau_seconds=6.0, temporal_fold=False)
                        ds_fold = NiftiTxtDataset(list_file, return_torch=False, T_prime=6, tau_seconds=6.0, temporal_fold=True)
                        s_raw = ds_raw[0]
                        s_fold = ds_fold[0]

                    exp_folded, exp_nz = _fold_4d_array(s_raw["data"], kt=3)
                    self.assertEqual(s_fold["folded_kt"], 3)
                    self.assertEqual(s_fold["T_selected"], 14)
                    self.assertEqual(s_fold["data"].shape, (3, 4, 5, 6))
                    self.assertTrue(s_fold["data"].flags.writeable)
                    np.testing.assert_allclose(s_fold["data"], exp_folded, atol=1e-6, rtol=1e-6)
                    np.testing.assert_array_equal(s_fold["nonzero_mask"], exp_nz)
                    np.testing.assert_array_equal(s_fold["affine"], s_raw["affine"].astype(np.float32))
                    self.assertEqual(s_fold["voxel"], s_raw["voxel"])
                    self.assertEqual(s_fold["tr"], s_raw["tr"])

    def test_temporal_fold_matches_real_nifti_gz_and_disk_cache_invalidation(self):
        if nifti_mod._igzip is None:
            self.skipTest("python-isal is not installed")
        if not REAL_NIFTI_LIST.exists():
            self.skipTest(f"Real dataset list not present: {REAL_NIFTI_LIST}")
        first_line = next(
            (ln.strip() for ln in REAL_NIFTI_LIST.read_text(encoding="utf-8").splitlines() if ln.strip()),
            None,
        )
        if not first_line or not Path(first_line).exists():
            self.skipTest("Real NIfTI file not present")
        real_path = Path(first_line)

        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            list_file = td / "real.txt"
            list_file.write_text(f"{real_path}\n", encoding="utf-8")
            cache_dir = td / "folded_cache"

            ds_raw = NiftiTxtDataset(list_file, return_torch=False, T_prime=30, tau_seconds=6.0, temporal_fold=False)
            ds_fold = NiftiTxtDataset(
                list_file, return_torch=False, T_prime=30, tau_seconds=6.0,
                temporal_fold=True, folded_cache_dir=cache_dir,
            )
            s_raw = ds_raw[0]
            s_fold_miss = ds_fold[0]
            kt = s_fold_miss["folded_kt"]
            exp_folded, exp_nz = _fold_4d_array(s_raw["data"], kt=kt)
            np.testing.assert_array_equal(s_fold_miss["data"], exp_folded)
            np.testing.assert_array_equal(s_fold_miss["nonzero_mask"], exp_nz)

            cache_files = list(cache_dir.glob("*.fbc"))
            self.assertEqual(len(cache_files), 1)

            # Second dataset reads from .fbc cache without invoking _load_nifti_folded
            ds_fold2 = NiftiTxtDataset(
                list_file, return_torch=False, T_prime=30, tau_seconds=6.0,
                temporal_fold=True, folded_cache_dir=cache_dir,
            )
            with mock.patch.object(nifti_mod, "_load_nifti_folded", side_effect=AssertionError("should hit cache")):
                s_fold_hit = ds_fold2[0]
            np.testing.assert_array_equal(s_fold_hit["data"], exp_folded)
            np.testing.assert_array_equal(s_fold_hit["nonzero_mask"], exp_nz)
            np.testing.assert_array_equal(s_fold_hit["affine"], s_fold_miss["affine"])
            self.assertEqual(s_fold_hit["voxel"], s_fold_miss["voxel"])
            self.assertEqual(s_fold_hit["tr"], s_fold_miss["tr"])
            self.assertEqual(s_fold_hit["T_selected"], s_fold_miss["T_selected"])
            self.assertEqual(s_fold_hit["folded_kt"], s_fold_miss["folded_kt"])

            # Changing T_prime or source file mtime invalidates cache key
            synth = td / "synth.nii.gz"
            rng = np.random.default_rng(31)
            _write_synthetic_nifti(synth, rng.normal(size=(4, 4, 4, 12)).astype(np.float32), tr=2.0)
            synth_list = td / "synth.txt"
            synth_list.write_text(f"{synth}\n", encoding="utf-8")
            synth_cache = td / "synth_cache"
            ds_s1 = NiftiTxtDataset(synth_list, return_torch=False, T_prime=2, tau_seconds=6.0, temporal_fold=True, folded_cache_dir=synth_cache)
            _ = ds_s1[0]
            self.assertEqual(len(list(synth_cache.glob("*.fbc"))), 1)

            # Touch mtime + change content -> new cache entry
            _write_synthetic_nifti(synth, rng.normal(size=(4, 4, 4, 12)).astype(np.float32), tr=2.0)
            st = synth.stat()
            os.utime(synth, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
            _ = ds_s1[0]
            self.assertEqual(len(list(synth_cache.glob("*.fbc"))), 2)


if __name__ == "__main__":
    unittest.main()
