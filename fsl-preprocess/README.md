# General FSL Native-Space fMRI Preprocessing (`fsl-preprocess`)

Minimal, resolution-preserving native-space fMRI preprocessing for **FlexiBrain**, using modular FSL tools (`slicetimer`, `mcflirt`, `bet`/`bet2`, `fslmaths`) followed by FlexiBrain's `data_process.py` (`96x96x96` center pad/crop preserving world affine + foreground global z-scoring).

## 1. FSL Installation (Disk-Safe on `/mnt/ironwolf`)

```bash
mkdir -p /mnt/ironwolf/conda-envs /mnt/ironwolf/conda-pkgs
CONDA_PKGS_DIRS=/mnt/ironwolf/conda-pkgs conda create -y --override-channels \
  -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \
  -c conda-forge \
  -p /mnt/ironwolf/conda-envs/fsl-tools \
  python=3.11 fsl-base fsl-bet2 fsl-mcflirt fsl-slicetimer fsl-avwutils
```

## 2. Pipeline Steps Per Run

0. **Dummy / Non-Steady-State TR Trimming**: Drops protocol-specified initial dummy volumes (`default_drop_initial_trs`, e.g., `5` for ABIDE1 to remove T1 equilibration transients) and optionally extends past contiguous outlier prefix frames when `--auto-drop-nonsteady` is set, prior to slice-timing and motion correction.
1. **Slice-Timing Correction (`slicetimer`)**: Uses BIDS JSON `SliceTiming` when present and matching `num_slices`, or site protocol from `protocols/<dataset>.yaml` (converted via `slice_timing.py` to FSL `--tcustom` TR-fraction shifts centered on the acquisition midpoint). Set scheme to `skip` for multiband sequences without `SliceTiming`.
2. **Head-Motion Correction (`mcflirt`)**: Rigid 6-DOF motion correction in native space (`-meanvol -plots`), keeping native voxel resolution and saving `*.par` + framewise displacement stats.
3. **Brain Extraction (`bet` / `bet2`)**: Skull-strips the motion-corrected mean volume (`-f 0.3 -m`, with automatic fallback if foreground mask fraction is degenerate).
4. **Native Masking & Header Preservation (`fslmaths` + `nibabel`)**: Applies the native brain mask so background voxels are strictly `0.0`, restores exact native `(vx, vy, vz, TR)` in `('mm', 'sec')` and `qform`/`sform`, and writes `<output_root>/fsl_native/nativespace/<out_stem>.nii.gz`.
5. **FlexiBrain `96x96x96` Conversion (`data_process.py`)**: Center pads/crops to `96x96x96` while updating `affine[:3, 3]` and applies sample-wise global foreground z-scoring (`<output_root>/flexibrain_ready/nativespace/*_global_zscore.nii.gz`).

## 3. Running ABIDE1

```bash
/home/ubuntu/projects/FlexiBrain/.venv/bin/python fsl-preprocess/run_pipeline.py \
  --dataset abide1 \
  --input-root /mnt/barracuda/ABIDE1 \
  --output-root /mnt/ironwolf/abide1-fsl \
  --index-json /mnt/barracuda/abide1-prep/status/abide_index.json \
  --protocol-yaml fsl-preprocess/protocols/abide1.yaml \
  --fsl-dir /mnt/ironwolf/conda-envs/fsl-tools \
  --workers 20 \
  --stage-workers 8 \
  --auto-drop-nonsteady

/home/ubuntu/projects/FlexiBrain/.venv/bin/python fsl-preprocess/make_splits.py \
  --nifti-dir /mnt/ironwolf/abide1-fsl/flexibrain_ready/nativespace \
  --phenotypic-csv /mnt/barracuda/abide1-prep/Phenotypic_V1_0b.csv \
  --manifest-csv /mnt/ironwolf/abide1-fsl/qc/pipeline_manifest.csv \
  --max-mean-fd 0.5 \
  --output-dir /mnt/ironwolf/abide1-fsl
```

## 4. Extending to ADHD200 or HCP-YA

- **ADHD200 (BIDS tree)**:
  ```bash
  /home/ubuntu/projects/FlexiBrain/.venv/bin/python fsl-preprocess/run_pipeline.py \
    --dataset bids \
    --input-root /mnt/barracuda/ADHD200 \
    --output-root /mnt/ironwolf/adhd200-fsl \
    --protocol-yaml fsl-preprocess/protocols/adhd200.yaml \
    --fsl-dir /mnt/ironwolf/conda-envs/fsl-tools \
    --workers 20 \
    --auto-drop-nonsteady

  /home/ubuntu/projects/FlexiBrain/.venv/bin/python fsl-preprocess/make_splits.py \
    --nifti-dir /mnt/ironwolf/adhd200-fsl/flexibrain_ready/nativespace \
    --phenotypic-csv /mnt/barracuda/ADHD200/adhd200_preprocessed_phenotypics.tsv \
    --manifest-csv /mnt/ironwolf/adhd200-fsl/qc/pipeline_manifest.csv \
    --label-preset adhd200_dx \
    --id-column "ScanDir ID" \
    --label-column DX \
    --site-column Site \
    --max-mean-fd 0.5 \
    --include-unlabeled-in-pretrain \
    --output-dir /mnt/ironwolf/adhd200-fsl
  ```
- **HCP-YA (unprocessed ZIPs)**:
  ```bash
  /home/ubuntu/projects/FlexiBrain/.venv/bin/python fsl-preprocess/run_pipeline.py \
    --dataset hcp_ya \
    --input-root /mnt/barracuda/HCP_YA/HCP_YA_Functional \
    --output-root /mnt/ironwolf/hcp-ya-fsl \
    --workers 16
  ```
  (Set `default_scheme: skip` in an `hcp_ya.yaml` protocol or pass no protocol for BIDS `SliceTiming` auto-detection.)
