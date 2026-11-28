from __future__ import annotations
import glob
import hashlib
import io
import os
import random
import struct
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

try:
    from isal import igzip as _igzip
except ImportError:
    _igzip = None


class fMRIDataset(Dataset):
    def __init__(self, 
                 data_root, datasets, split_suffixes, crop_length=40, downstream=False):

        self.file_paths = []
        self.crop_length = crop_length
        self.downstream = downstream

        for dataset_name in datasets:
            for suffix in split_suffixes:
                folder_name = f"{dataset_name}_{suffix}"
                folder_path = os.path.join(data_root, folder_name)
                if not os.path.exists(folder_path):
                    print(f"Warning: Folder not found: {folder_path}")
                    continue

                for root, dirs, files in os.walk(folder_path):
                    npz_files = glob.glob(os.path.join(root, "*.npz"))
                    if len(npz_files) > 1:
                        sample_size = max(1, int(len(npz_files))) 
                        npz_files = random.sample(npz_files, sample_size)
                    self.file_paths.extend(npz_files)

        print(f"Dataset loaded. Total files found: {len(self.file_paths)}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            with np.load(file_path) as data_file:
                key = list(data_file.keys())[0]
                fmri_data = data_file[key] 
                fmri_data = fmri_data.astype(np.float32)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None

        total_time_frames = fmri_data.shape[-1]
        if total_time_frames > self.crop_length:
            start_idx = np.random.randint(0, total_time_frames - self.crop_length + 1)
            end_idx = start_idx + self.crop_length
            cropped_data = fmri_data[..., start_idx:end_idx]
        else:
            cropped_data = fmri_data[..., :self.crop_length]

        data_tensor = torch.from_numpy(cropped_data)
        data_tensor = data_tensor.permute(3, 0, 1, 2)

        return data_tensor


def _read_list_files(txt_files: Union[str, Path, Sequence[Union[str, Path]]]) -> List[Path]:
    """Read one or many .txt files and collect absolute paths listed in them.

    Each line should contain a path to a .nii or .nii.gz file. Empty lines and lines
    starting with '#' are ignored. Paths are expanded and normalized to absolute Paths.
    """
    if isinstance(txt_files, (str, Path)):
        txt_files = [txt_files]
    paths: List[Path] = []
    for f in txt_files:  # type: ignore[assignment]
        f = Path(f)
        if not f.exists():
            raise FileNotFoundError(f"List file not found: {f}")
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(os.path.expanduser(line)).resolve()
            # allow relative paths inside list files (relative to the list file dir)
            if not p.exists():
                p = (f.parent / line).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Path from list file does not exist: {line} (resolved: {p})")
            if p.suffix not in {".nii", ".gz"} and not str(p).endswith(".nii.gz"):
                raise ValueError(f"Not a NIfTI file: {p}")
            paths.append(p)
    # deduplicate while preserving order
    seen = set()
    deduped = []
    for p in paths:
        if p not in seen:
            deduped.append(p)
            seen.add(p)
    return deduped


def _space_time_units_to_mm_s(header: nib.nifti1.Nifti1Header) -> Tuple[Tuple[float, float, float], float]:
    """Return (vx, vy, vz) in millimeters and TR in seconds from a NIfTI header.

    Uses header.get_zooms() and header.get_xyzt_units(). Safely handles cases with
    missing time dimension or unusual units.
    """
    zooms = header.get_zooms()
    # space-time units, e.g. ("mm", "sec")
    space_u, time_u = header.get_xyzt_units()

    # Spatial voxel sizes
    vx, vy, vz = (zooms + (1.0, 1.0, 1.0, 1.0))[:3]
    # Convert to mm if needed
    if space_u == "m":
        vx, vy, vz = vx * 1000.0, vy * 1000.0, vz * 1000.0
    elif space_u in ("mm", None, "unknown"):
        pass
    else:
        # Fallback: assume values already in mm
        pass

    # Temporal resolution (TR)
    tr = 0.0
    if len(zooms) >= 4:
        tr = float(zooms[3])
        if time_u == "msec":
            tr = tr / 1000.0
        elif time_u in ("usec", "microsec"):
            tr = tr / 1e6
        elif time_u in ("sec", None, "unknown"):
            pass
        else:
            # Unknown -> leave as-is
            pass
    return (float(vx), float(vy), float(vz)), float(tr)


def _is_unscaled_float32(img: nib.spatialimages.SpatialImage) -> bool:
    if img.header.get_data_dtype() != np.float32:
        return False
    dataobj = getattr(img, "dataobj", None)
    if dataobj is None:
        return False
    slope = getattr(dataobj, "slope", 1.0)
    inter = getattr(dataobj, "inter", 0.0)
    slope_identity = slope is None or (isinstance(slope, float) and np.isnan(slope)) or float(slope) == 1.0
    inter_identity = inter is None or (isinstance(inter, float) and np.isnan(inter)) or float(inter) == 0.0
    return slope_identity and inter_identity


def _read_nifti_data(img: nib.spatialimages.SpatialImage, max_frames: Optional[int] = None) -> np.ndarray:
    if max_frames is not None and len(img.shape) >= 4 and max_frames < img.shape[3]:
        if _is_unscaled_float32(img):
            # Slice directly through ArrayProxy so gzip decompression stops after max_frames.
            raw_slice = img.dataobj[..., :max_frames]
            return np.array(raw_slice, dtype=np.float32, order="F", copy=True)
        full_data = img.get_fdata(dtype=np.float32)
        return np.array(full_data[..., :max_frames], order="F", copy=True)

    full_data = img.get_fdata(dtype=np.float32)
    if (not full_data.flags.writeable) or (full_data.base is not None):
        return np.array(full_data, order="F", copy=True)
    return full_data


def _load_nifti(
    path: Union[str, Path],
    mmap: bool = True,
    max_frames_fn: Optional[Callable[[nib.nifti1.Nifti1Header], Optional[int]]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[nib.nifti1.Nifti1Header]]:
    path_str = str(path)
    use_igzip = _igzip is not None and path_str.lower().endswith(".gz")

    if use_igzip:
        fallback_to_nib = False
        try:
            with _igzip.open(path_str, "rb") as f:
                try:
                    fh = nib.FileHolder(fileobj=f)
                    img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
                    affine = img.affine.copy()
                    header = img.header.copy()
                except (nib.filebasedimages.ImageFileError, nib.spatialimages.HeaderDataError):
                    fallback_to_nib = True
                    img = None
                except Exception:
                    return None, None, None

                if not fallback_to_nib:
                    # Run metadata/frame callback outside the file-I/O error handler so validation errors propagate.
                    max_frames = max_frames_fn(header) if max_frames_fn is not None else None
                    try:
                        data = _read_nifti_data(img, max_frames=max_frames)
                        return data, affine, header
                    except Exception:
                        return None, None, None
        except OSError:
            return None, None, None
        if not fallback_to_nib:
            return None, None, None

    try:
        img = nib.load(path_str, mmap=mmap)
        affine = img.affine.copy()
        header = img.header.copy()
    except Exception:
        return None, None, None

    max_frames = max_frames_fn(header) if max_frames_fn is not None else None
    try:
        data = _read_nifti_data(img, max_frames=max_frames)
        return data, affine, header
    except Exception:
        return None, None, None


_FOLDED_CACHE_MAGIC = b"FBFOLD01"
_FOLDED_CACHE_HEADER = struct.Struct("<8s5I4d16f")


def _read_exact_into(fileobj, mv: memoryview) -> None:
    offset = 0
    total = len(mv)
    while offset < total:
        n = fileobj.readinto(mv[offset:])
        if n is None or n == 0:
            raise EOFError(f"Unexpected EOF after {offset}/{total} bytes")
        offset += n


def _fold_4d_array(data: np.ndarray, kt: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fold (X, Y, Z, T) volume into (kt, Z, Y, X) float32 sum and (Z, Y, X) bool mask."""
    if data.ndim < 4:
        data = data[..., None]
    X, Y, Z, T_sel = (int(d) for d in data.shape[:4])
    folded = np.zeros((kt, Z, Y, X), dtype=np.float32)
    nonzero = np.zeros((Z, Y, X), dtype=bool)
    if T_sel <= 0:
        return folded, nonzero

    data_tzyx = np.ascontiguousarray(data.transpose(3, 2, 1, 0), dtype=np.float32)
    full_w = T_sel // kt
    rem = T_sel % kt
    if full_w > 0:
        folded += data_tzyx[:full_w * kt].reshape(full_w, kt, Z, Y, X).sum(axis=0, dtype=np.float32)
    if rem > 0:
        folded[:rem] += data_tzyx[full_w * kt:]
    nonzero = np.any(data_tzyx != 0, axis=0)
    return folded, nonzero


def _is_fast_streamable_nifti1(header: nib.nifti1.Nifti1Header) -> bool:
    if header.endianness != "<" or header.get_data_dtype() != np.dtype("<f4"):
        return False
    shape = header.get_data_shape()
    if len(shape) != 4:
        return False
    if int(header["vox_offset"]) < 348:
        return False
    slope = float(header["scl_slope"])
    inter = float(header["scl_inter"])
    slope_ok = np.isnan(slope) or slope in (0.0, 1.0)
    inter_ok = np.isnan(inter) or inter == 0.0
    return slope_ok and inter_ok


def _load_nifti_folded(
    path: Union[str, Path],
    mmap: bool = True,
    select_fn: Optional[Callable[[nib.nifti1.Nifti1Header], Tuple[Tuple[float, float, float], float, int, int]]] = None,
    reuse_bufs: Optional[Dict[int, bytearray]] = None,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[nib.nifti1.Nifti1Header],
    Optional[Tuple[float, float, float]],
    Optional[float],
    Optional[int],
    Optional[int],
]:
    path_str = str(path)
    is_gz = path_str.lower().endswith(".gz")
    if (not is_gz) or (_igzip is not None):
        opener = (lambda p: _igzip.open(p, "rb")) if is_gz else (lambda p: open(p, "rb"))
        fallback_to_nib = False
        try:
            with opener(path_str) as f:
                hdr_buf = bytearray(348)
                try:
                    _read_exact_into(f, memoryview(hdr_buf))
                except EOFError:
                    fallback_to_nib = True
                if not fallback_to_nib:
                    if int.from_bytes(hdr_buf[:4], "little") != 348:
                        fallback_to_nib = True
                    else:
                        try:
                            header = nib.Nifti1Header.from_fileobj(io.BytesIO(hdr_buf))
                        except Exception:
                            fallback_to_nib = True
                        if not fallback_to_nib and not _is_fast_streamable_nifti1(header):
                            fallback_to_nib = True

                if not fallback_to_nib:
                    # Run metadata/frame validation outside the I/O handler so config errors propagate.
                    assert select_fn is not None
                    voxel, tr, t_sel, kt = select_fn(header)
                    try:
                        vox_offset = int(header["vox_offset"])
                        if vox_offset > 348:
                            skip = bytearray(vox_offset - 348)
                            _read_exact_into(f, memoryview(skip))
                        shape = header.get_data_shape()
                        X, Y, Z = int(shape[0]), int(shape[1]), int(shape[2])
                        frame_elems = X * Y * Z
                        frame_bytes = frame_elems * 4

                        folded = np.zeros((kt, Z, Y, X), dtype=np.float32)
                        nonzero = np.zeros((Z, Y, X), dtype=bool)

                        buf_size = kt * frame_bytes
                        win_buf = reuse_bufs.get(buf_size) if reuse_bufs is not None else None
                        if win_buf is None:
                            win_buf = bytearray(buf_size)
                            if reuse_bufs is not None:
                                reuse_bufs[buf_size] = win_buf
                        win_mv = memoryview(win_buf)
                        win_arr = np.frombuffer(win_buf, dtype="<f4").reshape(kt, Z, Y, X)

                        full_w = t_sel // kt
                        rem = t_sel % kt
                        for _ in range(full_w):
                            _read_exact_into(f, win_mv)
                            folded += win_arr
                            nonzero |= (win_arr != 0).any(axis=0)
                        if rem > 0:
                            _read_exact_into(f, win_mv[:rem * frame_bytes])
                            rem_arr = np.frombuffer(win_buf, dtype="<f4", count=rem * frame_elems).reshape(rem, Z, Y, X)
                            folded[:rem] += rem_arr
                            nonzero |= (rem_arr != 0).any(axis=0)

                        affine = header.get_best_affine().astype(np.float32)
                        return folded, nonzero, affine, header, voxel, tr, t_sel, kt
                    except (OSError, EOFError, ValueError):
                        return None, None, None, None, None, None, None, None
        except OSError:
            return None, None, None, None, None, None, None, None
        if not fallback_to_nib:
            return None, None, None, None, None, None, None, None

    resolved_meta: Dict[str, object] = {}

    def _max_frames_cb(hdr: nib.nifti1.Nifti1Header) -> Optional[int]:
        assert select_fn is not None
        voxel_v, tr_v, t_sel_v, kt_v = select_fn(hdr)
        resolved_meta["voxel"] = voxel_v
        resolved_meta["tr"] = tr_v
        resolved_meta["T_selected"] = t_sel_v
        resolved_meta["kt"] = kt_v
        shape = hdr.get_data_shape()
        t_total = int(shape[3]) if len(shape) >= 4 else 1
        return t_sel_v if (len(shape) >= 4 and t_sel_v < t_total) else None

    data, affine, header = _load_nifti(path_str, mmap=mmap, max_frames_fn=_max_frames_cb)
    if data is None:
        return None, None, None, None, None, None, None, None

    voxel = resolved_meta["voxel"]  # type: ignore[assignment]
    tr = float(resolved_meta["tr"])  # type: ignore[arg-type]
    t_sel = int(resolved_meta["T_selected"])  # type: ignore[arg-type]
    kt = int(resolved_meta["kt"])  # type: ignore[arg-type]
    if data.ndim >= 4 and t_sel < data.shape[3]:
        data = data[..., :t_sel]
    folded, nonzero = _fold_4d_array(data, kt)
    return folded, nonzero, affine.astype(np.float32), header, voxel, tr, t_sel, kt


def _folded_cache_key(
    path: Path,
    t_prime: Optional[int],
    tau_seconds: float,
    default_tr: Optional[float],
) -> str:
    st = path.stat()
    raw = f"{path}|{st.st_size}|{st.st_mtime_ns}|{t_prime}|{tau_seconds}|{default_tr}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _read_folded_cache(cache_path: Path):
    try:
        file_size = cache_path.stat().st_size
        if file_size < _FOLDED_CACHE_HEADER.size:
            return None
        with open(cache_path, "rb") as f:
            hdr_buf = bytearray(_FOLDED_CACHE_HEADER.size)
            _read_exact_into(f, memoryview(hdr_buf))
            unpacked = _FOLDED_CACHE_HEADER.unpack(hdr_buf)
            magic = unpacked[0]
            if magic != _FOLDED_CACHE_MAGIC:
                return None
            kt, Z, Y, X, t_sel = unpacked[1:6]
            tr, vx, vy, vz = unpacked[6:10]
            affine_vals = unpacked[10:26]
            folded_bytes = kt * Z * Y * X * 4
            nz_bytes = Z * Y * X
            if file_size != _FOLDED_CACHE_HEADER.size + folded_bytes + nz_bytes:
                return None
            folded_buf = bytearray(folded_bytes)
            nz_buf = bytearray(nz_bytes)
            _read_exact_into(f, memoryview(folded_buf))
            _read_exact_into(f, memoryview(nz_buf))
        folded = np.frombuffer(folded_buf, dtype=np.float32).reshape(kt, Z, Y, X)
        nonzero = np.frombuffer(nz_buf, dtype=bool).reshape(Z, Y, X)
        affine = np.asarray(affine_vals, dtype=np.float32).reshape(4, 4)
        return folded, nonzero, affine, (float(vx), float(vy), float(vz)), float(tr), int(t_sel), int(kt)
    except (OSError, EOFError, struct.error, ValueError):
        return None


def _write_folded_cache(
    cache_path: Path,
    folded: np.ndarray,
    nonzero: np.ndarray,
    affine: np.ndarray,
    voxel: Tuple[float, float, float],
    tr: float,
    t_sel: int,
    kt: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    Z, Y, X = (int(d) for d in nonzero.shape)
    affine_f32 = np.ascontiguousarray(affine, dtype=np.float32).reshape(-1)
    hdr_bytes = _FOLDED_CACHE_HEADER.pack(
        _FOLDED_CACHE_MAGIC,
        int(kt),
        Z,
        Y,
        X,
        int(t_sel),
        float(tr),
        float(voxel[0]),
        float(voxel[1]),
        float(voxel[2]),
        *[float(v) for v in affine_f32],
    )
    folded_c = np.ascontiguousarray(folded, dtype=np.float32)
    nz_c = np.ascontiguousarray(nonzero, dtype=np.uint8)
    tmp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp_path, "wb") as f:
            f.write(hdr_bytes)
            f.write(memoryview(folded_c))
            f.write(memoryview(nz_c))
        os.replace(tmp_path, cache_path)
    except OSError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


class NiftiTxtDataset(Dataset):
    """Dataset that loads NIfTI volumes listed in one or more .txt files.

    Each item returns a dict with:
      - 'data': np.ndarray (from get_fdata())
      - 'affine': np.ndarray (4x4)
      - 'header': nibabel header
      - 'voxel': (vx, vy, vz) in millimeters
      - 'tr': float, seconds (0.0 if not present)
      - 'path': pathlib.Path to the NIfTI file
      - 'subject_idx': integer index inside this dataset
      - 'T_selected': int, number of time frames selected based on T_prime and tau_seconds

    Parameters
    ----------
    txt_files: str | Path | Sequence[str|Path]
        One or more text files containing absolute (or relative) paths to NIfTI files.
    transform: Optional[callable]
        Optional transform applied to the sample dict (after loading).
    return_torch: bool
        If True, converts 'data' and 'affine' to torch tensors.
    memory_map: bool
        If True, enables nibabel's memory mapping. Disable to force full load into RAM.
    cache_meta: bool
        If True, caches voxel/TR in memory to avoid recomputing for repeated access.
    T_prime: Optional[int]
        Target number of time patches after TAPE (Time-to-space patch embedding).
        If provided, dataset will automatically select appropriate time frames to ensure
        all samples have the same T_prime after TAPE processing.
        Formula: T_selected = T_prime * tau_seconds / TR
    tau_seconds: float
        Time window in seconds for TAPE kernel (default: 6.0).
        Used to calculate T_selected when T_prime is specified.
    """

    def __init__(
        self,
        txt_files: Union[str, Path, Sequence[Union[str, Path]]],
        transform: Optional[callable] = None,
        return_torch: bool = False,
        memory_map: bool = True,
        cache_meta: bool = True,
        T_prime: Optional[int] = None,
        tau_seconds: float = 6.0,
        default_tr: Optional[float] = None,
        temporal_fold: bool = False,
        folded_cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        self.paths: List[Path] = _read_list_files(txt_files)
        if len(self.paths) == 0:
            raise ValueError("No NIfTI paths found in the provided list files.")
        self.transform = transform
        self.return_torch = bool(return_torch)
        self.memory_map = bool(memory_map)
        self.cache_meta = bool(cache_meta)
        self.T_prime = T_prime
        self.tau_seconds = float(tau_seconds)
        self.default_tr = float(default_tr) if default_tr is not None else None
        if self.default_tr is not None and self.default_tr <= 0:
            raise ValueError("default_tr must be positive when provided")
        self.temporal_fold = bool(temporal_fold)
        self.folded_cache_dir = Path(folded_cache_dir) if folded_cache_dir else None
        if self.folded_cache_dir is not None:
            self.folded_cache_dir.mkdir(parents=True, exist_ok=True)
        self._meta_cache: Dict[int, Tuple[Tuple[float, float, float], float]] = {}
        self._reuse_bufs: Dict[int, bytearray] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def _get_meta(self, idx: int, header: Optional[nib.nifti1.Nifti1Header] = None) -> Tuple[Tuple[float, float, float], float]:
        if self.cache_meta and idx in self._meta_cache:
            return self._meta_cache[idx]
        if header is None:
            _, _, header = _load_nifti(self.paths[idx], mmap=self.memory_map)
        voxel, tr = _space_time_units_to_mm_s(header)
        voxel = tuple(float(v) for v in voxel)
        if any((not np.isfinite(v)) or v <= 0 for v in voxel):
            raise ValueError(f"Invalid voxel spacing for {self.paths[idx]}: {voxel}")
        tr = float(tr)
        if (not np.isfinite(tr)) or tr <= 0:
            if self.default_tr is None:
                raise ValueError(
                    f"Invalid or missing TR for {self.paths[idx]}: {tr}. "
                    "Set data.default_tr or pass --default-tr to use an explicit fallback."
                )
            tr = self.default_tr
        if self.cache_meta:
            self._meta_cache[idx] = (voxel, tr)
        return voxel, tr

    def _calculate_T_selected(self, tr: float, T_total: int) -> int:
        """
        Calculate the number of time frames to select based on T_prime and tau_seconds.

        Formula:
            kt = round(tau_seconds / tr)  # kernel size in time dimension
            T_selected = T_prime * kt

        This ensures that after TAPE (Time-to-space patch embedding), all samples
        will have the same number of time patches (T_prime).

        Args:
            tr: Temporal resolution (TR) in seconds
            T_total: Total number of time frames available in the data

        Returns:
            T_selected: Number of time frames to use (min with T_total)
        """
        if self.T_prime is None:
            return T_total
        if tr <= 0:
            raise ValueError("TR must be positive when T_prime is set")

        # Calculate kernel size in time dimension
        kt = max(1, round(self.tau_seconds / tr))

        # Calculate required time frames to get T_prime patches
        T_selected = self.T_prime * kt

        # Ensure we don't exceed available data
        T_selected = min(T_selected, T_total)

        return T_selected

    def __getitem__(self, idx: int) -> Dict:
        # Try to load file, skip to next valid file if current is invalid
        attempt = 0
        max_attempts = len(self.paths)

        while attempt < max_attempts:
            current_idx = (idx + attempt) % len(self.paths)
            p = self.paths[current_idx]

            if self.temporal_fold:
                cache_path: Optional[Path] = None
                if self.folded_cache_dir is not None:
                    try:
                        key = _folded_cache_key(p, self.T_prime, self.tau_seconds, self.default_tr)
                        cache_path = self.folded_cache_dir / f"{key}.fbc"
                        cached = _read_folded_cache(cache_path)
                        if cached is not None:
                            folded, nonzero, affine, voxel, tr, t_sel, kt = cached
                            sample = {
                                "data": torch.from_numpy(folded) if self.return_torch else folded,
                                "nonzero_mask": torch.from_numpy(nonzero) if self.return_torch else nonzero,
                                "folded_kt": int(kt),
                                "affine": torch.from_numpy(affine) if self.return_torch else affine,
                                "header": None,
                                "voxel": voxel,
                                "tr": tr,
                                "path": p,
                                "subject_idx": current_idx,
                                "T_selected": int(t_sel),
                                "T_prime": self.T_prime,
                                "tau_seconds": self.tau_seconds,
                            }
                            if self.transform is not None:
                                sample = self.transform(sample)
                            return sample
                    except OSError:
                        cache_path = None

                def _select_folded_params(hdr: nib.nifti1.Nifti1Header) -> Tuple[Tuple[float, float, float], float, int, int]:
                    voxel_v, tr_v = self._get_meta(current_idx, hdr)
                    shape = hdr.get_data_shape()
                    t_total = int(shape[3]) if len(shape) >= 4 else 1
                    t_sel_v = self._calculate_T_selected(tr_v, t_total)
                    kt_v = max(1, round(self.tau_seconds / tr_v))
                    return voxel_v, tr_v, t_sel_v, kt_v

                folded, nonzero, affine, header, voxel, tr, t_sel, kt = _load_nifti_folded(
                    p,
                    mmap=self.memory_map,
                    select_fn=_select_folded_params,
                    reuse_bufs=self._reuse_bufs,
                )
                if folded is not None:
                    if cache_path is not None:
                        _write_folded_cache(cache_path, folded, nonzero, affine, voxel, tr, t_sel, kt)
                    sample = {
                        "data": torch.from_numpy(folded) if self.return_torch else folded,
                        "nonzero_mask": torch.from_numpy(nonzero) if self.return_torch else nonzero,
                        "folded_kt": int(kt),
                        "affine": torch.from_numpy(affine) if self.return_torch else affine,
                        "header": header,
                        "voxel": voxel,
                        "tr": tr,
                        "path": p,
                        "subject_idx": current_idx,
                        "T_selected": int(t_sel),
                        "T_prime": self.T_prime,
                        "tau_seconds": self.tau_seconds,
                    }
                    if self.transform is not None:
                        sample = self.transform(sample)
                    return sample

                attempt += 1
                continue

            resolved_meta: Dict[str, object] = {}

            def _select_frames(hdr: nib.nifti1.Nifti1Header) -> Optional[int]:
                voxel_v, tr_v = self._get_meta(current_idx, hdr)
                shape = hdr.get_data_shape()
                t_total = int(shape[3]) if len(shape) >= 4 else 1
                t_sel = self._calculate_T_selected(tr_v, t_total)
                resolved_meta["voxel"] = voxel_v
                resolved_meta["tr"] = tr_v
                resolved_meta["T_selected"] = t_sel
                return t_sel if (len(shape) >= 4 and t_sel < t_total) else None

            data, affine, header = _load_nifti(p, mmap=self.memory_map, max_frames_fn=_select_frames)

            # If file is valid, process it
            if data is not None:
                if "voxel" in resolved_meta:
                    voxel = resolved_meta["voxel"]
                    tr = resolved_meta["tr"]
                    T_selected = resolved_meta["T_selected"]
                else:
                    voxel, tr = self._get_meta(current_idx, header)
                    T_total = data.shape[3] if len(data.shape) >= 4 else 1
                    T_selected = self._calculate_T_selected(tr, T_total)

                # Copy slice in F-order if data still holds trailing frames so the full volume can be freed immediately.
                if len(data.shape) >= 4 and T_selected < data.shape[3]:
                    data = np.array(data[..., :T_selected], order="F", copy=True)

                sample = {
                    "data": torch.from_numpy(data) if self.return_torch else data,
                    "affine": torch.from_numpy(affine) if self.return_torch else affine,
                    "header": header,
                    "voxel": voxel,
                    "tr": tr,
                    "path": p,
                    "subject_idx": current_idx,
                    "T_selected": T_selected,
                    "T_prime": self.T_prime,
                    "tau_seconds": self.tau_seconds,
                }
                if self.transform is not None:
                    sample = self.transform(sample)
                return sample

            # Try next file if current one is invalid
            attempt += 1

        # If all files are invalid, raise error
        raise RuntimeError(f"Could not find any valid file starting from index {idx}")

    def meta_dict(self) -> Dict[int, Dict[str, Union[Tuple[float, float, float], float]]]:
        """Return {subject_idx: {"voxel": (vx,vy,vz), "tr": tr}} for the whole dataset."""
        meta: Dict[int, Dict[str, Union[Tuple[float, float, float], float]]] = {}
        for i, p in enumerate(self.paths):
            if self.cache_meta and i in self._meta_cache:
                voxel, tr = self._meta_cache[i]
            else:
                # read header cheaply without loading full data
                img = nib.load(str(p), mmap=True)
                voxel, tr = _space_time_units_to_mm_s(img.header)
                if self.cache_meta:
                    self._meta_cache[i] = (voxel, tr)
            meta[i] = {"voxel": voxel, "tr": tr}
        return meta




def build_train_val_from_lists(
    train_txts: Union[str, Path, Sequence[Union[str, Path]]],
    val_txts: Union[str, Path, Sequence[Union[str, Path]]],
    *,
    return_torch: bool = False,
    memory_map: bool = True,
    T_prime: Optional[int] = None,
    tau_seconds: float = 6.0,
) -> Tuple[NiftiTxtDataset, NiftiTxtDataset, Dict[str, Dict[int, Dict[str, Union[Tuple[float, float, float], float]]]]]:
    """Convenience helper to create train/val datasets and collect their meta dicts.

    Parameters
    ----------
    train_txts, val_txts: str | Path | Sequence[str|Path]
        Text files containing paths to NIfTI files
    return_torch: bool
        If True, converts data and affine to torch tensors
    memory_map: bool
        If True, enables nibabel's memory mapping
    T_prime: Optional[int]
        Target number of time patches after TAPE. If provided, dataset will automatically
        select appropriate time frames to ensure all samples have the same T_prime.
    tau_seconds: float
        Time window in seconds for TAPE kernel (default: 6.0)

    Returns
    -------
    train_set, val_set, meta_all
      where meta_all = {"train": {...}, "val": {...}}
    """
    train_set = NiftiTxtDataset(
        train_txts,
        return_torch=return_torch,
        memory_map=memory_map,
        T_prime=T_prime,
        tau_seconds=tau_seconds,
    )
    val_set = NiftiTxtDataset(
        val_txts,
        return_torch=return_torch,
        memory_map=memory_map,
        T_prime=T_prime,
        tau_seconds=tau_seconds,
    )
    meta_all = {"train": train_set.meta_dict(), "val": val_set.meta_dict()}
    return train_set, val_set, meta_all


