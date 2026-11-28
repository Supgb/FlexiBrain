# stape_time_to_space.py
import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flexibrain.models.layers.pos_embed import FixedSinCos3DPE, stape_patch_world_coords_physical
from flexibrain.utils.weight_resize import pi_resize_weight_1d, pi_resize_weight_3d

class STAPE4D_TimeToSpace(nn.Module):
    """
    input:
        x: (B, 96, 96, 96, T_max) 
        meta: {subject_idx: {"voxel": (vx,vy,vz) mm, "tr": float s}}
        orig_T
        affine
    output:
        tokens:   (B, L_max, D_out)
        attn_mask:(B, L_max)  True=padding
        lengths:  List[int] 
    """
    def __init__(self,
                 d_mid: int = 128,     
                 d_out: int = 256,
                 kt_base: int = 6,
                 kx_base: int = 6,
                 ky_base: int = 6,
                 kz_base: int = 6,
                 tau_seconds: float = 6.0,
                 rho_mm: Tuple[float, float, float] = (12., 12., 12.),
                 ):
        super().__init__()
        self.Dm = d_mid
        self.Do = d_out
        self.kt0, self.kx0, self.ky0, self.kz0 = kt_base, kx_base, ky_base, kz_base
        self.tau = float(tau_seconds)
        self.rho = tuple(float(r) for r in rho_mm)

        # time based kernal [Dm, 1, kt0] 
        self.w_t_first_base = nn.Parameter(
            torch.randn(d_mid, 1, kt_base) * (1.0 / (1 * kt_base)) ** 0.5
        )
        self.b_t_first = nn.Parameter(torch.zeros(d_mid))

        # space base kernal: [Do, Dm, kx0, ky0, kz0] 
        self.w_xyz_after_base = nn.Parameter(
            torch.randn(d_out, d_mid, kx_base, ky_base, kz_base) *
            (1.0 / (d_mid * kx_base * ky_base * kz_base)) ** 0.5
        )
        self.b_xyz_after = nn.Parameter(torch.zeros(d_out))

        self._cache_t = {}     # key=(kt,dtype) -> w_t_first
        self._cache_xyz = {}   # key=(kx,ky,kz,dtype) -> w_xyz_after

        # physical pe
        self.pos_embed = FixedSinCos3DPE(
            embed_dim=d_out,           
            num_freq=12,               
            space_scale=0.01,          
            learnable_proj=True        
        )

    @torch.no_grad()
    def _k_from_meta(self, tr: float, voxel: Tuple[float, float, float]) -> Tuple[int, int, int, int]:
        vx, vy, vz = voxel
        if tr <= 0 or not math.isfinite(float(tr)):
            raise ValueError(f"TR must be positive for STAPE kernel sizing, got {tr!r}")
        if any(v <= 0 or not math.isfinite(float(v)) for v in (vx, vy, vz)):
            raise ValueError(f"Voxel spacing must be positive for STAPE kernel sizing, got {voxel!r}")
        kt = max(1, round(self.tau / tr))
        kx = max(1, round(self.rho[0] / vx))
        ky = max(1, round(self.rho[1] / vy))
        kz = max(1, round(self.rho[2] / vz))
        return int(kt), int(kx), int(ky), int(kz)

    def _get_wt_first(self, kt:int, device, dtype):
        wt = pi_resize_weight_1d(self.w_t_first_base.to(dtype), kt)  # [Dm,1,kt]
        return wt.to(device)

    def _get_wxyz_after(self, kx:int, ky:int, kz:int, device, dtype):
        w = pi_resize_weight_3d(self.w_xyz_after_base.to(dtype), kx, ky, kz)  # [Do,Dm,kx,ky,kz]
        return w.to(device)

    @staticmethod
    def _detect_true_T(x_b: torch.Tensor) -> int:

        with torch.no_grad():
            s = x_b.abs().sum(dim=(0,1,2))
            nz = torch.nonzero(s > 0, as_tuple=False)
            if nz.numel() == 0:
                return 0
            return int(nz.max().item() + 1)

    @staticmethod
    def _spatial_keep_mask_alltime(x_b: torch.Tensor, kx:int, ky:int, kz:int, T_true:int) -> torch.Tensor:
        X=Y=Z=96
        Lx, Ly, Lz = X//kx, Y//ky, Z//kz
        if T_true == 0:
            return torch.zeros(Lx*Ly*Lz, dtype=torch.bool, device=x_b.device)
        vol = (x_b[:,:,:,:T_true] != 0).any(dim=-1).float()  # [96,96,96] -> 1/0
        vol = vol[:Lx*kx, :Ly*ky, :Lz*kz].unsqueeze(0).unsqueeze(0)  # [1,1,X,Y,Z]
        keep = F.max_pool3d(vol, kernel_size=(kx,ky,kz), stride=(kx,ky,kz)) > 0  # [1,1,Lx,Ly,Lz]
        return keep.squeeze(0).squeeze(0).reshape(-1)

    def _compute_spatial_coords_for_group(self,
                                        group_idxs: List[int],
                                        affines: List[torch.Tensor],
                                        kx: int, ky: int, kz: int,
                                        device: torch.device) -> torch.Tensor:
        """
        compute patch physical coordinate
        """
        G = len(group_idxs)
        X, Y, Z = 96, 96, 96  
        Lx, Ly, Lz = X//kx, Y//ky, Z//kz

        coords_list = []
        for g, affine in enumerate(affines):
            coords = stape_patch_world_coords_physical(
                X=X, Y=Y, Z=Z,
                kx=kx, ky=ky, kz=kz,
                affine=affine,
                rho_mm=self.rho  
            )  # [Lx*Ly*Lz, 3]
            coords_list.append(coords)

        # [G, Lx*Ly*Lz, 3]
        coords_batch = torch.stack(coords_list, dim=0)
        return coords_batch.to(device)

    def _add_positional_encoding(self,
                               tokens_all: torch.Tensor,    # [G, Lx*Ly*Lz, Do]
                               group_idxs: List[int],
                               affines: List[torch.Tensor],
                               kx: int, ky: int, kz: int) -> torch.Tensor:
        device = tokens_all.device

        coords = self._compute_spatial_coords_for_group(
            group_idxs, affines, kx, ky, kz, device
        )  # [G, Lx*Ly*Lz, 3]

        pos_encoding = self.pos_embed(coords)  # [G, Lx*Ly*Lz, Do]  
        pos_encoding = pos_encoding.to(tokens_all.dtype)
        tokens_with_pos = tokens_all + pos_encoding

        return tokens_with_pos, pos_encoding

    def _run_group_from_folded(self,
                               fg: torch.Tensor,            # [G, kt, 96, 96, 96] (X, Y, Z)
                               nz_g: torch.Tensor,          # [G, 96, 96, 96] bool
                               orig_Ts: List[int],
                               kt: int, kx: int, ky: int, kz: int,
                               group_idxs: List[int],
                               affines: List[torch.Tensor],
                               return_grid_info: bool = False) -> Tuple[List[torch.Tensor], List[int], Dict, List[torch.Tensor]]:
        device, dtype = fg.device, fg.dtype
        G, _, X, Y, Z = fg.shape
        assert X == 96 and Y == 96 and Z == 96
        Lx, Ly, Lz = X // kx, Y // ky, Z // kz

        grid_data = {}
        if max((int(t) for t in orig_Ts), default=0) <= 0:
            if return_grid_info:
                empty_mask = torch.zeros(Lx * Ly * Lz, dtype=torch.bool)
                empty_idx = torch.empty(0, dtype=torch.long)
                for sample_idx in group_idxs:
                    grid_data[sample_idx] = {
                        'Lx': Lx, 'Ly': Ly, 'Lz': Lz,
                        'kx': kx, 'ky': ky, 'kz': kz,
                        'keep_mask': empty_mask.clone(),
                        'grid_to_token_idx': empty_idx.clone(),
                    }
            return (
                [fg.new_zeros((0, self.Do)) for _ in range(G)],
                [0] * G,
                grid_data,
                [fg.new_zeros((0, self.Do)) for _ in range(G)],
            )

        w_t = self._get_wt_first(kt, device, dtype).squeeze(1)             # [Dm, kt]
        w_xyz = self._get_wxyz_after(kx, ky, kz, device, dtype)            # [Do, Dm, kx, ky, kz]
        b_t = self.b_t_first.to(device=device, dtype=dtype)                # [Dm]
        b_xyz = self.b_xyz_after.to(device=device, dtype=dtype)            # [Do]

        # Algebraically fuse temporal conv1d + spatial conv3d over folded windows.
        W_eff = torch.einsum("odxyz,dk->okxyz", w_xyz, w_t)                # [Do, kt, kx, ky, kz]
        bias_t = torch.einsum("od,d->o", w_xyz.sum(dim=(2, 3, 4)), b_t)    # [Do]
        sfeat = F.conv3d(fg.contiguous(), W_eff, bias=b_xyz, stride=(kx, ky, kz))

        valid_windows = torch.tensor(
            [(int(t) + kt - 1) // kt if int(t) > 0 else 0 for t in orig_Ts],
            device=device,
            dtype=sfeat.dtype,
        )
        sfeat = sfeat + valid_windows[:, None, None, None, None] * bias_t.to(sfeat.dtype)[None, :, None, None, None]

        tokens_all = sfeat.permute(0, 2, 3, 4, 1).contiguous().view(G, Lx * Ly * Lz, self.Do)

        if affines is not None and len(affines) == len(group_idxs):
            tokens_all, pos_group = self._add_positional_encoding(tokens_all, group_idxs, affines, kx, ky, kz)
        else:
            pos_group = torch.zeros_like(tokens_all)

        vol = nz_g[:, :Lx * kx, :Ly * ky, :Lz * kz].unsqueeze(1).float()   # [G, 1, X', Y', Z']
        keep_masks = (F.max_pool3d(vol, kernel_size=(kx, ky, kz), stride=(kx, ky, kz)) > 0).view(G, -1)
        for g in range(G):
            if int(orig_Ts[g]) <= 0:
                keep_masks[g].zero_()

        g_lens = keep_masks.sum(dim=1).tolist()
        max_g_len = max(g_lens) if g_lens else 0

        if max_g_len > 0:
            order = (~keep_masks).to(torch.int32).argsort(dim=1, stable=True)[:, :max_g_len]
            idx_exp = order.unsqueeze(-1).expand(-1, -1, self.Do)
            gathered_tok = torch.gather(tokens_all, 1, idx_exp)
            gathered_pos = torch.gather(pos_group, 1, idx_exp)
        else:
            gathered_tok = None
            gathered_pos = None

        tokens_list, lengths, pos_list = [], [], []
        for g in range(G):
            ln = int(g_lens[g])
            if ln > 0:
                toks = gathered_tok[g, :ln]
                pe = gathered_pos[g, :ln]
            else:
                toks = tokens_all.new_zeros((0, self.Do))
                pe = pos_group.new_zeros((0, self.Do))

            tokens_list.append(toks)
            pos_list.append(pe)
            lengths.append(ln)

            if return_grid_info:
                keep_mask = keep_masks[g]
                sample_idx = group_idxs[g]
                grid_data[sample_idx] = {
                    'Lx': Lx,
                    'Ly': Ly,
                    'Lz': Lz,
                    'kx': kx,
                    'ky': ky,
                    'kz': kz,
                    'keep_mask': keep_mask.cpu(),
                    'grid_to_token_idx': torch.nonzero(keep_mask, as_tuple=False).squeeze(-1).cpu(),
                }

        return tokens_list, lengths, grid_data, pos_list

    def _run_group_time_first(self,
                              x_group: torch.Tensor,            # [G,96,96,96,T_max]
                              orig_Ts: List[int],               
                              kt:int, kx:int, ky:int, kz:int,
                              group_idxs: List[int],          
                              affines: List[torch.Tensor],    
                              return_grid_info: bool = False) -> Tuple[List[torch.Tensor], List[int], Dict, List[torch.Tensor]]:
        device, dtype = x_group.device, x_group.dtype
        G, X, Y, Z, T_max = x_group.shape
        assert X == 96 and Y == 96 and Z == 96

        T_true_max = max((int(t) for t in orig_Ts), default=0)
        if T_true_max <= 0:
            fg = x_group.new_zeros((G, kt, X, Y, Z))
            nz_g = torch.zeros((G, X, Y, Z), dtype=torch.bool, device=device)
            return self._run_group_from_folded(
                fg, nz_g, orig_Ts, kt, kx, ky, kz, group_idxs, affines,
                return_grid_info=return_grid_info,
            )

        T_pad = math.ceil(T_true_max / kt) * kt
        T_prime = T_pad // kt

        if T_max < T_pad:
            xg = F.pad(x_group, (0, T_pad - T_max), mode='constant', value=0.0)
        else:
            xg = x_group[..., :T_pad]

        if not all(int(t) == T_pad for t in orig_Ts):
            t_valid = torch.arange(T_pad, device=device)[None, :] < torch.tensor(
                [int(t) for t in orig_Ts], device=device
            )[:, None]
            xg = xg * t_valid[:, None, None, None, :].to(xg.dtype)

        fg = xg.view(G, X, Y, Z, T_prime, kt).sum(dim=4).permute(0, 4, 1, 2, 3)  # [G, kt, X, Y, Z]
        nz_g = (xg != 0).any(dim=-1)  # [G, X, Y, Z]
        return self._run_group_from_folded(
            fg, nz_g, orig_Ts, kt, kx, ky, kz, group_idxs, affines,
            return_grid_info=return_grid_info,
        )
    
    def forward(self,
                x: torch.Tensor,                     
                meta: Dict[int, Dict],               
                orig_Ts: Sequence[int] = None,
                affines: Sequence[torch.Tensor] = None,
                return_grid_info: bool = False):         
        B = x.size(0)
        device, dtype = x.device, x.dtype

        folded_info = meta.get("_stape_folded") if isinstance(meta, dict) else None

        if orig_Ts is None:
            if folded_info is not None:
                raise ValueError("orig_Ts is required when using pre-folded STAPE inputs")
            orig_Ts = [self._detect_true_T(x[b]) for b in range(B)]
        else:
            orig_Ts = [int(t) for t in orig_Ts]

        if affines is None:
            print("WARNING: not provide affine")
            affines = [torch.eye(4, device=device, dtype=dtype) for _ in range(B)]
        else:
            affines = [aff.to(device=device, dtype=dtype) for aff in affines]

        groups = defaultdict(list)
        for i in range(B):
            voxel = tuple(meta[i]["voxel"])
            tr = float(meta[i]["tr"])
            group_key = self._k_from_meta(tr, voxel)
            groups[group_key].append(i)

        if folded_info is not None:
            layout = folded_info.get("layout", "kzyx")
            nz_all = folded_info["nonzero_mask"]
            if layout == "kzyx":
                x_kxyz = x.permute(0, 1, 4, 3, 2)
                nz_xyz = nz_all.permute(0, 3, 2, 1)
            elif layout == "kxyz":
                x_kxyz = x
                nz_xyz = nz_all
            else:
                raise ValueError(f"Unsupported folded STAPE layout: {layout!r}")
            folded_kts = folded_info["folded_kt"]

        per_sample_tokens: List[torch.Tensor] = [None]*B
        per_sample_pos:    List[torch.Tensor] = [None]*B
        lengths: List[int] = [0]*B
        grid_info: Dict[int, Dict] = {}  

        for (kt, kx, ky, kz), idxs in groups.items():
            Ts_group = [orig_Ts[i] for i in idxs]
            affines_group = [affines[i] for i in idxs]

            if folded_info is not None:
                for i in idxs:
                    if int(folded_kts[i]) != kt:
                        raise ValueError(
                            f"Folded sample kt={int(folded_kts[i])} does not match STAPE kt={kt} "
                            f"(tr={meta[i]['tr']}, tau={self.tau})"
                        )
                toks, lens, grid_data, pos_list = self._run_group_from_folded(
                    x_kxyz[idxs, :kt],
                    nz_xyz[idxs],
                    Ts_group,
                    kt, kx, ky, kz,
                    idxs,
                    affines_group,
                    return_grid_info=return_grid_info,
                )
            else:
                x_group = x[idxs, ...]                              # [G,96,96,96,T_max]
                toks, lens, grid_data, pos_list = self._run_group_time_first(
                    x_group, Ts_group, kt, kx, ky, kz, idxs, affines_group,
                    return_grid_info=return_grid_info
                )
            for g_idx, (tok, ln) in enumerate(zip(toks, lens)):
                loc = idxs[g_idx]
                per_sample_tokens[loc] = tok
                lengths[loc] = ln
                per_sample_pos[loc]    = pos_list[g_idx]
                if return_grid_info and loc in grid_data:
                    grid_info[loc] = grid_data[loc]

        L_max = max(lengths) if lengths else 0
        out_dtype = next((t.dtype for t in per_sample_tokens if t is not None and t.numel() > 0), dtype)
        out = torch.zeros((B, L_max, self.Do), device=device, dtype=out_dtype)
        pos_out = torch.zeros((B, L_max, self.Do), device=device, dtype=out_dtype)
        attn_mask = torch.ones((B, L_max), dtype=torch.bool, device=device)

        for b, tok in enumerate(per_sample_tokens):
            n = lengths[b]
            if n > 0:
                out[b, :n] = tok
                pos_out[b, :n] = per_sample_pos[b]
                attn_mask[b, :n] = False

        if return_grid_info:
            return out, attn_mask, lengths, grid_info, pos_out
        else:
            return out, attn_mask, lengths, pos_out
