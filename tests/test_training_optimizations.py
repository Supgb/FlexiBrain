"""Equivalence and regression tests for FlexiBrain training optimizations."""
import copy
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from flexibrain.data.collate import custom_collate_fn, prepare_batch_data
from flexibrain.data.nifti import _fold_4d_array
from flexibrain.models.layers.stape import STAPE4D_TimeToSpace
from flexibrain.models.mamba_blocks import Mamba
from flexibrain.models.mamba_jepa import VolumeMambaJEPA
from flexibrain.models.transformer_block import Attention


def _stape_reference(model: STAPE4D_TimeToSpace, x: torch.Tensor, meta: dict, orig_Ts, affines):
    """Explicit un-folded reference matching the original pre-optimization STAPE math."""
    B = x.size(0)
    per_tok, per_pos, lengths = [], [], []
    for i in range(B):
        T_true = int(orig_Ts[i])
        kt, kx, ky, kz = model._k_from_meta(float(meta[i]["tr"]), tuple(meta[i]["voxel"]))
        Lx, Ly, Lz = 96 // kx, 96 // ky, 96 // kz
        if T_true <= 0:
            per_tok.append(x.new_zeros((0, model.Do)))
            per_pos.append(x.new_zeros((0, model.Do)))
            lengths.append(0)
            continue
        xi = x[i : i + 1, :, :, :, :T_true]
        rem = (-T_true) % kt
        if rem > 0:
            xi = F.pad(xi, (0, rem), mode="constant", value=0.0)
        wt = model._get_wt_first(kt, x.device, x.dtype)
        ws = model._get_wxyz_after(kx, ky, kz, x.device, x.dtype)
        temporal = F.conv1d(xi.reshape(-1, 1, xi.shape[-1]), wt, model.b_t_first.to(x.dtype), stride=kt)
        spatial = None
        for t in range(temporal.shape[-1]):
            vol = temporal[..., t].reshape(1, 96, 96, 96, model.Dm).permute(0, 4, 1, 2, 3)
            val = F.conv3d(vol, ws, bias=None, stride=(kx, ky, kz))
            spatial = val if spatial is None else spatial + val
        spatial = spatial + model.b_xyz_after.to(x.dtype)[None, :, None, None, None]
        toks = spatial.permute(0, 2, 3, 4, 1).contiguous().view(1, Lx * Ly * Lz, model.Do)
        toks, pos = model._add_positional_encoding(toks, [i], [affines[i]], kx, ky, kz)
        km = model._spatial_keep_mask_alltime(x[i], kx, ky, kz, T_true)
        per_tok.append(toks[0][km])
        per_pos.append(pos[0][km])
        lengths.append(int(km.sum().item()))

    L_max = max(lengths) if lengths else 0
    out = x.new_zeros((B, L_max, model.Do))
    pos_out = x.new_zeros((B, L_max, model.Do))
    attn_mask = torch.ones((B, L_max), dtype=torch.bool, device=x.device)
    for b in range(B):
        n = lengths[b]
        if n > 0:
            out[b, :n] = per_tok[b]
            pos_out[b, :n] = per_pos[b]
            attn_mask[b, :n] = False
    return out, attn_mask, lengths, pos_out


class TrainingOptimizationTests(unittest.TestCase):
    def test_fused_stape_raw_and_prefolded_match_reference(self):
        torch.manual_seed(7)
        model = STAPE4D_TimeToSpace(
            d_mid=4, d_out=8, kt_base=6, kx_base=6, ky_base=6, kz_base=6
        )
        with torch.no_grad():
            model.b_t_first.normal_(0.0, 0.4)
            model.b_xyz_after.normal_(0.0, 0.3)

        # Mixed (kt, kx, ky, kz) groups, partial window, sparse foreground, and T_true==0
        x = torch.zeros(3, 96, 96, 96, 10)
        x[0, 10:30, 10:30, 10:30, :7] = torch.randn(20, 20, 20, 7)
        x[1, 20:50, 20:50, 20:50, :10] = torch.randn(30, 30, 30, 10)
        # sample 2 stays all-zero with orig_T=0
        orig_Ts = [7, 10, 0]
        meta = {
            0: {"tr": 2.0, "voxel": (3.0, 3.0, 3.0)},  # kt=3, kxyz=(4,4,4)
            1: {"tr": 3.0, "voxel": (4.0, 4.0, 4.0)},  # kt=2, kxyz=(3,3,3)
            2: {"tr": 2.0, "voxel": (3.0, 3.0, 3.0)},  # kt=3, kxyz=(4,4,4)
        }
        affines = [torch.eye(4) for _ in range(3)]
        params = [
            model.w_t_first_base, model.b_t_first,
            model.w_xyz_after_base, model.b_xyz_after,
            model.pos_embed.proj.weight,
        ]

        out_ref, mask_ref, lens_ref, pos_ref = _stape_reference(model, x, meta, orig_Ts, affines)
        grads_ref = torch.autograd.grad(out_ref.square().mean(), params)

        # 1. Raw 4D forward through optimized STAPE
        out_raw, mask_raw, lens_raw, pos_raw = model(x, meta, orig_Ts, affines)
        grads_raw = torch.autograd.grad(out_raw.square().mean(), params)
        self.assertEqual(lens_raw, lens_ref)
        self.assertTrue(torch.equal(mask_raw, mask_ref))
        torch.testing.assert_close(out_raw, out_ref, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(pos_raw, pos_ref, atol=1e-5, rtol=1e-5)
        for g_got, g_exp in zip(grads_raw, grads_ref):
            torch.testing.assert_close(g_got, g_exp, atol=2e-4, rtol=2e-4)

        # 2. Pre-folded kzyx path via collate + prepare_batch_data
        batch_items = []
        for i in range(3):
            kt = model._k_from_meta(meta[i]["tr"], meta[i]["voxel"])[0]
            arr_i = x[i, ..., :orig_Ts[i]].numpy()
            folded, nz = _fold_4d_array(arr_i, kt=kt)
            batch_items.append({
                "data": torch.from_numpy(folded),
                "nonzero_mask": torch.from_numpy(nz),
                "folded_kt": kt,
                "affine": affines[i],
                "voxel": meta[i]["voxel"],
                "tr": meta[i]["tr"],
                "subject_idx": i,
                "T_selected": orig_Ts[i],
                "T_prime": 30,
                "tau_seconds": 6.0,
            })
        collated = custom_collate_fn(batch_items)
        x_f, meta_f, ts_f, aff_f = prepare_batch_data(collated, torch.device("cpu"))
        out_f, mask_f, lens_f, pos_f = model(x_f, meta_f, ts_f, aff_f)
        grads_f = torch.autograd.grad(out_f.square().mean(), params)
        self.assertEqual(lens_f, lens_ref)
        self.assertTrue(torch.equal(mask_f, mask_ref))
        torch.testing.assert_close(out_f, out_ref, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(pos_f, pos_ref, atol=1e-5, rtol=1e-5)
        for g_got, g_exp in zip(grads_f, grads_ref):
            torch.testing.assert_close(g_got, g_exp, atol=2e-4, rtol=2e-4)

    def test_attention_sdpa_matches_explicit_attention(self):
        torch.manual_seed(13)
        attn = Attention(dim=64, num_heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0)
        attn.eval()
        x = torch.randn(2, 20, 64, requires_grad=True)
        valid_mask = torch.ones(2, 20, dtype=torch.bool)
        valid_mask[0, 13:] = False
        valid_mask[1, 17:] = False

        out_ref, weights_ref = attn(x, attention_mask=valid_mask, output_attentions=True)
        self.assertIsNotNone(weights_ref)
        grads_ref = torch.autograd.grad(out_ref.square().mean(), (x, *attn.parameters()))

        out_sdpa, weights_sdpa = attn(x, attention_mask=valid_mask, output_attentions=False)
        self.assertIsNone(weights_sdpa)
        grads_sdpa = torch.autograd.grad(out_sdpa.square().mean(), (x, *attn.parameters()))

        torch.testing.assert_close(out_sdpa, out_ref, atol=1e-5, rtol=1e-5)
        for g_got, g_exp in zip(grads_sdpa, grads_ref):
            torch.testing.assert_close(g_got, g_exp, atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
    def test_mamba_validated_mask_matches_checked_path(self):
        torch.manual_seed(29)
        m = Mamba(32, bimamba_type="v2", if_devide_out=True, device="cuda")
        x = torch.randn(3, 16, 32, device="cuda", requires_grad=True)
        attn_mask = torch.zeros(3, 16, dtype=torch.bool, device="cuda")
        attn_mask[0, 10:] = True
        attn_mask[1, 14:] = True
        # Note: L_eff == 16 here (sample 2 has full length 16)
        out_checked = m(x, attn_mask=attn_mask, validated_mask=False)
        grads_checked = torch.autograd.grad(out_checked.square().mean(), (x, *m.parameters()))

        out_fast = m(x, attn_mask=attn_mask, validated_mask=True)
        grads_fast = torch.autograd.grad(out_fast.square().mean(), (x, *m.parameters()))

        torch.testing.assert_close(out_fast, out_checked, atol=1e-6, rtol=1e-5)
        for g_got, g_exp in zip(grads_fast, grads_checked):
            torch.testing.assert_close(g_got, g_exp, atol=1e-5, rtol=1e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
    def test_vectorized_jepa_and_foreach_ema_on_cuda(self):
        torch.manual_seed(37)
        model = VolumeMambaJEPA(
            embed_dim=64, depth=2, predictor_depth=1,
            bimamba_type="v2", if_devide_out=True, device="cuda",
        )
        model.train()
        x = torch.zeros(2, 96, 96, 96, 6, device="cuda")
        x[0, 16:48, 16:48, 16:48, :4] = torch.randn(32, 32, 32, 4, device="cuda")
        x[1, 24:64, 24:64, 24:64, :6] = torch.randn(40, 40, 40, 6, device="cuda")
        meta = {
            0: {"tr": 2.0, "voxel": (3.0, 3.0, 3.0)},
            1: {"tr": 2.0, "voxel": (3.0, 3.0, 3.0)},
        }
        orig_Ts = np.array([4, 6])
        affines = torch.eye(4, device="cuda").unsqueeze(0).repeat(2, 1, 1)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, pred_m, tgt_f, mask_full = model(x, mask_ratio=0.65, meta=meta, orig_Ts=orig_Ts, affines=affines)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(pred_m.shape, tgt_f.shape)
        self.assertEqual(mask_full.shape[0], 2)
        loss.backward()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

        # Verify _foreach EMA update matches reference loop
        ref_targets = [p.detach().clone() for p in list(model.target_blocks.parameters()) + list(model.target_norm.parameters())]
        src_params = [p.detach().clone() for p in list(model.blocks.parameters()) + list(model.norm_f.parameters())]
        m = 0.996
        for pt, ps in zip(ref_targets, src_params):
            pt.mul_(m).add_(ps, alpha=1.0 - m)
        model.update_target_encoder(m=m)
        actual_targets = list(model.target_blocks.parameters()) + list(model.target_norm.parameters())
        for pt_got, pt_exp in zip(actual_targets, ref_targets):
            torch.testing.assert_close(pt_got, pt_exp, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
