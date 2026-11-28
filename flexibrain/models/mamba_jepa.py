# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath


from flexibrain.models.mamba_blocks import Mamba, RMSNorm, create_block, _init_weights, layer_norm_fn, rms_norm_fn


from flexibrain.models.layers.stape import STAPE4D_TimeToSpace
from flexibrain.models.layers.moe import MoE


class VolumeMambaJEPA(nn.Module):
    """ JEPA with VisionMamba backbone
    """
    def __init__(self,
                embed_dim=512,
                depth=24,
                predictor_depth=2,
                ssm_cfg=None, 
                encoder_attn_layer_idx=None,
                attn_cfg=None,
                drop_path_rate=0.1,
                norm_epsilon: float = 1e-5, 
                rms_norm: bool = False, 
                initializer_cfg=None,
                fused_add_norm=True,
                residual_in_fp32=True,
                device=None,
                dtype=None,
                bimamba_type="none",
                if_bimamba=False,
                mixer_type="mamba",
                if_devide_out=False,
                momentum: float = 0.996,
                norm_target: bool = True,

                **kwargs
                ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        super().__init__()

        self.embed_dim = embed_dim
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.momentum = float(momentum)
        self.norm_target = bool(norm_target)   

        self.patch_embed = STAPE4D_TimeToSpace(
            d_mid=16,
            d_out=embed_dim,
            kt_base=6,
            kx_base=6,
            ky_base=6,
            kz_base=6,
            tau_seconds=6.0,
            rho_mm=(12.0, 12.0, 12.0),
        )
        if device is not None:
            self.patch_embed = self.patch_embed.to(device=device)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.blocks = nn.ModuleList()
        block_idx = 0
        for i in range(depth):
            self.blocks.append(
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=encoder_attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=block_idx,
                    bimamba_type=bimamba_type,
                    if_bimamba=if_bimamba,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,   
                    mixer_type=mixer_type,
                    **factory_kwargs,
                )
            )
            block_idx += 1

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        self.target_blocks = copy.deepcopy(self.blocks) 
        self.target_norm = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)
        for p in self.target_blocks.parameters():
            p.requires_grad = False
        for p in self.target_norm.parameters():
            p.requires_grad = False


        # ------- MoE & ResolutionxTR embed-------
        self.moe_aux_coef = float(kwargs.get("moe_aux_coef", 0.1))
        self.moe = MoE(
            dim=embed_dim,
            hidden_dim=embed_dim * 4,
            num_indep=3,
            aux_loss_coef=self.moe_aux_coef,
            device=device,
            dtype=dtype,
        )

        self.mask_token_ctx = nn.Parameter(torch.zeros(1, 1, embed_dim, **factory_kwargs))
        torch.nn.init.normal_(self.mask_token_ctx, std=0.02)

        self.pred_depth = predictor_depth
        self.pred_dpr = [0.0 for _ in range(self.pred_depth)]
        self.predictor_blocks = nn.ModuleList()
        for i in range(self.pred_depth):
            self.predictor_blocks.append(
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=None,   
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba_type=bimamba_type,
                    if_bimamba=if_bimamba,
                    drop_path=self.pred_dpr[i],
                    if_devide_out=if_devide_out,
                    mixer_type=mixer_type,
                    **factory_kwargs,
                )
            )

        self.predictor_norm = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs)

        self.apply(self._init_linear_ln)
        self.initialize_jepa_weights()
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def _init_linear_ln(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif (isinstance(m, nn.LayerNorm) or m.__class__.__name__ == 'RMSNorm') and hasattr(m, 'weight'):
            nn.init.constant_(m.weight, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def initialize_jepa_weights(self):
        torch.nn.init.normal_(self.mask_token_ctx, std=0.02)
        if hasattr(self, 'predictor_blocks'):
            self.predictor_blocks.apply(self._init_linear_ln)
        if hasattr(self, 'predictor_norm'):
            self._init_linear_ln(self.predictor_norm)
        self.blocks.apply(self._init_linear_ln)
        self._init_linear_ln(self.norm_f)

        if hasattr(self.patch_embed, 'reset_parameters'):
            self.patch_embed.reset_parameters()
        else:
            for m in self.patch_embed.modules():
                if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        with torch.no_grad():
            for ps, pt in zip(self.blocks.parameters(), self.target_blocks.parameters()):
                pt.copy_(ps)
            if hasattr(self, 'target_norm'):
                for ps, pt in zip(self.norm_f.parameters(), self.target_norm.parameters()):
                    pt.copy_(ps)
    
    @torch.no_grad()
    def update_target_encoder(self, m: float = None):
        """EMA update of the target encoder"""
        m = float(m or self.momentum)
        src_params = [p.data for p in self.blocks.parameters()] + [p.data for p in self.norm_f.parameters()]
        tgt_params = [p.data for p in self.target_blocks.parameters()] + [p.data for p in self.target_norm.parameters()]
        if tgt_params:
            torch._foreach_mul_(tgt_params, m)
            torch._foreach_add_(tgt_params, src_params, alpha=1.0 - m)

    def _random_masking_impl(self, x, attn_mask, lengths, mask_ratio):
        N, Lmax, D = x.shape
        device = x.device
        keep_lengths = [max(1, int(Li * (1 - mask_ratio))) if Li > 0 else 0 for Li in lengths]
        tgt_lengths = [max(0, Li - Lk) for Li, Lk in zip(lengths, keep_lengths)]
        Lk_max = max(1, max(keep_lengths, default=0))
        Lt_max = max(tgt_lengths, default=0)

        if Lmax == 0:
            x_keep = x.new_zeros((N, Lk_max, D))
            mask_full = torch.ones((N, 0), device=device)
            ids_restore = torch.empty((N, 0), dtype=torch.long, device=device)
            attn_keep = torch.ones((N, Lk_max), dtype=torch.bool, device=device)
            ids_keep_pad = torch.full((N, Lk_max), -1, dtype=torch.long, device=device)
            ids_tgt = torch.empty((N, 0), dtype=torch.long, device=device)
            attn_tgt = torch.ones((N, 0), dtype=torch.bool, device=device)
            return x_keep, mask_full, ids_restore, attn_keep, keep_lengths, ids_keep_pad, ids_tgt, attn_tgt, tgt_lengths

        noise = torch.rand(N, Lmax, device=device)
        if attn_mask is not None:
            noise = noise.masked_fill(attn_mask, 2.0)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        keep_t = torch.tensor(keep_lengths, device=device, dtype=torch.long)
        ar_k = torch.arange(Lk_max, device=device)[None, :]
        attn_keep = ar_k >= keep_t[:, None]

        if Lk_max <= Lmax:
            ids_keep_raw = ids_shuffle[:, :Lk_max]
        else:
            ids_keep_raw = F.pad(ids_shuffle, (0, Lk_max - Lmax), value=0)
        ids_keep_pad = ids_keep_raw.masked_fill(attn_keep, -1)

        x_keep = torch.gather(x, 1, ids_keep_raw.clamp(0, Lmax - 1).unsqueeze(-1).expand(-1, -1, D))
        x_keep = x_keep.masked_fill(attn_keep.unsqueeze(-1), 0.0)

        ar_l = torch.arange(Lmax, device=device)[None, :]
        valid_sorted_mask = (ar_l >= keep_t[:, None]).to(torch.float32)
        mask_full = torch.gather(valid_sorted_mask, 1, ids_restore)

        if Lt_max > 0:
            tgt_t = torch.tensor(tgt_lengths, device=device, dtype=torch.long)
            ar_t = torch.arange(Lt_max, device=device)[None, :]
            attn_tgt = ar_t >= tgt_t[:, None]
            tgt_pos = (keep_t[:, None] + ar_t).clamp(max=Lmax - 1)
            ids_tgt = torch.gather(ids_shuffle, 1, tgt_pos)
        else:
            attn_tgt = torch.ones((N, 0), dtype=torch.bool, device=device)
            ids_tgt = torch.empty((N, 0), dtype=torch.long, device=device)

        return x_keep, mask_full, ids_restore, attn_keep, keep_lengths, ids_keep_pad, ids_tgt, attn_tgt, tgt_lengths

    def random_masking(self, x, attn_mask, lengths, mask_ratio):
        """
        x: [B, Lmax, D], attn_mask: [B, Lmax] (True=pad), lengths: list[int]
        Return:
        x_keep: [B, Lk_max, D]
        mask_full: [B, Lmax] (0=keep, 1=remove; pad 仍为1)
        ids_restore: [B, Lmax]
        attn_keep: [B, Lk_max] (True=pad)
        keep_lengths: list[int]
        ids_keep_pad: [B, Lk_max]  
        """
        x_keep, mask_full, ids_restore, attn_keep, keep_lengths, ids_keep_pad, _, _, _ = self._random_masking_impl(
            x, attn_mask, lengths, mask_ratio
        )
        return x_keep, mask_full, ids_restore, attn_keep, keep_lengths, ids_keep_pad

    def _build_context_visible(self, x_keep, attn_keep):

        return x_keep, attn_keep
    
    def _build_target_masked(self, x_full, mask_full, lengths):
        """
          x_tgt_pad:   [B, Lt_max, D]
          attn_tgt:    [B, Lt_max] (True=padding)
          tgt_lengths: list[int]
        """
        B, Lmax, D = x_full.shape
        device, dtype = x_full.device, x_full.dtype
        per_sample, tgt_lengths, Lt_max = [], [], 0
        for i in range(B):
            Li = lengths[i]
            if Li == 0:
                per_sample.append(x_full.new_empty((0, D)))
                tgt_lengths.append(0)
                continue
            sel = (mask_full[i, :Li] == 1) if mask_full.dtype != torch.bool else mask_full[i, :Li]
            xi = x_full[i, :Li][sel]
            per_sample.append(xi)
            tgt_lengths.append(xi.size(0))
            Lt_max = max(Lt_max, xi.size(0))

        x_tgt_pad = x_full.new_zeros((B, Lt_max, D))
        attn_tgt = torch.ones(B, Lt_max, dtype=torch.bool, device=device)
        for i, xi in enumerate(per_sample):
            if xi.numel() > 0:
                Lti = xi.size(0)
                x_tgt_pad[i, :Lti] = xi
                attn_tgt[i, :Lti] = False

        return x_tgt_pad, attn_tgt, tgt_lengths

    def _run_blocks(self, x, attn_mask, blocks, norm_layer, inference_params=None, unpack_buffer=None, mask_validated: bool = False):
        residual = None
        Lmax = x.size(1)
        if Lmax == 0:
            return x.new_zeros(x.shape)

        eff_mask = attn_mask
        pad_right = 0
        hidden_states = x
        if attn_mask is not None and not mask_validated:
            assert Mamba._is_right_padded(attn_mask), "attn_mask 必须为右侧 padding（中间不得有 True）"
            _, L_eff = Mamba._right_pad_lengths(attn_mask)
            if L_eff == 0:
                return x.new_zeros(x.shape)
            if L_eff < Lmax:
                pad_right = Lmax - L_eff
                hidden_states = x[:, :L_eff, :]
                eff_mask = attn_mask[:, :L_eff]

        for layer in blocks:
            if eff_mask is not None and isinstance(getattr(layer, "mixer", None), Mamba):
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    attn_mask=eff_mask, validated_mask=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    attn_mask=eff_mask
                )
        fused_norm_available = layer_norm_fn is not None and (
            RMSNorm is None or not isinstance(norm_layer, RMSNorm) or rms_norm_fn is not None
        )
        if not self.fused_add_norm or not fused_norm_available:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = norm_layer(residual.to(dtype=norm_layer.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if RMSNorm is not None and isinstance(norm_layer, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                norm_layer.weight,
                norm_layer.bias,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                eps=norm_layer.eps,
            )
        if pad_right > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_right))
            if attn_mask is not None:
                hidden_states = hidden_states.masked_fill(attn_mask.unsqueeze(-1), 0.0)
        return hidden_states

    def forward(self, x, mask_ratio=0.6, meta=None, orig_Ts=None, affines=None, inference_params=None, return_moe_features=False):

        x_full, attn_pad, lengths, _ = self.patch_embed(x, meta, orig_Ts, affines, return_grid_info=False)

        # random mask
        x_keep, mask_full, ids_restore, attn_keep, keep_lengths, ids_keep_pad, ids_tgt, attn_tgt, tgt_lengths = self._random_masking_impl(
            x_full, attn_pad, lengths, mask_ratio
        )

        # build ctx visible
        x_ctx, attn_ctx = self._build_context_visible(x_keep, attn_keep)
        ctx_keep_out = self._run_blocks(
            x_ctx, attn_ctx,
            blocks=self.blocks,
            norm_layer=self.norm_f,
            mask_validated=True,
        ) # [B, Lk_max, D]

        device = x_full.device
        B, Lmax, D = x_full.shape
        ctx_keep_out, moe_aux, gates = self.moe(ctx_keep_out, attn_mask=attn_ctx, cond_vec=None, return_gates=True)

        if return_moe_features:
            return ctx_keep_out, attn_keep, keep_lengths, meta

        if Lmax == 0:
            empty_pred = x_full.new_zeros((B, 0, D))
            return x_full.new_zeros(()), empty_pred, empty_pred, mask_full

        ctx_full = self.mask_token_ctx.to(device=device, dtype=ctx_keep_out.dtype).expand(B, Lmax, D).clone()
        ctx_full = ctx_full.masked_fill(attn_pad.unsqueeze(-1), 0.0)
        ctx_full_ext = F.pad(ctx_full, (0, 0, 0, 1))
        safe_ids_keep = ids_keep_pad.masked_fill(attn_keep, Lmax)
        ctx_full_ext.scatter_(1, safe_ids_keep.unsqueeze(-1).expand(-1, -1, D), ctx_keep_out)
        ctx_full = ctx_full_ext[:, :Lmax, :]
        attn_full = attn_pad

        pred_full = self._run_blocks(
            ctx_full, attn_full, 
            blocks=self.predictor_blocks,
            norm_layer=self.predictor_norm,
            mask_validated=True,
        ) # [B, Lmax, D]

        Lt_max = ids_tgt.size(1)
        if Lt_max == 0:
            empty_pred = pred_full.new_zeros((B, 0, D))
            return pred_full.new_zeros(()), empty_pred, empty_pred, mask_full

        ids_tgt_exp = ids_tgt.unsqueeze(-1).expand(-1, -1, D)
        x_tgt_pad = torch.gather(x_full, 1, ids_tgt_exp).masked_fill(attn_tgt.unsqueeze(-1), 0.0)
        with torch.no_grad():
            tgt_feat = self._run_blocks(
                x_tgt_pad, attn_tgt, 
                blocks=self.target_blocks,
                norm_layer=self.target_norm,
                mask_validated=True,
            ) # [B, Lt_max, D]

        pred_masked = torch.gather(pred_full, 1, ids_tgt_exp).masked_fill(attn_tgt.unsqueeze(-1), 0.0)

        if self.norm_target:
            tgt_norm = torch.linalg.norm(tgt_feat, dim=-1, keepdim=True).clamp_min(1e-6)
            tgt_feat = tgt_feat / tgt_norm
            pred_norm = torch.linalg.norm(pred_masked, dim=-1, keepdim=True).clamp_min(1e-6)
            pred_masked = pred_masked / pred_norm

        valid = (~attn_tgt).unsqueeze(-1).to(pred_masked.dtype)
        denom = (~attn_tgt).sum().clamp_min(1)
        loss = ((pred_masked - tgt_feat) * valid).pow(2).sum() / denom

        return loss, pred_masked, tgt_feat, mask_full
