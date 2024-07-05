import torch.nn as nn
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from pdb import set_trace as st

from ldm.modules.attention import MemoryEfficientCrossAttention
from .dit_models_xformers import DiT, get_2d_sincos_pos_embed, ImageCondDiTBlock, FinalLayer, CaptionEmbedder, approx_gelu, ImageCondDiTBlockPixelArt, t2i_modulate, ImageCondDiTBlockPixelArtRMSNorm, T2IFinalLayer
from apex.normalization import FusedLayerNorm as LayerNorm
from apex.normalization import FusedRMSNorm as RMSNorm
from timm.models.vision_transformer import Mlp

from vit.vit_triplane import XYZPosEmbed


class DiT_I23D(DiT):
    # DiT with 3D_aware operations
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        roll_out=False,
        vit_blk=ImageCondDiTBlock,
        final_layer_blk=T2IFinalLayer,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                         mixed_prediction, context_dim, roll_out, vit_blk,
                         T2IFinalLayer)

        assert self.roll_out

        # if context_dim is not None:
        # self.dino_proj = CaptionEmbedder(context_dim,
        self.clip_ctx_dim = 1024 # vit-l
        # self.dino_proj = CaptionEmbedder(self.clip_ctx_dim, # ! dino-vitl/14 here, for img-cond
        self.dino_proj = CaptionEmbedder(768, # ! dino-vitb/14 here, for MV-cond. hard coded for now...
        # self.dino_proj = CaptionEmbedder(1024, # ! dino-vitb/14 here, for MV-cond. hard coded for now...
                                                hidden_size,
                                                act_layer=approx_gelu)

        self.clip_spatial_proj = CaptionEmbedder(1024, # clip_I-L
                                                hidden_size,
                                                act_layer=approx_gelu)

    def init_PE_3D_aware(self):

        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.plane_n * self.x_embedder.num_patches, self.embed_dim),
                                      requires_grad=False)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        p = int(self.x_embedder.num_patches**0.5)
        D = self.pos_embed.shape[-1]
        grid_size = (self.plane_n, p * p)  # B n HW C

        pos_embed = get_2d_sincos_pos_embed(D, grid_size).reshape(
            self.plane_n * p * p, D)  # H*W, D

        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

    def initialize_weights(self):
        super().initialize_weights()

        # ! add 3d-aware PE
        self.init_PE_3D_aware()

    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert isinstance(context, dict)
        # context = self.clip_text_proj(context)
        clip_cls_token = self.clip_text_proj(context['vector'])
        clip_spatial_token, dino_spatial_token = context['crossattn'][..., :self.clip_ctx_dim], self.dino_proj(context['crossattn'][..., self.clip_ctx_dim:])

        t = self.t_embedder(timesteps) + clip_cls_token  # (N, D)
        # ! todo, return spatial clip features.

        # if self.roll_out:  # !
        x = rearrange(x, 'b (c n) h w->(b n) c h w',
                      n=3)  # downsample with same conv
        x = self.x_embedder(x)  # (b n) c h/f w/f

        x = rearrange(x, '(b n) l c -> b (n l) c', n=3)
        x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        # if self.roll_out:  # ! roll-out in the L dim, not B dim. add condition to all tokens.
        # x = rearrange(x, '(b n) l c ->b (n l) c', n=3)

        # assert context.ndim == 2
        # if isinstance(context, dict):
        #     context = context['crossattn']  # sgm conditioner compat


        # c = t + context
        # else:
        # c = t  # BS 1024

        for blk_idx, block in enumerate(self.blocks):
            x = block(x, t, dino_spatial_token=dino_spatial_token, clip_spatial_token=clip_spatial_token)  # (N, T, D)

        # todo later
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, 'b (n l) c ->(b n) l c', n=3)

        x = self.unpatchify(x)  # (N, out_channels, H, W)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, '(b n) c h w -> b (c n) h w', n=3)
            # x = rearrange(x, 'b n) c h w -> b (n c) h w', n=3)

        # cast to float32 for better accuracy
        x = x.to(torch.float32).contiguous()

        return x

    # ! compat issue
    def forward_with_cfg(self, x, t, context, cfg_scale):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        # half = x[: len(x) // 2]
        # combined = torch.cat([half, half], dim=0)
        eps = self.forward(x, t, context)
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps




class DiT_I23D_PixelArt(DiT_I23D):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        roll_out=False,
        vit_blk=ImageCondDiTBlockPixelArt,
        final_layer_blk=FinalLayer,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                        #  mixed_prediction, context_dim, roll_out, ImageCondDiTBlockPixelArt,
                         mixed_prediction, context_dim, roll_out, ImageCondDiTBlockPixelArtRMSNorm,
                         final_layer_blk)

        # ! a shared one
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        # ! single
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        del self.clip_text_proj
        self.cap_embedder = nn.Sequential( # TODO, init with zero here.
            LayerNorm(context_dim),
            nn.Linear(
                context_dim,
                hidden_size,
            ),
        )

        nn.init.constant_(self.cap_embedder[-1].weight, 0)
        nn.init.constant_(self.cap_embedder[-1].bias, 0)

        print(self) # check model arch

        self.attention_y_norm = RMSNorm(
            1024, eps=1e-5
        )  # https://github.com/Alpha-VLLM/Lumina-T2X/blob/0c8dd6a07a3b7c18da3d91f37b1e00e7ae661293/lumina_t2i/models/model.py#L570C9-L570C61


    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert isinstance(context, dict)
        # context = self.clip_text_proj(context)
        clip_cls_token = self.cap_embedder(context['vector'])
        clip_spatial_token, dino_spatial_token = context['crossattn'][..., :self.clip_ctx_dim], self.dino_proj(context['crossattn'][..., self.clip_ctx_dim:])
        clip_spatial_token = self.attention_y_norm(clip_spatial_token) # avoid re-normalization in each blk

        t = self.t_embedder(timesteps) + clip_cls_token  # (N, D)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144

        # if self.roll_out:  # !
        x = rearrange(x, 'b (c n) h w->(b n) c h w',
                      n=3)  # downsample with same conv
        x = self.x_embedder(x)  # (b n) c h/f w/f

        x = rearrange(x, '(b n) l c -> b (n l) c', n=3)
        x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        # if self.roll_out:  # ! roll-out in the L dim, not B dim. add condition to all tokens.
        # x = rearrange(x, '(b n) l c ->b (n l) c', n=3)

        # assert context.ndim == 2
        # if isinstance(context, dict):
        #     context = context['crossattn']  # sgm conditioner compat


        # c = t + context
        # else:
        # c = t  # BS 1024

        for blk_idx, block in enumerate(self.blocks):
            x = block(x, t0, dino_spatial_token=dino_spatial_token, clip_spatial_token=clip_spatial_token)  # (N, T, D)

        # todo later
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, 'b (n l) c ->(b n) l c', n=3)

        x = self.unpatchify(x)  # (N, out_channels, H, W)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, '(b n) c h w -> b (c n) h w', n=3)
            # x = rearrange(x, 'b n) c h w -> b (n c) h w', n=3)

        # cast to float32 for better accuracy
        x = x.to(torch.float32).contiguous()

        return x


class DiT_I23D_PixelArt_MVCond(DiT_I23D_PixelArt):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        roll_out=False,
        vit_blk=ImageCondDiTBlockPixelArt,
        final_layer_blk=FinalLayer,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                        #  mixed_prediction, context_dim, roll_out, ImageCondDiTBlockPixelArt,
                         mixed_prediction, context_dim, roll_out, ImageCondDiTBlockPixelArtRMSNorm,
                         final_layer_blk)


        # support multi-view img condition
        # DINO handles global pooling here; clip takes care of camera-cond with ModLN
        # Input DINO concat also + global pool. InstantMesh adopts DINO (but CA).
        # expected: support dynamic numbers of frames? since CA, shall be capable of. Any number of context window size.
        del self.dino_proj

    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert isinstance(context, dict)

        # st()
        # (Pdb) p context.keys()
        # dict_keys(['crossattn', 'vector', 'concat'])
        # (Pdb) p context['vector'].shape
        # torch.Size([2, 768])
        # (Pdb) p context['crossattn'].shape
        # torch.Size([2, 256, 1024])
        # (Pdb) p context['concat'].shape
        # torch.Size([2, 4, 256, 768]) # mv dino spatial features

        # ! clip spatial tokens for append self-attn, thus add a projection layer (self.dino_proj)
        # DINO features sent via crossattn, thus no proj required (already KV linear layers in crossattn blk)
        clip_cls_token, clip_spatial_token = self.cap_embedder(context['vector']), self.clip_spatial_proj(context['crossattn']) # no norm here required? QK norm is enough, since self.ln_post(x) in vit
        dino_spatial_token = rearrange(context['concat'], 'b v l c -> b (v l) c') # flatten MV dino features.

        t = self.t_embedder(timesteps) + clip_cls_token  # (N, D)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144

        # if self.roll_out:  # !
        x = rearrange(x, 'b (c n) h w->(b n) c h w',
                      n=3)  # downsample with same conv
        x = self.x_embedder(x)  # (b n) c h/f w/f

        x = rearrange(x, '(b n) l c -> b (n l) c', n=3)
        x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        for blk_idx, block in enumerate(self.blocks):
            # x = block(x, t0, dino_spatial_token=dino_spatial_token, clip_spatial_token=clip_spatial_token)  # (N, T, D)
            # ! DINO tokens for CA, CLIP tokens for append here.
            x = block(x, t0, dino_spatial_token=clip_spatial_token, clip_spatial_token=dino_spatial_token)  # (N, T, D)

        # todo later
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, 'b (n l) c ->(b n) l c', n=3)

        x = self.unpatchify(x)  # (N, out_channels, H, W)

        if self.roll_out:  # move n from L to B axis
            x = rearrange(x, '(b n) c h w -> b (c n) h w', n=3)

        x = x.to(torch.float32).contiguous()

        return x

# pcd-structured latent ddpm

class DiT_pcd_I23D_PixelArt_MVCond(DiT_I23D_PixelArt_MVCond):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        mixing_logit_init=-3,
        mixed_prediction=True,
        context_dim=False,
        roll_out=False,
        vit_blk=ImageCondDiTBlockPixelArt,
        final_layer_blk=FinalLayer,
    ):
        super().__init__(input_size, patch_size, in_channels, hidden_size,
                         depth, num_heads, mlp_ratio, class_dropout_prob,
                         num_classes, learn_sigma, mixing_logit_init,
                        #  mixed_prediction, context_dim, roll_out, ImageCondDiTBlockPixelArt,
                         mixed_prediction, context_dim, roll_out, ImageCondDiTBlockPixelArtRMSNorm,
                         final_layer_blk)
        # ! first, normalize xyz from [-0.45,0.45] to [-1,1]
        # Then, encode xyz with point fourier feat + MLP projection, serves as PE here.
        # a separate MLP for the KL feature
        # add them together in the feature space
        # use a single MLP (final_layer) to map them back to 16 + 3 dims.
        self.x_embedder = Mlp(in_features=in_channels,
                          hidden_features=hidden_size,
                          out_features=hidden_size,
                          act_layer=approx_gelu,
                          drop=0)
        del self.pos_embed


    def forward(self,
                x,
                timesteps=None,
                context=None,
                y=None,
                get_attr='',
                **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # t = timesteps
        assert isinstance(context, dict)

        # st()
        # (Pdb) p context.keys()
        # dict_keys(['crossattn', 'vector', 'concat'])
        # (Pdb) p context['vector'].shape
        # torch.Size([2, 768])
        # (Pdb) p context['crossattn'].shape
        # torch.Size([2, 256, 1024])
        # (Pdb) p context['concat'].shape
        # torch.Size([2, 4, 256, 768]) # mv dino spatial features

        # ! clip spatial tokens for append self-attn, thus add a projection layer (self.dino_proj)
        # DINO features sent via crossattn, thus no proj required (already KV linear layers in crossattn blk)
        clip_cls_token, clip_spatial_token = self.cap_embedder(context['vector']), self.clip_spatial_proj(context['crossattn']) # no norm here required? QK norm is enough, since self.ln_post(x) in vit
        dino_spatial_token = rearrange(context['concat'], 'b v l c -> b (v l) c') # flatten MV dino features.

        t = self.t_embedder(timesteps) + clip_cls_token  # (N, D)
        t0 = self.adaLN_modulation(t) # single-adaLN, B 6144

        x = self.x_embedder(x)

        for blk_idx, block in enumerate(self.blocks):
            # x = block(x, t0, dino_spatial_token=dino_spatial_token, clip_spatial_token=clip_spatial_token)  # (N, T, D)
            # ! DINO tokens for CA, CLIP tokens for append here.
            x = block(x, t0, dino_spatial_token=clip_spatial_token, clip_spatial_token=dino_spatial_token)  # (N, T, D)

        # todo later
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)

        x = x.to(torch.float32).contiguous()

        return x



#################################################################################
#                                   DiT_I23D Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT_I23D(depth=28,
                         hidden_size=1152,
                         patch_size=2,
                         num_heads=16,
                         **kwargs)


def DiT_L_2(**kwargs):
    return DiT_I23D(depth=24,
                         hidden_size=1024,
                         patch_size=2,
                         num_heads=16,
                         **kwargs)


def DiT_B_2(**kwargs):
    return DiT_I23D(depth=12,
                         hidden_size=768,
                         patch_size=2,
                         num_heads=12,
                         **kwargs)


def DiT_B_1(**kwargs):
    return DiT_I23D(depth=12,
                         hidden_size=768,
                         patch_size=1,
                         num_heads=12,
                         **kwargs)


def DiT_L_Pixelart_2(**kwargs):
    return DiT_I23D_PixelArt(depth=24,
                         hidden_size=1024,
                         patch_size=2,
                         num_heads=16,
                         **kwargs)


def DiT_B_Pixelart_2(**kwargs):
    return DiT_I23D_PixelArt(depth=12,
                         hidden_size=768,
                         patch_size=2,
                         num_heads=12,
                         **kwargs)

def DiT_L_Pixelart_MV_2(**kwargs):
    return DiT_I23D_PixelArt_MVCond(depth=24,
                         hidden_size=1024,
                         patch_size=2,
                         num_heads=16,
                         **kwargs)

def DiT_XL_Pixelart_MV_2(**kwargs):
    return DiT_I23D_PixelArt_MVCond(depth=28,
                         hidden_size=1152,
                         patch_size=2,
                         num_heads=16,
                         **kwargs)



def DiT_B_Pixelart_MV_2(**kwargs):
    return DiT_I23D_PixelArt_MVCond(depth=12,
                         hidden_size=768,
                         patch_size=2,
                         num_heads=12,
                         **kwargs)

# pcd latent 

def DiT_L_Pixelart_MV_pcd(**kwargs):
    return DiT_pcd_I23D_PixelArt_MVCond(depth=24,
                         hidden_size=1024,
                         patch_size=1, # no spatial compression here
                         num_heads=16,
                         **kwargs)



DiT_models = {
    'DiT-XL/2': DiT_XL_2,
    'DiT-L/2': DiT_L_2,
    'DiT-B/2': DiT_B_2,
    'DiT-B/1': DiT_B_1,
    'DiT-PixArt-L/2': DiT_L_Pixelart_2,
    'DiT-PixArt-MV-XL/2': DiT_XL_Pixelart_MV_2,
    'DiT-PixArt-MV-L/2': DiT_L_Pixelart_MV_2,
    'DiT-PixArt-MV-PCD-L': DiT_L_Pixelart_MV_pcd,
    'DiT-PixArt-MV-B/2': DiT_B_Pixelart_MV_2,
    'DiT-PixArt-B/2': DiT_B_Pixelart_2,
}
