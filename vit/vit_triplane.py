import math
import random
from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from tqdm import trange

from functools import partial

from nsr.networks_stylegan2 import Generator as StyleGAN2Backbone
from nsr.volumetric_rendering.renderer import ImportanceRenderer, ImportanceRendererfg_bg
from nsr.volumetric_rendering.ray_sampler import RaySampler
from nsr.triplane import OSGDecoder, Triplane, Triplane_fg_bg_plane
# from nsr.losses.helpers import ResidualBlock
# from vit.vision_transformer import TriplaneFusionBlockv4_nested, VisionTransformer, TriplaneFusionBlockv4_nested_init_from_dino
from vit.vision_transformer import TriplaneFusionBlockv4_nested, TriplaneFusionBlockv4_nested_init_from_dino_lite, TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout, VisionTransformer, TriplaneFusionBlockv4_nested_init_from_dino

from .vision_transformer import Block, VisionTransformer
from .utils import trunc_normal_

from guided_diffusion import dist_util, logger

from pdb import set_trace as st

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from utils.torch_utils.components import PixelShuffleUpsample, ResidualBlock, Upsample, PixelUnshuffleUpsample, Conv3x3TriplaneTransformation
from utils.torch_utils.distributions.distributions import DiagonalGaussianDistribution
from nsr.superresolution import SuperresolutionHybrid2X, SuperresolutionHybrid4X

from torch.nn.parameter import Parameter, UninitializedParameter, UninitializedBuffer

from nsr.common_blks import ResMlp
from .vision_transformer import *

from dit.dit_models import get_2d_sincos_pos_embed
from torch import _assert
from itertools import repeat
import collections.abc


# From PyTorch internals
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


class PatchEmbedTriplane(nn.Module):
    """ GroupConv patchembeder on triplane
    """

    def __init__(
        self,
        img_size=32,
        patch_size=2,
        in_chans=4,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans,
                              embed_dim * 3,
                              kernel_size=patch_size,
                              stride=patch_size,
                              bias=bias,
                              groups=3)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(
            H == self.img_size[0],
            f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        )
        _assert(
            W == self.img_size[1],
            f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        )
        x = self.proj(x)  # B 3*C token_H token_W

        x = x.reshape(B, x.shape[1] // 3, 3, x.shape[-2],
                      x.shape[-1])  # B C 3 H W

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BC3HW -> B 3HW C
        x = self.norm(x)
        return x


class PatchEmbedTriplaneRodin(PatchEmbedTriplane):

    def __init__(self,
                 img_size=32,
                 patch_size=2,
                 in_chans=4,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=True,
                 bias=True):
        super().__init__(img_size, patch_size, in_chans, embed_dim, norm_layer,
                         flatten, bias)
        self.proj = RodinRollOutConv3D_GroupConv(in_chans,
                                                 embed_dim * 3,
                                                 kernel_size=patch_size,
                                                 stride=patch_size,
                                                 padding=0)


class ViTTriplaneDecomposed(nn.Module):

    def __init__(
            self,
            vit_decoder,
            triplane_decoder: Triplane,
            cls_token=False,
            decoder_pred_size=-1,
            unpatchify_out_chans=-1,
            # * uvit arch
            channel_multiplier=4,
            use_fusion_blk=True,
            fusion_blk_depth=4,
            fusion_blk=TriplaneFusionBlock,
            fusion_blk_start=0,  # appy fusion blk start with?
            ldm_z_channels=4,  # 
            ldm_embed_dim=4,
            vae_p=2,
            token_size=None,
            w_avg=torch.zeros([512]),
            patch_size=None, 
            **kwargs,
    ) -> None:
        super().__init__()
        # self.superresolution = None
        self.superresolution = nn.ModuleDict({})

        self.decomposed_IN = False

        self.decoder_pred_3d = None
        self.transformer_3D_blk = None
        self.logvar = None
        self.channel_multiplier = channel_multiplier

        self.cls_token = cls_token
        self.vit_decoder = vit_decoder
        self.triplane_decoder = triplane_decoder

        if patch_size is None:
            self.patch_size = self.vit_decoder.patch_embed.patch_size
        else:
            self.patch_size = patch_size

        if isinstance(self.patch_size, tuple):  # dino-v2
            self.patch_size = self.patch_size[0]

        # self.img_size = self.vit_decoder.patch_embed.img_size

        if unpatchify_out_chans == -1:
            self.unpatchify_out_chans = self.triplane_decoder.out_chans
        else:
            self.unpatchify_out_chans = unpatchify_out_chans

        # ! mlp decoder from mae/dino
        if decoder_pred_size == -1:
            decoder_pred_size = self.patch_size**2 * self.triplane_decoder.out_chans

        self.decoder_pred = nn.Linear(
            self.vit_decoder.embed_dim,
            decoder_pred_size,
            #   self.patch_size**2 *
            #   self.triplane_decoder.out_chans,
            bias=True)  # decoder to pat
        # st()

        # triplane
        self.plane_n = 3

        # ! vae
        self.ldm_z_channels = ldm_z_channels
        self.ldm_embed_dim = ldm_embed_dim
        self.vae_p = vae_p
        self.token_size = 16  # use dino-v2 dim tradition here
        self.vae_res = self.vae_p * self.token_size

        # ! uvit
        # if token_size is None:
        #     token_size = 224 // self.patch_size
        #     logger.log('token_size: {}', token_size)

        self.vit_decoder.pos_embed = nn.Parameter(
            torch.zeros(1, 3 * (self.token_size**2 + self.cls_token),
                        vit_decoder.embed_dim))

        self.fusion_blk_start = fusion_blk_start
        self.create_fusion_blks(fusion_blk_depth, use_fusion_blk, fusion_blk)
        # self.vit_decoder.cls_token = self.vit_decoder.cls_token.clone().repeat_interleave(3, dim=0) # each plane has a separate cls token
        # translate

        # ! placeholder, not used here
        self.register_buffer('w_avg', w_avg)  # will replace externally
        self.rendering_kwargs = self.triplane_decoder.rendering_kwargs


    @torch.inference_mode()
    def forward_points(self, planes, points: torch.Tensor, chunk_size: int = 2**16):
        # planes: (N, 3, D', H', W')
        # points: (N, P, 3)
        N, P = points.shape[:2]
        if planes.ndim == 4:
            planes = planes.reshape(
                len(planes),
                3,
                -1,  # ! support background plane
                planes.shape[-2],
                planes.shape[-1])  # BS 96 256 256

        # query triplane in chunks
        outs = []
        for i in trange(0, points.shape[1], chunk_size):
            chunk_points = points[:, i:i+chunk_size]

            # query triplane
            # st()
            chunk_out = self.triplane_decoder.renderer._run_model( # type: ignore
                planes=planes,
                decoder=self.triplane_decoder.decoder,
                sample_coordinates=chunk_points,
                sample_directions=torch.zeros_like(chunk_points),
                options=self.rendering_kwargs,
            )
            # st()

            outs.append(chunk_out)
            torch.cuda.empty_cache()
        
        # st()

        # concatenate the outputs
        point_features = {
            k: torch.cat([out[k] for out in outs], dim=1)
            for k in outs[0].keys()
        }
        return point_features

    def triplane_decode_grid(self, vit_decode_out, grid_size, aabb: torch.Tensor = None, **kwargs):
                # planes: (N, 3, D', H', W')
        # grid_size: int

        assert isinstance(vit_decode_out, dict)
        planes = vit_decode_out['latent_after_vit']

        # aabb: (N, 2, 3)
        if aabb is None:
            if 'sampler_bbox_min' in self.rendering_kwargs:
                aabb = torch.tensor([
                    [self.rendering_kwargs['sampler_bbox_min']] * 3,
                    [self.rendering_kwargs['sampler_bbox_max']] * 3,
                ], device=planes.device, dtype=planes.dtype).unsqueeze(0).repeat(planes.shape[0], 1, 1)
            else: # shapenet dataset, follow eg3d
                aabb = torch.tensor([ # https://github.com/NVlabs/eg3d/blob/7cf1fd1e99e1061e8b6ba850f91c94fe56e7afe4/eg3d/gen_samples.py#L188
                    [-self.rendering_kwargs['box_warp']/2] * 3,
                    [self.rendering_kwargs['box_warp']/2] * 3,
                ], device=planes.device, dtype=planes.dtype).unsqueeze(0).repeat(planes.shape[0], 1, 1)

        assert planes.shape[0] == aabb.shape[0], "Batch size mismatch for planes and aabb"
        N = planes.shape[0]

        # create grid points for triplane query
        grid_points = []
        for i in range(N):
            grid_points.append(torch.stack(torch.meshgrid(
                torch.linspace(aabb[i, 0, 0], aabb[i, 1, 0], grid_size, device=planes.device),
                torch.linspace(aabb[i, 0, 1], aabb[i, 1, 1], grid_size, device=planes.device),
                torch.linspace(aabb[i, 0, 2], aabb[i, 1, 2], grid_size, device=planes.device),
                indexing='ij',
            ), dim=-1).reshape(-1, 3))
        cube_grid = torch.stack(grid_points, dim=0).to(planes.device) # 1 N 3
        # st()

        features = self.forward_points(planes, cube_grid)

        # reshape into grid
        features = {
            k: v.reshape(N, grid_size, grid_size, grid_size, -1)
            for k, v in features.items()
        }

        # st()

        return features


    def create_uvit_arch(self):
        # create skip linear
        logger.log(
            f'length of vit_decoder.blocks: {len(self.vit_decoder.blocks)}')
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2:]:
            blk.skip_linear = nn.Linear(2 * self.vit_decoder.embed_dim,
                                        self.vit_decoder.embed_dim)

            # trunc_normal_(blk.skip_linear.weight, std=.02)
            nn.init.constant_(blk.skip_linear.weight, 0)
            if isinstance(blk.skip_linear,
                          nn.Linear) and blk.skip_linear.bias is not None:
                nn.init.constant_(blk.skip_linear.bias, 0)


#

    def vit_decode_backbone(self, latent, img_size):
        return self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

    def init_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        p = self.token_size
        D = self.vit_decoder.pos_embed.shape[-1]
        grid_size = (3 * p, p)
        pos_embed = get_2d_sincos_pos_embed(D,
                                            grid_size).reshape(3 * p * p,
                                                               D)  # H*W, D
        self.vit_decoder.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))
        logger.log('init pos_embed with sincos')

    # !
    def create_fusion_blks(self, fusion_blk_depth, use_fusion_blk, fusion_blk):

        vit_decoder_blks = self.vit_decoder.blocks
        assert len(vit_decoder_blks) == 12, 'ViT-B by default'

        nh = self.vit_decoder.blocks[0].attn.num_heads
        dim = self.vit_decoder.embed_dim

        fusion_blk_start = self.fusion_blk_start
        triplane_fusion_vit_blks = nn.ModuleList()

        if fusion_blk_start != 0:
            for i in range(0, fusion_blk_start):
                triplane_fusion_vit_blks.append(
                    vit_decoder_blks[i])  # append all vit blocks in the front

        for i in range(fusion_blk_start, len(vit_decoder_blks),
                       fusion_blk_depth):
            vit_blks_group = vit_decoder_blks[i:i +
                                              fusion_blk_depth]  # moduleList
            triplane_fusion_vit_blks.append(
                # TriplaneFusionBlockv2(vit_blks_group, nh, dim, use_fusion_blk))
                fusion_blk(vit_blks_group, nh, dim, use_fusion_blk))

        self.vit_decoder.blocks = triplane_fusion_vit_blks

    def triplane_decode(self, latent, c):
        ret_dict = self.triplane_decoder(latent, c)  # triplane latent -> imgs
        ret_dict.update({'latent': latent})
        return ret_dict

    def triplane_renderer(self, latent, coordinates, directions):

        planes = latent.view(len(latent), 3,
                             self.triplane_decoder.decoder_in_chans,
                             latent.shape[-2],
                             latent.shape[-1])  # BS 96 256 256

        ret_dict = self.triplane_decoder.renderer.run_model(
            planes, self.triplane_decoder.decoder, coordinates, directions,
            self.triplane_decoder.rendering_kwargs)  # triplane latent -> imgs
        # ret_dict.update({'latent': latent})
        return ret_dict

        # * increase encoded encoded latent dim to match decoder

    # ! util functions
    def unpatchify_triplane(self, x, p=None, unpatchify_out_chans=None):
        """
        x: (N, L, patch_size**2 * self.out_chans)
        imgs: (N, self.out_chans, H, W)
        """
        if unpatchify_out_chans is None:
            unpatchify_out_chans = self.unpatchify_out_chans // 3
        # p = self.vit_decoder.patch_size
        if self.cls_token:  # TODO, how to better use cls token
            x = x[:, 1:]

        if p is None:  # assign upsample patch size
            p = self.patch_size
        h = w = int((x.shape[1] // 3)**.5)
        assert h * w * 3 == x.shape[1]

        x = x.reshape(shape=(x.shape[0], 3, h, w, p, p, unpatchify_out_chans))
        x = torch.einsum('ndhwpqc->ndchpwq',
                         x)  # nplanes, C order in the renderer.py
        triplanes = x.reshape(shape=(x.shape[0], unpatchify_out_chans * 3,
                                     h * p, h * p))
        return triplanes

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.vit_decoder.pos_embed.shape[1] - 1  # type: ignore
        # if npatch == N and w == h:
        # assert npatch == N and w == h
        return self.vit_decoder.pos_embed

        # pos_embed = self.vit_decoder.pos_embed.float()
        # return pos_embed
        class_pos_embed = pos_embed[:, 0]  # type: ignore
        patch_pos_embed = pos_embed[:, 1:]  # type: ignore
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        # patch_pos_embed = nn.functional.interpolate(
        #     patch_pos_embed.reshape(1, 3, int(math.sqrt(N//3)), int(math.sqrt(N//3)), dim).permute(0, 4, 1, 2, 3),
        #     scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        #     mode="bicubic",
        # ) # ! no interpolation needed, just add, since the resolution shall match

        # assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed),
                         dim=1).to(previous_dtype)

    def forward_vit_decoder(self, x, img_size=None):
        # latent: (N, L, C) from DINO/CLIP ViT encoder

        # * also dino ViT
        # add positional encoding to each token
        if img_size is None:
            img_size = self.img_size

        if self.cls_token:
            x = x + self.vit_decoder.interpolate_pos_encoding(
                x, img_size, img_size)[:, :]  # B, L, C
        else:
            x = x + self.vit_decoder.interpolate_pos_encoding(
                x, img_size, img_size)[:, 1:]  # B, L, C

        for blk in self.vit_decoder.blocks:
            x = blk(x)
        x = self.vit_decoder.norm(x)

        return x

    def unpatchify(self, x, p=None, unpatchify_out_chans=None):
        """
        x: (N, L, patch_size**2 * self.out_chans)
        imgs: (N, self.out_chans, H, W)
        """
        # st()
        if unpatchify_out_chans is None:
            unpatchify_out_chans = self.unpatchify_out_chans
        # p = self.vit_decoder.patch_size
        if self.cls_token:  # TODO, how to better use cls token
            x = x[:, 1:]

        if p is None:  # assign upsample patch size
            p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, unpatchify_out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], unpatchify_out_chans, h * p,
                                h * p))
        return imgs

    def forward(self, latent, c, img_size):
        latent = self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

        if self.cls_token:
            # latent, cls_token = latent[:, 1:], latent[:, :1]
            cls_token = latent[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent)  # pred_vit_latent -> patch or original size
        # st()
        latent = self.unpatchify(
            latent)  # spatial_vit_latent, B, C, H, W (B, 96, 256,256)

        # TODO 2D convolutions -> Triplane
        # * triplane rendering
        # ret_dict = self.forward_triplane_decoder(latent,
        #                                          c)  # triplane latent -> imgs
        ret_dict = self.triplane_decoder(planes=latent, c=c)
        ret_dict.update({'latent': latent, 'cls_token': cls_token})

        return ret_dict


class VAE_LDM_V4_vit3D_v3_conv3D_depth2_xformer_mha_PEinit_2d_sincos_uvit_RodinRollOutConv_4x4_lite_mlp_unshuffle_4XC_final(
        ViTTriplaneDecomposed):
    """
    1. reuse attention proj layer from dino
    2. reuse attention; first self then 3D cross attention
    """
    """ 4*4 SR with 2X channels
    """

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane,
            cls_token,
            #  normalize_feat=True,
            #  sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            channel_multiplier=4,
            fusion_blk=TriplaneFusionBlockv3,
            **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            #  normalize_feat,
            #  sr_ratio,
            fusion_blk=fusion_blk,  # type: ignore
            use_fusion_blk=use_fusion_blk,
            fusion_blk_depth=fusion_blk_depth,
            channel_multiplier=channel_multiplier,
            decoder_pred_size=(4 // 1)**2 *
            int(triplane_decoder.out_chans // 3 * channel_multiplier),
            **kwargs)

        patch_size = vit_decoder.patch_embed.patch_size  # type: ignore

        self.reparameterization_soft_clamp = False

        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]

        # ! todo, hard coded
        unpatchify_out_chans = triplane_decoder.out_chans * 1,

        if unpatchify_out_chans == -1:
            unpatchify_out_chans = triplane_decoder.out_chans * 3

        ldm_z_channels = triplane_decoder.out_chans
        # ldm_embed_dim = 16 # https://github.com/CompVis/latent-diffusion/blob/e66308c7f2e64cb581c6d27ab6fbeb846828253b/models/first_stage_models/kl-f16/config.yaml
        ldm_embed_dim = triplane_decoder.out_chans
        ldm_z_channels = ldm_embed_dim = triplane_decoder.out_chans

        self.superresolution.update(
            dict(
                after_vit_conv=nn.Conv2d(
                    int(triplane_decoder.out_chans * 2),
                    triplane_decoder.out_chans * 2,  # for vae features
                    3,
                    padding=1),
                quant_conv=torch.nn.Conv2d(2 * ldm_z_channels,
                                           2 * ldm_embed_dim, 1),
                ldm_downsample=nn.Linear(
                    384,
                    # vit_decoder.embed_dim,
                    self.vae_p * self.vae_p * 3 * self.ldm_z_channels *
                    2,  # 48
                    bias=True),
                ldm_upsample=nn.Linear(self.vae_p * self.vae_p *
                                       self.ldm_z_channels * 1,
                                       vit_decoder.embed_dim,
                                       bias=True),  # ? too high dim upsample
                quant_mlp=Mlp(2 * self.ldm_z_channels,
                              out_features=2 * self.ldm_embed_dim),
                conv_sr=RodinConv3D4X_lite_mlp_as_residual(
                    int(triplane_decoder.out_chans * channel_multiplier),
                    int(triplane_decoder.out_chans * 1))))

        has_token = bool(self.cls_token)
        self.vit_decoder.pos_embed = nn.Parameter(
            torch.zeros(1, 3 * 16 * 16 + has_token, vit_decoder.embed_dim))

        self.init_weights()
        self.reparameterization_soft_clamp = True  # some instability in training VAE

        self.create_uvit_arch()

    def vae_reparameterization(self, latent, sample_posterior):
        """input: latent from ViT encoder
        """
        # ! first downsample for VAE
        latents3D = self.superresolution['ldm_downsample'](latent)  # B L 24

        if self.vae_p > 1:
            latents3D = self.unpatchify3D(
                latents3D,
                p=self.vae_p,
                unpatchify_out_chans=self.ldm_z_channels *
                2)  # B 3 H W unpatchify_out_chans, H=W=16 now
            latents3D = latents3D.reshape(
                latents3D.shape[0], 3, -1, latents3D.shape[-1]
            )  # B 3 H*W C (H=self.vae_p*self.token_size)
        else:
            latents3D = latents3D.reshape(latents3D.shape[0],
                                          latents3D.shape[1], 3,
                                          2 * self.ldm_z_channels)  # B L 3 C
            latents3D = latents3D.permute(0, 2, 1, 3)  # B 3 L C

        # ! maintain the cls token here
        # latent3D = latent.reshape()

        # ! do VAE here
        posterior = self.vae_encode(latents3D)  # B self.ldm_z_channels 3 L

        if sample_posterior:
            latent = posterior.sample()
        else:
            latent = posterior.mode()  # B C 3 L

        log_q = posterior.log_p(latent)  # same shape as latent

        # latent = latent.permute(0, 2, 3, 4,
        #                         1)  # C to the last dim, B 3 16 16 4, for unpachify 3D

        # ! for LSGM KL code
        latent_normalized_2Ddiffusion = latent.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16
        log_q_2Ddiffusion = log_q.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16
        latent = latent.permute(0, 2, 3, 1)  # B C 3 L -> B 3 L C

        latent = latent.reshape(latent.shape[0], -1,
                                latent.shape[-1])  # B 3*L C

        ret_dict = dict(
            normal_entropy=posterior.normal_entropy(),
            latent_normalized=latent,
            latent_normalized_2Ddiffusion=latent_normalized_2Ddiffusion,  # 
            log_q_2Ddiffusion=log_q_2Ddiffusion,
            log_q=log_q,
            posterior=posterior,
            latent_name=
            'latent_normalized'  # for which latent to decode; could be modified externally
        )

        return ret_dict

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit
        )  # pred_vit_latent -> patch or original size; B 768 384

        latent = self.unpatchify_triplane(
            latent,
            p=4,
            unpatchify_out_chans=int(
                self.channel_multiplier * self.unpatchify_out_chans //
                3))  # spatial_vit_latent, B, C, H, W (B, 96*2, 16, 16)

        # 4X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr'](latent)  # still B 3C H W

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent_from_vit.shape[0], 0), ))  # type: ignore

        return ret_dict

    def forward_vit_decoder(self, x, img_size=None):
        # latent: (N, L, C) from DINO/CLIP ViT encoder

        # * also dino ViT
        # add positional encoding to each token
        if img_size is None:
            img_size = self.img_size

        # if self.cls_token:
        # st()
        x = x + self.interpolate_pos_encoding(x, img_size,
                                              img_size)[:, :]  # B, L, C

        B, L, C = x.shape  # has [cls] token in N
        x = x.view(B, 3, L // 3, C)

        skips = [x]
        assert self.fusion_blk_start == 0

        # in blks
        for blk in self.vit_decoder.blocks[0:len(self.vit_decoder.blocks) //
                                           2 - 1]:
            x = blk(x)  # B 3 N C
            skips.append(x)

        # mid blks
        # for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks)//2-1:len(self.vit_decoder.blocks)//2+1]:
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2 -
                                           1:len(self.vit_decoder.blocks) //
                                           2]:
            x = blk(x)  # B 3 N C

        # out blks
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2:]:
            x = x + blk.skip_linear(torch.cat([x, skips.pop()],
                                              dim=-1))  # long skip connections
            x = blk(x)  # B 3 N C

        x = self.vit_decoder.norm(x)

        # post process shape
        x = x.view(B, L, C)
        return x

    def triplane_decode(self,
                        vit_decode_out,
                        c,
                        return_raw_only=False,
                        **kwargs):

        if isinstance(vit_decode_out, dict):
            latent_after_vit, sr_w_code = (vit_decode_out.get(k, None)
                                           for k in ('latent_after_vit',
                                                     'sr_w_code'))

        else:
            latent_after_vit = vit_decode_out
            sr_w_code = None
            vit_decode_out = dict(latent_normalized=latent_after_vit
                                  )  # for later dict update compatability

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent_after_vit,
                                         c,
                                         ws=sr_w_code,
                                         return_raw_only=return_raw_only,
                                         **kwargs)  # triplane latent -> imgs
        ret_dict.update({
            'latent_after_vit': latent_after_vit,
            **vit_decode_out
        })

        return ret_dict

    def vit_decode_backbone(self, latent, img_size):
        # assert x.ndim == 3  # N L C
        if isinstance(latent, dict):
            if 'latent_normalized' not in latent:
                latent = latent[
                    'latent_normalized_2Ddiffusion']  # B, C*3, H, W
            else:
                latent = latent[
                    'latent_normalized']  # TODO, just for compatability now

        # st()
        if latent.ndim != 3:  # B 3*4 16 16
            latent = latent.reshape(latent.shape[0], latent.shape[1] // 3, 3,
                                    (self.vae_p * self.token_size)**2).permute(
                                        0, 2, 3, 1)  # B C 3 L => B 3 L C
            latent = latent.reshape(latent.shape[0], -1,
                                    latent.shape[-1])  # B 3*L C

        assert latent.shape == (
            # latent.shape[0], 3 * (self.token_size**2),
            latent.shape[0],
            3 * ((self.vae_p * self.token_size)**2),
            self.ldm_z_channels), f'latent.shape: {latent.shape}'

        latent = self.superresolution['ldm_upsample'](latent)

        return super().vit_decode_backbone(
            latent, img_size)  # torch.Size([8, 3072, 768])


class RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn(
        ViTTriplaneDecomposed):
    # lite version, no sd-bg, use TriplaneFusionBlockv4_nested_init_from_dino
    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            # normalize_feat=True,
            # sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino,
            channel_multiplier=4,
            ldm_z_channels=4,  # 
            ldm_embed_dim=4,
            vae_p=2,
            **kwargs) -> None:
        # st()
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            # normalize_feat,
            channel_multiplier=channel_multiplier,
            use_fusion_blk=use_fusion_blk,
            fusion_blk_depth=fusion_blk_depth,
            fusion_blk=fusion_blk,
            ldm_z_channels=ldm_z_channels,
            ldm_embed_dim=ldm_embed_dim,
            vae_p=vae_p,
            decoder_pred_size=(4 // 1)**2 *
            int(triplane_decoder.out_chans // 3 * channel_multiplier),
            **kwargs)

        logger.log(
            f'length of vit_decoder.blocks: {len(self.vit_decoder.blocks)}')

        # latent vae modules
        self.superresolution.update(
            dict(
                ldm_downsample=nn.Linear(
                    384,
                    self.vae_p * self.vae_p * 3 * self.ldm_z_channels *
                    2,  # 48
                    bias=True),
                ldm_upsample=PatchEmbedTriplane(
                    self.vae_p * self.token_size,
                    self.vae_p,
                    3 * self.ldm_embed_dim,  # B 3 L C
                    vit_decoder.embed_dim,
                    bias=True),
                quant_conv=nn.Conv2d(2 * 3 * self.ldm_z_channels,
                                     2 * self.ldm_embed_dim * 3,
                                     kernel_size=1,
                                     groups=3),
                conv_sr=RodinConv3D4X_lite_mlp_as_residual_lite(
                    int(triplane_decoder.out_chans * channel_multiplier),
                    int(triplane_decoder.out_chans * 1))))

        # ! initialize weights
        self.init_weights()
        self.reparameterization_soft_clamp = True  # some instability in training VAE

        self.create_uvit_arch()

        # create skip linear, adapted from uvit
        # for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2:]:
        #     blk.skip_linear = nn.Linear(2 * self.vit_decoder.embed_dim,
        #                                 self.vit_decoder.embed_dim)

        #     # trunc_normal_(blk.skip_linear.weight, std=.02)
        #     nn.init.constant_(blk.skip_linear.weight, 0)
        #     if isinstance(blk.skip_linear,
        #                   nn.Linear) and blk.skip_linear.bias is not None:
        #         nn.init.constant_(blk.skip_linear.bias, 0)

    def vit_decode(self, latent, img_size, sample_posterior=True):

        ret_dict = self.vae_reparameterization(latent, sample_posterior)
        # latent = ret_dict['latent_normalized']

        latent = self.vit_decode_backbone(ret_dict, img_size)
        return self.vit_decode_postprocess(latent, ret_dict)

    # # ! merge?
    def unpatchify3D(self, x, p, unpatchify_out_chans, plane_n=3):
        """
        x: (N, L, patch_size**2 * self.out_chans)
        return: 3D latents
        """

        if self.cls_token:  # TODO, how to better use cls token
            x = x[:, 1:]

        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, plane_n,
                             unpatchify_out_chans))

        x = torch.einsum(
            'nhwpqdc->ndhpwqc', x
        )  # nplanes, C little endian tradiition, as defined in the renderer.py

        latents3D = x.reshape(shape=(x.shape[0], plane_n, h * p, h * p,
                                     unpatchify_out_chans))
        return latents3D

    # ! merge?
    def vae_encode(self, h):
        # * smooth convolution before triplane
        # h = self.superresolution['after_vit_conv'](h)
        # h = h.permute(0, 2, 3, 1)  # B 64 64 6
        B, _, H, W = h.shape
        moments = self.superresolution['quant_conv'](h)

        moments = moments.reshape(
            B,
            # moments.shape[1] // 3,
            moments.shape[1] // self.plane_n,
            # 3,
            self.plane_n,
            H,
            W,
        )  # B C 3 H W

        moments = moments.flatten(-2)  # B C 3 L

        posterior = DiagonalGaussianDistribution(
            moments, soft_clamp=self.reparameterization_soft_clamp)
        return posterior

    def vae_reparameterization(self, latent, sample_posterior):
        """input: latent from ViT encoder
        """
        # ! first downsample for VAE
        # st() # latent: B 256 384
        latents3D = self.superresolution['ldm_downsample'](
            latent)  # latents3D: B 256 96

        assert self.vae_p > 1
        latents3D = self.unpatchify3D(
            latents3D,
            p=self.vae_p,
            unpatchify_out_chans=self.ldm_z_channels *
            2)  # B 3 H W unpatchify_out_chans, H=W=16 now
        #     latents3D = latents3D.reshape(
        #         latents3D.shape[0], 3, -1, latents3D.shape[-1]
        #     )  # B 3 H*W C (H=self.vae_p*self.token_size)
        # else:
        #     latents3D = latents3D.reshape(latents3D.shape[0],
        #                                   latents3D.shape[1], 3,
        #                                   2 * self.ldm_z_channels)  # B L 3 C
        #     latents3D = latents3D.permute(0, 2, 1, 3)  # B 3 L C

        B, _, H, W, C = latents3D.shape
        latents3D = latents3D.permute(0, 1, 4, 2, 3).reshape(B, -1, H,
                                                             W)  # B 3C H W

        # ! do VAE here
        posterior = self.vae_encode(latents3D)  # B self.ldm_z_channels 3 L

        if sample_posterior:
            latent = posterior.sample()
        else:
            latent = posterior.mode()  # B C 3 L

        log_q = posterior.log_p(latent)  # same shape as latent

        # ! for LSGM KL code
        latent_normalized_2Ddiffusion = latent.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16
        log_q_2Ddiffusion = log_q.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16
        # st()
        latent = latent.permute(0, 2, 3, 1)  # B C 3 L -> B 3 L C

        latent = latent.reshape(latent.shape[0], -1,
                                latent.shape[-1])  # B 3*L C

        ret_dict = dict(
            normal_entropy=posterior.normal_entropy(),
            latent_normalized=latent,
            latent_normalized_2Ddiffusion=latent_normalized_2Ddiffusion,  # 
            log_q_2Ddiffusion=log_q_2Ddiffusion,
            log_q=log_q,
            posterior=posterior,
        )

        return ret_dict

    def vit_decode_backbone(self, latent, img_size):
        # assert x.ndim == 3  # N L C
        if isinstance(latent, dict):
            latent = latent['latent_normalized_2Ddiffusion']  # B, C*3, H, W

        # assert latent.shape == (
        #     latent.shape[0], 3 * (self.token_size * self.vae_p)**2,
        #     self.ldm_z_channels), f'latent.shape: {latent.shape}'

        # st() # latent: B 12 32 32
        latent = self.superresolution['ldm_upsample'](  # ! B 768 (3*256) 768 
            latent)  # torch.Size([8, 12, 32, 32]) => torch.Size([8, 256, 768])
        # latent: torch.Size([8, 768, 768])

        # ! directly feed to vit_decoder
        return self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

    def triplane_decode(self,
                        vit_decode_out,
                        c,
                        return_raw_only=False,
                        **kwargs):

        if isinstance(vit_decode_out, dict):
            latent_after_vit, sr_w_code = (vit_decode_out.get(k, None)
                                           for k in ('latent_after_vit',
                                                     'sr_w_code'))

        else:
            latent_after_vit = vit_decode_out
            sr_w_code = None
            vit_decode_out = dict(latent_normalized=latent_after_vit
                                  )  # for later dict update compatability

        # * triplane rendering
        ret_dict = self.triplane_decoder(latent_after_vit,
                                         c,
                                         ws=sr_w_code,
                                         return_raw_only=return_raw_only,
                                         **kwargs)  # triplane latent -> imgs
        ret_dict.update({
            'latent_after_vit': latent_after_vit,
            **vit_decode_out
        })

        return ret_dict

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit
        )  # pred_vit_latent -> patch or original size; B 768 384

        latent = self.unpatchify_triplane(
            latent,
            p=4,
            unpatchify_out_chans=int(
                self.channel_multiplier * self.unpatchify_out_chans //
                3))  # spatial_vit_latent, B, C, H, W (B, 96*2, 16, 16)

        # 4X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr'](latent)  # still B 3C H W

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent_from_vit.shape[0], 0), ))  # type: ignore

        return ret_dict

    def forward_vit_decoder(self, x, img_size=None):
        # latent: (N, L, C) from DINO/CLIP ViT encoder

        # * also dino ViT
        # add positional encoding to each token
        if img_size is None:
            img_size = self.img_size

        # if self.cls_token:
        # st()
        x = x + self.interpolate_pos_encoding(x, img_size,
                                              img_size)[:, :]  # B, L, C

        B, L, C = x.shape  # has [cls] token in N
        x = x.view(B, 3, L // 3, C)

        skips = [x]
        assert self.fusion_blk_start == 0

        # in blks
        for blk in self.vit_decoder.blocks[0:len(self.vit_decoder.blocks) //
                                           2 - 1]:
            x = blk(x)  # B 3 N C
            skips.append(x)

        # mid blks
        # for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks)//2-1:len(self.vit_decoder.blocks)//2+1]:
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2 -
                                           1:len(self.vit_decoder.blocks) //
                                           2]:
            x = blk(x)  # B 3 N C

        # out blks
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2:]:
            x = x + blk.skip_linear(torch.cat([x, skips.pop()],
                                              dim=-1))  # long skip connections
            x = blk(x)  # B 3 N C

        x = self.vit_decoder.norm(x)

        # post process shape
        x = x.view(B, L, C)
        return x


# ! SD version
class RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_SD(
        RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane_fg_bg_plane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder,
                         triplane_decoder,
                         cls_token,
                        #  sr_ratio=sr_ratio, # not used
                         use_fusion_blk=use_fusion_blk,
                         fusion_blk_depth=fusion_blk_depth,
                         fusion_blk=fusion_blk,
                         channel_multiplier=channel_multiplier,
                         **kwargs)

        for k in [
                'ldm_downsample',
                # 'conv_sr'
        ]:
            del self.superresolution[k]

    def vae_reparameterization(self, latent, sample_posterior):
        # latent: B 24 32 32

        assert self.vae_p > 1
        # latents3D = self.unpatchify3D(
        #     latents3D,
        #     p=self.vae_p,
        #     unpatchify_out_chans=self.ldm_z_channels *
        #     2)  # B 3 H W unpatchify_out_chans, H=W=16 now
        # B, C3, H, W = latent.shape
        # latents3D = latent.reshape(B, 3, C3//3, H, W)

        # latents3D = latents3D.permute(0, 1, 4, 2, 3).reshape(B, -1, H,
        #                                                      W)  # B 3C H W

        # ! do VAE here
        posterior = self.vae_encode(latent)  # B self.ldm_z_channels 3 L

        if sample_posterior:
            latent = posterior.sample()
        else:
            latent = posterior.mode()  # B C 3 L

        log_q = posterior.log_p(latent)  # same shape as latent

        # ! for LSGM KL code
        latent_normalized_2Ddiffusion = latent.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16
        log_q_2Ddiffusion = log_q.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16

        latent = latent.permute(0, 2, 3, 1)  # B C 3 L -> B 3 L C

        latent = latent.reshape(latent.shape[0], -1,
                                latent.shape[-1])  # B 3*L C

        ret_dict = dict(
            normal_entropy=posterior.normal_entropy(),
            latent_normalized=latent,
            latent_normalized_2Ddiffusion=latent_normalized_2Ddiffusion,  # 
            log_q_2Ddiffusion=log_q_2Ddiffusion,
            log_q=log_q,
            posterior=posterior,
        )

        return ret_dict


class RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_SD_D(
        RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_SD):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane_fg_bg_plane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.decoder_pred = None  # directly un-patchembed

        self.superresolution.update(
            dict(conv_sr=Decoder(  # serve as Deconv
                resolution=128,
                in_channels=3,
                # ch=64,
                ch=32,
                ch_mult=[1, 2, 2, 4],
                # num_res_blocks=2,
                # ch_mult=[1,2,4],
                num_res_blocks=1,
                dropout=0.0,
                attn_resolutions=[],
                out_ch=32,
                # z_channels=vit_decoder.embed_dim//4,
                z_channels=vit_decoder.embed_dim,
            )))

    # ''' # for SD Decoder, verify encoder first
    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        def unflatten_token(x, p=None):
            B, L, C = x.shape
            x = x.reshape(B, 3, L // 3, C)

            if self.cls_token:  # TODO, how to better use cls token
                x = x[:, :, 1:]  # B 3 256 C

            h = w = int((x.shape[2])**.5)
            assert h * w == x.shape[2]

            if p is None:
                x = x.reshape(shape=(B, 3, h, w, -1))
                x = rearrange(
                    x, 'b n h w c->(b n) c h w'
                )  # merge plane into Batch and prepare for rendering
            else:
                x = x.reshape(shape=(B, 3, h, w, p, p, -1))
                x = rearrange(
                    x, 'b n h w p1 p2 c->(b n) c (h p1) (w p2)'
                )  # merge plane into Batch and prepare for rendering

            return x

        latent = unflatten_token(latent_from_vit)

        # latent = unflatten_token(latent_from_vit, p=2)

        # ! SD SR
        latent = self.superresolution['conv_sr'](latent)  # still B 3C H W
        latent = rearrange(latent, '(b n) c h w->b (n c) h w', n=3)

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        # sr_w_code = self.w_avg
        # assert sr_w_code is not None
        # ret_dict.update(
        #     dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
        #         latent_from_vit.shape[0], 0), ))  # type: ignore

        return ret_dict

    # '''


class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_lite3DAttn(
        RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_SD):

    def __init__(self,
                 vit_decoder: VisionTransformer,
                 triplane_decoder: Triplane_fg_bg_plane,
                 cls_token,
                 normalize_feat=True,
                 sr_ratio=2,
                 use_fusion_blk=True,
                 fusion_blk_depth=2,
                 fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite,
                 channel_multiplier=4,
                 **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)
        # 1. convert output plane token to B L 3 C//3 shape
        # 2. change vit decoder fusion arch (fusion block)
        # 3. output follow B L 3 C//3 with decoder input dim C//3
        # TODO: ablate basic decoder design, on the metrics (input/novelview both)
        self.decoder_pred = nn.Linear(self.vit_decoder.embed_dim // 3,
                                      2048,
                                      bias=True)  # decoder to patch

        # st()
        self.superresolution.update(
            dict(ldm_upsample=PatchEmbedTriplaneRodin(
                self.vae_p * self.token_size,
                self.vae_p,
                3 * self.ldm_embed_dim,  # B 3 L C
                vit_decoder.embed_dim // 3,
                bias=True)))

        # ! original pos_embed
        has_token = bool(self.cls_token)
        self.vit_decoder.pos_embed = nn.Parameter(
            torch.zeros(1, 16 * 16 + has_token, vit_decoder.embed_dim))

    def forward(self, latent, c, img_size):

        latent_normalized = self.vit_decode(latent, img_size)
        return self.triplane_decode(latent_normalized, c)

    def vae_reparameterization(self, latent, sample_posterior):
        # latent: B 24 32 32

        assert self.vae_p > 1

        # ! do VAE here
        # st()
        posterior = self.vae_encode(latent)  # B self.ldm_z_channels 3 L

        if sample_posterior:
            latent = posterior.sample()
        else:
            latent = posterior.mode()  # B C 3 L

        log_q = posterior.log_p(latent)  # same shape as latent

        # ! for LSGM KL code
        latent_normalized_2Ddiffusion = latent.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16
        log_q_2Ddiffusion = log_q.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16

        # TODO, add a conv_after_quant

        # ! reshape for ViT decoder
        latent = latent.permute(0, 3, 1, 2)  # B C 3 L -> B L C 3
        latent = latent.reshape(*latent.shape[:2], -1)  # B L C3

        ret_dict = dict(
            normal_entropy=posterior.normal_entropy(),
            latent_normalized=latent,
            latent_normalized_2Ddiffusion=latent_normalized_2Ddiffusion,  # 
            log_q_2Ddiffusion=log_q_2Ddiffusion,
            log_q=log_q,
            posterior=posterior,
        )

        return ret_dict

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        B, N, C = latent_from_vit.shape
        latent_from_vit = latent_from_vit.reshape(B, N, C // 3, 3).permute(
            0, 3, 1, 2)  # -> B 3 N C//3

        # ! remaining unchanged

        # ViT decoder projection, from MAE
        latent = self.decoder_pred(
            latent_from_vit
        )  # pred_vit_latent -> patch or original size; B 768 384

        latent = latent.reshape(B, 3 * N, -1)  # B L C

        latent = self.unpatchify_triplane(
            latent,
            p=4,
            unpatchify_out_chans=int(
                self.channel_multiplier * self.unpatchify_out_chans //
                3))  # spatial_vit_latent, B, C, H, W (B, 96*2, 16, 16)

        # 4X SR with Rodin Conv 3D
        latent = self.superresolution['conv_sr'](latent)  # still B 3C H W

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        sr_w_code = self.w_avg
        assert sr_w_code is not None
        ret_dict.update(
            dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
                latent_from_vit.shape[0], 0), ))  # type: ignore

        return ret_dict

    def vit_decode_backbone(self, latent, img_size):
        # assert x.ndim == 3  # N L C
        if isinstance(latent, dict):
            latent = latent['latent_normalized_2Ddiffusion']  # B, C*3, H, W

        # assert latent.shape == (
        #     latent.shape[0], 3 * (self.token_size * self.vae_p)**2,
        #     self.ldm_z_channels), f'latent.shape: {latent.shape}'

        # st() # latent: B 12 32 32
        latent = self.superresolution['ldm_upsample'](  # ! B 768 (3*256) 768 
            latent)  # torch.Size([8, 12, 32, 32]) => torch.Size([8, 256, 768])
        # latent: torch.Size([8, 768, 768])

        B, N3, C = latent.shape
        latent = latent.reshape(B, 3, N3 // 3,
                                C).permute(0, 2, 3, 1)  # B 3HW C -> B HW C 3
        latent = latent.reshape(*latent.shape[:2], -1)  # B HW C3

        # ! directly feed to vit_decoder
        return self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

    def forward_vit_decoder(self, x, img_size=None):
        # latent: (N, L, C) from DINO/CLIP ViT encoder

        # * also dino ViT
        # add positional encoding to each token
        if img_size is None:
            img_size = self.img_size

        # if self.cls_token:
        x = x + self.interpolate_pos_encoding(x, img_size,
                                              img_size)[:, :]  # B, L, C

        B, L, C = x.shape  # has [cls] token in N

        # ! no need to reshape here
        # x = x.view(B, 3, L // 3, C)

        skips = [x]
        assert self.fusion_blk_start == 0

        # in blks
        for blk in self.vit_decoder.blocks[0:len(self.vit_decoder.blocks) //
                                           2 - 1]:
            x = blk(x)  # B 3 N C
            skips.append(x)

        # mid blks
        # for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks)//2-1:len(self.vit_decoder.blocks)//2+1]:
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2 -
                                           1:len(self.vit_decoder.blocks) //
                                           2]:
            x = blk(x)  # B 3 N C

        # out blks
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2:]:
            x = x + blk.skip_linear(torch.cat([x, skips.pop()],
                                              dim=-1))  # long skip connections
            x = blk(x)  # B 3 N C

        x = self.vit_decoder.norm(x)

        # post process shape
        x = x.view(B, L, C)
        return x

    def create_fusion_blks(self, fusion_blk_depth, use_fusion_blk, fusion_blk):

        vit_decoder_blks = self.vit_decoder.blocks
        assert len(vit_decoder_blks) == 12, 'ViT-B by default'

        nh = self.vit_decoder.blocks[
            0].attn.num_heads // 3  # ! lighter, actually divisible by 4
        dim = self.vit_decoder.embed_dim // 3  # ! separate

        fusion_blk_start = self.fusion_blk_start
        triplane_fusion_vit_blks = nn.ModuleList()

        if fusion_blk_start != 0:
            for i in range(0, fusion_blk_start):
                triplane_fusion_vit_blks.append(
                    vit_decoder_blks[i])  # append all vit blocks in the front

        for i in range(fusion_blk_start, len(vit_decoder_blks),
                       fusion_blk_depth):
            vit_blks_group = vit_decoder_blks[i:i +
                                              fusion_blk_depth]  # moduleList
            triplane_fusion_vit_blks.append(
                # TriplaneFusionBlockv2(vit_blks_group, nh, dim, use_fusion_blk))
                fusion_blk(vit_blks_group, nh, dim, use_fusion_blk))

        self.vit_decoder.blocks = triplane_fusion_vit_blks
        # self.vit_decoder.blocks = triplane_fusion_vit_blks


# default for objaverse
class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S(
        RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(
            vit_decoder,
            triplane_decoder,
            cls_token,
            use_fusion_blk=use_fusion_blk,
            fusion_blk_depth=fusion_blk_depth,
            fusion_blk=fusion_blk,
            channel_multiplier=channel_multiplier,
            patch_size=-1,  # placeholder, since we use dit here
            token_size=2,
            **kwargs)
        self.D_roll_out_input = False

        for k in [
                'ldm_downsample',
                # 'conv_sr'
        ]:
            del self.superresolution[k]

        self.decoder_pred = None  # directly un-patchembed
        self.superresolution.update(
            dict(
                conv_sr=Decoder(  # serve as Deconv
                    resolution=128,
                    # resolution=256,
                    in_channels=3,
                    # ch=64,
                    ch=32,
                    # ch=16,
                    ch_mult=[1, 2, 2, 4],
                    # ch_mult=[1, 1, 2, 2],
                    # num_res_blocks=2,
                    # ch_mult=[1,2,4],
                    # num_res_blocks=0,
                    num_res_blocks=1,
                    dropout=0.0,
                    attn_resolutions=[],
                    out_ch=32,
                    # z_channels=vit_decoder.embed_dim//4,
                    z_channels=vit_decoder.embed_dim,
                    # z_channels=vit_decoder.embed_dim//2,
                ),
                # after_vit_upsampler=Upsample2D(channels=vit_decoder.embed_dim,use_conv=True, use_conv_transpose=False, out_channels=vit_decoder.embed_dim//2)
            ))

        # del skip_lienar
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2:]:
            del blk.skip_linear

    @torch.inference_mode()
    def forward_points(self,
                       planes,
                       points: torch.Tensor,
                       chunk_size: int = 2**16):
        # planes: (N, 3, D', H', W')
        # points: (N, P, 3)
        N, P = points.shape[:2]
        if planes.ndim == 4:
            planes = planes.reshape(
                len(planes),
                3,
                -1,  # ! support background plane
                planes.shape[-2],
                planes.shape[-1])  # BS 96 256 256

        # query triplane in chunks
        outs = []
        for i in trange(0, points.shape[1], chunk_size):
            chunk_points = points[:, i:i + chunk_size]

            # query triplane
            # st()
            chunk_out = self.triplane_decoder.renderer._run_model(  # type: ignore
                planes=planes,
                decoder=self.triplane_decoder.decoder,
                sample_coordinates=chunk_points,
                sample_directions=torch.zeros_like(chunk_points),
                options=self.rendering_kwargs,
            )
            # st()

            outs.append(chunk_out)
            torch.cuda.empty_cache()

        # st()

        # concatenate the outputs
        point_features = {
            k: torch.cat([out[k] for out in outs], dim=1)
            for k in outs[0].keys()
        }
        return point_features

    def triplane_decode_grid(self,
                             vit_decode_out,
                             grid_size,
                             aabb: torch.Tensor = None,
                             **kwargs):
        # planes: (N, 3, D', H', W')
        # grid_size: int

        assert isinstance(vit_decode_out, dict)
        planes = vit_decode_out['latent_after_vit']

        # aabb: (N, 2, 3)
        if aabb is None:
            if 'sampler_bbox_min' in self.rendering_kwargs:
                aabb = torch.tensor([
                    [self.rendering_kwargs['sampler_bbox_min']] * 3,
                    [self.rendering_kwargs['sampler_bbox_max']] * 3,
                ],
                                    device=planes.device,
                                    dtype=planes.dtype).unsqueeze(0).repeat(
                                        planes.shape[0], 1, 1)
            else:  # shapenet dataset, follow eg3d
                aabb = torch.tensor(
                    [  # https://github.com/NVlabs/eg3d/blob/7cf1fd1e99e1061e8b6ba850f91c94fe56e7afe4/eg3d/gen_samples.py#L188
                        [-self.rendering_kwargs['box_warp'] / 2] * 3,
                        [self.rendering_kwargs['box_warp'] / 2] * 3,
                    ],
                    device=planes.device,
                    dtype=planes.dtype).unsqueeze(0).repeat(
                        planes.shape[0], 1, 1)

        assert planes.shape[0] == aabb.shape[
            0], "Batch size mismatch for planes and aabb"
        N = planes.shape[0]

        # create grid points for triplane query
        grid_points = []
        for i in range(N):
            grid_points.append(
                torch.stack(torch.meshgrid(
                    torch.linspace(aabb[i, 0, 0],
                                   aabb[i, 1, 0],
                                   grid_size,
                                   device=planes.device),
                    torch.linspace(aabb[i, 0, 1],
                                   aabb[i, 1, 1],
                                   grid_size,
                                   device=planes.device),
                    torch.linspace(aabb[i, 0, 2],
                                   aabb[i, 1, 2],
                                   grid_size,
                                   device=planes.device),
                    indexing='ij',
                ),
                            dim=-1).reshape(-1, 3))
        cube_grid = torch.stack(grid_points, dim=0).to(planes.device)  # 1 N 3 

        features = self.forward_points(planes, cube_grid)

        # reshape into grid
        features = {
            k: v.reshape(N, grid_size, grid_size, grid_size, -1)
            for k, v in features.items()
        }

        # st()

        return features

    def create_fusion_blks(self, fusion_blk_depth, use_fusion_blk, fusion_blk):
        # no need to fuse anymore
        pass

    def forward_vit_decoder(self, x, img_size=None):
        # st()
        return self.vit_decoder(x)

    def vit_decode_backbone(self, latent, img_size):
        # assert x.ndim == 3  # N L C
        if isinstance(latent, dict):
            latent = latent['latent_normalized_2Ddiffusion']  # B, C*3, H, W

        # assert latent.shape == (
        #     latent.shape[0], 3 * (self.token_size * self.vae_p)**2,
        #     self.ldm_z_channels), f'latent.shape: {latent.shape}'

        # st() # latent: B 12 32 32
        # st()
        latent = self.superresolution['ldm_upsample'](  # ! B 768 (3*256) 768 
            latent)  # torch.Size([8, 12, 32, 32]) => torch.Size([8, 256, 768])
        # latent: torch.Size([8, 768, 768])

        # ! directly feed to vit_decoder
        return self.forward_vit_decoder(latent, img_size)  # pred_vit_latent

    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        def unflatten_token(x, p=None):
            B, L, C = x.shape
            x = x.reshape(B, 3, L // 3, C)

            if self.cls_token:  # TODO, how to better use cls token
                x = x[:, :, 1:]  # B 3 256 C

            h = w = int((x.shape[2])**.5)
            assert h * w == x.shape[2]

            if p is None:
                x = x.reshape(shape=(B, 3, h, w, -1))
                if not self.D_roll_out_input:
                    x = rearrange(
                        x, 'b n h w c->(b n) c h w'
                    )  # merge plane into Batch and prepare for rendering
                else:
                    x = rearrange(
                        x, 'b n h w c->b c h (n w)'
                    )  # merge plane into Batch and prepare for rendering
            else:
                x = x.reshape(shape=(B, 3, h, w, p, p, -1))
                if self.D_roll_out_input:
                    x = rearrange(
                        x, 'b n h w p1 p2 c->b c (h p1) (n w p2)'
                    )  # merge plane into Batch and prepare for rendering
                else:
                    x = rearrange(
                        x, 'b n h w p1 p2 c->(b n) c (h p1) (w p2)'
                    )  # merge plane into Batch and prepare for rendering

            return x

        latent = unflatten_token(
            latent_from_vit)  # B 3 h w vit_decoder.embed_dim

        # ! x2 upsapmle, 16 -32 before sending into SD Decoder
        # latent = self.superresolution['after_vit_upsampler'](latent) # B*3 192 32 32

        # latent = unflatten_token(latent_from_vit, p=2)

        # ! SD SR
        latent = self.superresolution['conv_sr'](latent)  # still B 3C H W
        if not self.D_roll_out_input:
            latent = rearrange(latent, '(b n) c h w->b (n c) h w', n=3)
        else:
            latent = rearrange(latent, 'b c h (n w)->b (n c) h w', n=3)

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        # sr_w_code = self.w_avg
        # assert sr_w_code is not None
        # ret_dict.update(
        #     dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
        #         latent_from_vit.shape[0], 0), ))  # type: ignore

        return ret_dict

    def vae_reparameterization(self, latent, sample_posterior):
        # latent: B 24 32 32

        assert self.vae_p > 1
        # latents3D = self.unpatchify3D(
        #     latents3D,
        #     p=self.vae_p,
        #     unpatchify_out_chans=self.ldm_z_channels *
        #     2)  # B 3 H W unpatchify_out_chans, H=W=16 now
        # B, C3, H, W = latent.shape
        # latents3D = latent.reshape(B, 3, C3//3, H, W)

        # latents3D = latents3D.permute(0, 1, 4, 2, 3).reshape(B, -1, H,
        #                                                      W)  # B 3C H W

        # ! do VAE here
        posterior = self.vae_encode(latent)  # B self.ldm_z_channels 3 L

        if sample_posterior:
            latent = posterior.sample()
        else:
            latent = posterior.mode()  # B C 3 L

        log_q = posterior.log_p(latent)  # same shape as latent

        # ! for LSGM KL code
        latent_normalized_2Ddiffusion = latent.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16
        log_q_2Ddiffusion = log_q.reshape(
            latent.shape[0], -1, self.token_size * self.vae_p,
            self.token_size * self.vae_p)  # B, 3*4, 16 16
        # st()

        latent = latent.permute(0, 2, 3, 1)  # B C 3 L -> B 3 L C

        latent = latent.reshape(latent.shape[0], -1,
                                latent.shape[-1])  # B 3*L C

        ret_dict = dict(
            normal_entropy=posterior.normal_entropy(),
            latent_normalized=latent,
            latent_normalized_2Ddiffusion=latent_normalized_2Ddiffusion,  # 
            log_q_2Ddiffusion=log_q_2Ddiffusion,
            log_q=log_q,
            posterior=posterior,
        )

        return ret_dict

    def vit_decode(self, latent, img_size, sample_posterior=True, **kwargs):
        return super().vit_decode(latent, img_size, sample_posterior)

# objv class


class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout(
        RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn_SD):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)


# final version, above + SD-Decoder
class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D(
        RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout
):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         **kwargs)

        self.decoder_pred = None  # directly un-patchembed
        self.superresolution.update(
            dict(
                conv_sr=Decoder(  # serve as Deconv
                    resolution=128,
                    # resolution=256,
                    in_channels=3,
                    # ch=64,
                    ch=32,
                    # ch=16,
                    ch_mult=[1, 2, 2, 4],
                    # ch_mult=[1, 1, 2, 2],
                    # num_res_blocks=2,
                    # ch_mult=[1,2,4],
                    # num_res_blocks=0,
                    num_res_blocks=1,
                    dropout=0.0,
                    attn_resolutions=[],
                    out_ch=32,
                    # z_channels=vit_decoder.embed_dim//4,
                    z_channels=vit_decoder.embed_dim,
                    # z_channels=vit_decoder.embed_dim//2,
                ),
                # after_vit_upsampler=Upsample2D(channels=vit_decoder.embed_dim,use_conv=True, use_conv_transpose=False, out_channels=vit_decoder.embed_dim//2)
            ))
        self.D_roll_out_input = False

    # ''' # for SD Decoder
    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):

        if self.cls_token:
            cls_token = latent_from_vit[:, :1]
        else:
            cls_token = None

        def unflatten_token(x, p=None):
            B, L, C = x.shape
            x = x.reshape(B, 3, L // 3, C)

            if self.cls_token:  # TODO, how to better use cls token
                x = x[:, :, 1:]  # B 3 256 C

            h = w = int((x.shape[2])**.5)
            assert h * w == x.shape[2]

            if p is None:
                x = x.reshape(shape=(B, 3, h, w, -1))
                if not self.D_roll_out_input:
                    x = rearrange(
                        x, 'b n h w c->(b n) c h w'
                    )  # merge plane into Batch and prepare for rendering
                else:
                    x = rearrange(
                        x, 'b n h w c->b c h (n w)'
                    )  # merge plane into Batch and prepare for rendering
            else:
                x = x.reshape(shape=(B, 3, h, w, p, p, -1))
                if self.D_roll_out_input:
                    x = rearrange(
                        x, 'b n h w p1 p2 c->b c (h p1) (n w p2)'
                    )  # merge plane into Batch and prepare for rendering
                else:
                    x = rearrange(
                        x, 'b n h w p1 p2 c->(b n) c (h p1) (w p2)'
                    )  # merge plane into Batch and prepare for rendering

            return x

        latent = unflatten_token(
            latent_from_vit)  # B 3 h w vit_decoder.embed_dim

        # ! x2 upsapmle, 16 -32 before sending into SD Decoder
        # latent = self.superresolution['after_vit_upsampler'](latent) # B*3 192 32 32

        # latent = unflatten_token(latent_from_vit, p=2)

        # ! SD SR
        latent = self.superresolution['conv_sr'](latent)  # still B 3C H W
        if not self.D_roll_out_input:
            latent = rearrange(latent, '(b n) c h w->b (n c) h w', n=3)
        else:
            latent = rearrange(latent, 'b c h (n w)->b (n c) h w', n=3)

        ret_dict.update(dict(cls_token=cls_token, latent_after_vit=latent))

        # include the w_avg for now
        # sr_w_code = self.w_avg
        # assert sr_w_code is not None
        # ret_dict.update(
        #     dict(sr_w_code=sr_w_code.reshape(1, 1, -1).repeat_interleave(
        #         latent_from_vit.shape[0], 0), ))  # type: ignore

        return ret_dict

    # '''


class RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder(
        RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D
):

    def __init__(
            self,
            vit_decoder: VisionTransformer,
            triplane_decoder: Triplane_fg_bg_plane,
            cls_token,
            normalize_feat=True,
            sr_ratio=2,
            use_fusion_blk=True,
            fusion_blk_depth=2,
            fusion_blk=TriplaneFusionBlockv4_nested_init_from_dino_lite_merge_B_3L_C_withrollout,
            channel_multiplier=4,
            **kwargs) -> None:
        super().__init__(vit_decoder, triplane_decoder, cls_token,
                         normalize_feat, sr_ratio, use_fusion_blk,
                         fusion_blk_depth, fusion_blk, channel_multiplier,
                         patch_size=-1,
                         **kwargs)

        # del skip_lienar
        for blk in self.vit_decoder.blocks[len(self.vit_decoder.blocks) // 2:]:
            del blk.skip_linear

    @torch.inference_mode()
    def forward_points(self,
                       planes,
                       points: torch.Tensor,
                       chunk_size: int = 2**16):
        # planes: (N, 3, D', H', W')
        # points: (N, P, 3)
        N, P = points.shape[:2]
        if planes.ndim == 4:
            planes = planes.reshape(
                len(planes),
                3,
                -1,  # ! support background plane
                planes.shape[-2],
                planes.shape[-1])  # BS 96 256 256

        # query triplane in chunks
        outs = []
        for i in trange(0, points.shape[1], chunk_size):
            chunk_points = points[:, i:i + chunk_size]

            # query triplane
            # st()
            chunk_out = self.triplane_decoder.renderer._run_model(  # type: ignore
                planes=planes,
                decoder=self.triplane_decoder.decoder,
                sample_coordinates=chunk_points,
                sample_directions=torch.zeros_like(chunk_points),
                options=self.rendering_kwargs,
            )
            # st()

            outs.append(chunk_out)
            torch.cuda.empty_cache()

        # st()

        # concatenate the outputs
        point_features = {
            k: torch.cat([out[k] for out in outs], dim=1)
            for k in outs[0].keys()
        }
        return point_features

    def triplane_decode_grid(self,
                             vit_decode_out,
                             grid_size,
                             aabb: torch.Tensor = None,
                             **kwargs):
        # planes: (N, 3, D', H', W')
        # grid_size: int

        assert isinstance(vit_decode_out, dict)
        planes = vit_decode_out['latent_after_vit']

        # aabb: (N, 2, 3)
        if aabb is None:
            if 'sampler_bbox_min' in self.rendering_kwargs:
                aabb = torch.tensor([
                    [self.rendering_kwargs['sampler_bbox_min']] * 3,
                    [self.rendering_kwargs['sampler_bbox_max']] * 3,
                ],
                                    device=planes.device,
                                    dtype=planes.dtype).unsqueeze(0).repeat(
                                        planes.shape[0], 1, 1)
            else:  # shapenet dataset, follow eg3d
                aabb = torch.tensor(
                    [  # https://github.com/NVlabs/eg3d/blob/7cf1fd1e99e1061e8b6ba850f91c94fe56e7afe4/eg3d/gen_samples.py#L188
                        [-self.rendering_kwargs['box_warp'] / 2] * 3,
                        [self.rendering_kwargs['box_warp'] / 2] * 3,
                    ],
                    device=planes.device,
                    dtype=planes.dtype).unsqueeze(0).repeat(
                        planes.shape[0], 1, 1)

        assert planes.shape[0] == aabb.shape[
            0], "Batch size mismatch for planes and aabb"
        N = planes.shape[0]

        # create grid points for triplane query
        grid_points = []
        for i in range(N):
            grid_points.append(
                torch.stack(torch.meshgrid(
                    torch.linspace(aabb[i, 0, 0],
                                   aabb[i, 1, 0],
                                   grid_size,
                                   device=planes.device),
                    torch.linspace(aabb[i, 0, 1],
                                   aabb[i, 1, 1],
                                   grid_size,
                                   device=planes.device),
                    torch.linspace(aabb[i, 0, 2],
                                   aabb[i, 1, 2],
                                   grid_size,
                                   device=planes.device),
                    indexing='ij',
                ),
                            dim=-1).reshape(-1, 3))
        cube_grid = torch.stack(grid_points, dim=0).to(planes.device)  # 1 N 3
        # st()

        features = self.forward_points(planes, cube_grid)

        # reshape into grid
        features = {
            k: v.reshape(N, grid_size, grid_size, grid_size, -1)
            for k, v in features.items()
        }

        # st()

        return features

    def create_fusion_blks(self, fusion_blk_depth, use_fusion_blk, fusion_blk):
        # no need to fuse anymore
        pass

    def forward_vit_decoder(self, x, img_size=None):
        # st()
        return self.vit_decoder(x)

    def vit_decode_backbone(self, latent, img_size):
        return super().vit_decode_backbone(latent, img_size)

    # ! flag2
    def vit_decode_postprocess(self, latent_from_vit, ret_dict: dict):
        return super().vit_decode_postprocess(latent_from_vit, ret_dict)

    def vae_reparameterization(self, latent, sample_posterior):
        return super().vae_reparameterization(latent, sample_posterior)
