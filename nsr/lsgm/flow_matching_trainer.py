"""
https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/diffusion/ddpm.py#L30
"""
import copy
import functools
import json
import os
from pathlib import Path
from pdb import set_trace as st
from typing import Any
from click import prompt
import einops
import blobfile as bf
import imageio
import numpy as np
import torch as th
import torch.distributed as dist
import torchvision
from PIL import Image
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
from guided_diffusion.resample import LossAwareSampler, UniformSampler
# from .train_util import TrainLoop3DRec
from guided_diffusion.train_util import (TrainLoop, calc_average_loss,
                                         find_ema_checkpoint,
                                         find_resume_checkpoint,
                                         get_blob_logdir, log_loss_dict,
                                         log_rec3d_loss_dict,
                                         parse_resume_step_from_filename)
from guided_diffusion.gaussian_diffusion import ModelMeanType

from ldm.modules.encoders.modules import FrozenClipImageEmbedder, TextEmbedder, FrozenCLIPTextEmbedder, FrozenOpenCLIPImagePredictionEmbedder, FrozenOpenCLIPImageEmbedder

import dnnlib
from dnnlib.util import requires_grad
from dnnlib.util import calculate_adaptive_weight

from ..train_util_diffusion import TrainLoop3DDiffusion
from ..cvD.nvsD_canoD import TrainLoop3DcvD_nvsD_canoD

from guided_diffusion.continuous_diffusion_utils import get_mixed_prediction, different_p_q_objectives, kl_per_group_vada, kl_balancer
# from .train_util_diffusion_lsgm_noD_joint import TrainLoop3DDiffusionLSGMJointnoD  # joint diffusion and rec class
# from .controlLDM import TrainLoop3DDiffusionLSGM_Control  # joint diffusion and rec class
from .train_util_diffusion_lsgm_noD_joint import TrainLoop3DDiffusionLSGMJointnoD  # joint diffusion and rec class

# ! add new schedulers from https://github.com/Stability-AI/generative-models

from .crossattn_cldm import TrainLoop3DDiffusionLSGM_crossattn

# import SD stuffs
from typing import Any, Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
from omegaconf import ListConfig, OmegaConf
from sgm.modules import UNCONDITIONAL_CONFIG

from sgm.util import (default, disabled_train, get_obj_from_str,
                      instantiate_from_config, log_txt_as_img)

from transport import create_transport, Sampler

# from sgm.sampling_utils.demo.streamlit_helpers import init_sampling


class FlowMatchingEngine(TrainLoop3DDiffusionLSGM_crossattn):

    def __init__(
        self,
        *,
        rec_model,
        denoise_model,
        diffusion,
        sde_diffusion,
        control_model,
        control_key,
        only_mid_control,
        loss_class,
        data,
        eval_data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        eval_interval,
        save_interval,
        resume_checkpoint,
        resume_cldm_checkpoint=None,
        use_fp16=False,
        fp16_scale_growth=0.001,
        schedule_sampler=None,
        weight_decay=0,
        lr_anneal_steps=0,
        iterations=10001,
        ignore_resume_opt=False,
        freeze_ae=False,
        denoised_ae=True,
        triplane_scaling_divider=10,
        use_amp=False,
        diffusion_input_size=224,
        normalize_clip_encoding=False,
        scale_clip_encoding=1,
        cfg_dropout_prob=0,
        cond_key='img_sr',
        use_eos_feature=False,
        compile=False,
        snr_type='lognorm',
        # denoiser_config,
        # conditioner_config: Union[None, Dict, ListConfig,
        #                           OmegaConf] = None,
        # sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        # loss_fn_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        **kwargs):
        super().__init__(rec_model=rec_model,
                         denoise_model=denoise_model,
                         diffusion=diffusion,
                         sde_diffusion=sde_diffusion,
                         control_model=control_model,
                         control_key=control_key,
                         only_mid_control=only_mid_control,
                         loss_class=loss_class,
                         data=data,
                         eval_data=eval_data,
                         batch_size=batch_size,
                         microbatch=microbatch,
                         lr=lr,
                         ema_rate=ema_rate,
                         log_interval=log_interval,
                         eval_interval=eval_interval,
                         save_interval=save_interval,
                         resume_checkpoint=resume_checkpoint,
                         resume_cldm_checkpoint=resume_cldm_checkpoint,
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         schedule_sampler=schedule_sampler,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         ignore_resume_opt=ignore_resume_opt,
                         freeze_ae=freeze_ae,
                         denoised_ae=denoised_ae,
                         triplane_scaling_divider=triplane_scaling_divider,
                         use_amp=use_amp,
                         diffusion_input_size=diffusion_input_size,
                         normalize_clip_encoding=normalize_clip_encoding,
                         scale_clip_encoding=scale_clip_encoding,
                         cfg_dropout_prob=cfg_dropout_prob,
                         cond_key=cond_key,
                         use_eos_feature=use_eos_feature,
                         compile=compile,
                         **kwargs)

        #  ! sgm diffusion pipeline
        # ! reuse the conditioner
        if self.cond_key == 'caption':
            ldm_configs = OmegaConf.load(
                'sgm/configs/t23d-clipl-compat-fm.yaml')['ldm_configs']
        else:
            assert 'lognorm' in snr_type
            if snr_type == 'lognorm': # by default
                ldm_configs = OmegaConf.load(
                    'sgm/configs/img23d-clipl-compat-fm-lognorm.yaml')['ldm_configs']
            elif snr_type == 'lognorm-mv':
                ldm_configs = OmegaConf.load(
                    'sgm/configs/mv23d-clipl-compat-fm-lognorm.yaml')['ldm_configs']
            else:
                ldm_configs = OmegaConf.load(
                    'sgm/configs/img23d-clipl-compat-fm.yaml')['ldm_configs']

        self.loss_fn = (
            instantiate_from_config(ldm_configs.loss_fn_config)
            # if loss_fn_config is not None
            # else None
        )

        # self.denoiser = instantiate_from_config(
        #     ldm_configs.denoiser_config).to(dist_util.dev())

        self.transport_sampler = Sampler(self.loss_fn.transport, guider_config=ldm_configs.guider_config)

        self.conditioner = instantiate_from_config(
            default(ldm_configs.conditioner_config,
                    UNCONDITIONAL_CONFIG)).to(dist_util.dev())

        # ! setup optimizer (with cond embedder params here)
        self._setup_opt2()
        self._load_model2()

    def _setup_opt(self):
        pass # see below

    def _setup_opt2(self):
        # ! add trainable conditioner parameters
        # https://github.com/Stability-AI/generative-models/blob/fbdc58cab9f4ee2be7a5e1f2e2787ecd9311942f/sgm/models/diffusion.py#L219

        # params = list(self.ddpm_model.parameters())

        self.opt = AdamW([{
            'name': 'ddpm',
            'params': self.ddpm_model.parameters(),
        },
        ],
                         lr=self.lr,
                         weight_decay=self.weight_decay)
        
        
        embedder_params = []
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                embedder_params = embedder_params + list(embedder.parameters())


        if len(embedder_params) != 0:
            self.opt.add_param_group(
                {
                    'name': 'embedder',
                    'params': embedder_params,
                    'lr': self.lr*0.1, # smaller lr to finetune dino/clip
                }
            )

        # if self.train_vae:
        #     for rec_param_group in self._init_optim_groups(self.rec_model):
        #         self.opt.add_param_group(rec_param_group)

        print(self.opt)

    def save(self, mp_trainer=None, model_name='ddpm'):
        # save embedder params also
        super().save(mp_trainer, model_name)

        # save embedder ckpt
        if dist_util.get_rank() == 0:
            for embedder in self.conditioner.embedders:
                if embedder.is_trainable:
                    # embedder_params = embedder_params + list(embedder.parameters())
                    model_name = embedder.__class__.__name__
                    filename = f"embedder_{model_name}{(self.step+self.resume_step):07d}.pt"
                    with bf.BlobFile(bf.join(get_blob_logdir(), filename),
                                        "wb") as f:
                        th.save(embedder.state_dict(), f)

        dist_util.synchronize()

    def _load_model2(self):

        # ! load embedder
        for embedder in self.conditioner.embedders:
            if embedder.is_trainable:
                # embedder_params = embedder_params + list(embedder.parameters())
                model_name = embedder.__class__.__name__
                filename = f"embedder_{model_name}{(self.step+self.resume_step):07d}.pt"
                # embedder_FrozenDinov2ImageEmbedderMV2115000.pt

                # with bf.BlobFile(bf.join(get_blob_logdir(), filename),
                #                     "wb") as f:
                #     th.save(embedder.state_dict(), f)

                split = self.resume_checkpoint.split("model")
                resume_checkpoint = str(
                    Path(split[0]) / filename)
                if os.path.exists(resume_checkpoint):
                    if dist.get_rank() == 0:
                        logger.log(
                            f"loading cond embedder from checkpoint: {resume_checkpoint}...")
                        # if model is None:
                        #     model = self.model
                        embedder.load_state_dict(
                            dist_util.load_state_dict(
                                resume_checkpoint,
                                map_location=dist_util.dev(),
                            ))
                else:
                    logger.log(f'{resume_checkpoint} not found.')

                if dist_util.get_world_size() > 1:
                    dist_util.sync_params(embedder.parameters())


    def instantiate_cond_stage(self, normalize_clip_encoding,
                               scale_clip_encoding, cfg_dropout_prob,
                               use_eos_feature):
        # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/diffusion/ddpm.py#L509C1-L509C46
        # self.cond_stage_model.train = disabled_train  # type: ignore
        # if self.cond_key == 'caption':
        #     self.cond_txt_model = TextEmbedder(dropout_prob=cfg_dropout_prob,
        #                                        use_eos_feature=use_eos_feature)
        # elif self.cond_key == 'img':
        #     self.cond_img_model = FrozenOpenCLIPImagePredictionEmbedder(
        #         1, 1,
        #         FrozenOpenCLIPImageEmbedder(freeze=True,
        #                                     device=dist_util.dev(),
        #                                     init_device=dist_util.dev()))

        # else:  # zero-shot Text to 3D using normalized clip latent
        #     self.cond_stage_model = FrozenClipImageEmbedder(
        #         'ViT-L/14',
        #         dropout_prob=cfg_dropout_prob,
        #         normalize_encoding=normalize_clip_encoding,
        #         scale_clip_encoding=scale_clip_encoding)
        #     self.cond_stage_model.freeze()

        #     self.cond_txt_model = FrozenCLIPTextEmbedder(
        #         dropout_prob=cfg_dropout_prob,
        #         scale_clip_encoding=scale_clip_encoding)
        #     self.cond_txt_model.freeze()
        pass # initialized in the self.__init__() using SD api



    # ! already merged
    def prepare_ddpm(self, eps, mode='p'):
        raise NotImplementedError('already implemented in self.denoiser')

    # merged from noD.py

    # use sota denoiser, loss_fn etc.
    def ldm_train_step(self, batch, behaviour='cano', *args, **kwargs):
        """
        add sds grad to all ae predicted x_0 
        """

        # ! enable the gradient of both models
        requires_grad(self.ddpm_model, True)

        self.mp_trainer.zero_grad()  # !!!!

        if 'img' in batch:
            batch_size = batch['img'].shape[0]
        else:
            batch_size = len(batch['caption'])

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k:
                v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
                    v, th.Tensor) else v
                for k, v in batch.items()
            }

            # move condition to self.dtype
            # =================================== ae part ===================================
            # with th.cuda.amp.autocast(dtype=th.bfloat16,
            with th.cuda.amp.autocast(dtype=self.dtype,
                                      enabled=self.mp_trainer.use_amp):

                loss = th.tensor(0.).to(dist_util.dev())

                assert 'latent' in micro
                vae_out = {self.latent_name: micro['latent']}
                # else:
                #     vae_out = self.ddp_rec_model(
                #         img=micro['img_to_encoder'],
                #         c=micro['c'],
                #         behaviour='encoder_vae',
                #     )  # pred: (B, 3, 64, 64)

                eps = vae_out[self.latent_name] / self.triplane_scaling_divider
                # eps = vae_out.pop(self.latent_name)

                # if 'bg_plane' in vae_out:
                #     eps = th.cat((eps, vae_out['bg_plane']),
                #                  dim=1)  # include background, B 12+4 32 32

                # ! SD loss
                # cond = self.get_c_input(micro, bs=eps.shape[0])

                micro['img-c'] = {
                    'img': micro['img'].to(self.dtype),
                    'c': micro['c'].to(self.dtype),
                }

                loss, loss_other_info = self.loss_fn(self.ddp_ddpm_model,
                                                    #  self.denoiser,
                                                     self.conditioner, 
                                                     eps.to(self.dtype),
                                                     micro)  # type: ignore
                loss = loss.mean()
                log_rec3d_loss_dict({})

                log_rec3d_loss_dict({
                    # 'eps_mean':
                    # eps.mean(),
                    # 'eps_std':
                    # eps.std([1, 2, 3]).mean(0),
                    # 'pred_x0_std':
                    # loss_other_info['model_output'].std([1, 2, 3]).mean(0),
                    "p_loss":
                    loss,
                })

            self.mp_trainer.backward(loss)  # joint gradient descent

        # update ddpm accordingly
        self.mp_trainer.optimize(self.opt)

        # ! directly eval_cldm() for sampling.
        # if dist_util.get_rank() == 0 and self.step % 500 == 0:
        #     self.log_control_images(vae_out, micro, loss_other_info)

    @th.inference_mode()
    def log_control_images(self, vae_out, micro, ddpm_ret):

        if 'posterior' in vae_out:
            vae_out.pop('posterior')  # for calculating kl loss
        vae_out_for_pred = {self.latent_name: vae_out[self.latent_name][0:1].to(self.dtype)}

        with th.cuda.amp.autocast(dtype=self.dtype,
                                    enabled=self.mp_trainer.use_amp):
            pred = self.ddp_rec_model(latent=vae_out_for_pred,
                                    c=micro['c'][0:1],
                                    behaviour=self.render_latent_behaviour)

        assert isinstance(pred, dict)

        pred_img = pred['image_raw']
        if 'img' in micro:
            gt_img = micro['img']
        else:
            gt_img = th.zeros_like(pred['image_raw'])

        if 'depth' in micro:
            gt_depth = micro['depth']
            if gt_depth.ndim == 3:
                gt_depth = gt_depth.unsqueeze(1)
            gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
                                                      gt_depth.min())
        else:
            gt_depth = th.zeros_like(gt_img[:, 0:1, ...])

        if 'image_depth' in pred:
            pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())
        else:
            pred_depth = th.zeros_like(gt_depth)

        gt_img = self.pool_128(gt_img)
        gt_depth = self.pool_128(gt_depth)
        # cond = self.get_c_input(micro)
        # hint = th.cat(cond['c_concat'], 1)

        gt_vis = th.cat(
            [
                gt_img,
                gt_img,
                gt_img,
                # self.pool_128(hint),
                # gt_img,
                gt_depth.repeat_interleave(3, dim=1)
            ],
            dim=-1)[0:1]  # TODO, fail to load depth. range [0, 1]

        # eps_t_p_3D = eps_t_p.reshape(batch_size, eps_t_p.shape[1]//3, 3, -1) # B C 3 L

        # self.sampler

        noised_latent, sigmas, x_start = [
            ddpm_ret[k] for k in ['noised_input', 'sigmas', 'model_output']
        ]

        noised_latent = {
            'latent_normalized_2Ddiffusion':
            noised_latent[0:1].to(self.dtype) * self.triplane_scaling_divider,
        }

        denoised_latent = {
            'latent_normalized_2Ddiffusion':
            x_start[0:1].to(self.dtype) * self.triplane_scaling_divider,
        }
         
        with th.cuda.amp.autocast(dtype=self.dtype,
                                    enabled=self.mp_trainer.use_amp):
            noised_ae_pred = self.ddp_rec_model(
                img=None,
                c=micro['c'][0:1],
                latent=noised_latent,
                behaviour=self.render_latent_behaviour)

            # pred_x0 = self.sde_diffusion._predict_x0_from_eps(
            # eps_t_p, pred_eps_p, logsnr_p)  # for VAE loss, denosied latent

            # pred_xstart_3D
            denoised_ae_pred = self.ddp_rec_model(
                img=None,
                c=micro['c'][0:1],
                latent=denoised_latent,
                # latent=pred_x0[0:1] * self.
                # triplane_scaling_divider,  # TODO, how to define the scale automatically?
                behaviour=self.render_latent_behaviour)

        pred_vis = th.cat(
            [
                self.pool_128(img) for img in (
                    pred_img[0:1],
                    noised_ae_pred['image_raw'][0:1],
                    denoised_ae_pred['image_raw'][0:1],  # controlnet result
                    pred_depth[0:1].repeat_interleave(3, dim=1))
            ],
            dim=-1)  # B, 3, H, W

        if 'img' in micro:
            vis = th.cat([gt_vis, pred_vis],
                         dim=-2)[0].permute(1, 2,
                                            0).cpu()  # ! pred in range[-1, 1]
        else:
            vis = pred_vis[0].permute(1, 2, 0).cpu()

        # vis_grid = torchvision.utils.make_grid(vis) # HWC
        vis = vis.numpy() * 127.5 + 127.5
        vis = vis.clip(0, 255).astype(np.uint8)
        img_save_path = f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{sigmas[0].item():3}.jpg'
        Image.fromarray(vis).save(img_save_path)

        # if self.cond_key == 'caption':
        #     with open(f'{logger.get_dir()}/{self.step+self.resume_step}caption_{t_p[0].item():3}.txt', 'w') as f:
        #         f.write(micro['caption'][0])

        print('log denoised vis to: ', img_save_path)

        th.cuda.empty_cache()

    @th.no_grad()
    def sample(
        self,
        cond: Dict,
        uc: Union[Dict, None] = None,
        batch_size: int = 16,
        shape: Union[None, Tuple, List] = None,
        use_cfg=True,
        # cfg_scale=4, # default value in SiT
        # cfg_scale=1.5, # default value in SiT
        cfg_scale=4.0, # default value in SiT
        **kwargs,
    ):
        # self.sampler
        sample_fn = self.transport_sampler.sample_ode(num_steps=250, cfg=True) # default ode sampling setting.

        zs = th.randn(batch_size, *shape).to(dist_util.dev()).to(self.dtype)
        assert use_cfg
        # sample_model_kwargs = {'uc': uc, 'cond': cond}       
        model_fn = self.ddpm_model.forward_with_cfg # default

        # ! prepare_inputs in VanillaCFG, for compat issue
        c_out = {}
        for k in cond:
            if k in ["vector", "crossattn", "concat"]:
                c_out[k] = th.cat((cond[k], uc[k]), 0)
            else:
                assert cond[k] == uc[k]
                c_out[k] = cond[k]
        sample_model_kwargs = {'context': c_out, 'cfg_scale': cfg_scale}
        zs = th.cat([zs, zs], 0)

        with th.cuda.amp.autocast(dtype=self.dtype,
                                    enabled=self.mp_trainer.use_amp):

            samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        return samples

    @th.inference_mode()
    def eval_cldm(
        self,
        prompt="",
        # use_ddim=False,
        # unconditional_guidance_scale=1.0,
        save_img=False,
        use_train_trajectory=False,
        camera=None,
        num_samples=1,
        num_instances=1,
    ):
        # ! slightly modified for new API. combined with
        # /cpfs01/shared/V2V/V2V_hdd/yslan/Repo/generative-models/sgm/models/diffusion.py:249 log_images()
        # TODO, support batch_size > 1

        self.ddpm_model.eval()

        args = dnnlib.EasyDict(
            dict(
                batch_size=1,
                image_size=self.diffusion_input_size,
                denoise_in_channels=self.rec_model.decoder.triplane_decoder.
                out_chans,  # type: ignore
                clip_denoised=False,
                class_cond=False))

        model_kwargs = {}

        uc = None
        log = dict()

        ucg_keys = [self.cond_key] # i23d

        if self.cond_key == 'caption':
            batch_c = {self.cond_key: prompt}
        else: 
            batch = next(self.data) # random cond here
            if self.cond_key == 'img-c':
                batch_c = {
                    self.cond_key: {
                        'img': batch['img'].to(self.dtype).to(dist_util.dev()),
                        'c': batch['c'].to(self.dtype).to(dist_util.dev()),
                    },
                    'img': batch['img'].to(self.dtype).to(dist_util.dev()) # required by clip
                }

            else:
                batch_c = {self.cond_key: batch[self.cond_key].to(dist_util.dev()).to(self.dtype)}

        with th.cuda.amp.autocast(dtype=self.dtype,
                                    enabled=self.mp_trainer.use_amp):

            c, uc = self.conditioner.get_unconditional_conditioning(
                batch_c,
                force_uc_zero_embeddings=ucg_keys
                if len(self.conditioner.embedders) > 0 else [],
            )

        sampling_kwargs = {}

        N = 1  # hard coded, to update
        z_shape = (
            N,
            self.ddpm_model.in_channels if not self.ddpm_model.roll_out else
            3 * self.ddpm_model.in_channels,  # type: ignore
            self.diffusion_input_size,
            self.diffusion_input_size)

        for k in c:
            if isinstance(c[k], th.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(dist_util.dev()),
                                  (c, uc))
    
        samples = self.sample(c,
                              shape=z_shape[1:],
                              uc=uc,
                              batch_size=N,
                              **sampling_kwargs)
        # st() # do rendering first

        # ! get c
        if 'img' in self.cond_key:
            img_save_path = f'{logger.get_dir()}/{self.step+self.resume_step}_imgcond.jpg'
            st()
            if 'c' in self.cond_key:
                torchvision.utils.save_image(batch_c['img'][0], img_save_path, value_range=(-1,1), normalize=True) # torch.Size([24, 6, 3, 256, 256])
            else:
                torchvision.utils.save_image(batch_c['img'], img_save_path, value_range=(-1,1), normalize=True)

        assert camera is not None
        batch = {'c': camera.clone()}

        # else:
        #     if use_train_trajectory:
        #         batch = next(iter(self.data))
        #     else:
        #         try:
        #             batch = next(self.eval_data)
        #         except Exception as e:
        #             self.eval_data = iter(self.eval_data)
        #             batch = next(self.eval_data)

        #     if camera is not None:
        #         batch['c'] = camera.clone()

        # rendering
        for i in range(samples.shape[0]):
            th.cuda.empty_cache()

            # ! render sampled latent
            name_prefix = f'{self.step + self.resume_step}_{i}'

            if self.cond_key == 'caption':
                name_prefix = f'{name_prefix}_{prompt}'

            with th.cuda.amp.autocast(dtype=self.dtype,
                                        enabled=self.mp_trainer.use_amp):

                self.render_video_given_triplane(
                    samples[i:i+1].to(self.dtype),
                    self.rec_model,  # compatible with join_model
                    name_prefix=name_prefix,
                    save_img=save_img,
                    render_reference=batch,
                   export_mesh=False)

        self.ddpm_model.train()


class FlowMatchingEngine_gs(FlowMatchingEngine):

    def __init__(
        self,
        *,
        rec_model,
        denoise_model,
        diffusion,
        sde_diffusion,
        control_model,
        control_key,
        only_mid_control,
        loss_class,
        data,
        eval_data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        eval_interval,
        save_interval,
        resume_checkpoint,
        resume_cldm_checkpoint=None,
        use_fp16=False,
        fp16_scale_growth=0.001,
        schedule_sampler=None,
        weight_decay=0,
        lr_anneal_steps=0,
        iterations=10001,
        ignore_resume_opt=False,
        freeze_ae=False,
        denoised_ae=True,
        triplane_scaling_divider=10,
        use_amp=False,
        diffusion_input_size=224,
        normalize_clip_encoding=False,
        scale_clip_encoding=1,
        cfg_dropout_prob=0,
        cond_key='img_sr',
        use_eos_feature=False,
        compile=False,
        snr_type='lognorm',
        **kwargs):
        super().__init__(rec_model=rec_model,
                         denoise_model=denoise_model,
                         diffusion=diffusion,
                         sde_diffusion=sde_diffusion,
                         control_model=control_model,
                         control_key=control_key,
                         only_mid_control=only_mid_control,
                         loss_class=loss_class,
                         data=data,
                         eval_data=eval_data,
                         batch_size=batch_size,
                         microbatch=microbatch,
                         lr=lr,
                         ema_rate=ema_rate,
                         log_interval=log_interval,
                         eval_interval=eval_interval,
                         save_interval=save_interval,
                         resume_checkpoint=resume_checkpoint,
                         resume_cldm_checkpoint=resume_cldm_checkpoint,
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         schedule_sampler=schedule_sampler,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         ignore_resume_opt=ignore_resume_opt,
                         freeze_ae=freeze_ae,
                         denoised_ae=denoised_ae,
                         triplane_scaling_divider=triplane_scaling_divider,
                         use_amp=use_amp,
                         diffusion_input_size=diffusion_input_size,
                         normalize_clip_encoding=normalize_clip_encoding,
                         scale_clip_encoding=scale_clip_encoding,
                         cfg_dropout_prob=cfg_dropout_prob,
                         cond_key=cond_key,
                         use_eos_feature=use_eos_feature,
                         compile=compile,
                         snr_type=snr_type,
                         **kwargs)
        self.gs_bg_color=th.tensor([1,1,1], dtype=th.float32, device=dist_util.dev())
        self.latent_name = 'latent_normalized'  # normalized triplane latent
        self.pcd_unnormalize_fn = lambda x: x.clip(-1,1) * 0.45 # [-1,1] -> [-0.45, 0.45] as in g-buffer dataset.


    @th.inference_mode()
    def eval_cldm(
        self,
        prompt="",
        # use_ddim=False,
        # unconditional_guidance_scale=1.0,
        save_img=False,
        use_train_trajectory=False,
        camera=None,
        num_samples=1,
        num_instances=1,
    ):
        self.ddpm_model.eval()

        args = dnnlib.EasyDict(
            dict(
                batch_size=1,
                image_size=self.diffusion_input_size,
                denoise_in_channels=self.rec_model.decoder.triplane_decoder.
                out_chans,  # type: ignore
                clip_denoised=False,
                class_cond=False))

        model_kwargs = {}

        uc = None
        log = dict()

        ucg_keys = [self.cond_key] # i23d

        if self.cond_key == 'caption':
            batch_c = {self.cond_key: prompt}
        else: 
            batch = next(self.data) # random cond here
            if self.cond_key == 'img-c':
                batch_c = {
                    self.cond_key: {
                        'img': batch['img'].to(self.dtype).to(dist_util.dev()),
                        'c': batch['c'].to(self.dtype).to(dist_util.dev()),
                    },
                    'img': batch['img'].to(self.dtype).to(dist_util.dev()) # required by clip
                }

            else:
                batch_c = {self.cond_key: batch[self.cond_key].to(dist_util.dev()).to(self.dtype)}

        with th.cuda.amp.autocast(dtype=self.dtype,
                                    enabled=self.mp_trainer.use_amp):

            c, uc = self.conditioner.get_unconditional_conditioning(
                batch_c,
                force_uc_zero_embeddings=ucg_keys
                if len(self.conditioner.embedders) > 0 else [],
            )

        sampling_kwargs = {}

        N = 1  # hard coded, to update
        z_shape = (N, 768, self.ddpm_model.in_channels)

        for k in c:
            if isinstance(c[k], th.Tensor):
                c[k], uc[k] = map(lambda y: y[k][:N].to(dist_util.dev()),
                                  (c, uc))
    
        samples = self.sample(c,
                              shape=z_shape[1:],
                              uc=uc,
                              batch_size=N,
                              **sampling_kwargs)

        # ! get c
        if 'img' in self.cond_key:
            img_save_path = f'{logger.get_dir()}/{self.step+self.resume_step}_imgcond.jpg'
            if 'c' in self.cond_key:
                torchvision.utils.save_image(batch_c['img'][0], img_save_path, value_range=(-1,1), normalize=True) # torch.Size([24, 6, 3, 256, 256])
            else:
                torchvision.utils.save_image(batch_c['img'], img_save_path, value_range=(-1,1), normalize=True)

        assert camera is not None
        batch = {'c': camera.clone()}

        # rendering
        for i in range(samples.shape[0]):
            th.cuda.empty_cache()

            # ! render sampled latent
            name_prefix = f'{self.step + self.resume_step}_{i}'

            if self.cond_key == 'caption':
                name_prefix = f'{name_prefix}_{prompt}'

            with th.cuda.amp.autocast(dtype=self.dtype,
                                        enabled=self.mp_trainer.use_amp):

                # ! todo, transform to gs camera
                self.render_gs_video_given_latent(
                    samples[i:i+1].to(self.dtype),
                    self.rec_model,  # compatible with join_model
                    name_prefix=name_prefix,
                    save_img=save_img,
                    render_reference=batch,
                    export_mesh=False)

        self.ddpm_model.train()



    @th.inference_mode()
    def render_gs_video_given_latent(self,
                                    planes,
                                    rec_model,
                                    name_prefix='0',
                                    save_img=False, 
                                    render_reference=None, 
                                    export_mesh=False):

        batch_size, L, C = planes.shape

        ddpm_latent = { self.latent_name: planes[..., :-3] * self.triplane_scaling_divider,  # kl-reg latent
                        'query_pcd_xyz': self.pcd_unnormalize_fn(planes[..., -3:]) }
        
        ddpm_latent.update(rec_model(latent=ddpm_latent, behaviour='decode_gs_after_vae_no_render')) 

        video_out = imageio.get_writer(
            f'{logger.get_dir()}/gs_{name_prefix}.mp4',
            mode='I',
            fps=15,
            codec='libx264')

        render_reference = th.load('eval_pose.pt', map_location='cpu').numpy()

        # assert render_reference is None
            # render_reference = self.eval_data # compat
        # else: # use train_traj

        # for key in ['ins', 'bbox', 'caption']:
        #     if key in render_reference:
        #         render_reference.pop(key)

        # render_reference = [ { k:v[idx:idx+1] for k, v in render_reference.items() } for idx in range(40) ]

        # for i, batch in enumerate(tqdm(self.eval_data)):
        for i, micro_c in enumerate(tqdm(render_reference)):
            # micro = {
            #     k: v.to(dist_util.dev()) if isinstance(v, th.Tensor) else v
            #     for k, v in batch.items() 
            # }

            c = self.eval_data.post_process.c_to_3dgs_format(micro_c)
            for k in c.keys(): # to cuda
                if isinstance(c[k], th.Tensor) and k != 'tanfov':
                    c[k] = c[k].unsqueeze(0).unsqueeze(0).to(dist_util.dev()) # actually, could render 40 views together.
            c['tanfov'] = th.tensor(c['tanfov']).to(dist_util.dev())

            pred = rec_model(
                img=None,
                c=c, # TODO, to dict
                latent=ddpm_latent, # render gs
                behaviour='triplane_dec',
                bg_color=self.gs_bg_color,
                )

            for k in pred.keys():
                pred[k] = einops.rearrange(pred[k], 'B V ... -> (B V) ...') # merge 
            
            pred_vis = self._make_vis_img(pred)

            if save_img:
                for batch_idx in range(gen_img.shape[0]):
                    sampled_img = Image.fromarray(
                        (gen_img[batch_idx].permute(1, 2, 0).cpu().numpy() *
                         127.5 + 127.5).clip(0, 255).astype(np.uint8))
                    if sampled_img.size != (512, 512):
                        sampled_img = sampled_img.resize(
                            (128, 128), Image.HAMMING)  # for shapenet
                    sampled_img.save(logger.get_dir() +
                                     '/FID_Cals/{}_{}.png'.format(
                                         int(name_prefix) * batch_size +
                                         batch_idx, i))
                    # print('FID_Cals/{}_{}.png'.format(int(name_prefix)*batch_size+batch_idx, i))

            vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
            vis = vis * 127.5 + 127.5
            vis = vis.clip(0, 255).astype(np.uint8)

            # if not save_img:
            for j in range(vis.shape[0]
                        ):  # ! currently only export one plane at a time
                video_out.append_data(vis[j])

        # if not save_img:
        video_out.close()
        print('logged video to: ',
            f'{logger.get_dir()}/triplane_{name_prefix}.mp4')

        del video_out, pred, pred_vis, vis

