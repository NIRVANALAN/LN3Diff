"""
https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/diffusion/ddpm.py#L30
"""
import copy

from matplotlib import pyplot as plt
import functools
import json
import os
from pathlib import Path
from pdb import set_trace as st
from typing import Any
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

from ldm.modules.encoders.modules import FrozenClipImageEmbedder, TextEmbedder, FrozenCLIPTextEmbedder

import dnnlib
from dnnlib.util import requires_grad
from dnnlib.util import calculate_adaptive_weight

from ..train_util_diffusion import TrainLoop3DDiffusion
from ..cvD.nvsD_canoD import TrainLoop3DcvD_nvsD_canoD

from guided_diffusion.continuous_diffusion_utils import get_mixed_prediction, different_p_q_objectives, kl_per_group_vada, kl_balancer
# from .train_util_diffusion_lsgm_noD_joint import TrainLoop3DDiffusionLSGMJointnoD  # joint diffusion and rec class
# from .controlLDM import TrainLoop3DDiffusionLSGM_Control  # joint diffusion and rec class
from .train_util_diffusion_lsgm_noD_joint import TrainLoop3DDiffusionLSGMJointnoD  # joint diffusion and rec class

__conditioning_keys__ = {
    'concat': 'c_concat',
    'crossattn': 'c_crossattn',
    'adm': 'y'
}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class TrainLoop3DDiffusionLSGM_crossattn(TrainLoop3DDiffusionLSGMJointnoD):

    def __init__(self,
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
                 scale_clip_encoding=1.0,
                 cfg_dropout_prob=0.,
                 cond_key='img_sr',
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
                         **kwargs)
        self.conditioning_key = 'c_crossattn'
        self.cond_key = cond_key
        self.instantiate_cond_stage(normalize_clip_encoding,
                                    scale_clip_encoding, cfg_dropout_prob)
        requires_grad(self.rec_model, False)
        self.rec_model.eval()
        # self.normalize_clip_encoding = normalize_clip_encoding
        # self.cfg_dropout_prob = cfg_dropout_prob

    def instantiate_cond_stage(self, normalize_clip_encoding,
                               scale_clip_encoding, cfg_dropout_prob):
        # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/diffusion/ddpm.py#L509C1-L509C46
        # self.cond_stage_model.train = disabled_train  # type: ignore
        # st()
        if self.cond_key == 'caption':  # for objaverse training (with extracted cap3d caption)
            self.cond_txt_model = TextEmbedder(dropout_prob=cfg_dropout_prob)
        else:  # zero-shot Text to 3D using normalized clip latent
            self.cond_stage_model = FrozenClipImageEmbedder(
                'ViT-L/14',
                dropout_prob=cfg_dropout_prob,
                normalize_encoding=normalize_clip_encoding,
                scale_clip_encoding=scale_clip_encoding)
            self.cond_stage_model.freeze()

            self.cond_txt_model = FrozenCLIPTextEmbedder(
                dropout_prob=cfg_dropout_prob,
                scale_clip_encoding=scale_clip_encoding)
            self.cond_txt_model.freeze()

    @th.no_grad()
    def get_c_input(self,
                    batch,
                    bs=None,
                    use_text=False,
                    prompt="",
                    *args,
                    **kwargs):

        # using clip to transform control to tokens for crossattn
        cond_inp = None

        if self.cond_key == 'caption':
            c = self.cond_txt_model(
                cond_inp, train=self.ddpm_model.training
            )  # ! SD training text condition injection layer
            # st() # check whether context repeat?
        else:  # zero shot
            if use_text:  # for test
                assert prompt != ""
                c = self.cond_txt_model.encode(prompt)  # ! for test
                # st()
            else:

                cond_inp = batch[self.cond_key]
                if bs is not None:
                    cond_inp = cond_inp[:bs]

                cond_inp = cond_inp.to(
                    memory_format=th.contiguous_format).float()
                c = self.cond_stage_model(cond_inp)  # BS 768

        # return dict(c_concat=[control])
        # return  dict(c_crossattn=[c], c_concat=[control])
        # return dict(__conditioning_keys__[self.cond_key]=)
        # return {self.conditioning_key: [c], 'c_concat': [cond_inp]}
        return {self.conditioning_key: c, 'c_concat': [cond_inp]}

    # TODO, merge the APIs
    def apply_model_inference(self, x_noisy, t, c, model_kwargs={}):
        pred_params = self.ddp_ddpm_model(
            x_noisy, t, **{
                **model_kwargs, 'context': c['c_crossattn']
            })
        return pred_params

    def apply_model(self, p_sample_batch, cond, model_kwargs={}):
        return super().apply_model(
            p_sample_batch, **{
                **model_kwargs, 'context': cond['c_crossattn']
            })

    def run_step(self, batch, step='ldm_step'):

        # if step == 'diffusion_step_rec':

        if step == 'ldm_step':
            self.ldm_train_step(batch)

        # if took_step_ddpm:
        # self._update_cldm_ema()

        self._anneal_lr()
        self.log_step()

    def run_loop(self):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            # dist_util.synchronize()

            batch = next(self.data)
            self.run_step(batch, step='ldm_step')

            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                out = logger.dumpkvs()
                # * log to tensorboard
                for k, v in out.items():
                    self.writer.add_scalar(f'Loss/{k}', v,
                                           self.step + self.resume_step)

            # if self.step % self.eval_interval == 0 and self.step != 0:
            if self.step % self.eval_interval == 0:
                if dist_util.get_rank() == 0:
                    # self.eval_ddpm_sample()
                    # self.eval_cldm(use_ddim=True, unconditional_guidance_scale=7.5, prompt="") # during training, use image as condition
                    self.eval_cldm(use_ddim=False,
                                   prompt="")  # fix condition bug first
                    # if self.sde_diffusion.args.train_vae:
                    #     self.eval_loop()

                th.cuda.empty_cache()
                dist_util.synchronize()

            if self.step % self.save_interval == 0:
                self.save(self.mp_trainer, self.mp_trainer.model_name)
                if os.environ.get("DIFFUSION_TRAINING_TEST",
                                  "") and self.step > 0:
                    return

            self.step += 1

            if self.step > self.iterations:
                print('reached maximum iterations, exiting')

                # Save the last checkpoint if it wasn't already saved.
                if (self.step - 1) % self.save_interval != 0:

                    self.save(self.mp_trainer, self.mp_trainer.model_name)
                    # if self.sde_diffusion.args.train_vae:
                    #     self.save(self.mp_trainer_rec,
                    #               self.mp_trainer_rec.model_name)

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save(self.mp_trainer,
                      self.mp_trainer.model_name)  # rec and ddpm all fixed.
            # st()
            # self.save(self.mp_trainer_canonical_cvD, 'cvD')

    # ddpm + rec loss
    def ldm_train_step(self, batch, behaviour='cano', *args, **kwargs):
        """
        add sds grad to all ae predicted x_0 
        """

        # ! enable the gradient of both models
        requires_grad(self.ddpm_model, True)

        self.mp_trainer.zero_grad()  # !!!!

        batch_size = batch['img'].shape[0]

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k:
                v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
                    v, th.Tensor) else v
                for k, v in batch.items()
            }

            # =================================== ae part ===================================
            with th.cuda.amp.autocast(dtype=th.float16,
                                      enabled=self.mp_trainer.use_amp):

                loss = th.tensor(0.).to(dist_util.dev())

                vae_out = self.ddp_rec_model(
                    img=micro['img_to_encoder'],
                    c=micro['c'],
                    behaviour='encoder_vae',
                )  # pred: (B, 3, 64, 64)
                eps = vae_out[self.latent_name]
                # eps = vae_out.pop(self.latent_name)

                if 'bg_plane' in vae_out:
                    eps = th.cat((eps, vae_out['bg_plane']),
                                 dim=1)  # include background, B 12+4 32 32

                p_sample_batch = self.prepare_ddpm(eps)
                cond = self.get_c_input(micro)

                # ! running diffusion forward
                ddpm_ret = self.apply_model(p_sample_batch, cond)
                if self.sde_diffusion.args.p_rendering_loss:

                    target = micro
                    pred = self.ddp_rec_model(
                        # latent=vae_out,
                        latent={
                            # **vae_out,
                            self.latent_name: ddpm_ret['pred_x0_p'],
                            'latent_name': self.latent_name
                        },
                        c=micro['c'],
                        behaviour=self.render_latent_behaviour)

                    # vae reconstruction loss
                    with self.ddp_control_model.no_sync():  # type: ignore
                        p_vae_recon_loss, rec_loss_dict = self.loss_class(
                            pred, target, test_mode=False)
                    log_rec3d_loss_dict(rec_loss_dict)
                    # log_rec3d_loss_dict(
                    #     dict(p_vae_recon_loss=p_vae_recon_loss, ))
                    loss = p_vae_recon_loss + ddpm_ret[
                        'p_eps_objective']  # TODO, add obj_weight_t_p?
                else:
                    loss = ddpm_ret['p_eps_objective'].mean()

                # =====================================================================

            self.mp_trainer.backward(loss)  # joint gradient descent

        # update ddpm accordingly
        self.mp_trainer.optimize(self.opt)

        if dist_util.get_rank() == 0 and self.step % 500 == 0:
            self.log_control_images(vae_out, p_sample_batch, micro, ddpm_ret)

    @th.inference_mode()
    def log_control_images(self, vae_out, p_sample_batch, micro, ddpm_ret):

        eps_t_p, t_p, logsnr_p = (p_sample_batch[k] for k in (
            'eps_t_p',
            't_p',
            'logsnr_p',
        ))
        pred_eps_p = ddpm_ret['pred_eps_p']

        vae_out.pop('posterior')  # for calculating kl loss
        vae_out_for_pred = {
            k: v[0:1].to(dist_util.dev()) if isinstance(v, th.Tensor) else v
            for k, v in vae_out.items()
        }

        pred = self.ddp_rec_model(latent=vae_out_for_pred,
                                  c=micro['c'][0:1],
                                  behaviour=self.render_latent_behaviour)
        assert isinstance(pred, dict)

        pred_img = pred['image_raw']
        gt_img = micro['img']

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

        if 'bg_plane' in vae_out:
            noised_latent = {
                'latent_normalized_2Ddiffusion':
                eps_t_p[0:1, :12] * self.triplane_scaling_divider,
                'bg_plane':
                eps_t_p[0:1, 12:16] * self.triplane_scaling_divider,
            }
        else:
            noised_latent = {
                'latent_normalized_2Ddiffusion':
                eps_t_p[0:1] * self.triplane_scaling_divider,
            }

        noised_ae_pred = self.ddp_rec_model(
            img=None,
            c=micro['c'][0:1],
            latent=noised_latent,
            # latent=eps_t_p[0:1] * self.
            # triplane_scaling_divider,  # TODO, how to define the scale automatically
            behaviour=self.render_latent_behaviour)

        pred_x0 = self.sde_diffusion._predict_x0_from_eps(
            eps_t_p, pred_eps_p, logsnr_p)  # for VAE loss, denosied latent

        if 'bg_plane' in vae_out:
            denoised_latent = {
                'latent_normalized_2Ddiffusion':
                pred_x0[0:1, :12] * self.triplane_scaling_divider,
                'bg_plane':
                pred_x0[0:1, 12:16] * self.triplane_scaling_divider,
            }
        else:
            denoised_latent = {
                'latent_normalized_2Ddiffusion':
                pred_x0[0:1] * self.triplane_scaling_divider,
            }

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

        vis = th.cat([gt_vis, pred_vis],
                     dim=-2)[0].permute(1, 2,
                                        0).cpu()  # ! pred in range[-1, 1]

        # vis_grid = torchvision.utils.make_grid(vis) # HWC
        vis = vis.numpy() * 127.5 + 127.5
        vis = vis.clip(0, 255).astype(np.uint8)
        Image.fromarray(vis).save(
            f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t_p[0].item():3}.jpg'
        )

        if self.cond_key == 'caption':
            with open(
                    f'{logger.get_dir()}/{self.step+self.resume_step}caption_{t_p[0].item():3}.txt',
                    'w') as f:
                f.write(micro['caption'][0])

        print(
            'log denoised vis to: ',
            f'{logger.get_dir()}/{self.step+self.resume_step}denoised_{t_p[0].item():3}.jpg'
        )

        th.cuda.empty_cache()

    @th.inference_mode()
    def eval_cldm(self,
                  prompt="",
                  use_ddim=False,
                  unconditional_guidance_scale=1.0,
                  save_img=False,
                  use_train_trajectory=False,
                  export_mesh=False,
                  camera=None, 
                  overwrite_diff_inp_size=None):
        self.ddpm_model.eval()

        args = dnnlib.EasyDict(
            dict(
                batch_size=self.batch_size,
                image_size=self.diffusion_input_size,
                denoise_in_channels=self.rec_model.decoder.triplane_decoder.
                out_chans,  # type: ignore
                clip_denoised=False,
                class_cond=False,
                use_ddim=use_ddim))

        model_kwargs = {}

        if args.class_cond:
            classes = th.randint(low=0,
                                 high=NUM_CLASSES,
                                 size=(args.batch_size, ),
                                 device=dist_util.dev())
            model_kwargs["y"] = classes

        diffusion = self.diffusion
        sample_fn = (diffusion.p_sample_loop
                     if not args.use_ddim else diffusion.ddim_sample_loop)
        extra_kwargs = {}
        if args.use_ddim:
            extra_kwargs.update(
                dict(
                    unconditional_guidance_scale=unconditional_guidance_scale))

        # for i, batch in enumerate(tqdm(self.eval_data)):
        # if use_train_trajectory:
        #     batch = next(iter(self.data))
        # else:
        # batch = next(iter(self.eval_data))

        # st() # th.save(batch['c'].cpu(), 'assets/shapenet_eval_pose.pt')

        assert camera is not None  # for evaluation
        batch = {'c': camera.clone()}
        # st()

        # use the first frame as the condition now
        novel_view_cond = {
            k:
            v[0:1].to(dist_util.dev()) if isinstance(v, th.Tensor) else v[0:1]
            # micro['img'].shape[0], 0)
            for k, v in batch.items()
        }
        cond = self.get_c_input(novel_view_cond,
                                use_text=prompt != "",
                                prompt=prompt)  # use specific prompt for debug

        # broadcast to args.batch_size
        cond = {
            k: cond_v.repeat_interleave(args.batch_size, 0)
            for k, cond_v in cond.items() if k == self.conditioning_key
        }

        for i in range(1):
            # st()
            noise_size = (
                args.batch_size,
                self.ddpm_model.in_channels,
                self.diffusion_input_size if not overwrite_diff_inp_size else int(overwrite_diff_inp_size),
                self.diffusion_input_size if not overwrite_diff_inp_size else int(overwrite_diff_inp_size)
            )

            triplane_sample = sample_fn(
                self,
                noise_size,
                cond=cond,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                mixing_normal=True,  # !
                device=dist_util.dev(),
                **extra_kwargs)
            # triplane_sample = th.zeros((args.batch_size, self.ddpm_model.in_channels, self.diffusion_input_size, self.diffusion_input_size), device=dist_util.dev())
            th.cuda.empty_cache()

            for sub_idx in range(triplane_sample.shape[0]):

                self.render_video_given_triplane(
                    triplane_sample[sub_idx:sub_idx + 1],
                    self.rec_model,  # compatible with join_model
                    name_prefix=f'{self.step + self.resume_step}_{i+sub_idx}',
                    save_img=save_img,
                    render_reference=batch,
                    # render_reference=None,
                    export_mesh=export_mesh,
                    render_all=True,
                )

            del triplane_sample
            th.cuda.empty_cache()

        self.ddpm_model.train()

    @th.inference_mode()
    # def eval_loop(self, c_list:list):
    def eval_novelview_loop(self, rec_model):
        # novel view synthesis given evaluation camera trajectory
        video_out = imageio.get_writer(
            f'{logger.get_dir()}/video_novelview_{self.step+self.resume_step}.mp4',
            mode='I',
            fps=60,
            codec='libx264')

        all_loss_dict = []
        novel_view_micro = {}

        # for i in range(0, len(c_list), 1): # TODO, larger batch size for eval
        for i, batch in enumerate(tqdm(self.eval_data)):
            # for i in range(0, 8, self.microbatch):
            # c = c_list[i].to(dist_util.dev()).reshape(1, -1)
            micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

            if i == 0:
                novel_view_micro = {
                    k:
                    v[0:1].to(dist_util.dev()).repeat_interleave(
                        micro['img'].shape[0], 0)
                    for k, v in batch.items()
                }

                torchvision.utils.save_image(
                    self.pool_128(novel_view_micro['img']),
                    logger.get_dir() + '/FID_Cals/gt.png',
                    normalize=True,
                    val_range=(0, 1),
                    padding=0)

            else:
                # if novel_view_micro['c'].shape[0] < micro['img'].shape[0]:
                novel_view_micro = {
                    k:
                    v[0:1].to(dist_util.dev()).repeat_interleave(
                        micro['img'].shape[0], 0)
                    for k, v in novel_view_micro.items()
                }

            th.manual_seed(0)  # avoid vae re-sampling changes results
            pred = rec_model(img=novel_view_micro['img_to_encoder'],
                             c=micro['c'])  # pred: (B, 3, 64, 64)

            # ! move to other places, add tensorboard

            # pred_vis = th.cat([
            #     pred['image_raw'],
            #     -pred['image_depth'].repeat_interleave(3, dim=1)
            # ],
            #                   dim=-1)

            # normalize depth
            # if True:
            pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())

            # ! save

            pooled_depth = self.pool_128(pred_depth).repeat_interleave(3,
                                                                       dim=1)
            pred_vis = th.cat([
                self.pool_128(micro['img']),
                self.pool_128(pred['image_raw']),
                pooled_depth,
            ],
                              dim=-1)  # B, 3, H, W

            # ! save depth
            name_prefix = i

            torchvision.utils.save_image(self.pool_128(pred['image_raw']),
                                         logger.get_dir() +
                                         '/FID_Cals/{}.png'.format(i),
                                         normalize=True,
                                         val_range=(0, 1),
                                         padding=0)

            torchvision.utils.save_image(self.pool_128(pooled_depth),
                                         logger.get_dir() +
                                         '/FID_Cals/{}_depth.png'.format(i),
                                         normalize=True,
                                         val_range=(0, 1),
                                         padding=0)

            vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
            vis = vis * 127.5 + 127.5
            vis = vis.clip(0, 255).astype(np.uint8)

            for j in range(vis.shape[0]):
                video_out.append_data(vis[j])

        video_out.close()

        del video_out
        # del pred_vis
        # del pred

        th.cuda.empty_cache()
