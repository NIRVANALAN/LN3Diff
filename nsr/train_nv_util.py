import copy
# import imageio.v3
import functools
import json
import os
from pathlib import Path
from pdb import set_trace as st
from einops import rearrange
import webdataset as wds

import traceback
import blobfile as bf
import imageio
import numpy as np
# from sympy import O
import torch as th
import torch.distributed as dist
import torchvision
from PIL import Image
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.nn import update_ema
from guided_diffusion.resample import LossAwareSampler, UniformSampler
from guided_diffusion.train_util import (calc_average_loss,
                                         find_ema_checkpoint,
                                         find_resume_checkpoint,
                                         get_blob_logdir, log_rec3d_loss_dict,
                                         parse_resume_step_from_filename)

from .camera_utils import LookAtPoseSampler, FOV_to_intrinsics

from .train_util import TrainLoop3DRec


class TrainLoop3DRecNV(TrainLoop3DRec):
    # supervise the training of novel view
    def __init__(self,
                 *,
                 rec_model,
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
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
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
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)
        self.rec_cano = True

    def forward_backward(self, batch, *args, **kwargs):
        # return super().forward_backward(batch, *args, **kwargs)

        self.mp_trainer_rec.zero_grad()
        batch_size = batch['img_to_encoder'].shape[0]

        for i in range(0, batch_size, self.microbatch):

            # st()
            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev())
                for k, v in batch.items()
            }

            # ! concat novel-view? next version. also add self reconstruction, patch-based loss in the next version. verify novel-view prediction first.

            # wrap forward within amp
            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                target_nvs = {}
                target_cano = {}

                latent = self.rec_model(img=micro['img_to_encoder'],
                                        behaviour='enc_dec_wo_triplane')

                pred = self.rec_model(
                    latent=latent,
                    c=micro['nv_c'],  # predict novel view here
                    behaviour='triplane_dec')

                for k, v in micro.items():
                    if k[:2] == 'nv':
                        orig_key = k.replace('nv_', '')
                        target_nvs[orig_key] = v
                        target_cano[orig_key] = micro[orig_key]

                with self.rec_model.no_sync():  # type: ignore
                    loss, loss_dict, fg_mask = self.loss_class(
                        pred,
                        target_nvs,
                        step=self.step + self.resume_step,
                        test_mode=False,
                        return_fg_mask=True,
                        conf_sigma_l1=None,
                        conf_sigma_percl=None)
                    log_rec3d_loss_dict(loss_dict)

                if self.rec_cano:

                    pred_cano = self.rec_model(latent=latent,
                                               c=micro['c'],
                                               behaviour='triplane_dec')

                    with self.rec_model.no_sync():  # type: ignore

                        fg_mask = target_cano['depth_mask'].unsqueeze(
                            1).repeat_interleave(3, 1).float()

                        loss_cano, loss_cano_dict = self.loss_class.calc_2d_rec_loss(
                            pred_cano['image_raw'],
                            target_cano['img'],
                            fg_mask,
                            step=self.step + self.resume_step,
                            test_mode=False,
                        )

                    loss = loss + loss_cano

                    # remove redundant log
                    log_rec3d_loss_dict({
                        f'cano_{k}': v
                        for k, v in loss_cano_dict.items()
                        #  if "loss" in k
                    })

            self.mp_trainer_rec.backward(loss)

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                if self.rec_cano:
                    self.log_img(micro, pred, pred_cano)
                else:
                    self.log_img(micro, pred, None)

    @th.inference_mode()
    def log_img(self, micro, pred, pred_cano):
        # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

        def norm_depth(pred_depth):  # to [-1,1]
            # pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())
            return -(pred_depth * 2 - 1)

        pred_img = pred['image_raw']
        gt_img = micro['img']

        # infer novel view also
        # if self.loss_class.opt.symmetry_loss:
        #     pred_nv_img = nvs_pred
        # else:
        # ! replace with novel view prediction

        # ! log another novel-view prediction
        # pred_nv_img = self.rec_model(
        #     img=micro['img_to_encoder'],
        #     c=self.novel_view_poses)  # pred: (B, 3, 64, 64)

        # if 'depth' in micro:
        gt_depth = micro['depth']
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        gt_depth = norm_depth(gt_depth)
        # gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
        #                                           gt_depth.min())
        # if True:
        fg_mask = pred['image_mask'] * 2 - 1  # 0-1
        input_fg_mask = pred_cano['image_mask'] * 2 - 1  # 0-1
        if 'image_depth' in pred:
            pred_depth = norm_depth(pred['image_depth'])
            pred_nv_depth = norm_depth(pred_cano['image_depth'])
        else:
            pred_depth = th.zeros_like(gt_depth)
            pred_nv_depth = th.zeros_like(gt_depth)

        if 'image_sr' in pred:
            if pred['image_sr'].shape[-1] == 512:
                pred_img = th.cat([self.pool_512(pred_img), pred['image_sr']],
                                  dim=-1)
                gt_img = th.cat([self.pool_512(micro['img']), micro['img_sr']],
                                dim=-1)
                pred_depth = self.pool_512(pred_depth)
                gt_depth = self.pool_512(gt_depth)

            elif pred['image_sr'].shape[-1] == 256:
                pred_img = th.cat([self.pool_256(pred_img), pred['image_sr']],
                                  dim=-1)
                gt_img = th.cat([self.pool_256(micro['img']), micro['img_sr']],
                                dim=-1)
                pred_depth = self.pool_256(pred_depth)
                gt_depth = self.pool_256(gt_depth)

            else:
                pred_img = th.cat([self.pool_128(pred_img), pred['image_sr']],
                                  dim=-1)
                gt_img = th.cat([self.pool_128(micro['img']), micro['img_sr']],
                                dim=-1)
                gt_depth = self.pool_128(gt_depth)
                pred_depth = self.pool_128(pred_depth)
        else:
            gt_img = self.pool_64(gt_img)
            gt_depth = self.pool_64(gt_depth)

        pred_vis = th.cat([
            pred_img,
            pred_depth.repeat_interleave(3, dim=1),
            fg_mask.repeat_interleave(3, dim=1),
        ],
                          dim=-1)  # B, 3, H, W

        pred_vis_nv = th.cat([
            pred_cano['image_raw'],
            pred_nv_depth.repeat_interleave(3, dim=1),
            input_fg_mask.repeat_interleave(3, dim=1),
        ],
                             dim=-1)  # B, 3, H, W

        pred_vis = th.cat([pred_vis, pred_vis_nv], dim=-2)  # cat in H dim

        gt_vis = th.cat([
            gt_img,
            gt_depth.repeat_interleave(3, dim=1),
            th.zeros_like(gt_img)
        ],
                        dim=-1)  # TODO, fail to load depth. range [0, 1]

        if 'conf_sigma' in pred:
            gt_vis = th.cat([gt_vis, fg_mask], dim=-1)  # placeholder

        # vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
        vis = th.cat([gt_vis, pred_vis], dim=-2)
        # .permute(
        #     0, 2, 3, 1).cpu()
        vis_tensor = torchvision.utils.make_grid(vis, nrow=vis.shape[-1] //
                                                 64)  # HWC
        torchvision.utils.save_image(
            vis_tensor,
            f'{logger.get_dir()}/{self.step+self.resume_step}.jpg',
            value_range=(-1, 1),
            normalize=True)
        # vis = vis.numpy() * 127.5 + 127.5
        # vis = vis.clip(0, 255).astype(np.uint8)

        # Image.fromarray(vis).save(
        #     f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')

        logger.log('log vis to: ',
                   f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')

        # self.writer.add_image(f'images',
        #                       vis,
        #                       self.step + self.resume_step,
        #                       dataformats='HWC')


# return pred


class TrainLoop3DRecNVPatch(TrainLoop3DRecNV):
    # add patch rendering
    def __init__(self,
                 *,
                 rec_model,
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
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
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
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)
        # the rendrer
        self.eg3d_model = self.rec_model.module.decoder.triplane_decoder  # type: ignore
        # self.rec_cano = False
        self.rec_cano = True

    def forward_backward(self, batch, *args, **kwargs):
        # add patch sampling

        self.mp_trainer_rec.zero_grad()
        batch_size = batch['img_to_encoder'].shape[0]

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k: v[i:i + self.microbatch].to(dist_util.dev())
                for k, v in batch.items()
            }

            # ! sample rendering patch
            target = {
                **self.eg3d_model(
                    c=micro['nv_c'],  # type: ignore
                    ws=None,
                    planes=None,
                    sample_ray_only=True,
                    fg_bbox=micro['nv_bbox']),  # rays o / dir
            }

            patch_rendering_resolution = self.eg3d_model.rendering_kwargs[
                'patch_rendering_resolution']  # type: ignore
            cropped_target = {
                k:
                th.empty_like(v)
                [..., :patch_rendering_resolution, :patch_rendering_resolution]
                if k not in [
                    'ins_idx', 'img_to_encoder', 'img_sr', 'nv_img_to_encoder',
                    'nv_img_sr', 'c'
                ] else v
                for k, v in micro.items()
            }

            # crop according to uv sampling
            for j in range(micro['img'].shape[0]):
                top, left, height, width = target['ray_bboxes'][
                    j]  # list of tuple
                # for key in ('img', 'depth_mask', 'depth', 'depth_mask_sr'): # type: ignore
                for key in ('img', 'depth_mask', 'depth'):  # type: ignore
                    # target[key][i:i+1] = torchvision.transforms.functional.crop(
                    # cropped_target[key][
                    #     j:j + 1] = torchvision.transforms.functional.crop(
                    #         micro[key][j:j + 1], top, left, height, width)

                    cropped_target[f'{key}'][  # ! no nv_ here
                        j:j + 1] = torchvision.transforms.functional.crop(
                            micro[f'nv_{key}'][j:j + 1], top, left, height,
                            width)

            # target.update(cropped_target)

            # wrap forward within amp
            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                # target_nvs = {}
                # target_cano = {}

                latent = self.rec_model(img=micro['img_to_encoder'],
                                        behaviour='enc_dec_wo_triplane')

                pred_nv = self.rec_model(
                    latent=latent,
                    c=micro['nv_c'],  # predict novel view here
                    behaviour='triplane_dec',
                    ray_origins=target['ray_origins'],
                    ray_directions=target['ray_directions'],
                )

                # ! directly retrieve from target
                # for k, v in target.items():
                #     if k[:2] == 'nv':
                #         orig_key = k.replace('nv_', '')
                #         target_nvs[orig_key] = v
                #         target_cano[orig_key] = target[orig_key]

                with self.rec_model.no_sync():  # type: ignore
                    loss, loss_dict, _ = self.loss_class(pred_nv,
                                                         cropped_target,
                                                         step=self.step +
                                                         self.resume_step,
                                                         test_mode=False,
                                                         return_fg_mask=True,
                                                         conf_sigma_l1=None,
                                                         conf_sigma_percl=None)
                    log_rec3d_loss_dict(loss_dict)

                if self.rec_cano:

                    cano_target = {
                        **self.eg3d_model(
                            c=micro['c'],  # type: ignore
                            ws=None,
                            planes=None,
                            sample_ray_only=True,
                            fg_bbox=micro['bbox']),  # rays o / dir
                    }

                    cano_cropped_target = {
                        k: th.empty_like(v)
                        for k, v in cropped_target.items()
                    }

                    for j in range(micro['img'].shape[0]):
                        top, left, height, width = cano_target['ray_bboxes'][
                            j]  # list of tuple
                        # for key in ('img', 'depth_mask', 'depth', 'depth_mask_sr'): # type: ignore
                        for key in ('img', 'depth_mask',
                                    'depth'):  # type: ignore
                            # target[key][i:i+1] = torchvision.transforms.functional.crop(
                            cano_cropped_target[key][
                                j:j +
                                1] = torchvision.transforms.functional.crop(
                                    micro[key][j:j + 1], top, left, height,
                                    width)

                    # cano_target.update(cano_cropped_target)

                    pred_cano = self.rec_model(
                        latent=latent,
                        c=micro['c'],
                        behaviour='triplane_dec',
                        ray_origins=cano_target['ray_origins'],
                        ray_directions=cano_target['ray_directions'],
                    )

                    with self.rec_model.no_sync():  # type: ignore

                        fg_mask = cano_cropped_target['depth_mask'].unsqueeze(
                            1).repeat_interleave(3, 1).float()

                        loss_cano, loss_cano_dict = self.loss_class.calc_2d_rec_loss(
                            pred_cano['image_raw'],
                            cano_cropped_target['img'],
                            fg_mask,
                            step=self.step + self.resume_step,
                            test_mode=False,
                        )

                    loss = loss + loss_cano

                    # remove redundant log
                    log_rec3d_loss_dict({
                        f'cano_{k}': v
                        for k, v in loss_cano_dict.items()
                        #  if "loss" in k
                    })

            self.mp_trainer_rec.backward(loss)

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                self.log_patch_img(cropped_target, pred_nv, pred_cano)

    @th.inference_mode()
    def log_patch_img(self, micro, pred, pred_cano):
        # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

        def norm_depth(pred_depth):  # to [-1,1]
            # pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())
            return -(pred_depth * 2 - 1)

        pred_img = pred['image_raw']
        gt_img = micro['img']

        # infer novel view also
        # if self.loss_class.opt.symmetry_loss:
        #     pred_nv_img = nvs_pred
        # else:
        # ! replace with novel view prediction

        # ! log another novel-view prediction
        # pred_nv_img = self.rec_model(
        #     img=micro['img_to_encoder'],
        #     c=self.novel_view_poses)  # pred: (B, 3, 64, 64)

        # if 'depth' in micro:
        gt_depth = micro['depth']
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        gt_depth = norm_depth(gt_depth)
        # gt_depth = (gt_depth - gt_depth.min()) / (gt_depth.max() -
        #                                           gt_depth.min())
        # if True:
        fg_mask = pred['image_mask'] * 2 - 1  # 0-1
        input_fg_mask = pred_cano['image_mask'] * 2 - 1  # 0-1
        if 'image_depth' in pred:
            pred_depth = norm_depth(pred['image_depth'])
            pred_cano_depth = norm_depth(pred_cano['image_depth'])
        else:
            pred_depth = th.zeros_like(gt_depth)
            pred_cano_depth = th.zeros_like(gt_depth)

        # if 'image_sr' in pred:
        #     if pred['image_sr'].shape[-1] == 512:
        #         pred_img = th.cat([self.pool_512(pred_img), pred['image_sr']],
        #                           dim=-1)
        #         gt_img = th.cat([self.pool_512(micro['img']), micro['img_sr']],
        #                         dim=-1)
        #         pred_depth = self.pool_512(pred_depth)
        #         gt_depth = self.pool_512(gt_depth)

        #     elif pred['image_sr'].shape[-1] == 256:
        #         pred_img = th.cat([self.pool_256(pred_img), pred['image_sr']],
        #                           dim=-1)
        #         gt_img = th.cat([self.pool_256(micro['img']), micro['img_sr']],
        #                         dim=-1)
        #         pred_depth = self.pool_256(pred_depth)
        #         gt_depth = self.pool_256(gt_depth)

        #     else:
        #         pred_img = th.cat([self.pool_128(pred_img), pred['image_sr']],
        #                           dim=-1)
        #         gt_img = th.cat([self.pool_128(micro['img']), micro['img_sr']],
        #                         dim=-1)
        #         gt_depth = self.pool_128(gt_depth)
        #         pred_depth = self.pool_128(pred_depth)
        # else:
        #     gt_img = self.pool_64(gt_img)
        #     gt_depth = self.pool_64(gt_depth)

        pred_vis = th.cat([
            pred_img,
            pred_depth.repeat_interleave(3, dim=1),
            fg_mask.repeat_interleave(3, dim=1),
        ],
                          dim=-1)  # B, 3, H, W

        pred_vis_nv = th.cat([
            pred_cano['image_raw'],
            pred_cano_depth.repeat_interleave(3, dim=1),
            input_fg_mask.repeat_interleave(3, dim=1),
        ],
                             dim=-1)  # B, 3, H, W

        pred_vis = th.cat([pred_vis, pred_vis_nv], dim=-2)  # cat in H dim

        gt_vis = th.cat([
            gt_img,
            gt_depth.repeat_interleave(3, dim=1),
            th.zeros_like(gt_img)
        ],
                        dim=-1)  # TODO, fail to load depth. range [0, 1]

        # if 'conf_sigma' in pred:
        #     gt_vis = th.cat([gt_vis, fg_mask], dim=-1)  # placeholder

        # vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
        # st()
        vis = th.cat([gt_vis, pred_vis], dim=-2)
        # .permute(
        #     0, 2, 3, 1).cpu()
        vis_tensor = torchvision.utils.make_grid(vis, nrow=vis.shape[-1] //
                                                 64)  # HWC
        torchvision.utils.save_image(
            vis_tensor,
            f'{logger.get_dir()}/{self.step+self.resume_step}.jpg',
            value_range=(-1, 1),
            normalize=True)

        logger.log('log vis to: ',
                   f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')

        # self.writer.add_image(f'images',
        #                       vis,
        #                       self.step + self.resume_step,
        #                       dataformats='HWC')


class TrainLoop3DRecNVPatchSingleForward(TrainLoop3DRecNVPatch):

    def __init__(self,
                 *,
                 rec_model,
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
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
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
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)

    def forward_backward(self, batch, *args, **kwargs):
        # add patch sampling

        self.mp_trainer_rec.zero_grad()
        batch_size = batch['img_to_encoder'].shape[0]

        batch.pop('caption')  # not required
        batch.pop('ins')  # not required
        # batch.pop('nv_caption') # not required

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k:
                v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
                    v, th.Tensor) else v[i:i + self.microbatch]
                for k, v in batch.items()
            }

            # ! sample rendering patch
            target = {
                **self.eg3d_model(
                    c=micro['nv_c'],  # type: ignore
                    ws=None,
                    planes=None,
                    sample_ray_only=True,
                    fg_bbox=micro['nv_bbox']),  # rays o / dir
            }

            patch_rendering_resolution = self.eg3d_model.rendering_kwargs[
                'patch_rendering_resolution']  # type: ignore
            cropped_target = {
                k:
                th.empty_like(v)
                [..., :patch_rendering_resolution, :patch_rendering_resolution]
                if k not in [
                    'ins_idx', 'img_to_encoder', 'img_sr', 'nv_img_to_encoder',
                    'nv_img_sr', 'c', 'caption', 'nv_caption'
                ] else v
                for k, v in micro.items()
            }

            # crop according to uv sampling
            for j in range(micro['img'].shape[0]):
                top, left, height, width = target['ray_bboxes'][
                    j]  # list of tuple
                # for key in ('img', 'depth_mask', 'depth', 'depth_mask_sr'): # type: ignore
                for key in ('img', 'depth_mask', 'depth'):  # type: ignore
                    # target[key][i:i+1] = torchvision.transforms.functional.crop(
                    # cropped_target[key][
                    #     j:j + 1] = torchvision.transforms.functional.crop(
                    #         micro[key][j:j + 1], top, left, height, width)

                    cropped_target[f'{key}'][  # ! no nv_ here
                        j:j + 1] = torchvision.transforms.functional.crop(
                            micro[f'nv_{key}'][j:j + 1], top, left, height,
                            width)

            # ! cano view loss
            cano_target = {
                **self.eg3d_model(
                    c=micro['c'],  # type: ignore
                    ws=None,
                    planes=None,
                    sample_ray_only=True,
                    fg_bbox=micro['bbox']),  # rays o / dir
            }

            # cano_cropped_target = {
            #     k: th.empty_like(v)
            #     for k, v in cropped_target.items()
            # }

            # for j in range(micro['img'].shape[0]):
            #     top, left, height, width = cano_target['ray_bboxes'][
            #         j]  # list of tuple
            #     # for key in ('img', 'depth_mask', 'depth', 'depth_mask_sr'): # type: ignore
            #     for key in ('img', 'depth_mask', 'depth'):  # type: ignore
            #         # target[key][i:i+1] = torchvision.transforms.functional.crop(
            #         cano_cropped_target[key][
            #             j:j + 1] = torchvision.transforms.functional.crop(
            #                 micro[key][j:j + 1], top, left, height, width)

            # ! vit no amp
            latent = self.rec_model(img=micro['img_to_encoder'],
                                    behaviour='enc_dec_wo_triplane')

            # wrap forward within amp
            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                # c = th.cat([micro['nv_c'], micro['c']]),  # predict novel view here
                # c = th.cat([micro['nv_c'].repeat(3, 1), micro['c']]),  # predict novel view here
                instance_mv_num = batch_size // 4  # 4 pairs by default
                # instance_mv_num = 4
                # ! roll views for multi-view supervision
                c = th.cat([
                    micro['nv_c'].roll(instance_mv_num * i, dims=0)
                    for i in range(1, 4)
                ]
                           # + [micro['c']]
                           )  # predict novel view here

                ray_origins = th.cat(
                    [
                        target['ray_origins'].roll(instance_mv_num * i, dims=0)
                        for i in range(1, 4)
                    ]
                    # + [cano_target['ray_origins'] ]
                    ,
                    0)

                ray_directions = th.cat([
                    target['ray_directions'].roll(instance_mv_num * i, dims=0)
                    for i in range(1, 4)
                ]
                                        # + [cano_target['ray_directions'] ]
                                        )

                pred_nv_cano = self.rec_model(
                    # latent=latent.expand(2,),
                    latent={
                        'latent_after_vit': # ! triplane for rendering
                        # latent['latent_after_vit'].repeat(2, 1, 1, 1)
                        latent['latent_after_vit'].repeat(3, 1, 1, 1)
                    },
                    c=c,
                    behaviour='triplane_dec',
                    # ray_origins=target['ray_origins'],
                    # ray_directions=target['ray_directions'],
                    ray_origins=ray_origins,
                    ray_directions=ray_directions,
                )

                pred_nv_cano.update(
                    latent
                )  # torchvision.utils.save_image(pred_nv_cano['image_raw'], 'pred.png', normalize=True)
                # gt = {
                #     k: th.cat([v, cano_cropped_target[k]], 0)
                #     for k, v in cropped_target.items()
                # }
                gt = {
                    k:
                    th.cat(
                        [
                            v.roll(instance_mv_num * i, dims=0)
                            for i in range(1, 4)
                        ]
                        # + [cano_cropped_target[k] ]
                        ,
                        0)
                    for k, v in cropped_target.items()
                }  # torchvision.utils.save_image(gt['img'], 'gt.png', normalize=True)

                with self.rec_model.no_sync():  # type: ignore
                    loss, loss_dict, _ = self.loss_class(
                        pred_nv_cano,
                        gt,  # prepare merged data
                        step=self.step + self.resume_step,
                        test_mode=False,
                        return_fg_mask=True,
                        conf_sigma_l1=None,
                        conf_sigma_percl=None)
                    log_rec3d_loss_dict(loss_dict)

            self.mp_trainer_rec.backward(loss)

            # for name, p in self.rec_model.named_parameters():
            #     if p.grad is None:
            #         logger.log(f"found rec unused param: {name}")

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                micro_bs = micro['img_to_encoder'].shape[0]
                self.log_patch_img( # record one cano view and one novel view
                    cropped_target,
                    {
                        k: pred_nv_cano[k][-micro_bs:]
                        for k in ['image_raw', 'image_depth', 'image_mask']
                    },
                    {
                        k: pred_nv_cano[k][:micro_bs]
                        for k in ['image_raw', 'image_depth', 'image_mask']
                    },
                )

    def eval_loop(self):
        return super().eval_loop()

    @th.inference_mode()
    # def eval_loop(self, c_list:list):
    def eval_novelview_loop_old(self, camera=None):
        # novel view synthesis given evaluation camera trajectory

        all_loss_dict = []
        novel_view_micro = {}

        # ! randomly inference an instance

        export_mesh = True
        if export_mesh:
            Path(f'{logger.get_dir()}/FID_Cals/').mkdir(parents=True,
                                                        exist_ok=True)

        # for i in range(0, len(c_list), 1): # TODO, larger batch size for eval

        batch = {}
        # if camera is not None:
        #     # batch['c'] = camera.to(batch['c'].device())
        #     batch['c'] = camera.clone()
        # else:
        #     batch =

        for eval_idx, render_reference in enumerate(tqdm(self.eval_data)):

            if eval_idx > 500:
                break

            video_out = imageio.get_writer(
                f'{logger.get_dir()}/video_novelview_{self.step+self.resume_step}_{eval_idx}.mp4',
                mode='I',
                fps=25,
                codec='libx264')

            with open(
                    f'{logger.get_dir()}/triplane_{self.step+self.resume_step}_{eval_idx}_caption.txt',
                    'w') as f:
                f.write(render_reference['caption'])

            for key in ['ins', 'bbox', 'caption']:
                if key in render_reference:
                    render_reference.pop(key)

            real_flag = False
            mv_flag = False  # TODO, use full-instance for evaluation? Calculate the metrics.
            if render_reference['c'].shape[:2] == (1, 40):
                real_flag = True
                # real img monocular reconstruction
                # compat lst for enumerate
                render_reference = [{
                    k: v[0][idx:idx + 1]
                    for k, v in render_reference.items()
                } for idx in range(40)]

            elif render_reference['c'].shape[0] == 8:
                mv_flag = True

                render_reference = {
                    k: v[:4]
                    for k, v in render_reference.items()
                }

                # save gt
                torchvision.utils.save_image(
                    render_reference[0:4]['img'],
                    logger.get_dir() + '/FID_Cals/{}_inp.png'.format(eval_idx),
                    padding=0,
                    normalize=True,
                    value_range=(-1, 1),
                )
                # torchvision.utils.save_image(render_reference[4:8]['img'],
                #     logger.get_dir() + '/FID_Cals/{}_inp2.png'.format(eval_idx),
                #     padding=0,
                #     normalize=True,
                #     value_range=(-1,1),
                # )

            else:
                # compat lst for enumerate
                st()
                render_reference = [{
                    k: v[idx:idx + 1]
                    for k, v in render_reference.items()
                } for idx in range(40)]

                # ! single-view version
                render_reference[0]['img_to_encoder'] = render_reference[14][
                    'img_to_encoder']  # encode side view
                render_reference[0]['img'] = render_reference[14][
                    'img']  # encode side view

                # save gt
                torchvision.utils.save_image(
                    render_reference[0]['img'],
                    logger.get_dir() + '/FID_Cals/{}_gt.png'.format(eval_idx),
                    padding=0,
                    normalize=True,
                    value_range=(-1, 1))

            # ! TODO, merge with render_video_given_triplane later
            for i, batch in enumerate(render_reference):
                # for i in range(0, 8, self.microbatch):
                # c = c_list[i].to(dist_util.dev()).reshape(1, -1)
                micro = {k: v.to(dist_util.dev()) for k, v in batch.items()}

                st()
                if i == 0:
                    if mv_flag:
                        novel_view_micro = None
                    else:
                        novel_view_micro = {
                            k:
                            v[0:1].to(dist_util.dev()).repeat_interleave(
                                # v[14:15].to(dist_util.dev()).repeat_interleave(
                                micro['img'].shape[0],
                                0) if isinstance(v, th.Tensor) else v[0:1]
                            for k, v in batch.items()
                        }

                else:
                    if i == 1:

                        # ! output mesh
                        if export_mesh:

                            # ! get planes first
                            # self.latent_name = 'latent_normalized'  # normalized triplane latent

                            # ddpm_latent = {
                            #     self.latent_name: planes,
                            # }
                            # ddpm_latent.update(self.rec_model(latent=ddpm_latent, behaviour='decode_after_vae_no_render'))

                            # mesh_size = 512
                            # mesh_size = 256
                            mesh_size = 384
                            # mesh_size = 320
                            # mesh_thres = 3 # TODO, requires tuning
                            # mesh_thres = 5 # TODO, requires tuning
                            mesh_thres = 10  # TODO, requires tuning
                            import mcubes
                            import trimesh
                            dump_path = f'{logger.get_dir()}/mesh/'

                            os.makedirs(dump_path, exist_ok=True)

                            grid_out = self.rec_model(
                                latent=pred,
                                grid_size=mesh_size,
                                behaviour='triplane_decode_grid',
                            )

                            vtx, faces = mcubes.marching_cubes(
                                grid_out['sigma'].squeeze(0).squeeze(
                                    -1).cpu().numpy(), mesh_thres)
                            vtx = vtx / (mesh_size - 1) * 2 - 1

                            # vtx_tensor = th.tensor(vtx, dtype=th.float32, device=dist_util.dev()).unsqueeze(0)
                            # vtx_colors = self.model.synthesizer.forward_points(planes, vtx_tensor)['rgb'].squeeze(0).cpu().numpy()  # (0, 1)
                            # vtx_colors = (vtx_colors * 255).astype(np.uint8)

                            # mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)
                            mesh = trimesh.Trimesh(
                                vertices=vtx,
                                faces=faces,
                            )

                            mesh_dump_path = os.path.join(
                                dump_path, f'{eval_idx}.ply')
                            mesh.export(mesh_dump_path, 'ply')

                            print(f"Mesh dumped to {dump_path}")
                            del grid_out, mesh
                            th.cuda.empty_cache()
                            # return
                            # st()

                    # if novel_view_micro['c'].shape[0] < micro['img'].shape[0]:
                    novel_view_micro = {
                        k:
                        v[0:1].to(dist_util.dev()).repeat_interleave(
                            micro['img'].shape[0], 0)
                        for k, v in novel_view_micro.items()
                    }

                pred = self.rec_model(img=novel_view_micro['img_to_encoder'],
                                      c=micro['c'])  # pred: (B, 3, 64, 64)
                # target = {
                #     'img': micro['img'],
                #     'depth': micro['depth'],
                #     'depth_mask': micro['depth_mask']
                # }
                # targe

                # if not export_mesh:
                if not real_flag:
                    _, loss_dict = self.loss_class(pred, micro, test_mode=True)
                    all_loss_dict.append(loss_dict)

                # ! move to other places, add tensorboard

                # pred_vis = th.cat([
                #     pred['image_raw'],
                #     -pred['image_depth'].repeat_interleave(3, dim=1)
                # ],
                #                   dim=-1)

                # normalize depth
                # if True:
                pred_depth = pred['image_depth']
                pred_depth = (pred_depth - pred_depth.min()) / (
                    pred_depth.max() - pred_depth.min())
                if 'image_sr' in pred:

                    if pred['image_sr'].shape[-1] == 512:

                        pred_vis = th.cat([
                            micro['img_sr'],
                            self.pool_512(pred['image_raw']), pred['image_sr'],
                            self.pool_512(pred_depth).repeat_interleave(3,
                                                                        dim=1)
                        ],
                                          dim=-1)

                    elif pred['image_sr'].shape[-1] == 256:

                        pred_vis = th.cat([
                            micro['img_sr'],
                            self.pool_256(pred['image_raw']), pred['image_sr'],
                            self.pool_256(pred_depth).repeat_interleave(3,
                                                                        dim=1)
                        ],
                                          dim=-1)

                    else:
                        pred_vis = th.cat([
                            micro['img_sr'],
                            self.pool_128(pred['image_raw']),
                            self.pool_128(pred['image_sr']),
                            self.pool_128(pred_depth).repeat_interleave(3,
                                                                        dim=1)
                        ],
                                          dim=-1)

                else:
                    # pred_vis = th.cat([
                    #     self.pool_64(micro['img']), pred['image_raw'],
                    #     pred_depth.repeat_interleave(3, dim=1)
                    # ],
                    #                   dim=-1)  # B, 3, H, W

                    pooled_depth = self.pool_128(pred_depth).repeat_interleave(
                        3, dim=1)
                    pred_vis = th.cat(
                        [
                            # self.pool_128(micro['img']),
                            self.pool_128(novel_view_micro['img']
                                          ),  # use the input here
                            self.pool_128(pred['image_raw']),
                            pooled_depth,
                        ],
                        dim=-1)  # B, 3, H, W

                vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
                vis = vis * 127.5 + 127.5
                vis = vis.clip(0, 255).astype(np.uint8)

                if export_mesh:
                    # save image
                    torchvision.utils.save_image(
                        pred['image_raw'],
                        logger.get_dir() +
                        '/FID_Cals/{}_{}.png'.format(eval_idx, i),
                        padding=0,
                        normalize=True,
                        value_range=(-1, 1))

                    torchvision.utils.save_image(
                        pooled_depth,
                        logger.get_dir() +
                        '/FID_Cals/{}_{}_dpeth.png'.format(eval_idx, i),
                        padding=0,
                        normalize=True,
                        value_range=(0, 1))

                # st()

                for j in range(vis.shape[0]):
                    video_out.append_data(vis[j])

            video_out.close()

        # if not export_mesh:
        if not real_flag or mv_flag:
            val_scores_for_logging = calc_average_loss(all_loss_dict)
            with open(os.path.join(logger.get_dir(), 'scores_novelview.json'),
                      'a') as f:
                json.dump({'step': self.step, **val_scores_for_logging}, f)

            # * log to tensorboard
            for k, v in val_scores_for_logging.items():
                self.writer.add_scalar(f'Eval/NovelView/{k}', v,
                                       self.step + self.resume_step)

        del video_out
        # del pred_vis
        # del pred

        th.cuda.empty_cache()

    @th.inference_mode()
    # def eval_loop(self, c_list:list):
    def eval_novelview_loop(self, camera=None, save_latent=False):
        # novel view synthesis given evaluation camera trajectory
        if save_latent:  # for diffusion learning
            latent_dir = Path(f'{logger.get_dir()}/latent_dir')
            latent_dir.mkdir(exist_ok=True, parents=True)

            # wds_path = os.path.join(logger.get_dir(), 'latent_dir',
            #                         f'wds-%06d.tar')
            # sink = wds.ShardWriter(wds_path, start_shard=0)

        # eval_batch_size = 20
        # eval_batch_size = 1
        eval_batch_size = 40  # ! for i23d

        for eval_idx, micro in enumerate(tqdm(self.eval_data)):

            # if eval_idx > 500:
            #     break

            latent = self.rec_model(
                img=micro['img_to_encoder'][:4],
                behaviour='encoder_vae')  # pred: (B, 3, 64, 64)
            # torchvision.utils.save_image(micro['img'], 'inp.jpg')
            if micro['img'].shape[0] == 40:
                assert eval_batch_size == 40

            if save_latent:
                # np.save(f'{logger.get_dir()}/latent_dir/{eval_idx}.npy', latent[self.latent_name].cpu().numpy())

                latent_save_dir = f'{logger.get_dir()}/latent_dir/{micro["ins"][0]}'
                Path(latent_save_dir).mkdir(parents=True, exist_ok=True)

                np.save(f'{latent_save_dir}/latent.npy',
                        latent[self.latent_name][0].cpu().numpy())
                assert all([
                    micro['ins'][0] == micro['ins'][i]
                    for i in range(micro['c'].shape[0])
                ])  # ! assert same instance

                # for i in range(micro['img'].shape[0]):

                #     compressed_sample = {
                #         'latent':latent[self.latent_name][0].cpu().numpy(), # 12 32 32
                #         'caption': micro['caption'][0].encode('utf-8'),
                #         'ins': micro['ins'][0].encode('utf-8'),
                #         'c': micro['c'][i].cpu().numpy(),
                #         'img': micro['img'][i].cpu().numpy() # 128x128, for diffusion log
                #     }

                #     sink.write({
                #         "__key__": f"sample_{eval_idx*eval_batch_size+i:07d}",
                #         'sample.pyd': compressed_sample
                #     })

            if eval_idx < 50:
                # if False:
                self.render_video_given_triplane(
                    latent[self.latent_name], # B 12 32 32
                    self.rec_model,  # compatible with join_model
                    name_prefix=f'{self.step + self.resume_step}_{eval_idx}',
                    save_img=False,
                    render_reference={'c': camera},
                    save_mesh=True)


class TrainLoop3DRecNVPatchSingleForwardMV(TrainLoop3DRecNVPatchSingleForward):

    def __init__(self,
                 *,
                 rec_model,
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
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
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
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)

    def forward_backward(self, batch, behaviour='g_step', *args, **kwargs):
        # add patch sampling

        self.mp_trainer_rec.zero_grad()
        batch_size = batch['img_to_encoder'].shape[0]

        batch.pop('caption')  # not required
        batch.pop('ins')  # not required
        if '__key__' in batch.keys():
            batch.pop('__key__')

        for i in range(0, batch_size, self.microbatch):

            micro = {
                k:
                v[i:i + self.microbatch].to(dist_util.dev()) if isinstance(
                    v, th.Tensor) else v[i:i + self.microbatch]
                for k, v in batch.items()
            }

            # ! sample rendering patch
            # nv_c = th.cat([micro['nv_c'], micro['c']])
            nv_c = th.cat([micro['nv_c'], micro['c']])
            # nv_c = micro['nv_c']
            target = {
                **self.eg3d_model(
                    c=nv_c,  # type: ignore
                    ws=None,
                    planes=None,
                    sample_ray_only=True,
                    fg_bbox=th.cat([micro['nv_bbox'], micro['bbox']])),  # rays o / dir
            }

            patch_rendering_resolution = self.eg3d_model.rendering_kwargs[
                'patch_rendering_resolution']  # type: ignore
            cropped_target = {
                k:
                th.empty_like(v).repeat_interleave(2, 0)
                # th.empty_like(v).repeat_interleave(1, 0)
                [..., :patch_rendering_resolution, :patch_rendering_resolution]
                if k not in [
                    'ins_idx', 'img_to_encoder', 'img_sr', 'nv_img_to_encoder',
                    'nv_img_sr', 'c', 'caption', 'nv_caption'
                ] else v
                for k, v in micro.items()
            }

            # crop according to uv sampling
            for j in range(2 * self.microbatch):
                top, left, height, width = target['ray_bboxes'][
                    j]  # list of tuple
                # for key in ('img', 'depth_mask', 'depth', 'depth_mask_sr'): # type: ignore
                for key in ('img', 'depth_mask', 'depth'):  # type: ignore

                    if j < self.microbatch:
                        cropped_target[f'{key}'][  # ! no nv_ here
                            j:j + 1] = torchvision.transforms.functional.crop(
                                micro[f'nv_{key}'][j:j + 1], top, left, height,
                                width)
                    else:
                        cropped_target[f'{key}'][  # ! no nv_ here
                            j:j + 1] = torchvision.transforms.functional.crop(
                                micro[f'{key}'][j - self.microbatch:j -
                                                self.microbatch + 1], top,
                                left, height, width)

            # for j in range(batch_size, 2*batch_size, 1):
            #     top, left, height, width = target['ray_bboxes'][
            #         j]  # list of tuple
            #     # for key in ('img', 'depth_mask', 'depth', 'depth_mask_sr'): # type: ignore
            #     for key in ('img', 'depth_mask', 'depth'):  # type: ignore

            #         cropped_target[f'{key}'][  # ! no nv_ here
            #             j:j + 1] = torchvision.transforms.functional.crop(
            #                 micro[f'{key}'][j-batch_size:j-batch_size + 1], top, left, height,
            #                 width)

            # ! vit no amp
            latent = self.rec_model(img=micro['img_to_encoder'],
                                    behaviour='enc_dec_wo_triplane')

            # wrap forward within amp
            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                # c = th.cat([micro['nv_c'], micro['c']]),  # predict novel view here
                # c = th.cat([micro['nv_c'].repeat(3, 1), micro['c']]),  # predict novel view here
                # instance_mv_num = batch_size // 4  # 4 pairs by default
                # instance_mv_num = 4
                # ! roll views for multi-view supervision
                # c = micro['nv_c']
                ray_origins = target['ray_origins']
                ray_directions = target['ray_directions']

                pred_nv_cano = self.rec_model(
                    # latent=latent.expand(2,),
                    latent={
                        'latent_after_vit':  # ! triplane for rendering
                        latent['latent_after_vit'].repeat_interleave(4, dim=0).repeat(2,1,1,1)  # NV=4
                        # latent['latent_after_vit'].repeat_interleave(8, dim=0)  # NV=4
                    },
                    c=nv_c,
                    behaviour='triplane_dec',
                    ray_origins=ray_origins,
                    ray_directions=ray_directions,
                )

                pred_nv_cano.update(
                    latent
                )  # torchvision.utils.save_image(pred_nv_cano['image_raw'], 'pred.png', normalize=True)
                gt = cropped_target

                with self.rec_model.no_sync():  # type: ignore
                    loss, loss_dict, _ = self.loss_class(
                        pred_nv_cano,
                        gt,  # prepare merged data
                        step=self.step + self.resume_step,
                        test_mode=False,
                        return_fg_mask=True,
                        behaviour=behaviour,
                        conf_sigma_l1=None,
                        conf_sigma_percl=None)
                    log_rec3d_loss_dict(loss_dict)

            self.mp_trainer_rec.backward(loss)

            # for name, p in self.rec_model.named_parameters():
            #     if p.grad is None:
            #         logger.log(f"found rec unused param: {name}")
            # torchvision.utils.save_image(cropped_target['img'], 'gt.png', normalize=True)
            # torchvision.utils.save_image( pred_nv_cano['image_raw'], 'pred.png', normalize=True)

            if dist_util.get_rank() == 0 and self.step % 500 == 0 and i == 0:
                try:
                    torchvision.utils.save_image(
                        th.cat(
                            [cropped_target['img'], pred_nv_cano['image_raw']
                             ], ),
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg',
                        normalize=True)

                    logger.log(
                        'log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')
                except Exception as e:
                    logger.log(e)

                # micro_bs = micro['img_to_encoder'].shape[0]
                # self.log_patch_img( # record one cano view and one novel view
                #     cropped_target,
                #     {
                #         k: pred_nv_cano[k][0:1]
                #         for k in ['image_raw', 'image_depth', 'image_mask']
                #     },
                #     {
                #         k: pred_nv_cano[k][1:2]
                #         for k in ['image_raw', 'image_depth', 'image_mask']
                #     },
                # )

    # def save(self):
    #     return super().save()


class TrainLoop3DRecNVPatchSingleForwardMVAdvLoss(
        TrainLoop3DRecNVPatchSingleForwardMV):

    def __init__(self,
                 *,
                 rec_model,
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
                 use_fp16=False,
                 fp16_scale_growth=0.001,
                 weight_decay=0,
                 lr_anneal_steps=0,
                 iterations=10001,
                 load_submodule_name='',
                 ignore_resume_opt=False,
                 model_name='rec',
                 use_amp=False,
                 **kwargs):
        super().__init__(rec_model=rec_model,
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
                         use_fp16=use_fp16,
                         fp16_scale_growth=fp16_scale_growth,
                         weight_decay=weight_decay,
                         lr_anneal_steps=lr_anneal_steps,
                         iterations=iterations,
                         load_submodule_name=load_submodule_name,
                         ignore_resume_opt=ignore_resume_opt,
                         model_name=model_name,
                         use_amp=use_amp,
                         **kwargs)

        # create discriminator
        disc_params = self.loss_class.get_trainable_parameters()

        self.mp_trainer_disc = MixedPrecisionTrainer(
            model=self.loss_class.discriminator,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
            model_name='disc',
            use_amp=use_amp,
            model_params=disc_params)

        # st() # check self.lr
        self.opt_disc = AdamW(
            self.mp_trainer_disc.master_params,
            lr=self.lr,  # follow sd code base
            betas=(0, 0.999),
            eps=1e-8)

        # TODO, is loss cls already in the DDP?
        if self.use_ddp:
            self.ddp_disc = DDP(
                self.loss_class.discriminator,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            self.ddp_disc = self.loss_class.discriminator

    # def run_st

    # def run_step(self, batch, *args):
    #     self.forward_backward(batch)
    #     took_step = self.mp_trainer_rec.optimize(self.opt)
    #     if took_step:
    #         self._update_ema()
    #     self._anneal_lr()
    #     self.log_step()

    def save(self, mp_trainer=None, model_name='rec'):
        if mp_trainer is None:
            mp_trainer = self.mp_trainer_rec

        def save_checkpoint(rate, params):
            state_dict = mp_trainer.master_params_to_state_dict(params)
            if dist_util.get_rank() == 0:
                logger.log(f"saving model {model_name} {rate}...")
                if not rate:
                    filename = f"model_{model_name}{(self.step+self.resume_step):07d}.pt"
                else:
                    filename = f"ema_{model_name}_{rate}_{(self.step+self.resume_step):07d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename),
                                 "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, mp_trainer.master_params)

        dist.barrier()

    def run_step(self, batch, step='g_step'):
        # self.forward_backward(batch)

        if step == 'g_step':
            self.forward_backward(batch, behaviour='g_step')
            took_step_g_rec = self.mp_trainer_rec.optimize(self.opt)

            if took_step_g_rec:
                self._update_ema()  # g_ema

        elif step == 'd_step':
            self.forward_backward(batch, behaviour='d_step')
            _ = self.mp_trainer_disc.optimize(self.opt_disc)

        self._anneal_lr()
        self.log_step()

    def run_loop(self, batch=None):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            batch = next(self.data)
            self.run_step(batch, 'g_step')

            batch = next(self.data)
            self.run_step(batch, 'd_step')

            if self.step % 1000 == 0:
                dist_util.synchronize()
                if self.step % 10000 == 0:
                    th.cuda.empty_cache()  # avoid memory leak

            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                out = logger.dumpkvs()
                # * log to tensorboard
                for k, v in out.items():
                    self.writer.add_scalar(f'Loss/{k}', v,
                                           self.step + self.resume_step)

            if self.step % self.eval_interval == 0 and self.step != 0:
                if dist_util.get_rank() == 0:
                    try:
                        self.eval_loop()
                    except Exception as e:
                        logger.log(e)
                dist_util.synchronize()

            # if self.step % self.save_interval == 0 and self.step != 0:
            if self.step % self.save_interval == 0:
                self.save()
                self.save(self.mp_trainer_disc,
                          self.mp_trainer_disc.model_name)
                dist_util.synchronize()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST",
                                  "") and self.step > 0:
                    return

            self.step += 1

            if self.step > self.iterations:
                logger.log('reached maximum iterations, exiting')

                # Save the last checkpoint if it wasn't already saved.
                if (self.step -
                        1) % self.save_interval != 0 and self.step != 1:
                    self.save()

                exit()

        # Save the last checkpoint if it wasn't already saved.
        # if (self.step - 1) % self.save_interval != 0 and self.step != 1:
        if (self.step - 1) % self.save_interval != 0:
            self.save()  # save rec
            self.save(self.mp_trainer_disc, self.mp_trainer_disc.model_name)
