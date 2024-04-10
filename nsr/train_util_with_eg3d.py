import copy
import functools
import json
import os
from pathlib import Path
from pdb import set_trace as st

import blobfile as bf
import imageio
import numpy as np
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

from .train_util import TrainLoop3DRec


class TrainLoop3DRecEG3D(TrainLoop3DRec):

    def __init__(self,
                 *,
                 G,
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
                #  hybrid_training=False,
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
        self.G = G
        # self.hybrid_training = hybrid_training

        self.pool_224 = th.nn.AdaptiveAvgPool2d((224, 224))

    @th.no_grad()
    def run_G(
        self,
        z,
        c,
        swapping_prob,
        neural_rendering_resolution,
        update_emas=False,
        return_raw_only=False,
    ):
        """add truncation psi

        Args:
            z (_type_): _description_
            c (_type_): _description_
            swapping_prob (_type_): _description_
            neural_rendering_resolution (_type_): _description_
            update_emas (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        c_gen_conditioning = th.zeros_like(c)

        # ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)

        ws = self.G.mapping(
            z,
            c_gen_conditioning,
            truncation_psi=0.7,
            truncation_cutoff=None,
            update_emas=update_emas,
        )

        gen_output = self.G.synthesis(
            ws,  # BS * 14 * 512
            c,
            neural_rendering_resolution=neural_rendering_resolution,
            update_emas=update_emas,
            noise_mode='const',
            return_raw_only=return_raw_only
            # return_meta=True # return feature_volume
        )  # fix the SynthesisLayer modulation noise, otherviwe the same latent code may output two different ID

        return gen_output, ws

    def run_loop(self, batch=None):
        while (not self.lr_anneal_steps
               or self.step + self.resume_step < self.lr_anneal_steps):

            # let all processes sync up before starting with a new epoch of training
            dist_util.synchronize()

            # batch, cond = next(self.data)
            # if batch is None:
            batch = next(self.data)
            # batch = self.run_G()

            self.run_step(batch)
            if self.step % self.log_interval == 0 and dist_util.get_rank(
            ) == 0:
                out = logger.dumpkvs()
                # * log to tensorboard
                for k, v in out.items():
                    self.writer.add_scalar(f'Loss/{k}', v,
                                           self.step + self.resume_step)

            if self.step % self.eval_interval == 0 and self.step != 0:
                # if dist_util.get_rank() == 0:
                    # self.eval_loop()
                    # self.eval_novelview_loop()
                    # let all processes sync up before starting with a new epoch of training
                dist_util.synchronize()

            if self.step % self.save_interval == 0:
                self.save()
                dist_util.synchronize()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST",
                                  "") and self.step > 0:
                    return

            self.step += 1

            if self.step > self.iterations:
                print('reached maximum iterations, exiting')

                # Save the last checkpoint if it wasn't already saved.
                if (self.step - 1) % self.save_interval != 0:
                    self.save()

                exit()

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, *args):
        self.forward_backward(batch)
        took_step = self.mp_trainer_rec.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, *args, **kwargs):

        self.mp_trainer_rec.zero_grad()

        batch_size = batch['c'].shape[0]

        for i in range(0, batch_size, self.microbatch):

            micro = {'c': batch['c'].to(dist_util.dev())}

            with th.no_grad():  # * infer gt
                eg3d_batch, ws = self.run_G(
                    z=th.randn(micro['c'].shape[0],
                                512).to(dist_util.dev()),
                    c=micro['c'].to(dist_util.dev(
                    )),  # use real img pose here? or synthesized pose.
                    swapping_prob=0,
                    neural_rendering_resolution=128)

            micro.update({
                'img':
                eg3d_batch['image_raw'],  # gt
                'img_to_encoder':
                self.pool_224(eg3d_batch['image']),
                'depth':
                eg3d_batch['image_depth'],
                'img_sr': eg3d_batch['image'],
            })

            last_batch = (i + self.microbatch) >= batch_size

            # wrap forward within amp
            with th.autocast(device_type='cuda',
                             dtype=th.float16,
                             enabled=self.mp_trainer_rec.use_amp):

                pred_gen_output = self.rec_model(
                    img=micro['img_to_encoder'],  # pool from 512
                    c=micro['c'])  # pred: (B, 3, 64, 64)

                # target = micro
                target = dict(
                    img=eg3d_batch['image_raw'],
                    shape_synthesized=eg3d_batch['shape_synthesized'],
                    img_sr=eg3d_batch['image'],
                )

                pred_gen_output['shape_synthesized_query'] = {
                    'coarse_densities':
                    pred_gen_output['shape_synthesized']['coarse_densities'],
                    'image_depth': pred_gen_output['image_depth'],
                }

                eg3d_batch['shape_synthesized']['image_depth'] = eg3d_batch['image_depth']

                batch_size, num_rays, _, _ = pred_gen_output[
                    'shape_synthesized']['coarse_densities'].shape


                for coord_key in ['fine_coords']:  # TODO add surface points

                    sigma = self.rec_model(
                        latent=pred_gen_output['latent_denormalized'],
                        coordinates=eg3d_batch['shape_synthesized'][coord_key],
                        directions=th.randn_like(
                            eg3d_batch['shape_synthesized'][coord_key]),
                        behaviour='triplane_renderer',
                    )['sigma']

                    rendering_kwargs = self.rec_model(
                        behaviour='get_rendering_kwargs')

                    sigma = sigma.reshape(
                        batch_size, num_rays,
                        rendering_kwargs['depth_resolution_importance'], 1)

                    pred_gen_output['shape_synthesized_query'][
                        f"{coord_key.split('_')[0]}_densities"] = sigma

                # * 2D reconstruction loss
                if last_batch or not self.use_ddp:
                    loss, loss_dict = self.loss_class(pred_gen_output,
                                                      target,
                                                      test_mode=False)
                else:
                    with self.rec_model.no_sync():  # type: ignore
                        loss, loss_dict = self.loss_class(pred_gen_output,
                                                          target,
                                                          test_mode=False)

                # * fully mimic 3D geometry output

                loss_shape = self.calc_shape_rec_loss(
                    pred_gen_output['shape_synthesized_query'],
                    eg3d_batch['shape_synthesized'])

                loss += loss_shape.mean()

                # * add feature loss on feature_image
                loss_feature_volume = th.nn.functional.mse_loss(
                    eg3d_batch['feature_volume'],
                    pred_gen_output['feature_volume'])
                loss += loss_feature_volume * 0.1

                loss_ws = th.nn.functional.mse_loss(
                    ws[:, -1:, :],
                    pred_gen_output['sr_w_code'])
                loss += loss_ws * 0.1

                loss_dict.update(
                    dict(loss_feature_volume=loss_feature_volume,
                         loss=loss,
                         loss_shape=loss_shape, 
                         loss_ws=loss_ws))

                loss_dict.update(dict(loss_feature_volume=loss_feature_volume, loss=loss, loss_shape=loss_shape))

                log_rec3d_loss_dict(loss_dict)


            self.mp_trainer_rec.backward(loss)

            # for name, p in self.ddp_model.named_parameters():
            #     if p.grad is None:
            #         print(f"found rec unused param: {name}")

            if dist_util.get_rank() == 0 and self.step % 500 == 0:
                with th.no_grad():
                    # gt_vis = th.cat([batch['img'], batch['depth']], dim=-1)

                    pred_img = pred_gen_output['image_raw']
                    gt_img = micro['img']

                    if 'depth' in micro:
                        gt_depth = micro['depth']
                        if gt_depth.ndim == 3:
                            gt_depth = gt_depth.unsqueeze(1)
                        gt_depth = (gt_depth - gt_depth.min()) / (
                            gt_depth.max() - gt_depth.min())

                        pred_depth = pred_gen_output['image_depth']
                        pred_depth = (pred_depth - pred_depth.min()) / (
                            pred_depth.max() - pred_depth.min())

                        gt_vis = th.cat(
                            [gt_img,
                             gt_depth.repeat_interleave(3, dim=1)],
                            dim=-1)  # TODO, fail to load depth. range [0, 1]
                    else:

                        gt_vis = th.cat(
                            [gt_img],
                            dim=-1)  # TODO, fail to load depth. range [0, 1]

                    if 'image_sr' in pred_gen_output:
                        pred_img = th.cat([
                            self.pool_512(pred_img),
                            pred_gen_output['image_sr']
                        ],
                                          dim=-1)
                        pred_depth = self.pool_512(pred_depth)
                        gt_depth = self.pool_512(gt_depth)

                        gt_vis = th.cat(
                            [self.pool_512(micro['img']), micro['img_sr'], gt_depth.repeat_interleave(3, dim=1)],
                            dim=-1)

                    pred_vis = th.cat(
                        [pred_img,
                         pred_depth.repeat_interleave(3, dim=1)],
                        dim=-1)  # B, 3, H, W

                    vis = th.cat([gt_vis, pred_vis], dim=-2)[0].permute(
                        1, 2, 0).cpu()  # ! pred in range[-1, 1]
                    # vis_grid = torchvision.utils.make_grid(vis) # HWC
                    vis = vis.numpy() * 127.5 + 127.5
                    vis = vis.clip(0, 255).astype(np.uint8)
                    Image.fromarray(vis).save(
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')
                    print(
                        'log vis to: ',
                        f'{logger.get_dir()}/{self.step+self.resume_step}.jpg')

                    # self.writer.add_image(f'images',
                    #                       vis,
                    #                       self.step + self.resume_step,
                    #                       dataformats='HWC')
            return pred_gen_output

    def calc_shape_rec_loss(
        self,
        pred_shape: dict,
        gt_shape: dict,
    ):

        loss_shape, loss_shape_dict = self.loss_class.calc_shape_rec_loss(
            pred_shape,
            gt_shape,
            dist_util.dev(),
        )

        for loss_k, loss_v in loss_shape_dict.items():
            # training_stats.report('Loss/E/3D/{}'.format(loss_k), loss_v)
            log_rec3d_loss_dict({'Loss/3D/{}'.format(loss_k): loss_v})

        return loss_shape

    # @th.inference_mode()
    def eval_novelview_loop(self):
        # novel view synthesis given evaluation camera trajectory
        video_out = imageio.get_writer(
            f'{logger.get_dir()}/video_novelview_real_{self.step+self.resume_step}.mp4',
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
                    k: v[0:1].to(dist_util.dev()).repeat_interleave(
                        micro['img'].shape[0], 0)
                    for k, v in batch.items()
                }

            else:
                # if novel_view_micro['c'].shape[0] < micro['img'].shape[0]:
                novel_view_micro = {
                    k: v[0:1].to(dist_util.dev()).repeat_interleave(
                        micro['img'].shape[0], 0)
                    for k, v in novel_view_micro.items()
                }
            
            # st()

            pred = self.rec_model(img=novel_view_micro['img_to_encoder'],
                              c=micro['c'])  # pred: (B, 3, 64, 64)

            # _, loss_dict = self.loss_class(pred, micro, test_mode=True)
            # all_loss_dict.append(loss_dict)

            # ! move to other places, add tensorboard

            pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())
            if 'image_sr' in pred:
                pred_vis = th.cat([
                    micro['img_sr'],
                    self.pool_512(pred['image_raw']), pred['image_sr'],
                    self.pool_512(pred_depth).repeat_interleave(3, dim=1)
                ],
                                  dim=-1)
            else:
                pred_vis = th.cat([
                    self.pool_128(micro['img']), pred['image_raw'],
                    pred_depth.repeat_interleave(3, dim=1)
                ],
                                  dim=-1)  # B, 3, H, W

            vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
            vis = vis * 127.5 + 127.5
            vis = vis.clip(0, 255).astype(np.uint8)

            for j in range(vis.shape[0]):
                video_out.append_data(vis[j])

        video_out.close()

        # val_scores_for_logging = calc_average_loss(all_loss_dict)
        # with open(os.path.join(logger.get_dir(), 'scores_novelview.json'),
        #           'a') as f:
        #     json.dump({'step': self.step, **val_scores_for_logging}, f)

        # * log to tensorboard
        # for k, v in val_scores_for_logging.items():
        #     self.writer.add_scalar(f'Eval/NovelView/{k}', v,
        #                            self.step + self.resume_step)
        del video_out
        # del pred_vis
        # del pred

        th.cuda.empty_cache()
        # self.eval_novelview_loop_eg3d()


    @th.inference_mode()
    def eval_novelview_loop_eg3d(self):
        # novel view synthesis given evaluation camera trajectory
        video_out = imageio.get_writer(
            f'{logger.get_dir()}/video_novelview_synthetic_{self.step+self.resume_step}.mp4',
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
                # novel_view_micro = {
                #     k: v[0:1].to(dist_util.dev()).repeat_interleave(
                #         micro['img'].shape[0], 0)
                #     for k, v in batch.items()
                # }

                with th.no_grad():  # * infer gt
                    eg3d_batch, _ = self.run_G(
                        z=th.randn(micro['c'].shape[0],
                                    512).to(dist_util.dev()),
                        c=micro['c'].to(dist_util.dev(
                        )),  # use real img pose here? or synthesized pose.
                        swapping_prob=0,
                        neural_rendering_resolution=128)

                novel_view_micro.update({
                    'img':
                    eg3d_batch['image_raw'],  # gt
                    'img_to_encoder':
                    self.pool_224(eg3d_batch['image']),
                    'depth':
                    eg3d_batch['image_depth'],
                })

            else:
                # if novel_view_micro['c'].shape[0] < micro['img'].shape[0]:
                novel_view_micro = {
                    k: v[0:1].to(dist_util.dev()).repeat_interleave(
                        micro['img'].shape[0], 0)
                    for k, v in novel_view_micro.items()
                }

            # st()

            pred = self.rec_model(img=novel_view_micro['img_to_encoder'],
                              c=micro['c'])  # pred: (B, 3, 64, 64)

            # _, loss_dict = self.loss_class(pred, micro, test_mode=True)
            # all_loss_dict.append(loss_dict)

            # ! move to other places, add tensorboard

            pred_depth = pred['image_depth']
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() -
                                                            pred_depth.min())
            if 'image_sr' in pred:
                pred_vis = th.cat([
                    micro['img_sr'],
                    self.pool_512(pred['image_raw']), pred['image_sr'],
                    self.pool_512(pred_depth).repeat_interleave(3, dim=1)
                ],
                                  dim=-1)
            else:
                pred_vis = th.cat([
                    self.pool_128(micro['img']), pred['image_raw'],
                    pred_depth.repeat_interleave(3, dim=1)
                ],
                                  dim=-1)  # B, 3, H, W

            vis = pred_vis.permute(0, 2, 3, 1).cpu().numpy()
            vis = vis * 127.5 + 127.5
            vis = vis.clip(0, 255).astype(np.uint8)

            for j in range(vis.shape[0]):
                video_out.append_data(vis[j])

        video_out.close()

        # val_scores_for_logging = calc_average_loss(all_loss_dict)
        # with open(os.path.join(logger.get_dir(), 'scores_novelview.json'),
        #           'a') as f:
        #     json.dump({'step': self.step, **val_scores_for_logging}, f)

        # # * log to tensorboard
        # for k, v in val_scores_for_logging.items():
        #     self.writer.add_scalar(f'Eval/NovelView/{k}', v,
        #                            self.step + self.resume_step)
        del video_out
        # del pred_vis
        # del pred

        th.cuda.empty_cache()