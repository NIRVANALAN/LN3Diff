"""
Train a diffusion model on images.
"""
import json
import sys
import os

sys.path.append('.')
import torch.distributed as dist

import traceback

import torch as th
import torch.multiprocessing as mp
import numpy as np

import argparse
import dnnlib
from dnnlib.util import EasyDict, InfiniteSampler
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)

# from nsr.train_util import TrainLoop3DRec as TrainLoop

import nsr
from nsr.script_util import create_3DAE_model, encoder_and_nsr_defaults, loss_defaults, rendering_options_defaults, eg3d_options_default
from datasets.shapenet import load_data, load_eval_data, load_memory_data
from nsr.losses.builder import E3DGELossClass
from torch.utils.data import Subset
from datasets.eg3d_dataset import init_dataset_kwargs
from utils.torch_utils import legacy, misc

from pdb import set_trace as st

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# th.backends.cuda.matmul.allow_tf32 = True # https://huggingface.co/docs/diffusers/optimization/fp16

SEED = 0


def training_loop(args):
    # def training_loop(args):
    dist_util.setup_dist(args)

    # dist.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank, world_size=th.cuda.device_count())
    print(f"{args.local_rank=} init complete")
    th.cuda.set_device(args.local_rank)
    th.cuda.empty_cache()

    th.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    # logger.configure(dir=args.logdir, format_strs=["tensorboard", "csv"])
    logger.configure(dir=args.logdir)

    logger.log("creating encoder and NSR decoder...")
    # device = dist_util.dev()
    device = th.device("cuda", args.local_rank)

    # shared eg3d opts
    opts = eg3d_options_default()

    # if args.sr_training:
    #     args.sr_kwargs = dnnlib.EasyDict(
    #         channel_base=opts.cbase,
    #         channel_max=opts.cmax,
    #         fused_modconv_default='inference_only',
    #         use_noise=True
    #     )  # ! close noise injection? since noise_mode='none' in eg3d

    logger.log("creating data loader...")
    # data = load_data(
    # if args.overfitting:
    #     data = load_memory_data(
    #         file_path=args.data_dir,
    #         batch_size=args.batch_size,
    #         reso=args.image_size,
    #         reso_encoder=args.image_size_encoder,  # 224 -> 128
    #         num_workers=args.num_workers,
    #         # load_depth=args.depth_lambda > 0
    #         load_depth=True  # for evaluation
    #     )
    # else:
    #     data = load_data(
    #         dataset_size=args.dataset_size,
    #         file_path=args.data_dir,
    #         batch_size=args.batch_size,
    #         reso=args.image_size,
    #         reso_encoder=args.image_size_encoder,  # 224 -> 128
    #         num_workers=args.num_workers,
    #         load_depth=True,
    #         preprocess=auto_encoder.preprocess  # clip
    #         # load_depth=True # for evaluation
    #     )
    # eval_data = load_eval_data(
    #     file_path=args.eval_data_dir,
    #     batch_size=args.eval_batch_size,
    #     reso=args.image_size,
    #     reso_encoder=args.image_size_encoder,  # 224 -> 128
    #     num_workers=2,
    #     load_depth=True,  # for evaluation
    #     preprocess=auto_encoder.preprocess)
    # ! load pre-trained SR in G
    common_kwargs = dict(c_dim=25, img_resolution=512, img_channels=3)

    G_kwargs = EasyDict(class_name=None,
                        z_dim=512,
                        w_dim=512,
                        mapping_kwargs=EasyDict())
    G_kwargs.channel_base = opts.cbase
    G_kwargs.channel_max = opts.cmax
    G_kwargs.mapping_kwargs.num_layers = opts.map_depth
    G_kwargs.class_name = opts.g_class_name
    G_kwargs.fused_modconv_default = 'inference_only'  # Speed up training by using regular convolutions instead of grouped convolutions.
    G_kwargs.rendering_kwargs = args.rendering_kwargs
    G_kwargs.num_fp16_res = 0
    G_kwargs.sr_num_fp16_res = 4

    G_kwargs.sr_kwargs = EasyDict(channel_base=opts.cbase,
                                  channel_max=opts.cmax,
                                  fused_modconv_default='inference_only',
                                  use_noise=True) # ! close noise injection? since noise_mode='none' in eg3d

    G_kwargs.num_fp16_res = opts.g_num_fp16_res
    G_kwargs.conv_clamp = 256 if opts.g_num_fp16_res > 0 else None

    # creating G
    resume_data = th.load(args.resume_checkpoint_EG3D, map_location='cuda:{}'.format(args.local_rank))
    G_ema = dnnlib.util.construct_class_by_name(
        **G_kwargs, **common_kwargs).train().requires_grad_(False).to(
            dist_util.dev())  # subclass of th.nn.Module
    for name, module in [
        ('G_ema', G_ema),
        # ('D', D),
    ]:
        misc.copy_params_and_buffers(
            resume_data[name],  # type: ignore
            module,
            require_all=True,
            # load_except=d_load_except if name == 'D' else [],
            )
    

    G_ema.requires_grad_(False)
    G_ema.eval()

    if args.sr_training:
        args.sr_kwargs = G_kwargs.sr_kwargs # uncomment if needs to train with SR module

    auto_encoder = create_3DAE_model(
        **args_to_dict(args,
                       encoder_and_nsr_defaults().keys()))
    auto_encoder.to(device)
    auto_encoder.train()

    # * clone G_ema.decoder to auto_encoder triplane
    logger.log("AE triplane decoder reuses G_ema decoder...")
    auto_encoder.decoder.register_buffer('w_avg', G_ema.backbone.mapping.w_avg)

    auto_encoder.decoder.triplane_decoder.decoder.load_state_dict(  # type: ignore
        G_ema.decoder.state_dict())  # type: ignore

    # set grad=False in this manner suppresses the DDP forward no grad error.
    for param in auto_encoder.decoder.triplane_decoder.decoder.parameters(): # type: ignore
        param.requires_grad_(False)
    
    if args.sr_training:
        logger.log("AE triplane decoder reuses G_ema SR module...")
        auto_encoder.decoder.triplane_decoder.superresolution.load_state_dict(  # type: ignore
            G_ema.superresolution.state_dict())  # type: ignore
        # set grad=False in this manner suppresses the DDP forward no grad error.
        for param in auto_encoder.decoder.triplane_decoder.superresolution.parameters(): # type: ignore
            param.requires_grad_(False)

    del resume_data, G_ema
    th.cuda.empty_cache()

    auto_encoder.to(dist_util.dev())
    auto_encoder.train()

    # ! load FFHQ/AFHQ
    # Training set.
    # training_set_kwargs, dataset_name = init_dataset_kwargs(data=args.data_dir, class_name='datasets.eg3d_dataset.ImageFolderDatasetPose') # only load pose here
    training_set_kwargs, dataset_name = init_dataset_kwargs(data=args.data_dir, class_name='datasets.eg3d_dataset.ImageFolderDataset') # only load pose here
    # if args.cond and not training_set_kwargs.use_labels:
    # raise Exception('check here')

    # training_set_kwargs.use_labels = args.cond
    training_set_kwargs.use_labels = True
    training_set_kwargs.xflip = False
    training_set_kwargs.random_seed = SEED
    # desc = f'{args.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'

    # * construct ffhq/afhq dataset
    training_set = dnnlib.util.construct_class_by_name(
        **training_set_kwargs)  # subclass of training.dataset.Dataset

    training_set = dnnlib.util.construct_class_by_name(
        **training_set_kwargs)  # subclass of training.dataset.Dataset

    training_set_sampler = InfiniteSampler(
        dataset=training_set,
        rank=dist_util.get_rank(),
        num_replicas=dist_util.get_world_size(),
        seed=SEED)

    data = iter(
        th.utils.data.DataLoader(dataset=training_set,
                                 sampler=training_set_sampler,
                                 batch_size=args.batch_size,
                                 pin_memory=True,
                                 num_workers=args.num_workers,))
                                #  prefetch_factor=2))

    eval_data = th.utils.data.DataLoader(dataset=Subset(training_set, np.arange(10)),
                                 batch_size=args.eval_batch_size,
                                 num_workers=1)

    args.img_size = [args.image_size_encoder]
    # try dry run
    # batch = next(data)
    # batch = None

    # logger.log("creating model and diffusion...")

    # let all processes sync up before starting with a new epoch of training
    dist_util.synchronize()

    # schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    opt = dnnlib.EasyDict(args_to_dict(args, loss_defaults().keys()))
    loss_class = E3DGELossClass(device, opt).to(device)

    # writer = SummaryWriter() # TODO, add log dir

    logger.log("training...")

    TrainLoop = {
        'cvD': nsr.TrainLoop3DcvD,
        'nvsD': nsr.TrainLoop3DcvD_nvsD,
        'cano_nvs_cvD': nsr.TrainLoop3DcvD_nvsD_canoD,
        'canoD': nsr.TrainLoop3DcvD_canoD
    }[args.trainer_name]

    TrainLoop(rec_model=auto_encoder,
              loss_class=loss_class,
              data=data,
              eval_data=eval_data,
              **vars(args)).run_loop()  # ! overfitting


def create_argparser(**kwargs):
    # defaults.update(model_and_diffusion_defaults())

    defaults = dict(
        dataset_size=-1,
        trainer_name='cvD',
        use_amp=False,
        overfitting=False,
        num_workers=4,
        image_size=128,
        image_size_encoder=224,
        iterations=150000,
        anneal_lr=False,
        lr=5e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        eval_batch_size=12,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        eval_interval=2500,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        data_dir="",
        eval_data_dir="",
        # load_depth=False, # TODO
        logdir="/mnt/lustre/yslan/logs/nips23/",
        resume_checkpoint_EG3D="",
    )

    defaults.update(encoder_and_nsr_defaults())  # type: ignore
    defaults.update(loss_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"

    # master_addr = '127.0.0.1'
    # master_port = dist_util._find_free_port()
    # master_port = 31323

    args = create_argparser().parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpus = th.cuda.device_count()

    opts = args

    args.rendering_kwargs = rendering_options_defaults(opts)

    # print(args)
    with open(os.path.join(args.logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Launch processes.
    print('Launching processes...')

    try:
        training_loop(args)
    # except KeyboardInterrupt as e:
    except Exception as e:
        # print(e)
        traceback.print_exc()
        dist_util.cleanup()  # clean port and socket when ctrl+c
