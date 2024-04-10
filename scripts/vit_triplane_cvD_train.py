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

    if args.sr_training:
        args.sr_kwargs = dnnlib.EasyDict(
            channel_base=opts.cbase,
            channel_max=opts.cmax,
            fused_modconv_default='inference_only',
            use_noise=True
        )  # ! close noise injection? since noise_mode='none' in eg3d

    auto_encoder = create_3DAE_model(
        **args_to_dict(args,
                       encoder_and_nsr_defaults().keys()))
    auto_encoder.to(device)
    auto_encoder.train()

    logger.log("creating data loader...")
    # data = load_data(
    if args.overfitting:
        data = load_memory_data(
            file_path=args.data_dir,
            batch_size=args.batch_size,
            reso=args.image_size,
            reso_encoder=args.image_size_encoder,  # 224 -> 128
            num_workers=args.num_workers,
            # trainer_name=args.trainer_name,
            # load_depth=args.depth_lambda > 0
            load_depth=True  # for evaluation
        )
    else:
        data = load_data(
            dataset_size=args.dataset_size,
            file_path=args.data_dir,
            batch_size=args.batch_size,
            reso=args.image_size,
            reso_encoder=args.image_size_encoder,  # 224 -> 128
            num_workers=args.num_workers,
            load_depth=True,
            preprocess=auto_encoder.preprocess,  # clip
            trainer_name=args.trainer_name,
            use_lmdb=args.use_lmdb
            # load_depth=True # for evaluation
        )
    eval_data = load_eval_data(
        file_path=args.eval_data_dir,
        batch_size=args.eval_batch_size,
        reso=args.image_size,
        reso_encoder=args.image_size_encoder,  # 224 -> 128
        num_workers=2,
        load_depth=True,  # for evaluation
        preprocess=auto_encoder.preprocess)
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
        'nvsD_nosr': nsr.TrainLoop3DcvD_nvsD_noSR,
        'cano_nvsD_nosr': nsr.TrainLoop3DcvD_nvsD_noSR,
        'cano_nvs_cvD': nsr.TrainLoop3DcvD_nvsD_canoD,
        'cano_nvs_cvD_nv': nsr.TrainLoop3DcvD_nvsD_canoD_multiview,
        'cvD_nvsD_canoD_canomask': nsr.TrainLoop3DcvD_nvsD_canoD_canomask,
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
        pose_warm_up_iter=-1,
        use_lmdb=False,
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
