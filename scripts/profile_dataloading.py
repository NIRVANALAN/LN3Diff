"""
Train a diffusion model on images.
"""
import cv2
from pathlib import Path
import imageio
import random
import json
import sys
import os

from tqdm import tqdm
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
from nsr.train_nv_util import TrainLoop3DRecNV, TrainLoop3DRec, TrainLoop3DRecNVPatch
from nsr.script_util import create_3DAE_model, encoder_and_nsr_defaults, loss_defaults, rendering_options_defaults, eg3d_options_default
# from datasets.shapenet import load_data, load_eval_data, load_memory_data, load_dataset
from nsr.losses.builder import E3DGELossClass
from datasets.eg3d_dataset import LMDBDataset_MV_Compressed_eg3d
from dnnlib.util import EasyDict, InfiniteSampler

from pdb import set_trace as st

# th.backends.cuda.matmul.allow_tf32 = True # https://huggingface.co/docs/diffusers/optimization/fp16



def training_loop(args):
    # def training_loop(args):
    dist_util.setup_dist(args)
    # th.autograd.set_detect_anomaly(True) # type: ignore
    th.autograd.set_detect_anomaly(False)  # type: ignore
    # https://blog.csdn.net/qq_41682740/article/details/126304613

    SEED = args.seed

    # dist.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank, world_size=th.cuda.device_count())
    # logger.log(f"{args.local_rank=} init complete, seed={SEED}")
    th.cuda.set_device(args.local_rank)
    th.cuda.empty_cache()

    # * deterministic algorithms flags
    th.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # logger.configure(dir=args.logdir, format_strs=["tensorboard", "csv"])
    logger.configure(dir=args.logdir)

    logger.log("creating encoder and NSR decoder...")
    # device = dist_util.dev()
    # device = th.device("cuda", args.local_rank)

    # shared eg3d opts
    opts = eg3d_options_default()

    if args.sr_training:
        args.sr_kwargs = dnnlib.EasyDict(
            channel_base=opts.cbase,
            channel_max=opts.cmax,
            fused_modconv_default='inference_only',
            use_noise=True
        )  # ! close noise injection? since noise_mode='none' in eg3d

    # auto_encoder = create_3DAE_model(
    #     **args_to_dict(args,
    #                    encoder_and_nsr_defaults().keys()))
    # auto_encoder.to(device)
    # auto_encoder.train()

    logger.log("creating data loader...")
    # data = load_data(
    # st()

    # st()
    if args.objv_dataset:
        from datasets.g_buffer_objaverse import load_data, load_dataset, load_eval_data, load_memory_data
    else: # shapenet
        from datasets.shapenet import load_data, load_eval_data, load_memory_data, load_dataset

    # st()
    if args.overfitting:
        data = load_memory_data(
            file_path=args.data_dir,
            batch_size=args.batch_size,
            reso=args.image_size,
            reso_encoder=args.image_size_encoder,  # 224 -> 128
            num_workers=args.num_workers,
            # load_depth=args.depth_lambda > 0
            load_depth=True  # for evaluation
        )
    else:
        if args.cfg in ['ffhq' ]:
            training_set = LMDBDataset_MV_Compressed_eg3d(
                args.data_dir,
                args.image_size,
                args.image_size_encoder,
            )
            training_set_sampler = InfiniteSampler(
                dataset=training_set,
                rank=dist_util.get_rank(),
                num_replicas=dist_util.get_world_size(),
                seed=SEED)

            data = iter(
                th.utils.data.DataLoader(
                    dataset=training_set,
                    sampler=training_set_sampler,
                    batch_size=args.batch_size,
                    pin_memory=True,
                    num_workers=args.num_workers,
                    persistent_workers=args.num_workers>0,
                    prefetch_factor=max(8//args.batch_size, 2),
                ))

        else:
            # st()
            # loader = load_data(
            loader = load_dataset(
                file_path=args.data_dir,
                batch_size=args.batch_size,
                reso=args.image_size,
                reso_encoder=args.image_size_encoder,  # 224 -> 128
                num_workers=args.num_workers,
                load_depth=True,
                preprocess=None,
                dataset_size=args.dataset_size,
                trainer_name=args.trainer_name,
                use_lmdb=args.use_lmdb,
                infi_sampler=False,
                # infi_sampler=True,
                # load_depth=True # for evaluation
            )
            if args.pose_warm_up_iter > 0:
                overfitting_dataset = load_memory_data(
                    file_path=args.data_dir,
                    batch_size=args.batch_size,
                    reso=args.image_size,
                    reso_encoder=args.image_size_encoder,  # 224 -> 128
                    num_workers=args.num_workers,
                    # load_depth=args.depth_lambda > 0
                    load_depth=True  # for evaluation
                )
                data = [data, overfitting_dataset, args.pose_warm_up_iter]
    # eval_data = load_eval_data(
    #     file_path=args.eval_data_dir,
    #     batch_size=args.eval_batch_size,
    #     reso=args.image_size,
    #     reso_encoder=args.image_size_encoder,  # 224 -> 128
    #     num_workers=args.num_workers,
    #     load_depth=True,  # for evaluation
    #     preprocess=None,
    args.img_size = [args.image_size_encoder]
    # try dry run
    # batch = next(data)
    # batch = None

    # logger.log("creating model and diffusion...")

    # let all processes sync up before starting with a new epoch of training
    dist_util.synchronize()

    # schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    opt = dnnlib.EasyDict(args_to_dict(args, loss_defaults().keys()))
    # opt.max_depth, opt.min_depth = args.rendering_kwargs.ray_end, args.rendering_kwargs.ray_start
    # loss_class = E3DGELossClass(device, opt).to(device)

    # writer = SummaryWriter() # TODO, add log dir

    logger.log("training...")

    # TrainLoop = {
    #     'input_rec': TrainLoop3DRec,
    #     'nv_rec': TrainLoop3DRecNV,
    #     'nv_rec_patch': TrainLoop3DRecNVPatch,
    # }[args.trainer_name]

    # TrainLoop(rec_model=auto_encoder,
    #           loss_class=loss_class,
    #           data=data,
    #           eval_data=eval_data,
    #           **vars(args)).run_loop()  # ! overfitting
    number = 0
    # tgt_dir = Path(f'/mnt/lustre/yslan/3D_Dataset/resized_for_fid/chair/{args.image_size}')
    # tgt_dir = Path(f'/mnt/lustre/yslan/3D_Dataset/resized_for_fid/chair-new/{args.image_size}')
    # tgt_dir.mkdir(parents=True, exist_ok=True)
    for idx, batch in enumerate(tqdm(loader)): 
    # for idx in tqdm(len(loader)): # ! dataset here, direct reference
        # batch = loader[idx]
        # worker=3: 2.5it/s; worker=8: 1.47it/s; worker=4, 2.3it/s; worker=1, 1.45it/s
        # ! save to target folder for FID/KID
        # for idx in range(batch['img'].shape[0]):
        #     # imageio.v3.imwrite(tgt_dir / f'{number}.png' ,(127.5+127.5*batch['img'][idx].cpu().numpy()).astype(np.uint8))
        #     cv2.imwrite(str(tgt_dir / f'{number}.png') ,(127.5+127.5*batch['img'][idx].cpu().permute(1,2,0).numpy()).astype(np.uint8))
        #     number += 1

        pass


def create_argparser(**kwargs):
    # defaults.update(model_and_diffusion_defaults())

    defaults = dict(
        seed=0,
        dataset_size=-1,
        trainer_name='input_rec',
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
        # test warm up pose sampling training
        pose_warm_up_iter=-1,
        use_lmdb=False,
        objv_dataset=False,
    )

    defaults.update(encoder_and_nsr_defaults())  # type: ignore
    defaults.update(loss_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser


if __name__ == "__main__":
    # os.environ[
    # "TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
    # os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    # os.environ["NCCL_DEBUG"]="INFO"

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
        dist_util.cleanup() # clean port and socket when ctrl+c
