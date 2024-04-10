"""
Train a diffusion model on images.
"""
# import imageio
import gzip
import random
import json
import sys
import os
import lmdb
from tqdm import tqdm
sys.path.append('.')
import torch.distributed as dist
import pickle
import traceback
from PIL import Image
import torch as th
import torch.multiprocessing as mp
import lzma
import webdataset as wds
import numpy as np

from torch.utils.data import DataLoader, Dataset
import imageio.v3 as iio

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
from datasets.shapenet import load_data, load_data_for_lmdb, load_eval_data, load_memory_data
from nsr.losses.builder import E3DGELossClass
from datasets.eg3d_dataset import init_dataset_kwargs

# from .lmdb_create import encode_and_compress_image

def encode_and_compress_image(inp_array, is_image=False, compress=True):
    # Read the image using imageio
    # image = imageio.v3.imread(image_path)

    # Convert the image to bytes
    # with io.BytesIO() as byte_buffer:
    #     imageio.imsave(byte_buffer, image, format="png")
    #     image_bytes = byte_buffer.getvalue()
    if is_image:
        inp_bytes = iio.imwrite("<bytes>", inp_array, extension=".png")
    else:
        inp_bytes = inp_array.tobytes()

    # Compress the image data using gzip
    if compress:
        compressed_data = gzip.compress(inp_bytes)
        return compressed_data
    else:
        return inp_bytes




from pdb import set_trace as st
import bz2

# th.backends.cuda.matmul.allow_tf32 = True # https://huggingface.co/docs/diffusers/optimization/fp16



def training_loop(args):
    # def training_loop(args):
    dist_util.setup_dist(args)
    # th.autograd.set_detect_anomaly(True) # type: ignore
    th.autograd.set_detect_anomaly(False)  # type: ignore
    # https://blog.csdn.net/qq_41682740/article/details/126304613

    SEED = args.seed

    # dist.init_process_group(backend='nccl', init_method='env://', rank=args.local_rank, world_size=th.cuda.device_count())
    logger.log(f"{args.local_rank=} init complete, seed={SEED}")
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


    if args.objv_dataset:
        from datasets.g_buffer_objaverse import load_data, load_eval_data, load_memory_data, load_data_for_lmdb
    else: # shapenet
        from datasets.shapenet import load_data, load_eval_data, load_memory_data, load_data_for_lmdb

    # auto_encoder = create_3DAE_model(
    #     **args_to_dict(args,
    #                    encoder_and_nsr_defaults().keys()))
    # auto_encoder.to(device)
    # auto_encoder.train()

    logger.log("creating data loader...")
    # data = load_data(
    # st()
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
    if args.cfg in ('afhq', 'ffhq'):
        # ! load data
        logger.log("creating eg3d data loader...")
        training_set_kwargs, dataset_name = init_dataset_kwargs(data=args.data_dir, 
                                                                class_name='datasets.eg3d_dataset.ImageFolderDatasetLMDB',
                                                                reso_gt=args.image_size) # only load pose here
        # if args.cond and not training_set_kwargs.use_labels:
        # raise Exception('check here')

        # training_set_kwargs.use_labels = args.cond
        training_set_kwargs.use_labels = True
        training_set_kwargs.xflip = False
        training_set_kwargs.random_seed = SEED
        # training_set_kwargs.max_size = args.dataset_size
        # desc = f'{args.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'

        # * construct ffhq/afhq dataset
        training_set = dnnlib.util.construct_class_by_name(
            **training_set_kwargs)  # subclass of training.dataset.Dataset
        dataset_size = len(training_set)

        # training_set_sampler = InfiniteSampler(
        #     dataset=training_set,
        #     rank=dist_util.get_rank(),
        #     num_replicas=dist_util.get_world_size(),
        #     seed=SEED)

        data = DataLoader(
            training_set,
            shuffle=False,
            batch_size=1,
            num_workers=16,
            drop_last=False,
            # prefetch_factor=2,
            pin_memory=True,
            persistent_workers=True,
        )

    else:
        data, dataset_name, dataset_size, dataset = load_data_for_lmdb(
            file_path=args.data_dir,
            batch_size=args.batch_size,
            reso=args.image_size,
            reso_encoder=args.image_size_encoder,  # 224 -> 128
            num_workers=args.num_workers,
            load_depth=True,
            preprocess=None,
            dataset_size=args.dataset_size,
            trainer_name=args.trainer_name,
            wds_output_path=os.path.join(logger.get_dir(), f'wds-%06d.tar')
            # load_depth=True # for evaluation
        )
    #     if args.pose_warm_up_iter > 0:
    #         overfitting_dataset = load_memory_data(
    #             file_path=args.data_dir,
    #             batch_size=args.batch_size,
    #             reso=args.image_size,
    #             reso_encoder=args.image_size_encoder,  # 224 -> 128
    #             num_workers=args.num_workers,
    #             # load_depth=args.depth_lambda > 0
    #             load_depth=True  # for evaluation
    #         )
    #         data = [data, overfitting_dataset, args.pose_warm_up_iter]
    # eval_data = load_eval_data(
    #     file_path=args.eval_data_dir,
    #     batch_size=args.eval_batch_size,
    #     reso=args.image_size,
    #     reso_encoder=args.image_size_encoder,  # 224 -> 128
    #     num_workers=args.num_workers,
    #     load_depth=True,  # for evaluation
    #     preprocess=auto_encoder.preprocess)
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


    # Function to compress an image using gzip
    # def compress_image_gzip(image_path):
    # def encode_and_compress_image(inp_array, is_image=False):
    #     # Read the image using imageio
    #     # image = imageio.v3.imread(image_path)

    #     # Convert the image to bytes
    #     # with io.BytesIO() as byte_buffer:
    #     #     imageio.imsave(byte_buffer, image, format="png")
    #     #     image_bytes = byte_buffer.getvalue()
    #     if is_image:
    #         inp_bytes = iio.imwrite("<bytes>", inp_array, extension=".png")
    #     else:
    #         inp_bytes = inp_array.tobytes()

    #     # Compress the image data using gzip
    #     compressed_data = gzip.compress(inp_bytes)

    #     return compressed_data


    def convert_to_wds_compressed(dataset,dataset_loader, dataset_size, lmdb_path):
        """
        Convert a PyTorch dataset to LMDB format.

        Parameters:
        - dataset: PyTorch dataset
        - lmdb_path: Path to store the LMDB database
        """
        # env = lmdb.open(lmdb_path, map_size=1024 ** 4, readahead=False)  # Adjust map_size based on your dataset size
        sink = wds.ShardWriter(lmdb_path)

        # with env.begin(write=True) as txn:

        # with env.begin(write=True) as txn:
            # txn.put("length".encode("utf-8"), str(dataset_size).encode("utf-8"))

        for idx, sample in enumerate(tqdm(dataset_loader)):
            pass
            # remove the batch index of returned dict sample

            sample = {
                k:v.squeeze(0).cpu().numpy() if isinstance(v, th.Tensor) else v[0] for k, v in sample.items()
                # k:v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in sample.items()
            }

            # sample = dataset_loader[idx]
            compressed_sample = {}
            for k, v in sample.items():

                # key = f'{idx}-{k}'.encode('utf-8')

                if 'img' in k: # only bytes required? laod the 512 depth bytes only.
                    v = encode_and_compress_image(v, is_image=True, compress=False)
                # elif 'depth' in k:
                elif isinstance(v, str):
                    v = v.encode('utf-8') # caption
                else: # regular bytes encoding
                    v = encode_and_compress_image(v.astype(np.float32), is_image=False, compress=False)
                
                compressed_sample[k] = v

            sink.write({
                "__key__": "sample%08d" % idx,
                # **{f'{k}.pyd': v for k, v in compressed_sample.items()}, # store as pickle, already compressed
                'sample.pyd': compressed_sample
            })

            # break
            if idx > 100:
                break

        sink.close()


    # convert_to_lmdb(data, os.path.join(logger.get_dir(), dataset_name)) convert_to_lmdb_compressed(data, os.path.join(logger.get_dir(), dataset_name))
    # convert_to_lmdb_compressed(data, os.path.join(logger.get_dir()), dataset_size) 
    convert_to_wds_compressed(dataset, data, dataset_size, os.path.join(logger.get_dir(), f'wds-%06d.tar')) 



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
        objv_dataset=False,
        pose_warm_up_iter=-1,
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
