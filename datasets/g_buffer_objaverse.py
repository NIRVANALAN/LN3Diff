import os
import collections
import math
import time
import itertools
import pickle
from typing import Any
import lmdb
import cv2
import imageio
import numpy as np
from PIL import Image
import Imath
import OpenEXR
from pdb import set_trace as st
from pathlib import Path
import torchvision

from einops import rearrange, repeat
from functools import partial
import io
import gzip
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import lz4.frame

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

from utils.general_utils import PILtoTorch, matrix_to_quaternion

from guided_diffusion import logger
import json

import webdataset as wds

from .shapenet import LMDBDataset, LMDBDataset_MV_Compressed, decompress_and_open_image_gzip, decompress_array
from kiui.op import safe_normalize

from utils.gs_utils.graphics_utils import getWorld2View2, getProjectionMatrix, getView2World


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def resize_depth_mask(depth_to_resize, resolution):
    depth_resized = cv2.resize(depth_to_resize, (resolution, resolution),
                               interpolation=cv2.INTER_LANCZOS4)
    #    interpolation=cv2.INTER_AREA)
    return depth_resized, depth_resized > 0  # type: ignore


def resize_depth_mask_Tensor(depth_to_resize, resolution):

    if depth_to_resize.shape[-1] != resolution:
        depth_resized = torch.nn.functional.interpolate(
            input=depth_to_resize.unsqueeze(1),
            size=(resolution, resolution),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)
    else:
        depth_resized = depth_to_resize

    return depth_resized, depth_resized > 0  # type: ignore


def load_dataset(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        #   shuffle=True,
        num_workers=6,
        load_depth=False,
        preprocess=None,
        imgnet_normalize=True,
        dataset_size=-1,
        trainer_name='input_rec',
        use_lmdb=False,
        use_wds=False,
        use_lmdb_compressed=False,
        infi_sampler=True):
    # st()
    # dataset_cls = {
    #     'input_rec': MultiViewDataset,
    #     'nv': NovelViewDataset,
    # }[trainer_name]
    # st()
    if use_wds:
        return load_wds_data(file_path, reso, reso_encoder, batch_size,
                             num_workers)

    if use_lmdb:
        logger.log('using LMDB dataset')
        # dataset_cls = LMDBDataset_MV #  2.5-3iter/s, but unstable, drops to 1 later.

        if use_lmdb_compressed:
            if 'nv' in trainer_name:
                dataset_cls = Objv_LMDBDataset_NV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
            else:
                dataset_cls = Objv_LMDBDataset_MV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
        else:
            if 'nv' in trainer_name:
                dataset_cls = Objv_LMDBDataset_NV_NoCompressed  #  2.5-3iter/s, but unstable, drops to 1 later.
            else:
                dataset_cls = Objv_LMDBDataset_MV_NoCompressed  #  2.5-3iter/s, but unstable, drops to 1 later.

        # dataset = dataset_cls(file_path)
    else:
        if 'nv' in trainer_name:
            dataset_cls = NovelViewObjverseDataset
        else:
            dataset_cls = MultiViewObjverseDataset  # 1.5-2iter/s

    dataset = dataset_cls(file_path,
                          reso,
                          reso_encoder,
                          test=False,
                          preprocess=preprocess,
                          load_depth=load_depth,
                          imgnet_normalize=imgnet_normalize,
                          dataset_size=dataset_size)

    logger.log('dataset_cls: {}, dataset size: {}'.format(
        trainer_name, len(dataset)))

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        drop_last=False,
                        pin_memory=True,
                        persistent_workers=num_workers > 0,
                        shuffle=False)
    return loader


def load_data(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        #   shuffle=True,
        num_workers=6,
        load_depth=False,
        preprocess=None,
        imgnet_normalize=True,
        dataset_size=-1,
        trainer_name='input_rec',
        use_lmdb=False,
        use_wds=False,
        use_lmdb_compressed=False,
        plucker_embedding=False,
        infi_sampler=True):
    # st()
    # dataset_cls = {
    #     'input_rec': MultiViewDataset,
    #     'nv': NovelViewDataset,
    # }[trainer_name]
    # st()
    # if use_lmdb:
    #     logger.log('using LMDB dataset')
    #     # dataset_cls = LMDBDataset_MV #  2.5-3iter/s, but unstable, drops to 1 later.
    #     if 'nv' in trainer_name:
    #         dataset_cls = Objv_LMDBDataset_NV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
    #     else:
    #         dataset_cls = Objv_LMDBDataset_MV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
    #     # dataset = dataset_cls(file_path)

    if use_lmdb:
        logger.log('using LMDB dataset')
        # dataset_cls = LMDBDataset_MV #  2.5-3iter/s, but unstable, drops to 1 later.

        if use_lmdb_compressed:
            if 'nv' in trainer_name:
                dataset_cls = Objv_LMDBDataset_NV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
            else:
                dataset_cls = Objv_LMDBDataset_MV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
        else:
            if 'nv' in trainer_name:
                dataset_cls = Objv_LMDBDataset_NV_NoCompressed  #  2.5-3iter/s, but unstable, drops to 1 later.
            else:
                dataset_cls = Objv_LMDBDataset_MV_NoCompressed  #  2.5-3iter/s, but unstable, drops to 1 later.

    else:
        if 'nv' in trainer_name:
            dataset_cls = NovelViewObjverseDataset  # 1.5-2iter/s
        else:
            dataset_cls = MultiViewObjverseDataset

    dataset = dataset_cls(file_path,
                          reso,
                          reso_encoder,
                          test=False,
                          preprocess=preprocess,
                          load_depth=load_depth,
                          imgnet_normalize=imgnet_normalize,
                          dataset_size=dataset_size,
                          plucker_embedding=plucker_embedding)

    logger.log('dataset_cls: {}, dataset size: {}'.format(
        trainer_name, len(dataset)))

    # st()

    if infi_sampler:
        train_sampler = DistributedSampler(dataset=dataset,
                                           shuffle=True,
                                           drop_last=True)

        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            drop_last=True,
                            pin_memory=True,
                            persistent_workers=num_workers > 0,
                            sampler=train_sampler)

        while True:
            yield from loader

    # else:
    #     # loader = DataLoader(dataset,
    #     #                     batch_size=batch_size,
    #     #                     num_workers=num_workers,
    #     #                     drop_last=False,
    #     #                     pin_memory=True,
    #     #                     persistent_workers=num_workers > 0,
    #     #                     shuffle=False)
    #     st()
    #     return dataset


def load_eval_data(
    file_path="",
    reso=64,
    reso_encoder=224,
    batch_size=1,
    num_workers=1,
    load_depth=False,
    preprocess=None,
    imgnet_normalize=True,
    interval=1,
    use_lmdb=False,
    plucker_embedding=False,
    load_real=False,
    four_view_for_latent=False,
    shuffle_across_cls=False,
    load_extra_36_view=False,
    gs_cam_format=False,
    single_view_for_i23d=False,
    **kwargs,
):

    if use_lmdb:
        logger.log('using LMDB dataset')
        dataset_cls = Objv_LMDBDataset_MV_Compressed  #  2.5-3iter/s, but unstable, drops to 1 later.
        dataset = dataset_cls(file_path,
                              reso,
                              reso_encoder,
                              test=True,
                              preprocess=preprocess,
                              load_depth=load_depth,
                              imgnet_normalize=imgnet_normalize,
                              interval=interval)

    elif load_real:
        dataset = RealDataset(file_path,
                              reso,
                              reso_encoder,
                              preprocess=preprocess,
                              load_depth=load_depth,
                              test=True,
                              imgnet_normalize=imgnet_normalize,
                              interval=interval,
                              plucker_embedding=plucker_embedding)

    else:
        dataset = MultiViewObjverseDataset(
            file_path,
            reso,
            reso_encoder,
            preprocess=preprocess,
            load_depth=load_depth,
            test=True,
            imgnet_normalize=imgnet_normalize,
            interval=interval,
            plucker_embedding=plucker_embedding,
            four_view_for_latent=four_view_for_latent,
            load_extra_36_view=load_extra_36_view,
            shuffle_across_cls=shuffle_across_cls,
            gs_cam_format=gs_cam_format,
            single_view_for_i23d=single_view_for_i23d,
        )

    print('eval dataset size: {}'.format(len(dataset)))
    # train_sampler = DistributedSampler(dataset=dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )
    # sampler=train_sampler)
    return loader


def load_data_for_lmdb(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        #   shuffle=True,
        num_workers=6,
        load_depth=False,
        preprocess=None,
        imgnet_normalize=True,
        dataset_size=-1,
        trainer_name='input_rec',
        shuffle_across_cls=False,
        four_view_for_latent=False,
        wds_split=1):
    # st()
    # dataset_cls = {
    #     'input_rec': MultiViewDataset,
    #     'nv': NovelViewDataset,
    # }[trainer_name]
    # if 'nv' in trainer_name:
    #     dataset_cls = NovelViewDataset
    # else:
    # dataset_cls = MultiViewDataset
    # st()
    dataset_cls = MultiViewObjverseDatasetforLMDB

    dataset = dataset_cls(file_path,
                          reso,
                          reso_encoder,
                          test=False,
                          preprocess=preprocess,
                          load_depth=load_depth,
                          imgnet_normalize=imgnet_normalize,
                          dataset_size=dataset_size,
                          shuffle_across_cls=shuffle_across_cls,
                          wds_split=wds_split,
                          four_view_for_latent=four_view_for_latent)

    logger.log('dataset_cls: {}, dataset size: {}'.format(
        trainer_name, len(dataset)))
    # train_sampler = DistributedSampler(dataset=dataset, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        # prefetch_factor=2,
        # prefetch_factor=3,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    # sampler=train_sampler)

    # while True:
    #     yield from loader
    return loader, dataset.dataset_name, len(dataset)


def load_lmdb_for_lmdb(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        #   shuffle=True,
        num_workers=6,
        load_depth=False,
        preprocess=None,
        imgnet_normalize=True,
        dataset_size=-1,
        trainer_name='input_rec'):
    # st()
    # dataset_cls = {
    #     'input_rec': MultiViewDataset,
    #     'nv': NovelViewDataset,
    # }[trainer_name]
    # if 'nv' in trainer_name:
    #     dataset_cls = NovelViewDataset
    # else:
    # dataset_cls = MultiViewDataset
    # st()
    dataset_cls = Objv_LMDBDataset_MV_Compressed_for_lmdb

    dataset = dataset_cls(file_path,
                          reso,
                          reso_encoder,
                          test=False,
                          preprocess=preprocess,
                          load_depth=load_depth,
                          imgnet_normalize=imgnet_normalize,
                          dataset_size=dataset_size)

    logger.log('dataset_cls: {}, dataset size: {}'.format(
        trainer_name, len(dataset)))
    # train_sampler = DistributedSampler(dataset=dataset, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        prefetch_factor=2,
        # prefetch_factor=3,
        pin_memory=True,
        persistent_workers=True,
    )
    # sampler=train_sampler)

    # while True:
    #     yield from loader
    return loader, len(dataset)


def load_memory_data(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        num_workers=1,
        #  load_depth=True,
        preprocess=None,
        imgnet_normalize=True,
        **kwargs):
    # load a single-instance into the memory to speed up training IO
    # dataset = MultiViewObjverseDataset(file_path,
    dataset = NovelViewObjverseDataset(file_path,
                                       reso,
                                       reso_encoder,
                                       preprocess=preprocess,
                                       load_depth=True,
                                       test=False,
                                       overfitting=True,
                                       imgnet_normalize=imgnet_normalize,
                                       overfitting_bs=batch_size,
                                       **kwargs)
    logger.log('!!!!!!! memory dataset size: {} !!!!!!'.format(len(dataset)))
    # train_sampler = DistributedSampler(dataset=dataset)
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
    )

    all_data: dict = next(
        iter(loader)
    )  # torchvision.utils.save_image(all_data['img'], 'gt.jpg', normalize=True, value_range=(-1,1))
    if kwargs.get('gs_cam_format', False):  # gs rendering pipeline
        # ! load V=4 images for training in a batch.
        while True:
            # indices = torch.randperm(len(dataset))[:4]
            indices = torch.randperm(
                len(dataset))[:batch_size]  # all instances
            # indices2 = torch.randperm(len(dataset))[:] # all instances

            batch_c = collections.defaultdict(dict)
            for k in ['c', 'nv_c']:
                for k_c, v_c in all_data[k].items():
                    batch_c[k][k_c] = torch.index_select(
                        v_c, dim=0, index=indices).reshape(
                            batch_size //
                            4, 4, *v_c.shape[1:]).float() if isinstance(
                                v_c, torch.Tensor) else v_c.float()  # float

            batch_c['c']['tanfov'] = batch_c['c']['tanfov'][0][0].item()
            batch_c['nv_c']['tanfov'] = batch_c['nv_c']['tanfov'][0][0].item()

            batch_data = {}
            for k, v in all_data.items():
                if k not in ['c', 'nv_c']:
                    batch_data[k] = torch.index_select(
                        v, dim=0, index=indices).float() if isinstance(
                            v, torch.Tensor) else v  # float

            yield {
                **batch_data,
                **batch_c,
            }

    else:
        while True:
            start_idx = np.random.randint(0, len(dataset) - batch_size + 1)
            yield {
                k: v[start_idx:start_idx + batch_size]
                for k, v in all_data.items()
            }


def read_dnormal(normald_path, cond_pos, h=None, w=None):
    cond_cam_dis = np.linalg.norm(cond_pos, 2)

    near = 0.867  #sqrt(3) * 0.5
    near_distance = cond_cam_dis - near

    normald = cv2.imread(normald_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth = normald[..., 3:]

    depth[depth < near_distance] = 0

    if h is not None:
        assert w is not None
        depth = cv2.resize(depth, (h, w))  # 512,512, 1 -> self.reso, self.reso

    else:
        depth = depth[..., 0]

    return torch.from_numpy(depth).float()


def get_intri(target_im=None, h=None, w=None, normalize=False):
    if target_im is None:
        assert (h is not None and w is not None)
    else:
        h, w = target_im.shape[:2]

    fx = fy = 1422.222
    res_raw = 1024
    f_x = f_y = fx * h / res_raw
    K = np.array([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
    if normalize:  # center is [0.5, 0.5], eg3d renderer tradition
        K[:6] /= h
    # print("intr: ", K)
    return K


def convert_pose(C2W):
    # https://github.com/modelscope/richdreamer/blob/c3d9a77fa15fc42dbae12c2d41d64aaec14efd37/dataset/gobjaverse/depth_warp_example.py#L402
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return torch.from_numpy(C2W)


def read_camera_matrix_single(json_file):
    with open(json_file, 'r', encoding='utf8') as reader:
        json_content = json.load(reader)
    '''
    # NOTE that different from unity2blender experiments.
    camera_matrix = np.eye(4)
    camera_matrix[:3, 0] = np.array(json_content['x'])
    camera_matrix[:3, 1] = -np.array(json_content['y'])
    camera_matrix[:3, 2] = -np.array(json_content['z'])
    camera_matrix[:3, 3] = np.array(json_content['origin'])


    '''
    camera_matrix = np.eye(4)  # blender-based
    camera_matrix[:3, 0] = np.array(json_content['x'])
    camera_matrix[:3, 1] = np.array(json_content['y'])
    camera_matrix[:3, 2] = np.array(json_content['z'])
    camera_matrix[:3, 3] = np.array(json_content['origin'])
    # print(camera_matrix)
    # '''

    # return convert_pose(camera_matrix)
    return camera_matrix


def unity2blender(normal):
    normal_clone = normal.copy()
    normal_clone[..., 0] = -normal[..., -1]
    normal_clone[..., 1] = -normal[..., 0]
    normal_clone[..., 2] = normal[..., 1]

    return normal_clone


def blender2midas(img):
    '''Blender: rub
    midas: lub
    '''
    img[..., 0] = -img[..., 0]
    img[..., 1] = -img[..., 1]
    img[..., -1] = -img[..., -1]
    return img


def current_milli_time():
    return round(time.time() * 1000)


# modified from ShapeNet class
class MultiViewObjverseDataset(Dataset):

    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,  # 4 splits to accelerate preprocessing
            four_view_for_latent=False,
            single_view_for_i23d=False,
            load_extra_36_view=False,
            gs_cam_format=False,
            **kwargs):
        self.load_extra_36_view = load_extra_36_view
        # st()
        self.gs_cam_format = gs_cam_format
        self.four_view_for_latent = four_view_for_latent  # export 0 12 30 36, 4 views for reconstruction
        self.single_view_for_i23d = single_view_for_i23d
        self.file_path = file_path
        self.overfitting = overfitting
        self.scene_scale = scene_scale
        self.reso = reso
        self.reso_encoder = reso_encoder
        self.classes = False
        self.load_depth = load_depth
        self.preprocess = preprocess
        self.plucker_embedding = plucker_embedding
        self.intrinsics = get_intri(h=self.reso, w=self.reso,
                                    normalize=True).reshape(9)

        assert not self.classes, "Not support class condition now."

        dataset_name = Path(self.file_path).stem.split('_')[0]
        self.dataset_name = dataset_name

        self.zfar = 100.0
        self.znear = 0.01

        # if test:
        #     self.ins_list = sorted(os.listdir(self.file_path))[0:1]  # the first 1 instance for evaluation reference.
        # else:
        # ! TODO, read from list?

        def load_single_cls_instances(file_path):
            ins_list = []  # the first 1 instance for evaluation reference.
            for dict_dir in os.listdir(file_path)[:]:
                for ins_dir in os.listdir(os.path.join(file_path, dict_dir)):
                    # self.ins_list.append(os.path.join(self.file_path, dict_dir, ins_dir,))
                    ins_list.append(
                        os.path.join(file_path, dict_dir, ins_dir,
                                     'campos_512_v4'))
            return ins_list

        # st()
        if shuffle_across_cls:
            self.ins_list = []
            # for subset in ['Animals', 'Transportations_tar', 'Furnitures']:
            # for subset in ['Furnitures']:
                # selected subset for training
            for subset in [ # ! around 17W instances in total. MVImageNet is the next thing to deal with? Later.
                        # 'daily-used',
                        # 'Food',
                        # 'Plants',
                        # 'Electronics',
                        # 'BuildingsOutdoor',
                        # 'Human-Shape',
                        'Animals',
                        # 'Transportations_tar',
                        # 'Furnitures',
                ]:  # selected subset for training
                self.ins_list += load_single_cls_instances(
                    os.path.join(self.file_path, subset))
                # st()
                current_time = int(current_milli_time()
                                   )  # randomly shuffle given current time
                random.seed(current_time)
                random.shuffle(self.ins_list)

        else:  # preprocess single class
            self.ins_list = load_single_cls_instances(self.file_path)
            self.ins_list = sorted(self.ins_list)

        # if test:
        #     self.ins_list = self.ins_list[0:1]

        if overfitting:
            self.ins_list = self.ins_list[:1]

        self.rgb_list = []
        self.pose_list = []
        self.depth_list = []
        self.data_ins_list = []
        self.instance_data_length = -1

        with open(
                # '/cpfs01/shared/V2V/V2V_hdd/yslan/aigc3d/text_captions_cap3d.json'
                '/mnt/yslan/objaverse/richdreamer/dataset/text_captions_cap3d.json',
        ) as f:
            self.caption_data = json.load(f)

        self.shuffle_across_cls = shuffle_across_cls

        # for ins in self.ins_list[47000:]:
        if four_view_for_latent:
            self.wds_split_all = 1  # ! when dumping latent
            ins_list_to_process = self.ins_list
        else:
            self.wds_split_all = 4
            # self.wds_split_all = 8  # ! 8 cls in total
            all_ins_size = len(self.ins_list)
            ratio_size = all_ins_size // self.wds_split_all + 1

            ins_list_to_process = self.ins_list[ratio_size *
                                                (wds_split - 1):ratio_size *
                                                wds_split]

        # st()
        for ins in ins_list_to_process:
            # ins = os.path.join(
            #     # self.file_path, ins , 'campos_512_v4'
            #     self.file_path, ins ,
            #     # 'compos_512_v4'
            # )
            # cur_rgb_path = os.path.join(self.file_path, ins, 'compos_512_v4')
            # cur_pose_path = os.path.join(self.file_path, ins, 'pose')

            # st()
            # ][:27])

            if self.four_view_for_latent:
                # cur_all_fname = [t.split('.')[0] for t in os.listdir(ins)
                #                  ]  # use full set for training
                # cur_all_fname = [f'{idx:05d}' for idx in [0, 12, 30, 36]
                # cur_all_fname = [f'{idx:05d}' for idx in [6,12,18,24]
                # cur_all_fname = [f'{idx:05d}' for idx in [7,16,24,25]
                cur_all_fname = [f'{idx:05d}' for idx in [4,12,20,25]
                                 ]  # ! four views for inference
                # cur_all_fname += [f'{idx:05d}' for idx in range(40) if idx not in [0,12,30,36]] # ! four views for inference
            elif self.single_view_for_i23d:
                # cur_all_fname = [f'{idx:05d}'
                #                  for idx in [16]]  # 20 is also fine
                cur_all_fname = [f'{idx:05d}'
                                 for idx in [2]]  # ! furniture side view

            else:
                cur_all_fname = [t.split('.')[0] for t in os.listdir(ins)
                                 ]  # use full set for training

                if shuffle_across_cls:
                    random.seed(current_time)
                    random.shuffle(cur_all_fname)
                else:
                    cur_all_fname = sorted(cur_all_fname)

                if self.instance_data_length == -1:
                    self.instance_data_length = len(cur_all_fname)
                else:
                    try:  # data missing?
                        assert len(cur_all_fname) == self.instance_data_length
                    except:
                        # with open('error_log.txt', 'a') as f:
                        #     f.write(str(e) + '\n')
                        with open('missing_ins_new2.txt', 'a') as f:
                            f.write(str(Path(ins.parent)) +
                                    '\n')  # remove the "campos_512_v4"
                        continue

            # if test: # use middle image as the novel view model input
            #     mid_index = len(cur_all_fname) // 3 * 2
            #     cur_all_fname.insert(0, cur_all_fname[mid_index])

            self.pose_list += ([
                os.path.join(ins, fname, fname + '.json')
                for fname in cur_all_fname
            ])
            self.rgb_list += ([
                os.path.join(ins, fname, fname + '.png')
                for fname in cur_all_fname
            ])

            self.depth_list += ([
                os.path.join(ins, fname, fname + '_nd.exr')
                for fname in cur_all_fname
            ])
            self.data_ins_list += ([ins] * len(cur_all_fname))

        # check

        # ! setup normalizataion
        transformations = [
            transforms.ToTensor(),  # [0,1] range
        ]
        if imgnet_normalize:
            transformations.append(
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))  # type: ignore
            )
        else:
            transformations.append(
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5)))  # type: ignore

        self.normalize = transforms.Compose(transformations)

    def get_source_cw2wT(self, source_cameras_view_to_world):
        return matrix_to_quaternion(
            source_cameras_view_to_world[:3, :3].transpose(0, 1))

    def c_to_3dgs_format(self, pose):
        # TODO, switch to torch version (batched later)

        c2w = pose[:16].reshape(4, 4)  # 3x4

        # ! load cam
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        fx = pose[16]
        FovX = focal2fov(fx, 1)
        FovY = focal2fov(fx, 1)

        tanfovx = math.tan(FovX * 0.5)
        tanfovy = math.tan(FovY * 0.5)

        assert tanfovx == tanfovy

        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0

        world_view_transform = torch.tensor(getWorld2View2(R, T, trans,
                                                           scale)).transpose(
                                                               0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear,
                                                zfar=self.zfar,
                                                fovX=FovX,
                                                fovY=FovY).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
            projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        view_world_transform = torch.tensor(getView2World(R, T, trans,
                                                          scale)).transpose(
                                                              0, 1)

        # item.update(viewpoint_cam=[viewpoint_cam])
        c = {}
        c["source_cv2wT_quat"] = self.get_source_cw2wT(view_world_transform)
        c.update(
            # projection_matrix=projection_matrix, # K
            cam_view=world_view_transform,  # world_view_transform
            cam_view_proj=full_proj_transform,  # full_proj_transform
            cam_pos=camera_center,
            tanfov=tanfovx,  # TODO, fix in the renderer
            # orig_c2w=c2w,
            # orig_w2c=w2c,
            orig_pose=torch.from_numpy(pose),
            orig_c2w=torch.from_numpy(c2w),
            orig_w2c=torch.from_numpy(w2c),
            # tanfovy=tanfovy,
        )

        return c  # dict for gs rendering

    def __len__(self):
        return len(self.rgb_list)

    def load_bbox(self, mask):
        # st()
        nonzero_value = torch.nonzero(mask)
        height, width = nonzero_value.max(dim=0)[0]
        top, left = nonzero_value.min(dim=0)[0]
        bbox = torch.tensor([top, left, height, width], dtype=torch.float32)
        return bbox

    def __getitem__(self, idx):
        # try:
        data = self._read_data(idx)
        return data
        # except Exception as e:
        #     with open('error_log.txt', 'a') as f:
        #         f.write(str(e) + '\n')
        #     with open('error_idx.txt', 'a') as f:
        #         f.write(str(self.data_ins_list[idx]) + '\n')
        #     print(e, flush=True)
        #     return {}

    def gen_rays(self, c2w):
        # Generate rays
        self.h = self.reso_encoder
        self.w = self.reso_encoder
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
            indexing='ij')

        # normalize to 0-1 pixel range
        yy = yy / self.h
        xx = xx / self.w

        # K = np.array([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
        cx, cy, fx, fy = self.intrinsics[2], self.intrinsics[
            5], self.intrinsics[0], self.intrinsics[4]
        # cx *= self.w
        # cy *= self.h

        # f_x = f_y = fx * h / res_raw
        c2w = torch.from_numpy(c2w).float()

        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        del xx, yy, zz
        # st()
        dirs = (c2w[None, :3, :3] @ dirs)[..., 0]

        origins = c2w[None, :3, 3].expand(self.h * self.w, -1).contiguous()
        origins = origins.view(self.h, self.w, 3)
        dirs = dirs.view(self.h, self.w, 3)

        return origins, dirs

    def _read_data(self, idx):
        rgb_fname = self.rgb_list[idx]
        pose_fname = self.pose_list[idx]

        raw_img = imageio.imread(rgb_fname)

        # ! RGBD
        alpha_mask = raw_img[..., -1:] / 255
        raw_img = alpha_mask * raw_img[..., :3] + (
            1 - alpha_mask) * np.ones_like(raw_img[..., :3]) * 255

        raw_img = raw_img.astype(
            np.uint8)  # otherwise, float64 won't call ToTensor()

        # return raw_img
        # st()

        if self.preprocess is None:
            img_to_encoder = cv2.resize(raw_img,
                                        (self.reso_encoder, self.reso_encoder),
                                        interpolation=cv2.INTER_LANCZOS4)
            # interpolation=cv2.INTER_AREA)
            img_to_encoder = img_to_encoder[
                ..., :3]  #[3, reso_encoder, reso_encoder]
            img_to_encoder = self.normalize(img_to_encoder)
        else:
            img_to_encoder = self.preprocess(Image.open(rgb_fname))  # clip

        # return img_to_encoder

        img = cv2.resize(raw_img, (self.reso, self.reso),
                         interpolation=cv2.INTER_LANCZOS4)

        #  interpolation=cv2.INTER_AREA)

        # img_sr = cv2.resize(raw_img, (512, 512), interpolation=cv2.INTER_AREA)
        # img_sr = cv2.resize(raw_img, (256, 256), interpolation=cv2.INTER_AREA) # just as refinement, since eg3d uses 64->128 final resolution
        # img_sr = cv2.resize(raw_img, (128, 128), interpolation=cv2.INTER_AREA) # just as refinement, since eg3d uses 64->128 final resolution

        # img_sr = cv2.resize(
        #     raw_img, (128, 128), interpolation=cv2.INTER_LANCZOS4
        # )  # just as refinement, since eg3d uses 64->128 final resolution

        # img = torch.from_numpy(img)[..., :3].permute(
        #     2, 0, 1) / 255.0  #[3, reso, reso]

        img = torch.from_numpy(img)[..., :3].permute(
            2, 0, 1
        ) / 127.5 - 1  #[3, reso, reso], normalize to [-1,1], follow triplane range

        # img_sr = torch.from_numpy(img_sr)[..., :3].permute(
        #     2, 0, 1
        # ) / 127.5 - 1  #[3, reso, reso], normalize to [-1,1], follow triplane range

        c2w = read_camera_matrix_single(pose_fname)  #[1, 4, 4] -> [1, 16]
        # c = np.concatenate([c2w, self.intrinsics], axis=0).reshape(25)  # 25, no '1' dim needed.

        # return c2w

        # if self.load_depth:
        # depth, depth_mask, depth_mask_sr = read_dnormal(self.depth_list[idx],
        # try:
        depth = read_dnormal(self.depth_list[idx], c2w[:3, 3:], self.reso,
                             self.reso)
        # return depth
        # except:
        #     # print(self.depth_list[idx])
        #     raise NotImplementedError(self.depth_list[idx])
        # if depth

        # try:
        bbox = self.load_bbox(depth > 0)
        # except:
        #     print(rgb_fname)
        #     return {}
        #     st()

        # plucker
        rays_o, rays_d = self.gen_rays(c2w)
        rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d],
                                 dim=-1)  # [h, w, 6]

        img_to_encoder = torch.cat(
            [img_to_encoder, rays_plucker.permute(2, 0, 1)],
            0).float()  # concat in C dim

        # ! add depth as input

        normalized_depth = read_dnormal(self.depth_list[idx], c2w[:3, 3:],
                                        self.reso_encoder,
                                        self.reso_encoder).unsqueeze(0)
        # normalized_depth = depth.unsqueeze(0)  # min=0
        img_to_encoder = torch.cat([img_to_encoder, normalized_depth],
                                   0)  # concat in C dim

        c = np.concatenate([c2w.reshape(16), self.intrinsics],
                           axis=0).reshape(25).astype(
                               np.float32)  # 25, no '1' dim needed.

        if self.gs_cam_format:
            c = self.c_to_3dgs_format(c)
        else:
            c = torch.from_numpy(c)

        ret_dict = {
            # 'rgb_fname': rgb_fname,
            'img_to_encoder': img_to_encoder,
            'img': img,
            'c': c,
            # 'img_sr': img_sr,
            # 'ins_name': self.data_ins_list[idx]
        }

        ins = str(
            (Path(self.data_ins_list[idx]).relative_to(self.file_path)).parent)
        if self.shuffle_across_cls:
            caption = self.caption_data['/'.join(ins.split('/')[1:])]
        else:
            caption = self.caption_data[ins]

        ret_dict.update({
            'depth': depth,
            'depth_mask': depth > 0,
            # 'depth_mask_sr': depth_mask_sr,
            'bbox': bbox,
            'caption': caption,
            'rays_plucker': rays_plucker,  # cam embedding used in lgm
            'ins': ins,  # placeholder
        })

        return ret_dict


class RealDataset(Dataset):

    def __init__(
            self,
            file_path,
            reso,
            reso_encoder,
            preprocess=None,
            classes=False,
            load_depth=False,
            test=False,
            scene_scale=1,
            overfitting=False,
            imgnet_normalize=True,
            dataset_size=-1,
            overfitting_bs=-1,
            interval=1,
            plucker_embedding=False,
            shuffle_across_cls=False,
            wds_split=1,  # 4 splits to accelerate preprocessing
    ) -> None:
        super().__init__()

        self.file_path = file_path
        self.overfitting = overfitting
        self.scene_scale = scene_scale
        self.reso = reso
        self.reso_encoder = reso_encoder
        self.classes = False
        self.load_depth = load_depth
        self.preprocess = preprocess
        self.plucker_embedding = plucker_embedding

        self.rgb_list = []

        all_fname = [
            t for t in os.listdir(self.file_path)
            if t.split('.')[1] in ['png', 'jpg']
        ]
        self.rgb_list += ([
            os.path.join(self.file_path, fname) for fname in all_fname
        ])
        # if len(self.rgb_list) == 1:
        #     # placeholder
        #     self.rgb_list = self.rgb_list * 40

        # ! setup normalizataion
        transformations = [
            transforms.ToTensor(),  # [0,1] range
        ]

        assert imgnet_normalize
        if imgnet_normalize:
            transformations.append(
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))  # type: ignore
            )
        else:
            transformations.append(
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5)))  # type: ignore

        self.normalize = transforms.Compose(transformations)
        camera = torch.load('eval_pose.pt', map_location='cpu')
        self.eval_camera = camera

        # pre-cache
        self.calc_rays_plucker()

    def gen_rays(self, c):
        # Generate rays
        intrinsics, c2w = c[16:], c[:16].reshape(4, 4)
        self.h = self.reso_encoder
        self.w = self.reso_encoder
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
            indexing='ij')

        # normalize to 0-1 pixel range
        yy = yy / self.h
        xx = xx / self.w

        # K = np.array([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
        cx, cy, fx, fy = intrinsics[2], intrinsics[5], intrinsics[
            0], intrinsics[4]
        # cx *= self.w
        # cy *= self.h

        # f_x = f_y = fx * h / res_raw
        if not isinstance(c2w, torch.Tensor):
            c2w = torch.from_numpy(c2w)

        c2w = c2w.float()

        xx = (xx - cx) / fx
        yy = (yy - cy) / fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        del xx, yy, zz
        # st()
        dirs = (c2w[None, :3, :3] @ dirs)[..., 0]

        origins = c2w[None, :3, 3].expand(self.h * self.w, -1).contiguous()
        origins = origins.view(self.h, self.w, 3)
        dirs = dirs.view(self.h, self.w, 3)

        return origins, dirs

    def calc_rays_plucker(self):
        all_rays_plucker = []

        for c2w in self.eval_camera:
            rays_o, rays_d = self.gen_rays(c2w)
            rays_plucker = torch.cat(
                [torch.cross(rays_o, rays_d, dim=-1), rays_d],
                dim=-1)  # [h, w, 6]
            all_rays_plucker.append(rays_plucker)

        self.all_rays_plucker = torch.stack(all_rays_plucker,
                                            0).permute(0, 3, 1, 2)  # B 6 H W

        # st()
        pass

    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, index) -> Any:
        # return super().__getitem__(index)

        rgb_fname = self.rgb_list[index]
        # ! preprocess, normalize

        raw_img = imageio.imread(rgb_fname)

        # interpolation=cv2.INTER_AREA)
        if raw_img.shape[-1] == 4:
            alpha_mask = raw_img[..., 3:4] / 255.0
            bg_white = np.ones_like(alpha_mask) * 255.0
            raw_img = raw_img[..., :3] * alpha_mask + (
                1 - alpha_mask) * bg_white  #[3, reso_encoder, reso_encoder]
            raw_img = raw_img.astype(np.uint8)

        img_to_encoder = cv2.resize(raw_img,
                                    (self.reso_encoder, self.reso_encoder),
                                    interpolation=cv2.INTER_LANCZOS4)

        # img_to_encoder = img_to_encoder
        img_to_encoder = self.normalize(img_to_encoder)

        # ! concat plucker
        img_to_encoder = torch.cat(
            [img_to_encoder, self.all_rays_plucker[index]],
            0)  # concat in C dim

        # log gt
        img = cv2.resize(raw_img, (self.reso, self.reso),
                         interpolation=cv2.INTER_LANCZOS4)

        img = torch.from_numpy(img)[..., :3].permute(
            2, 0, 1
        ) / 127.5 - 1  #[3, reso, reso], normalize to [-1,1], follow triplane range

        ret_dict = {
            # 'rgb_fname': rgb_fname,
            'img_to_encoder':
            img_to_encoder.unsqueeze(0).repeat_interleave(40, 0),
            'img': img.unsqueeze(0).repeat_interleave(40, 0),
            'c': self.eval_camera,  # TODO, get pre-calculated samples
            'ins': 'placeholder',
            'bbox': 'placeholder',
            'caption': 'placeholder',
        }

        # ! repeat as a intance

        return ret_dict


class NovelViewObjverseDataset(MultiViewObjverseDataset):
    """novel view prediction version.
    """

    def __init__(self,
                 file_path,
                 reso,
                 reso_encoder,
                 preprocess=None,
                 classes=False,
                 load_depth=False,
                 test=False,
                 scene_scale=1,
                 overfitting=False,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 overfitting_bs=-1,
                 **kwargs):
        super().__init__(file_path, reso, reso_encoder, preprocess, classes,
                         load_depth, test, scene_scale, overfitting,
                         imgnet_normalize, dataset_size, overfitting_bs,
                         **kwargs)

    def __getitem__(self, idx):
        input_view = super().__getitem__(
            idx)  # get previous input view results

        # get novel view of the same instance
        novel_view = super().__getitem__(
            (idx // self.instance_data_length) * self.instance_data_length +
            random.randint(0, self.instance_data_length - 1))

        # assert input_view['ins_name'] == novel_view['ins_name'], 'should sample novel view from the same instance'

        input_view.update({f'nv_{k}': v for k, v in novel_view.items()})
        return input_view


class MultiViewObjverseDatasetforLMDB(MultiViewObjverseDataset):

    def __init__(
        self,
        file_path,
        reso,
        reso_encoder,
        preprocess=None,
        classes=False,
        load_depth=False,
        test=False,
        scene_scale=1,
        overfitting=False,
        imgnet_normalize=True,
        dataset_size=-1,
        overfitting_bs=-1,
        shuffle_across_cls=False,
        wds_split=1,
        four_view_for_latent=False,
    ):
        super().__init__(file_path,
                         reso,
                         reso_encoder,
                         preprocess,
                         classes,
                         load_depth,
                         test,
                         scene_scale,
                         overfitting,
                         imgnet_normalize,
                         dataset_size,
                         overfitting_bs,
                         shuffle_across_cls=shuffle_across_cls,
                         wds_split=wds_split,
                         four_view_for_latent=four_view_for_latent)

        assert self.reso == 256

        with open(
                '/cpfs01/shared/V2V/V2V_hdd/yslan/aigc3d/text_captions_cap3d.json'
        ) as f:
            self.caption_data = json.load(f)
        lmdb_path = '/cpfs01/user/yangpeiqing.p/yslan/data/Furnitures_uncompressed/'

        # with open(os.path.join(lmdb_path, 'idx_to_ins_mapping.json')) as f:
        #     self.idx_to_ins_mapping = json.load(f)

    def __len__(self):
        return super().__len__()
        # return 100 # for speed debug

    def __getitem__(self, idx):
        # ret_dict = super().__getitem__(idx)
        rgb_fname = self.rgb_list[idx]
        pose_fname = self.pose_list[idx]
        raw_img = imageio.imread(rgb_fname)  # [..., :3]

        # assert raw_img.shape[-1] == 4

        if raw_img.shape[-1] == 4:  # ! set bg to white
            alpha_mask = raw_img[..., -1:] / 255  # [0,1]
            raw_img = alpha_mask * raw_img[..., :3] + (
                1 - alpha_mask) * np.ones_like(raw_img[..., :3]) * 255
            raw_img = raw_img.astype(np.uint8)

        raw_img = cv2.resize(raw_img, (self.reso, self.reso),
                             interpolation=cv2.INTER_LANCZOS4)

        c2w = read_camera_matrix_single(pose_fname)  #[1, 4, 4] -> [1, 16]
        c = np.concatenate([c2w.reshape(16), self.intrinsics],
                           axis=0).reshape(25).astype(
                               np.float32)  # 25, no '1' dim needed.
        c = torch.from_numpy(c)
        # c = np.concatenate([c2w, self.intrinsics], axis=0).reshape(25)  # 25, no '1' dim needed.

        # if self.load_depth:
        # depth, depth_mask, depth_mask_sr = read_dnormal(self.depth_list[idx],
        # try:
        depth = read_dnormal(self.depth_list[idx], c2w[:3, 3:], self.reso,
                             self.reso)
        # except:
        #     # print(self.depth_list[idx])
        #     raise NotImplementedError(self.depth_list[idx])
        # if depth

        # try:
        bbox = self.load_bbox(depth > 0)

        ins = str(
            (Path(self.data_ins_list[idx]).relative_to(self.file_path)).parent)
        if self.shuffle_across_cls:
            caption = self.caption_data['/'.join(ins.split('/')[1:])]
        else:
            caption = self.caption_data[ins]

        ret_dict = {
            'raw_img': raw_img,
            'c': c,
            'depth': depth,
            # 'depth_mask': depth_mask, # 64x64 here?
            'bbox': bbox,
            'ins': ins,
            'caption': caption,
            # 'fname': rgb_fname,
        }
        return ret_dict


class Objv_LMDBDataset_MV_Compressed(LMDBDataset_MV_Compressed):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 test=False,
                 **kwargs):
        super().__init__(lmdb_path,
                         reso,
                         reso_encoder,
                         imgnet_normalize,
                         dataset_size=dataset_size,
                         **kwargs)
        self.instance_data_length = 40  # ! could save some key attributes in LMDB
        if test:
            self.length = self.instance_data_length
        elif dataset_size > 0:
            self.length = dataset_size * self.instance_data_length

        # load caption data, and idx-to-ins mapping
        with open(
                '/cpfs01/shared/V2V/V2V_hdd/yslan/aigc3d/text_captions_cap3d.json'
        ) as f:
            self.caption_data = json.load(f)
        with open(os.path.join(lmdb_path, 'idx_to_ins_mapping.json')) as f:
            self.idx_to_ins_mapping = json.load(f)

    def _load_data(self, idx):
        # '''
        raw_img, depth, c, bbox = self._load_lmdb_data(idx)
        # raw_img, depth, c, bbox  = self._load_lmdb_data_no_decompress(idx)

        # resize depth and bbox
        caption = self.caption_data[self.idx_to_ins_mapping[str(idx)]]

        return {
            **self._post_process_sample(raw_img, depth),
            'c': c,
            'bbox': (bbox * (self.reso / 512.0)).astype(np.uint8),
            # 'bbox': (bbox*(self.reso/256.0)).astype(np.uint8), # TODO, double check 512 in wds?
            'caption': caption
        }
        # '''
        # raw_img, depth, c, bbox  = self._load_lmdb_data_no_decompress(idx)
        # st()
        # return {}

    def __getitem__(self, idx):
        return self._load_data(idx)


class Objv_LMDBDataset_MV_NoCompressed(Objv_LMDBDataset_MV_Compressed):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 test=False,
                 **kwargs):
        super().__init__(lmdb_path, reso, reso_encoder, imgnet_normalize,
                         dataset_size, test, **kwargs)

    def _load_data(self, idx):
        # '''
        raw_img, depth, c, bbox = self._load_lmdb_data_no_decompress(idx)

        # resize depth and bbox
        caption = self.caption_data[self.idx_to_ins_mapping[str(idx)]]

        return {
            **self._post_process_sample(raw_img, depth), 'c': c,
            'bbox': (bbox * (self.reso / 512.0)).astype(np.uint8),
            'caption': caption
        }
        return {}


class Objv_LMDBDataset_NV_NoCompressed(Objv_LMDBDataset_MV_NoCompressed):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 test=False,
                 **kwargs):
        super().__init__(lmdb_path, reso, reso_encoder, imgnet_normalize,
                         dataset_size, test, **kwargs)

    def __getitem__(self, idx):
        input_view = self._load_data(idx)  # get previous input view results

        # get novel view of the same instance
        try:
            novel_view = self._load_data(
                (idx // self.instance_data_length) *
                self.instance_data_length +
                random.randint(0, self.instance_data_length - 1))
        except Exception as e:
            raise NotImplementedError(idx)

        # assert input_view['ins_name'] == novel_view['ins_name'], 'should sample novel view from the same instance'

        input_view.update({f'nv_{k}': v for k, v in novel_view.items()})
        return input_view


class Objv_LMDBDataset_MV_Compressed_for_lmdb(LMDBDataset_MV_Compressed):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 test=False,
                 **kwargs):
        super().__init__(lmdb_path,
                         reso,
                         reso_encoder,
                         imgnet_normalize,
                         dataset_size=dataset_size,
                         **kwargs)
        self.instance_data_length = 40  # ! could save some key attributes in LMDB
        if test:
            self.length = self.instance_data_length
        elif dataset_size > 0:
            self.length = dataset_size * self.instance_data_length

        # load caption data, and idx-to-ins mapping
        with open(
                '/cpfs01/shared/V2V/V2V_hdd/yslan/aigc3d/text_captions_cap3d.json'
        ) as f:
            self.caption_data = json.load(f)
        with open(os.path.join(lmdb_path, 'idx_to_ins_mapping.json')) as f:
            self.idx_to_ins_mapping = json.load(f)

    # def _load_data(self, idx):
    #     # '''
    #     raw_img, depth, c, bbox  = self._load_lmdb_data(idx)

    #     # resize depth and bbox
    #     caption = self.caption_data[self.idx_to_ins_mapping[str(idx)]]

    #     # st()

    #     return {
    #         **self._post_process_sample(raw_img, depth), 'c': c,
    #         'bbox': (bbox*(self.reso/512.0)).astype(np.uint8),
    #         'caption': caption
    #     }
    #     # '''
    #     # raw_img, depth, c, bbox  = self._load_lmdb_data_no_decompress(idx)
    #     # st()
    #     # return {}

    def load_bbox(self, mask):
        # st()
        nonzero_value = torch.nonzero(mask)
        height, width = nonzero_value.max(dim=0)[0]
        top, left = nonzero_value.min(dim=0)[0]
        bbox = torch.tensor([top, left, height, width], dtype=torch.float32)
        return bbox

    def __getitem__(self, idx):
        raw_img, depth, c, bbox = self._load_lmdb_data(idx)
        return {'raw_img': raw_img, 'depth': depth, 'c': c, 'bbox': bbox}


class Objv_LMDBDataset_NV_Compressed(Objv_LMDBDataset_MV_Compressed):

    def __init__(self,
                 lmdb_path,
                 reso,
                 reso_encoder,
                 imgnet_normalize=True,
                 dataset_size=-1,
                 **kwargs):
        super().__init__(lmdb_path, reso, reso_encoder, imgnet_normalize,
                         dataset_size, **kwargs)

    def __getitem__(self, idx):
        input_view = self._load_data(idx)  # get previous input view results

        # get novel view of the same instance
        try:
            novel_view = self._load_data(
                (idx // self.instance_data_length) *
                self.instance_data_length +
                random.randint(0, self.instance_data_length - 1))
        except Exception as e:
            raise NotImplementedError(idx)

        # assert input_view['ins_name'] == novel_view['ins_name'], 'should sample novel view from the same instance'

        input_view.update({f'nv_{k}': v for k, v in novel_view.items()})
        return input_view


#


# test tar loading
def load_wds_ResampledShard(file_path,
                            batch_size,
                            num_workers,
                            reso,
                            reso_encoder,
                            test=False,
                            preprocess=None,
                            imgnet_normalize=True,
                            plucker_embedding=False,
                            decode_encode_img_only=False,
                            load_instance=False,
                            mv_input=False,
                            split_chunk_input=False,
                            duplicate_sample=True,
                            append_depth=False,
                            gs_cam_format=False,
                            orthog_duplicate=False,
                            **kwargs):

    #     return raw_img, depth, c, bbox, sample_pyd['ins.pyd'], sample_pyd['fname.pyd']
    class PostProcess:

        def __init__(
            self,
            reso,
            reso_encoder,
            imgnet_normalize,
            plucker_embedding,
            decode_encode_img_only,
            mv_input,
            split_chunk_input,
            duplicate_sample,
            append_depth,
            gs_cam_format,
            orthog_duplicate,
        ) -> None:
            self.gs_cam_format = gs_cam_format
            self.append_depth = append_depth
            self.plucker_embedding = plucker_embedding
            self.decode_encode_img_only = decode_encode_img_only
            self.duplicate_sample = duplicate_sample
            self.orthog_duplicate = orthog_duplicate

            self.zfar = 100.0
            self.znear = 0.01

            transformations = []
            if not split_chunk_input:
                transformations.append(transforms.ToTensor())

            if imgnet_normalize:
                transformations.append(
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))  # type: ignore
                )
            else:
                transformations.append(
                    transforms.Normalize((0.5, 0.5, 0.5),
                                         (0.5, 0.5, 0.5)))  # type: ignore

            self.normalize = transforms.Compose(transformations)

            self.reso_encoder = reso_encoder
            self.reso = reso
            self.instance_data_length = 40
            # self.pair_per_instance = 1 # compat
            self.mv_input = mv_input
            self.split_chunk_input = split_chunk_input  # 8
            self.chunk_size = 8 if split_chunk_input else 40
            # st()
            if split_chunk_input:
                self.pair_per_instance = 1
            else:
                self.pair_per_instance = 4 if mv_input else 2  # check whether improves IO

        def gen_rays(self, c):
            # Generate rays
            intrinsics, c2w = c[16:], c[:16].reshape(4, 4)
            self.h = self.reso_encoder
            self.w = self.reso_encoder
            yy, xx = torch.meshgrid(
                torch.arange(self.h, dtype=torch.float32) + 0.5,
                torch.arange(self.w, dtype=torch.float32) + 0.5,
                indexing='ij')

            # normalize to 0-1 pixel range
            yy = yy / self.h
            xx = xx / self.w

            # K = np.array([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
            cx, cy, fx, fy = intrinsics[2], intrinsics[5], intrinsics[
                0], intrinsics[4]
            # cx *= self.w
            # cy *= self.h

            # f_x = f_y = fx * h / res_raw
            c2w = torch.from_numpy(c2w).float()

            xx = (xx - cx) / fx
            yy = (yy - cy) / fy
            zz = torch.ones_like(xx)
            dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
            dirs /= torch.norm(dirs, dim=-1, keepdim=True)
            dirs = dirs.reshape(-1, 3, 1)
            del xx, yy, zz
            # st()
            dirs = (c2w[None, :3, :3] @ dirs)[..., 0]

            origins = c2w[None, :3, 3].expand(self.h * self.w, -1).contiguous()
            origins = origins.view(self.h, self.w, 3)
            dirs = dirs.view(self.h, self.w, 3)

            return origins, dirs

        def _post_process_batch_sample(
                self, sample):  # sample is an instance batch here
            caption, ins = sample[-2:]
            instance_samples = []

            for instance_idx in range(sample[0].shape[0]):
                instance_samples.append(
                    self._post_process_sample(item[instance_idx]
                                              for item in sample[:-2]))

            return (*instance_samples, caption, ins)

        def _post_process_sample(self, data_sample):
            # raw_img, depth, c, bbox, caption, ins = data_sample
            raw_img, depth, c, bbox = data_sample

            bbox = (bbox * (self.reso / 256)).astype(
                np.uint8)  # normalize bbox to the reso range

            if raw_img.shape[-2] != self.reso_encoder:
                img_to_encoder = cv2.resize(
                    raw_img, (self.reso_encoder, self.reso_encoder),
                    interpolation=cv2.INTER_LANCZOS4)
            else:
                img_to_encoder = raw_img

            img_to_encoder = self.normalize(img_to_encoder)
            if self.plucker_embedding:
                rays_o, rays_d = self.gen_rays(c)
                rays_plucker = torch.cat(
                    [torch.cross(rays_o, rays_d, dim=-1), rays_d],
                    dim=-1).permute(2, 0, 1)  # [h, w, 6] -> 6,h,w
                img_to_encoder = torch.cat([img_to_encoder, rays_plucker], 0)

            img = cv2.resize(raw_img, (self.reso, self.reso),
                             interpolation=cv2.INTER_LANCZOS4)

            img = torch.from_numpy(img).permute(2, 0, 1) / 127.5 - 1

            if self.decode_encode_img_only:
                depth_reso, fg_mask_reso = depth, depth
            else:
                depth_reso, fg_mask_reso = resize_depth_mask(depth, self.reso)

            # return {
            #     # **sample,
            #     'img_to_encoder': img_to_encoder,
            #     'img': img,
            #     'depth_mask': fg_mask_reso,
            #     # 'img_sr': img_sr,
            #     'depth': depth_reso,
            #     'c': c,
            #     'bbox': bbox,
            #     'caption': caption,
            #     'ins': ins
            #     # ! no need to load img_sr for now
            # }
            # if len(data_sample) == 4:
            return (img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox)
            # else:
            #     return (img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox, data_sample[-2], data_sample[-1])

        def _post_process_sample_batch(self, data_sample):
            # raw_img, depth, c, bbox, caption, ins = data_sample
            raw_img, depth, c, bbox = data_sample

            bbox = (bbox * (self.reso / 256)).astype(
                np.uint8)  # normalize bbox to the reso range

            assert raw_img.shape[-2] == self.reso_encoder
            # img_to_encoder = cv2.resize(
            #     raw_img, (self.reso_encoder, self.reso_encoder),
            #     interpolation=cv2.INTER_LANCZOS4)
            # else:
            # img_to_encoder = raw_img

            raw_img = torch.from_numpy(raw_img).permute(0, 3, 1,
                                                        2) / 255.0  # [0,1]
            img_to_encoder = self.normalize(raw_img)

            if self.plucker_embedding:
                rays_plucker = []
                for idx in range(c.shape[0]):
                    rays_o, rays_d = self.gen_rays(c[idx])
                    rays_plucker.append(
                        torch.cat(
                            [torch.cross(rays_o, rays_d, dim=-1), rays_d],
                            dim=-1).permute(2, 0, 1))  # [h, w, 6] -> 6,h,w
                rays_plucker = torch.stack(rays_plucker, 0)
                img_to_encoder = torch.cat([img_to_encoder, rays_plucker],
                                           1)  # concat in C dim
            if self.append_depth:
                normalized_depth = torch.from_numpy(depth).clone().unsqueeze(
                    1)  # min=0
                # normalized_depth -= torch.min(normalized_depth) # always 0 here
                # normalized_depth /= torch.max(normalized_depth)
                # normalized_depth = normalized_depth.unsqueeze(1) * 2 - 1 # normalize to [-1,1]
                img_to_encoder = torch.cat([img_to_encoder, normalized_depth],
                                           1)  # concat in C dim

            # img = cv2.resize(raw_img, (self.reso, self.reso),
            #                  interpolation=cv2.INTER_LANCZOS4)

            # img = torch.from_numpy(raw_img).permute(2, 0, 1) / 127.5 - 1
            # st()
            if raw_img.shape[-1] != self.reso:
                img = torch.nn.functional.interpolate(
                    input=raw_img,
                    size=(self.reso, self.reso),
                    mode='bilinear',
                    align_corners=False,
                ) * 2 - 1  # [-1,1] range
            else:
                img = raw_img * 2 - 1

            if self.decode_encode_img_only:
                depth_reso, fg_mask_reso = depth, depth
            else:
                depth_reso, fg_mask_reso = resize_depth_mask_Tensor(
                    torch.from_numpy(depth), self.reso)

            # if not self.gs_cam_format: # otherwise still playing with np format later
            c = torch.from_numpy(c)

            return (img_to_encoder, img, fg_mask_reso, depth_reso, c,
                    torch.from_numpy(bbox))

        def rand_sample_idx(self):
            return random.randint(0, self.instance_data_length - 1)

        def rand_pair(self):
            return (self.rand_sample_idx() for _ in range(2))

        def paired_post_process(self, sample):
            # repeat n times?
            all_inp_list = []
            all_nv_list = []
            caption, ins = sample[-2:]
            # expanded_return = []
            for _ in range(self.pair_per_instance):
                cano_idx, nv_idx = self.rand_pair()
                cano_sample = self._post_process_sample(
                    item[cano_idx] for item in sample[:-2])
                nv_sample = self._post_process_sample(item[nv_idx]
                                                      for item in sample[:-2])
                all_inp_list.extend(cano_sample)
                all_nv_list.extend(nv_sample)
            return (*all_inp_list, *all_nv_list, caption, ins)
            # return [cano_sample, nv_sample, caption, ins]
            # return (*cano_sample, *nv_sample, caption, ins)

        def get_source_cw2wT(self, source_cameras_view_to_world):
            return matrix_to_quaternion(
                source_cameras_view_to_world[:3, :3].transpose(0, 1))

        def c_to_3dgs_format(self, pose):
            # TODO, switch to torch version (batched later)

            c2w = pose[:16].reshape(4, 4)  # 3x4

            # ! load cam
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            fx = pose[16]
            FovX = focal2fov(fx, 1)
            FovY = focal2fov(fx, 1)

            tanfovx = math.tan(FovX * 0.5)
            tanfovy = math.tan(FovY * 0.5)

            assert tanfovx == tanfovy

            trans = np.array([0.0, 0.0, 0.0])
            scale = 1.0

            view_world_transform = torch.tensor(
                getView2World(R, T, trans, scale)).transpose(0, 1)

            world_view_transform = torch.tensor(
                getWorld2View2(R, T, trans, scale)).transpose(0, 1)
            projection_matrix = getProjectionMatrix(znear=self.znear,
                                                    zfar=self.zfar,
                                                    fovX=FovX,
                                                    fovY=FovY).transpose(0, 1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
                projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            # item.update(viewpoint_cam=[viewpoint_cam])
            c = {}
            #
            c["source_cv2wT_quat"] = self.get_source_cw2wT(
                view_world_transform)
            c.update(
                # projection_matrix=projection_matrix, # K
                cam_view=world_view_transform,  # world_view_transform
                cam_view_proj=full_proj_transform,  # full_proj_transform
                cam_pos=camera_center,
                tanfov=tanfovx,  # TODO, fix in the renderer
                orig_pose=torch.from_numpy(pose),
                orig_c2w=torch.from_numpy(c2w),
                orig_w2c=torch.from_numpy(w2c),
                # tanfovy=tanfovy,
            )

            return c  # dict for gs rendering

        def paired_post_process_chunk(self, sample):
            # repeat n times?
            all_inp_list = []
            all_nv_list = []
            caption, ins = sample[-2:]
            assert sample[0].shape[0] == 8  # random chunks
            # expanded_return = []

            if self.duplicate_sample:
                processed_sample = self._post_process_sample_batch(
                    item for item in sample[:-2])

                if self.orthog_duplicate:
                    indices = torch.cat([torch.randperm(8),
                                         torch.randperm(8)])  # for now
                else:
                    indices = torch.randperm(8)

                shuffle_processed_sample = []

                for _, item in enumerate(processed_sample):
                    shuffle_processed_sample.append(
                        torch.index_select(item, dim=0, index=indices))
                processed_sample = shuffle_processed_sample

                if not self.orthog_duplicate:
                    all_inp_list.extend(item[:4] for item in processed_sample)
                    all_nv_list.extend(item[4:] for item in processed_sample)
                else:
                    all_inp_list.extend(item[:8] for item in processed_sample)
                    all_nv_list.extend(item[8:] for item in processed_sample)

                return (*all_inp_list, *all_nv_list, caption, ins)

            else:
                processed_sample = self._post_process_sample_batch(  # avoid shuffle shorten processing time
                    item[:4] for item in sample[:-2])

                all_inp_list.extend(item for item in processed_sample)
                all_nv_list.extend(
                    item for item in processed_sample)  # ! placeholder

            return (*all_inp_list, *all_nv_list, caption, ins)

            # randomly shuffle 8 views, avoid overfitting

        def single_sample_create_dict(self, sample, prefix=''):
            # if len(sample) == 1:
            #     sample = sample[0]
            # assert len(sample) == 6
            img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox = sample

            if self.gs_cam_format:
                # TODO, can optimize later after model converges
                B, V, _ = c.shape  # B 4 25
                c = rearrange(c, 'B V C -> (B V) C').cpu().numpy()
                all_gs_c = [self.c_to_3dgs_format(pose) for pose in c]
                c = {
                    k:
                    rearrange(torch.stack([gs_c[k] for gs_c in all_gs_c]),
                              '(B V) ... -> B V ...',
                              B=B,
                              V=V) if isinstance(all_gs_c[0][k], torch.Tensor)
                    else all_gs_c[0][k]
                    for k in all_gs_c[0].keys()
                }
                # c = collate_gs_c

            return {
                # **sample,
                f'{prefix}img_to_encoder': img_to_encoder,
                f'{prefix}img': img,
                f'{prefix}depth_mask': fg_mask_reso,
                f'{prefix}depth': depth_reso,
                f'{prefix}c': c,
                f'{prefix}bbox': bbox,
            }

        def single_instance_sample_create_dict(self, sample, prfix=''):
            assert len(sample) == 42

            inp_sample_list = [[] for _ in range(6)]

            for item in sample[:40]:
                for item_idx in range(6):
                    inp_sample_list[item_idx].append(item[0][item_idx])

            inp_sample = self.single_sample_create_dict(
                (torch.stack(item_list) for item_list in inp_sample_list),
                prefix='')

            return {
                **inp_sample,  # 
                'caption': sample[-2],
                'ins': sample[-1]
            }

        def decode_zip(self, sample_pyd, shape=(256, 256)):
            if isinstance(sample_pyd, tuple):
                sample_pyd = sample_pyd[0]
            assert isinstance(sample_pyd, dict)

            raw_img = decompress_and_open_image_gzip(
                sample_pyd['raw_img'],
                is_img=True,
                decompress=True,
                decompress_fn=lz4.frame.decompress)

            caption = sample_pyd['caption'].decode('utf-8')
            ins = sample_pyd['ins'].decode('utf-8')

            c = decompress_array(sample_pyd['c'], (
                self.chunk_size,
                25,
            ),
                                 np.float32,
                                 decompress=True,
                                 decompress_fn=lz4.frame.decompress)

            bbox = decompress_array(
                sample_pyd['bbox'],
                (
                    self.chunk_size,
                    4,
                ),
                np.float32,
                # decompress=False)
                decompress=True,
                decompress_fn=lz4.frame.decompress)

            if self.decode_encode_img_only:
                depth = np.zeros(shape=(self.chunk_size,
                                        *shape))  # save loading time
            else:
                depth = decompress_array(sample_pyd['depth'],
                                         (self.chunk_size, *shape),
                                         np.float32,
                                         decompress=True,
                                         decompress_fn=lz4.frame.decompress)

            # return {'raw_img': raw_img, 'depth': depth, 'bbox': bbox, 'caption': caption, 'ins': ins, 'c': c}
            # return raw_img, depth, c, bbox, caption, ins
            # return raw_img, bbox, caption, ins
            # return bbox, caption, ins
            return raw_img, depth, c, bbox, caption, ins
            # ! run single-instance pipeline first
            # return raw_img[0], depth[0], c[0], bbox[0], caption, ins

        def create_dict(self, sample):
            # sample = [item[0] for item in sample] # wds wrap items in []
            # st()
            cano_sample_list = [[] for _ in range(6)]
            nv_sample_list = [[] for _ in range(6)]
            # st()
            # bs = (len(sample)-2) // 6
            for idx in range(0, self.pair_per_instance):

                cano_sample = sample[6 * idx:6 * (idx + 1)]
                nv_sample = sample[6 * self.pair_per_instance +
                                   6 * idx:6 * self.pair_per_instance + 6 *
                                   (idx + 1)]

                for item_idx in range(6):
                    cano_sample_list[item_idx].append(cano_sample[item_idx])
                    nv_sample_list[item_idx].append(nv_sample[item_idx])

                    # ! cycle input/output view for more pairs
                    cano_sample_list[item_idx].append(nv_sample[item_idx])
                    nv_sample_list[item_idx].append(cano_sample[item_idx])

            # if self.split_chunk_input:
            #     cano_sample = self.single_sample_create_dict(
            #         (torch.cat(item_list, 0) for item_list in cano_sample_list),
            #         prefix='')
            #     nv_sample = self.single_sample_create_dict(
            #         (torch.cat(item_list, 0) for item_list in nv_sample_list),
            #         prefix='nv_')
            # else:
            cano_sample = self.single_sample_create_dict(
                (torch.cat(item_list, 0) for item_list in cano_sample_list),
                prefix='')
            nv_sample = self.single_sample_create_dict(
                (torch.cat(item_list, 0) for item_list in nv_sample_list),
                prefix='nv_')

            return {
                **cano_sample,
                **nv_sample, 'caption': sample[-2],
                'ins': sample[-1]
            }

        def prepare_mv_input(self, sample):
            # sample = [item[0] for item in sample] # wds wrap items in []
            bs = len(sample['caption'])  # number of instances
            chunk_size = sample['img'].shape[0] // bs

            if self.split_chunk_input:
                for k, v in sample.items():
                    if isinstance(v, torch.Tensor):
                        sample[k] = rearrange(v,
                                              "b f c ... -> (b f) c ...",
                                              f=4 if not self.orthog_duplicate
                                              else 8).contiguous()

            # img = rearrange(sample['img'], "(b f) c h w -> b f c h w", f=4).contiguous()
            # gt = rearrange(sample['nv_img'], "(b f) c h w -> b c (f h) w", f=4).contiguous()
            # img = rearrange(sample['img'], "b f c h w -> b c (f h) w", f=4).contiguous()
            # gt = rearrange(sample['nv_img'], "b f c h w -> b c (f h) w", f=4).contiguous()
            # torchvision.utils.save_image(img, 'inp.jpg', normalize=True)
            # torchvision.utils.save_image(gt, 'nv.jpg', normalize=True)

            # ! shift nv
            else:
                for k, v in sample.items():
                    if k not in ['ins', 'caption']:

                        rolled_idx = torch.LongTensor(
                            list(
                                itertools.chain.from_iterable(
                                    list(range(i, sample['img'].shape[0], bs))
                                    for i in range(bs))))

                        v = torch.index_select(v, dim=0, index=rolled_idx)
                    sample[k] = v

                # img = sample['img']
                # gt = sample['nv_img']
                # torchvision.utils.save_image(img[0], 'inp.jpg', normalize=True)
                # torchvision.utils.save_image(gt[0], 'nv.jpg', normalize=True)

                for k, v in sample.items():
                    if 'nv' in k:
                        rolled_idx = torch.LongTensor(
                            list(
                                itertools.chain.from_iterable(
                                    list(
                                        np.roll(
                                            np.arange(i * chunk_size, (i + 1) *
                                                      chunk_size), 4)
                                        for i in range(bs)))))

                        v = torch.index_select(v, dim=0, index=rolled_idx)
                        sample[k] = v

            # torchvision.utils.save_image(sample['nv_img'], 'nv.png', normalize=True)
            # torchvision.utils.save_image(sample['img'], 'inp.png', normalize=True)

            return sample

    post_process_cls = PostProcess(
        reso,
        reso_encoder,
        imgnet_normalize=imgnet_normalize,
        plucker_embedding=plucker_embedding,
        decode_encode_img_only=decode_encode_img_only,
        mv_input=mv_input,
        split_chunk_input=split_chunk_input,
        duplicate_sample=duplicate_sample,
        append_depth=append_depth,
        gs_cam_format=gs_cam_format,
        orthog_duplicate=orthog_duplicate,
    )

    # ! add shuffling

    if isinstance(file_path, list):  # lst of shard urls
        all_shards = []
        for url_path in file_path:
            all_shards.extend(wds.shardlists.expand_source(url_path))
        logger.log('all_shards', all_shards)
    else:
        all_shards = file_path  # to be expanded

    if not load_instance:  # during reconstruction training, load pair
        if not split_chunk_input:
            dataset = wds.DataPipeline(
                wds.ResampledShards(all_shards),  # url_shard
                # at this point we have an iterator over all the shards
                wds.shuffle(50),
                wds.split_by_worker,  # if multi-node
                wds.tarfile_to_samples(),
                # add wds.split_by_node here if you are using multiple nodes
                wds.shuffle(
                    1000
                ),  # shuffles in the memory, leverage large RAM for more efficient loading
                wds.decode(wds.autodecode.basichandlers),  # TODO
                wds.to_tuple(
                    "sample.pyd"),  # extract the pyd from top level dict
                wds.map(post_process_cls.decode_zip),
                wds.map(post_process_cls.paired_post_process
                        ),  # create input-novelview paired samples
                # wds.map(post_process_cls._post_process_sample),
                # wds.detshuffle(1000),  # shuffles in the memory, leverage large RAM for more efficient loading
                wds.batched(
                    16,
                    partial=True,
                    # collation_fn=collate
                )  # streaming more data at once, and rebatch later
            )

        else:
            dataset = wds.DataPipeline(
                wds.ResampledShards(all_shards),  # url_shard
                # at this point we have an iterator over all the shards
                wds.shuffle(100),
                wds.split_by_worker,  # if multi-node
                wds.tarfile_to_samples(),
                # add wds.split_by_node here if you are using multiple nodes
                wds.shuffle(
                    # 7500 if not duplicate_sample else 2500
                    # 7500 if not duplicate_sample else 5000
                    # 1000,
                    250,
                ),  # shuffles in the memory, leverage large RAM for more efficient loading
                wds.decode(wds.autodecode.basichandlers),  # TODO
                wds.to_tuple(
                    "sample.pyd"),  # extract the pyd from top level dict
                wds.map(post_process_cls.decode_zip),
                wds.map(post_process_cls.paired_post_process_chunk
                        ),  # create input-novelview paired samples
                # wds.map(post_process_cls._post_process_sample),
                # wds.detshuffle(1000),  # shuffles in the memory, leverage large RAM for more efficient loading
                wds.batched(
                    20,
                    partial=True,
                    # collation_fn=collate
                )  # streaming more data at once, and rebatch later
            )

        loader_shard = wds.WebLoader(
            dataset,
            num_workers=num_workers,
            drop_last=False,
            batch_size=None,
            shuffle=False,
            persistent_workers=num_workers
            # > 0).unbatched().shuffle(1000).batched(batch_size).map(
            > 0).unbatched().shuffle(250).batched(batch_size).map(
                post_process_cls.create_dict)

        if mv_input:
            loader_shard = loader_shard.map(post_process_cls.prepare_mv_input)

    else:  # load single instance during test/eval
        assert batch_size == 1

        dataset = wds.DataPipeline(
            wds.ResampledShards(all_shards),  # url_shard
            # at this point we have an iterator over all the shards
            wds.shuffle(50),
            wds.split_by_worker,  # if multi-node
            wds.tarfile_to_samples(),
            # add wds.split_by_node here if you are using multiple nodes
            wds.detshuffle(
                100
            ),  # shuffles in the memory, leverage large RAM for more efficient loading
            wds.decode(wds.autodecode.basichandlers),  # TODO
            wds.to_tuple("sample.pyd"),  # extract the pyd from top level dict
            wds.map(post_process_cls.decode_zip),
            # wds.map(post_process_cls.paired_post_process), # create input-novelview paired samples
            wds.map(post_process_cls._post_process_batch_sample),
            # wds.detshuffle(1000),  # shuffles in the memory, leverage large RAM for more efficient loading
            wds.batched(
                2,
                partial=True,
                # collation_fn=collate
            )  # streaming more data at once, and rebatch later
        )

        loader_shard = wds.WebLoader(
            dataset,
            num_workers=num_workers,
            drop_last=False,
            batch_size=None,
            shuffle=False,
            persistent_workers=num_workers
            > 0).unbatched().shuffle(200).batched(batch_size).map(
                post_process_cls.single_instance_sample_create_dict)

        # persistent_workers=num_workers > 0).unbatched().batched(batch_size).map(post_process_cls.create_dict)
        # 1000).batched(batch_size).map(post_process_cls.create_dict)
    # .map(collate)
    # .map(collate)

    # .batched(batch_size)
    #

    # .unbatched().shuffle(1000).batched(batch_size).map(post_process)
    #     # https://github.com/webdataset/webdataset/issues/187

    # return next(iter(loader_shard))
    #return dataset
    return loader_shard


# test tar loading
def load_wds_diff_ResampledShard(file_path,
                                 batch_size,
                                 num_workers,
                                 reso,
                                 reso_encoder,
                                 test=False,
                                 preprocess=None,
                                 imgnet_normalize=True,
                                 plucker_embedding=False,
                                 decode_encode_img_only=False,
                                 mv_latent_dir='',
                                 **kwargs):

    #     return raw_img, depth, c, bbox, sample_pyd['ins.pyd'], sample_pyd['fname.pyd']
    class PostProcess:

        def __init__(
            self,
            reso,
            reso_encoder,
            imgnet_normalize,
            plucker_embedding,
            decode_encode_img_only,
            mv_latent_dir,
        ) -> None:
            self.plucker_embedding = plucker_embedding

            self.mv_latent_dir = mv_latent_dir
            self.decode_encode_img_only = decode_encode_img_only

            transformations = [
                transforms.ToTensor(),  # [0,1] range
            ]
            if imgnet_normalize:
                transformations.append(
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))  # type: ignore
                )
            else:
                transformations.append(
                    transforms.Normalize((0.5, 0.5, 0.5),
                                         (0.5, 0.5, 0.5)))  # type: ignore

            self.normalize = transforms.Compose(transformations)

            self.reso_encoder = reso_encoder
            self.reso = reso
            self.instance_data_length = 40
            # self.pair_per_instance = 1 # compat
            self.pair_per_instance = 2  # check whether improves IO
            # self.pair_per_instance = 3 # check whether improves IO
            # self.pair_per_instance = 4 # check whether improves IO

        def get_rays_kiui(self, c, opengl=True):
            h, w = self.reso_encoder, self.reso_encoder
            intrinsics, pose = c[16:], c[:16].reshape(4, 4)
            # cx, cy, fx, fy = intrinsics[2], intrinsics[5]
            fx = fy = 525  # pixel space
            cx = cy = 256  # rendering default K
            factor = self.reso / (cx * 2)  # 128 / 512
            fx = fx * factor
            fy = fy * factor

            x, y = torch.meshgrid(
                torch.arange(w, device=pose.device),
                torch.arange(h, device=pose.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

            cx = w * 0.5
            cy = h * 0.5

            # focal = h * 0.5 / np.tan(0.5 * np.deg2rad(fovy))

            camera_dirs = F.pad(
                torch.stack(
                    [
                        (x - cx + 0.5) / fx,
                        (y - cy + 0.5) / fy * (-1.0 if opengl else 1.0),
                    ],
                    dim=-1,
                ),
                (0, 1),
                value=(-1.0 if opengl else 1.0),
            )  # [hw, 3]

            rays_d = camera_dirs @ pose[:3, :3].transpose(0, 1)  # [hw, 3]
            rays_o = pose[:3, 3].unsqueeze(0).expand_as(rays_d)  # [hw, 3]

            rays_o = rays_o.view(h, w, 3)
            rays_d = safe_normalize(rays_d).view(h, w, 3)

            return rays_o, rays_d

        def gen_rays(self, c):
            # Generate rays
            intrinsics, c2w = c[16:], c[:16].reshape(4, 4)
            self.h = self.reso_encoder
            self.w = self.reso_encoder
            yy, xx = torch.meshgrid(
                torch.arange(self.h, dtype=torch.float32) + 0.5,
                torch.arange(self.w, dtype=torch.float32) + 0.5,
                indexing='ij')

            # normalize to 0-1 pixel range
            yy = yy / self.h
            xx = xx / self.w

            # K = np.array([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
            cx, cy, fx, fy = intrinsics[2], intrinsics[5], intrinsics[
                0], intrinsics[4]
            # cx *= self.w
            # cy *= self.h

            # f_x = f_y = fx * h / res_raw
            c2w = torch.from_numpy(c2w).float()

            xx = (xx - cx) / fx
            yy = (yy - cy) / fy
            zz = torch.ones_like(xx)
            dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
            dirs /= torch.norm(dirs, dim=-1, keepdim=True)
            dirs = dirs.reshape(-1, 3, 1)
            del xx, yy, zz
            # st()
            dirs = (c2w[None, :3, :3] @ dirs)[..., 0]

            origins = c2w[None, :3, 3].expand(self.h * self.w, -1).contiguous()
            origins = origins.view(self.h, self.w, 3)
            dirs = dirs.view(self.h, self.w, 3)

            return origins, dirs

        def _post_process_sample(self, data_sample):
            # raw_img, depth, c, bbox, caption, ins = data_sample
            raw_img, c, caption, ins = data_sample

            # bbox = (bbox*(self.reso/256)).astype(np.uint8) # normalize bbox to the reso range

            # if raw_img.shape[-2] != self.reso_encoder:
            #     img_to_encoder = cv2.resize(
            #         raw_img, (self.reso_encoder, self.reso_encoder),
            #         interpolation=cv2.INTER_LANCZOS4)
            # else:
            #     img_to_encoder = raw_img

            # img_to_encoder = self.normalize(img_to_encoder)
            # if self.plucker_embedding:
            #     rays_o, rays_d = self.gen_rays(c)
            #     rays_plucker = torch.cat(
            #         [torch.cross(rays_o, rays_d, dim=-1), rays_d],
            #         dim=-1).permute(2, 0, 1)  # [h, w, 6] -> 6,h,w
            #     img_to_encoder = torch.cat([img_to_encoder, rays_plucker], 0)

            # img = cv2.resize(raw_img, (self.reso, self.reso),
            #                  interpolation=cv2.INTER_LANCZOS4)
            img = raw_img  # 256x256

            img = torch.from_numpy(img).permute(2, 0, 1) / 127.5 - 1

            # load latent

            latent_path = Path(self.mv_latent_dir, ins, 'latent.npy')
            latent = np.load(latent_path)

            # return (img_to_encoder, img, c, caption, ins)
            return (latent, img, c, caption, ins)

        def rand_sample_idx(self):
            return random.randint(0, self.instance_data_length - 1)

        def rand_pair(self):
            return (self.rand_sample_idx() for _ in range(2))

        def paired_post_process(self, sample):
            # repeat n times?
            all_inp_list = []
            all_nv_list = []
            caption, ins = sample[-2:]
            # expanded_return = []
            for _ in range(self.pair_per_instance):
                cano_idx, nv_idx = self.rand_pair()
                cano_sample = self._post_process_sample(
                    item[cano_idx] for item in sample[:-2])
                nv_sample = self._post_process_sample(item[nv_idx]
                                                      for item in sample[:-2])
                all_inp_list.extend(cano_sample)
                all_nv_list.extend(nv_sample)
            return (*all_inp_list, *all_nv_list, caption, ins)
            # return [cano_sample, nv_sample, caption, ins]
            # return (*cano_sample, *nv_sample, caption, ins)

        # def single_sample_create_dict(self, sample, prefix=''):
        #     # if len(sample) == 1:
        #     #     sample = sample[0]
        #     # assert len(sample) == 6
        #     img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox = sample
        #     return {
        #         # **sample,
        #         f'{prefix}img_to_encoder': img_to_encoder,
        #         f'{prefix}img': img,
        #         f'{prefix}depth_mask': fg_mask_reso,
        #         f'{prefix}depth': depth_reso,
        #         f'{prefix}c': c,
        #         f'{prefix}bbox': bbox,
        #     }

        def single_sample_create_dict(self, sample, prefix=''):
            # if len(sample) == 1:
            #     sample = sample[0]
            # assert len(sample) == 6
            # img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox = sample
            # img_to_encoder, img, c, caption, ins = sample
            # img, c, caption, ins = sample
            latent, img, c, caption, ins = sample
            # load latent
            return {
                # **sample,
                # 'img_to_encoder': img_to_encoder,
                'latent': latent,
                'img': img,
                'c': c,
                'caption': caption,
                'ins': ins
            }

        def decode_zip(self, sample_pyd, shape=(256, 256)):
            if isinstance(sample_pyd, tuple):
                sample_pyd = sample_pyd[0]
            assert isinstance(sample_pyd, dict)

            raw_img = decompress_and_open_image_gzip(
                sample_pyd['raw_img'],
                is_img=True,
                decompress=True,
                decompress_fn=lz4.frame.decompress)

            caption = sample_pyd['caption'].decode('utf-8')
            ins = sample_pyd['ins'].decode('utf-8')

            c = decompress_array(sample_pyd['c'], (25, ),
                                 np.float32,
                                 decompress=True,
                                 decompress_fn=lz4.frame.decompress)

            # bbox = decompress_array(
            #     sample_pyd['bbox'],
            #     (
            #         40,
            #         4,
            #     ),
            #     np.float32,
            #     # decompress=False)
            #     decompress=True,
            #     decompress_fn=lz4.frame.decompress)

            # if self.decode_encode_img_only:
            #     depth = np.zeros(shape=(40, *shape)) # save loading time
            # else:
            #     depth = decompress_array(sample_pyd['depth'], (40, *shape),
            #                             np.float32,
            #                             decompress=True,
            #                             decompress_fn=lz4.frame.decompress)

            # return {'raw_img': raw_img, 'depth': depth, 'bbox': bbox, 'caption': caption, 'ins': ins, 'c': c}
            # return raw_img, depth, c, bbox, caption, ins
            # return raw_img, bbox, caption, ins
            # return bbox, caption, ins
            return raw_img, c, caption, ins
            # ! run single-instance pipeline first
            # return raw_img[0], depth[0], c[0], bbox[0], caption, ins

        def create_dict(self, sample):
            # sample = [item[0] for item in sample] # wds wrap items in []
            # cano_sample_list = [[] for _ in range(6)]
            # nv_sample_list = [[] for _ in range(6)]
            # for idx in range(0, self.pair_per_instance):
            #     cano_sample = sample[6*idx:6*(idx+1)]
            #     nv_sample = sample[6*self.pair_per_instance+6*idx:6*self.pair_per_instance+6*(idx+1)]

            #     for item_idx in range(6):
            #         cano_sample_list[item_idx].append(cano_sample[item_idx])
            #         nv_sample_list[item_idx].append(nv_sample[item_idx])

            #         # ! cycle input/output view for more pairs
            #         cano_sample_list[item_idx].append(nv_sample[item_idx])
            #         nv_sample_list[item_idx].append(cano_sample[item_idx])

            cano_sample = self.single_sample_create_dict(sample, prefix='')
            # nv_sample = self.single_sample_create_dict((torch.cat(item_list) for item_list in nv_sample_list) , prefix='nv_')

            return cano_sample
            # return {
            #     **cano_sample,
            #     # **nv_sample,
            #     'caption': sample[-2],
            #     'ins': sample[-1]
            # }

    post_process_cls = PostProcess(
        reso,
        reso_encoder,
        imgnet_normalize=imgnet_normalize,
        plucker_embedding=plucker_embedding,
        decode_encode_img_only=decode_encode_img_only,
        mv_latent_dir=mv_latent_dir,
    )

    if isinstance(file_path, list):  # lst of shard urls
        all_shards = []
        for url_path in file_path:
            all_shards.extend(wds.shardlists.expand_source(url_path))
        logger.log('all_shards', all_shards)
    else:
        all_shards = file_path  # to be expanded

    dataset = wds.DataPipeline(
        wds.ResampledShards(all_shards),  # url_shard
        # at this point we have an iterator over all the shards
        wds.shuffle(50),
        wds.split_by_worker,  # if multi-node
        wds.tarfile_to_samples(),
        # add wds.split_by_node here if you are using multiple nodes
        wds.detshuffle(
            15000
        ),  # shuffles in the memory, leverage large RAM for more efficient loading
        wds.decode(wds.autodecode.basichandlers),  # TODO
        wds.to_tuple("sample.pyd"),  # extract the pyd from top level dict
        wds.map(post_process_cls.decode_zip),
        # wds.map(post_process_cls.paired_post_process), # create input-novelview paired samples
        wds.map(post_process_cls._post_process_sample),
        # wds.detshuffle(1000),  # shuffles in the memory, leverage large RAM for more efficient loading
        wds.batched(
            80,
            partial=True,
            # collation_fn=collate
        )  # streaming more data at once, and rebatch later
    )

    loader_shard = wds.WebLoader(
        dataset,
        num_workers=num_workers,
        drop_last=False,
        batch_size=None,
        shuffle=False,
        persistent_workers=num_workers
        > 0).unbatched().shuffle(2500).batched(batch_size).map(
            post_process_cls.create_dict)

    # persistent_workers=num_workers > 0).unbatched().batched(batch_size).map(post_process_cls.create_dict)
    # 1000).batched(batch_size).map(post_process_cls.create_dict)
    # .map(collate)
    # .map(collate)

    # .batched(batch_size)
    #

    # .unbatched().shuffle(1000).batched(batch_size).map(post_process)
    #     # https://github.com/webdataset/webdataset/issues/187

    # return next(iter(loader_shard))
    #return dataset
    return loader_shard


def load_wds_data(
        file_path="",
        reso=64,
        reso_encoder=224,
        batch_size=1,
        num_workers=6,
        plucker_embedding=False,
        decode_encode_img_only=False,
        load_wds_diff=False,
        load_wds_latent=False,
        load_instance=False,  # for evaluation
        mv_input=False,
        split_chunk_input=False,
        duplicate_sample=True,
        mv_latent_dir='',
        append_depth=False,
        gs_cam_format=False,
        orthog_duplicate=False,
        **args):

    if load_wds_diff:
        assert num_workers == 0  # on aliyun, worker=0 performs much much faster
        wds_loader = load_wds_diff_ResampledShard(
            file_path,
            batch_size=batch_size,
            num_workers=num_workers,
            reso=reso,
            reso_encoder=reso_encoder,
            plucker_embedding=plucker_embedding,
            decode_encode_img_only=decode_encode_img_only,
            mv_input=mv_input,
            split_chunk_input=split_chunk_input,
            append_depth=append_depth,
            mv_latent_dir=mv_latent_dir,
            gs_cam_format=gs_cam_format,
            orthog_duplicate=orthog_duplicate,
        )
    elif load_wds_latent:
        # for diffusion training, cache latent
        wds_loader = load_wds_latent_ResampledShard(
            file_path,
            batch_size=batch_size,
            num_workers=num_workers,
            reso=reso,
            reso_encoder=reso_encoder,
            plucker_embedding=plucker_embedding,
            decode_encode_img_only=decode_encode_img_only,
            mv_input=mv_input,
            split_chunk_input=split_chunk_input,
        )

    # elif load_instance:
    #     wds_loader = load_wds_instance_ResampledShard(
    #         file_path,
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #         reso=reso,
    #         reso_encoder=reso_encoder,
    #         plucker_embedding=plucker_embedding,
    #         decode_encode_img_only=decode_encode_img_only
    #     )

    else:
        wds_loader = load_wds_ResampledShard(
            file_path,
            batch_size=batch_size,
            num_workers=num_workers,
            reso=reso,
            reso_encoder=reso_encoder,
            plucker_embedding=plucker_embedding,
            decode_encode_img_only=decode_encode_img_only,
            load_instance=load_instance,
            mv_input=mv_input,
            split_chunk_input=split_chunk_input,
            duplicate_sample=duplicate_sample,
            append_depth=append_depth,
            gs_cam_format=gs_cam_format,
            orthog_duplicate=orthog_duplicate,
        )

    while True:
        yield from wds_loader
        # yield from wds_loader


class PostProcess_forlatent:
    def __init__(
        self,
        reso,
        reso_encoder,
        imgnet_normalize,
        plucker_embedding,
        decode_encode_img_only,
    ) -> None:
        self.plucker_embedding = plucker_embedding
        self.decode_encode_img_only = decode_encode_img_only

        transformations = [
            transforms.ToTensor(),  # [0,1] range
        ]
        if imgnet_normalize:
            transformations.append(
                transforms.Normalize((0.485, 0.456, 0.406),
                                        (0.229, 0.224, 0.225))  # type: ignore
            )
        else:
            transformations.append(
                transforms.Normalize((0.5, 0.5, 0.5),
                                        (0.5, 0.5, 0.5)))  # type: ignore

        self.normalize = transforms.Compose(transformations)

        self.reso_encoder = reso_encoder
        self.reso = reso
        self.instance_data_length = 40
        # self.pair_per_instance = 1 # compat
        self.pair_per_instance = 2  # check whether improves IO
        # self.pair_per_instance = 3 # check whether improves IO
        # self.pair_per_instance = 4 # check whether improves IO

    def _post_process_sample(self, data_sample):
        # raw_img, depth, c, bbox, caption, ins = data_sample
        raw_img, c, caption, ins = data_sample

        # bbox = (bbox*(self.reso/256)).astype(np.uint8) # normalize bbox to the reso range

        if raw_img.shape[-2] != self.reso_encoder:
            img_to_encoder = cv2.resize(
                raw_img, (self.reso_encoder, self.reso_encoder),
                interpolation=cv2.INTER_LANCZOS4)
        else:
            img_to_encoder = raw_img

        img_to_encoder = self.normalize(img_to_encoder)
        if self.plucker_embedding:
            rays_o, rays_d = self.gen_rays(c)
            rays_plucker = torch.cat(
                [torch.cross(rays_o, rays_d, dim=-1), rays_d],
                dim=-1).permute(2, 0, 1)  # [h, w, 6] -> 6,h,w
            img_to_encoder = torch.cat([img_to_encoder, rays_plucker], 0)

        img = cv2.resize(raw_img, (self.reso, self.reso),
                            interpolation=cv2.INTER_LANCZOS4)

        img = torch.from_numpy(img).permute(2, 0, 1) / 127.5 - 1

        return (img_to_encoder, img, c, caption, ins)

    def rand_sample_idx(self):
        return random.randint(0, self.instance_data_length - 1)

    def rand_pair(self):
        return (self.rand_sample_idx() for _ in range(2))

    def paired_post_process(self, sample):
        # repeat n times?
        all_inp_list = []
        all_nv_list = []
        caption, ins = sample[-2:]
        # expanded_return = []
        for _ in range(self.pair_per_instance):
            cano_idx, nv_idx = self.rand_pair()
            cano_sample = self._post_process_sample(
                item[cano_idx] for item in sample[:-2])
            nv_sample = self._post_process_sample(item[nv_idx]
                                                    for item in sample[:-2])
            all_inp_list.extend(cano_sample)
            all_nv_list.extend(nv_sample)
        return (*all_inp_list, *all_nv_list, caption, ins)
        # return [cano_sample, nv_sample, caption, ins]
        # return (*cano_sample, *nv_sample, caption, ins)

    # def single_sample_create_dict(self, sample, prefix=''):
    #     # if len(sample) == 1:
    #     #     sample = sample[0]
    #     # assert len(sample) == 6
    #     img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox = sample
    #     return {
    #         # **sample,
    #         f'{prefix}img_to_encoder': img_to_encoder,
    #         f'{prefix}img': img,
    #         f'{prefix}depth_mask': fg_mask_reso,
    #         f'{prefix}depth': depth_reso,
    #         f'{prefix}c': c,
    #         f'{prefix}bbox': bbox,
    #     }

    def single_sample_create_dict(self, sample, prefix=''):
        # if len(sample) == 1:
        #     sample = sample[0]
        # assert len(sample) == 6
        # img_to_encoder, img, fg_mask_reso, depth_reso, c, bbox = sample
        img_to_encoder, img, c, caption, ins = sample
        return {
            # **sample,
            'img_to_encoder': img_to_encoder,
            'img': img,
            'c': c,
            'caption': caption,
            'ins': ins
        }

    def decode_zip(self, sample_pyd, shape=(256, 256)):
        if isinstance(sample_pyd, tuple):
            sample_pyd = sample_pyd[0]
        assert isinstance(sample_pyd, dict)

        latent = sample_pyd['latent']
        caption = sample_pyd['caption'].decode('utf-8')
        c = sample_pyd['c']
        # img = sample_pyd['img']
        # st()

        return latent, caption, c

    def create_dict(self, sample):

        return {
            # **sample,
            'latent': sample[0],
            'caption': sample[1],
            'c': sample[2],
        }



# test tar loading
def load_wds_latent_ResampledShard(file_path,
                                   batch_size,
                                   num_workers,
                                   reso,
                                   reso_encoder,
                                   test=False,
                                   preprocess=None,
                                   imgnet_normalize=True,
                                   plucker_embedding=False,
                                   decode_encode_img_only=False,
                                   **kwargs):

    post_process_cls = PostProcess_forlatent(
        reso,
        reso_encoder,
        imgnet_normalize=imgnet_normalize,
        plucker_embedding=plucker_embedding,
        decode_encode_img_only=decode_encode_img_only,
    )

    if isinstance(file_path, list):  # lst of shard urls
        all_shards = []
        for url_path in file_path:
            all_shards.extend(wds.shardlists.expand_source(url_path))
        logger.log('all_shards', all_shards)
    else:
        all_shards = file_path  # to be expanded

    dataset = wds.DataPipeline(
        wds.ResampledShards(all_shards),  # url_shard
        # at this point we have an iterator over all the shards
        wds.shuffle(50),
        wds.split_by_worker,  # if multi-node
        wds.tarfile_to_samples(),
        # add wds.split_by_node here if you are using multiple nodes
        wds.detshuffle(
            2500
        ),  # shuffles in the memory, leverage large RAM for more efficient loading
        wds.decode(wds.autodecode.basichandlers),  # TODO
        wds.to_tuple("sample.pyd"),  # extract the pyd from top level dict
        wds.map(post_process_cls.decode_zip),
        # wds.map(post_process_cls._post_process_sample),
        # wds.detshuffle(1000),  # shuffles in the memory, leverage large RAM for more efficient loading
        wds.batched(
            150,
            partial=True,
            # collation_fn=collate
        )  # streaming more data at once, and rebatch later
    )

    loader_shard = wds.WebLoader(
        dataset,
        num_workers=num_workers,
        drop_last=False,
        batch_size=None,
        shuffle=False,
        persistent_workers=num_workers
        > 0).unbatched().shuffle(1000).batched(batch_size).map(
            post_process_cls.create_dict)

    # persistent_workers=num_workers > 0).unbatched().batched(batch_size).map(post_process_cls.create_dict)
    # 1000).batched(batch_size).map(post_process_cls.create_dict)
    # .map(collate)
    # .map(collate)

    # .batched(batch_size)
    #

    # .unbatched().shuffle(1000).batched(batch_size).map(post_process)
    #     # https://github.com/webdataset/webdataset/issues/187

    # return next(iter(loader_shard))
    #return dataset
    return loader_shard
