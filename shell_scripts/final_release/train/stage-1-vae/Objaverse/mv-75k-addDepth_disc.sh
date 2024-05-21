set -x 
# vit_decoder_lr=1.001

# lpips_lambda=0.8
# lpips_lambda=2.0 # ! lrm
lpips_lambda=2.0
# lpips_lambda=0.0
ssim_lambda=0.
l1_lambda=0. # following gaussian splatting
l2_lambda=1 # ! use_conf_map
# patchgan_disc=0.002

patchgan_disc_factor=1.0
patchgan_disc_g_weight=0.5

# NUM_GPUS=4
# NUM_GPUS=8
# NUM_GPUS=3
NUM_GPUS=8
# NUM_GPUS=1
# NUM_GPUS=7
# NUM_GPUS=5
# NUM_GPUS=6

image_size=128 # final rendered resolution
# image_size=64 # final rendered resolution
# image_size=80 # ! to alleviate memory issue
# image_size=128 # final rendered resolution

num_workers=0 # much faster, why?
# num_workers=3 # 
image_size_encoder=256
patch_size=14
kl_lambda=1.0e-06

# patch_rendering_resolution=64
# patch_rendering_resolution=68
# patch_rendering_resolution=58
# patch_rendering_resolution=48
patch_rendering_resolution=64 # ! render 8 views each, 64 crops given bs=4, 80gib
# patch_rendering_resolution=56 # ! render 8 views each, 64 crops given bs=4, 80gib
# patch_rendering_resolution=58 # ! OOM when BS=10

batch_size=4 # ! actuall BS will double
microbatch=32 # grad acc

# batch_size=1 # ! actuall BS will double
# microbatch=8 # grad acc

# data_dir=/cpfs01/user/yangpeiqing.p/yslan/data/Objaverse
# data_dir=/cpfs01/shared/V2V/V2V_hdd/yslan/Objaverse # hdd version for debug
# data_dir="/cpfs01/user/yangpeiqing.p/yslan/data/Furnitures_compressed_lz4/wds-{000000..000004}.tar"
# DATASET_FLAGS="
#  --data_dir ${data_dir} \
# "
# eval_data_dir=/cpfs01/shared/V2V/V2V_hdd/yslan/aigc3d/download_unzipped/Furnitures # to update later

# shards_lst="/cpfs01/user/lanyushi.p/Repo/diffusion-3d/shell_scripts/baselines/reconstruction/sr/final_mv/shards_lst_4w.txt"
# shards_lst="/cpfs01/user/lanyushi.p/Repo/diffusion-3d/shell_scripts/baselines/reconstruction/sr/final_mv/shards_lst.txt"
# shards_lst="/cpfs01/user/lanyushi.p/Repo/diffusion-3d/shell_scripts/baselines/reconstruction/sr/final_mv/shards_animals_lst.txt"
# shards_lst="/cpfs01/user/lanyushi.p/Repo/diffusion-3d/shell_scripts/baselines/reconstruction/sr/final_mv/shards_ani_trans_lst.txt"
# shards_lst="/cpfs01/user/lanyushi.p/Repo/diffusion-3d/shell_scripts/baselines/reconstruction/sr/final_mv/shards_furnitures_lst.txt"
# eval_data_dir=/cpfs01/shared/V2V/V2V_hdd/yslan/aigc3d/download_unzipped/Furnitures # to update later
# eval_data_dir="/cpfs01/user/lanyushi.p/data/Furnitures_compressed_lz4/wds-{000000..000004}.tar"

# shards_lst="/cpfs01/user/lanyushi.p/Repo/diffusion-3d/shell_scripts/baselines/reconstruction/sr/final_mv/shards_ani_trans_lst.txt"
# shards_lst="/cpfs01/user/lanyushi.p/Repo/diffusion-3d/shell_scripts/baselines/reconstruction/sr/final_mv/shards_lst_subset_shuffle.txt"
# shards_lst="/cpfs01/user/lanyushi.p/Repo/diffusion-3d/shell_scripts/baselines/reconstruction/sr/final_mv/shards_lst_subset_shuffle.txt"
shards_lst="shell_scripts/shards_list/shards_lst_75K_shuffle.txt"


DATASET_FLAGS="
 --data_dir "NONE" \
 --shards_lst ${shards_lst} \
 --eval_data_dir "NONE" \
 --eval_shards_lst ${shards_lst} \
"

conv_lr=2e-4
lr=1e-4 # 4e-4 for BS=256, we have 80 here.

vit_decoder_lr=$lr
encoder_lr=${conv_lr} # scaling version , could be larger when multi-nodes
triplane_decoder_lr=$conv_lr
super_resolution_lr=$conv_lr

# * above the best lr config

LR_FLAGS="--encoder_lr $encoder_lr \
--vit_decoder_lr $vit_decoder_lr \
--triplane_decoder_lr $triplane_decoder_lr \
--super_resolution_lr $super_resolution_lr \
--lr $lr"

TRAIN_FLAGS="--iterations 10001 --anneal_lr False \
 --batch_size $batch_size --save_interval 10000 \
 --microbatch ${microbatch} \
 --image_size_encoder $image_size_encoder \
 --dino_version mv-sd-dit \
 --sr_training False \
 --cls_token False \
 --weight_decay 0.05 \
 --image_size $image_size \
 --kl_lambda ${kl_lambda} \
 --no_dim_up_mlp True \
 --uvit_skip_encoder False \
 --fg_mse True \
 --bg_lamdba 1.0 \
 --lpips_delay_iter 100 \
 --sr_delay_iter 25000 \
 --kl_anneal True \
 --symmetry_loss False \
 --vae_p 2 \
 --plucker_embedding True \
 --encoder_in_channels 10 \
 --arch_dit_decoder DiT2-B/2 \
 --sd_E_ch 64 \
 --sd_E_num_res_blocks 1 \
 --lrm_decoder False \
 --resume_checkpoint /cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/Reconstruction/final/objav/vae/MV/75K/add-depth/adv-0.002/bs4-gpu8/model_rec1620000.pt \
 "


#  --resume_checkpoint /cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/Reconstruction/final/objav/vae/Ani-Trans-Furni/V=4/5-gpu8-lr2e-4vitlr-1e-4-plucker-patch46-ctd/model_rec0740000.pt \




# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/Reconstruction/final/objav/vae/Ani-Trans/V=4/MV/${batch_size}-gpu${NUM_GPUS}-lr${encoder_lr}vitlr-${vit_decoder_lr}-plucker-patch${patch_rendering_resolution}-V8-vtd-scaleinvDepth-noLRMDecoder-clipDepth-fullset_shuffle-fixdepth-shuffleInp/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/Reconstruction/final/objav/vae/MV/75K/bs${batch_size}-gpu${NUM_GPUS}-lr${encoder_lr}vitlr-${vit_decoder_lr}-patch${patch_rendering_resolution}
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/Reconstruction/final/objav/vae/MV/75K/bs${batch_size}-gpu${NUM_GPUS}-lr${encoder_lr}vitlr-${vit_decoder_lr}-patch${patch_rendering_resolution}-ctd
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/Reconstruction/final/objav/vae/MV/75K/add-depth/adv-${patchgan_disc}/bs${batch_size}-gpu${NUM_GPUS}
logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/Reconstruction/final/objav/vae/MV/75K/add-depth/adv-${patchgan_disc_factor}-${patchgan_disc_g_weight}/bs${batch_size}-gpu${NUM_GPUS}

SR_TRAIN_FLAGS_v1_2XC="
--decoder_in_chans 32 \
--out_chans 96 \
--alpha_lambda 1.0 \
--logdir $logdir \
--arch_encoder vits \
--arch_decoder vitb \
--vit_decoder_wd 0.001 \
--encoder_weight_decay 0.001 \
--color_criterion mse \
--decoder_output_dim 3 \
--ae_classname vit.vit_triplane.RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder_S \
"

SR_TRAIN_FLAGS=${SR_TRAIN_FLAGS_v1_2XC}


rm -rf "$logdir"/runs
mkdir -p "$logdir"/
cp "$0" "$logdir"/

# localedef -c -f UTF-8 -i en_US en_US.UTF-8
export LC_ALL=en_US.UTF-8

export OPENCV_IO_ENABLE_OPENEXR=1
export OMP_NUM_THREADS=12
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_GID_INDEX=3 # https://github.com/huggingface/accelerate/issues/314#issuecomment-1821973930
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=4,5,6
# export CUDA_VISIBLE_DEVICES=1,2,3,4
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=3,0,1
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=1
# export CUDA_VISIBLE_DEVICES=5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6


  # --master_addr=${MASTER_ADDR} \
  # --node_rank=${RANK} \

torchrun --nproc_per_node=$NUM_GPUS \
  --nnodes=1 \
  --rdzv-endpoint=localhost:15368 \
  --rdzv_backend=c10d \
 scripts/vit_triplane_train.py \
 --trainer_name nv_rec_patch_mvE_disc \
 --num_workers ${num_workers} \
 ${TRAIN_FLAGS}  \
 ${SR_TRAIN_FLAGS} \
 ${DATASET_FLAGS} \
 --lpips_lambda $lpips_lambda \
 --overfitting False \
 --load_pretrain_encoder False \
 --iterations 5000001 \
 --save_interval 10000 \
 --eval_interval 250000000 \
 --decomposed True \
 --logdir $logdir \
 --decoder_load_pretrained False \
 --cfg objverse_tuneray_aug_resolution_64_64_auto \
 --patch_size ${patch_size} \
 --use_amp False \
 --eval_batch_size 1 \
 ${LR_FLAGS} \
 --l1_lambda ${l1_lambda} \
 --l2_lambda ${l2_lambda} \
 --ssim_lambda ${ssim_lambda} \
 --depth_smoothness_lambda 0 \
 --use_conf_map False \
 --objv_dataset True \
 --depth_lambda 0.5 \
 --patch_rendering_resolution ${patch_rendering_resolution} \
 --use_lmdb_compressed False \
 --use_lmdb False \
 --mv_input True \
 --split_chunk_input True \
 --append_depth True \
 --patchgan_disc_factor ${patchgan_disc_factor} \
 --patchgan_disc_g_weight ${patchgan_disc_g_weight} \
 --use_wds True \

#  --inference True \

#  --dataset_size 1000 \

#  --trainer_name nv_rec_patch \
#  --cfg shapenet_tuneray_aug_resolution_64_64_nearestSR_patch \

#  --use_conf_map True

#  seed=0 fails to converge at the beginning

#  scripts/vit_triplane_train.py \

#  --rec_cvD_lambda 0.05 \
#  --nvs_cvD_lambda 0.2 \