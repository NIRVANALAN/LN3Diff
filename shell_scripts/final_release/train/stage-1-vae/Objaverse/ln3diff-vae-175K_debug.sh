set -x 
lpips_lambda=0.8
ssim_lambda=0.
l1_lambda=1. # following gaussian splatting
l2_lambda=0 # ! use_conf_map


patchgan_disc_factor=1.0 # d loss multiplier
patchgan_disc_g_weight=0.08

# NUM_GPUS=8
NUM_GPUS=1

image_size=256 # final rendered resolution

num_workers=0 # shall be larger

image_size_encoder=256
patch_size=14
kl_lambda=1.0e-06
patch_rendering_resolution=48 # just for debugging

batch_size=1
microbatch=12

shards_lst="shell_scripts/shards_list/shards_lst_170K_shuffle.txt"

# data_dir=/mnt/sfs-common/yslan/Dataset/Obajverse/chunk-png-normal/bs_12_shuffle/170K/256
data_dir=./assets/stage1_vae_reconstruction


DATASET_FLAGS="
 --data_dir ${data_dir} \
 --eval_data_dir ${data_dir} \
"

# conv_lr=2e-4
# lr=1e-4 # 4e-4 for BS=256, we have 80 here.

# ! somewhat slow convergence?
# conv_lr=1e-4
# lr=5e-5 # 4e-4 for BS=256, we have 80 here.

conv_lr=1.5e-4
lr=7.5e-5 # 4e-4 for BS=256, we have 80 here.

# slightly increase the LR? try later. 
# conv_lr=1.5e-4
# lr=7.5e-5 # 4e-4 for BS=256, we have 80 here.

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
 --dino_version mv-sd-dit-dynaInp-trilatent \
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
 --arch_dit_decoder DiT2-L/2 \
 --sd_E_ch 64 \
 --sd_E_num_res_blocks 1 \
 --lrm_decoder False \
 "

# logdir=/nas/shared/V2V/yslan/logs/nips23/Reconstruction/final/objav/vae/MV/170K/add-depth/frame0/adv/bs${batch_size}-gpu${NUM_GPUS}-img${image_size}-patch${patch_rendering_resolution}-disc-${patchgan_disc_factor}-${patchgan_disc_g_weight}-v=6-fixAttn-fixalpha-${image_size}
logdir='./logs/stage-1/vae_train_demo'

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
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_GID_INDEX=3 # https://github.com/huggingface/accelerate/issues/314#issuecomment-1821973930
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 8gpu
export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=$NUM_GPUS \
  --nnodes=1 \
  --rdzv-endpoint=localhost:15371 \
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
 --depth_lambda 0.1 \
 --patch_rendering_resolution ${patch_rendering_resolution} \
 --use_lmdb_compressed False \
 --use_lmdb False \
 --mv_input True \
 --split_chunk_input True \
 --append_depth True \
 --split_chunk_size 12 \
 --patchgan_disc_factor ${patchgan_disc_factor} \
 --patchgan_disc_g_weight ${patchgan_disc_g_weight} \
 --use_chunk True \
 --use_wds False \
 --num_frames 6 \

#  --frame_0_as_canonical False \