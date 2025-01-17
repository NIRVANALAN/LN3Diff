set -x 
# vit_decoder_lr=0.001

# lpips_lambda=0.8
lpips_lambda=2.0 # ! lrm
# lpips_lambda=0
ssim_lambda=0.
l1_lambda=0. # following gaussian splatting
l2_lambda=1 # ! use_conf_map

# NUM_GPUS=4
NUM_GPUS=8
# NUM_GPUS=1

# image_size=64 # final rendered resolution
image_size=128 # final rendered resolution

image_size_encoder=224
patch_size=14
kl_lambda=1.0e-06

# batch_size=16 # 30.704 GiB on patch=32; 
# batch_size=12 # 30.704 GiB on patch=32; 
batch_size=8 # 
# batch_size=9 # 30.704 GiB on patch=32; 
# ! only 20GiB using new decoder/SR arch; using ViTB, 29.76GiB

# data_dir=/mnt/cache/yslan/get3d/chair_upper_train
# data_dir=/mnt/cache/yslan/get3d/chair_train_upper_new

data_dir=/mnt/cache/yslan/get3d/lmdb_debug/car/ # ! lmdb 

eval_data_dir=/mnt/lustre/yslan/3D_Dataset/get3d/car_test

# data_dir=/mnt/lustre/share/fzhong/shapenet/renders_train_new/car_02958343_200_r1.2_rgb_depth # new 
# eval_data_dir=/mnt/lustre/share/fzhong/shapenet/renders_new/car_02958343_200_r1.2_rgb_depth

lr=1e-5
# lr=2e-5
encoder_lr=$lr
vit_decoder_lr=$lr
conv_lr=0.0005
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
 --image_size_encoder $image_size_encoder \
 --data_dir $data_dir \
 --eval_data_dir $eval_data_dir \
 --dino_version v2 \
 --sr_training False \
 --cls_token False \
 --weight_decay 0.05 \
 --image_size $image_size \
 --kl_lambda ${kl_lambda} \
 --no_dim_up_mlp True \
 --uvit_skip_encoder True \
 --fg_mse True \
 --bg_lamdba 1.0 \
 --lpips_delay_iter 100 \
 --sr_delay_iter 25000 \
 --kl_anneal True \
 --symmetry_loss False \
 --vae_p 2 \
 --resume_checkpoint /mnt/lustre/yslan/logs/nips23/Reconstruction/final/car/vae/get3d/RodinSR_256_fusionv5_ConvQuant/upper_new/mv/vitb-kl1.0e-06-64+64-rec_patch_newarch_oldSRModel_fullimg_clipgrad-128-gpu4/model_rec0290000.pt \
 "
#  --resume_checkpoint /mnt/lustre/yslan/logs/nips23/Reconstruction/final/chair/vae/get3d/RodinSR_256_fusionv5_ConvQuant/upper_new/mv/vits-kl1.0e-06-64+64/model_rec0040000.pt \


# logdir=/mnt/lustre/yslan/logs/nips23/Reconstruction/final/chair/vae/get3d/RodinSR_256_fusionv5_ConvQuant/upper_new/mv/vits-kl${kl_lambda}-64+64
# logdir=/mnt/lustre/yslan/logs/nips23/Reconstruction/final/chair/vae/get3d/RodinSR_256_fusionv5_ConvQuant/upper_new/mv/vits-kl${kl_lambda}-64+64-rec_cano
# logdir=/mnt/lustre/yslan/logs/nips23/Reconstruction/final/chair/vae/get3d/RodinSR_256_fusionv5_ConvQuant/upper_new/mv/vits-kl${kl_lambda}-64+64-rec_patch_lr2e-5_scratch
# logdir=/mnt/lustre/yslan/logs/nips23/Reconstruction/final/car/vae/get3d/RodinSR_256_fusionv5_ConvQuant/upper_new/mv/vitb-kl${kl_lambda}-64+64-rec_patch_newarch_oldSRModel
# logdir=/mnt/lustre/yslan/logs/nips23/Reconstruction/final/car/vae/get3d/RodinSR_256_fusionv5_ConvQuant/upper_new/mv/vitb-kl${kl_lambda}-64+64-rec_patch_newarch_oldSRModel_clipgrad-${image_size}-gpu${NUM_GPUS}-singleforward-patch45
logdir=/mnt/lustre/yslan/logs/nips23/Reconstruction/final/car/vae/get3d/RodinSR_256_fusionv5_ConvQuant/upper_new/mv/vitb-kl${kl_lambda}-64+64-rec_patch_newarch_oldSRModel_clipgrad-${image_size}-gpu${NUM_GPUS}-singleforward-patch45_debug

SR_TRAIN_FLAGS_v1_2XC="
--decoder_in_chans 32 \
--out_chans 96 \
--alpha_lambda 1 \
--logdir $logdir \
--arch_encoder vits \
--arch_decoder vitb \
--vit_decoder_wd 0.001 \
--encoder_weight_decay 0.001 \
--color_criterion mse \
--decoder_output_dim 32 \
--ae_classname vit.vit_triplane.RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn \
"

SR_TRAIN_FLAGS=${SR_TRAIN_FLAGS_v1_2XC}

# NUM_GPUS=1

rm -rf "$logdir"/runs
mkdir -p "$logdir"/
cp "$0" "$logdir"/

export OMP_NUM_THREADS=12

torchrun --nproc_per_node=$NUM_GPUS \
  --master_port=0 \
  --rdzv_backend=c10d \
  --rdzv-endpoint=localhost:14358 \
  --nnodes 1 \
 scripts/vit_triplane_train.py \
 --trainer_name nv_rec_patch \
 --num_workers 6 \
 --depth_lambda 0 \
 --data_dir $data_dir \
 ${TRAIN_FLAGS}  \
 ${SR_TRAIN_FLAGS} \
 --lpips_lambda $lpips_lambda \
 --overfitting False \
 --load_pretrain_encoder True \
 --iterations 5000001 \
 --save_interval 10000 \
 --eval_interval 2500 \
 --decomposed True \
 --logdir $logdir \
 --cfg shapenet_tuneray_aug_resolution_64_64_nearestSR_patch \
 --ray_start 0.6 \
 --ray_end 1.8 \
 --patch_size ${patch_size} \
 --use_amp False \
 --eval_batch_size 6 \
 ${LR_FLAGS} \
 --l1_lambda ${l1_lambda} \
 --l2_lambda ${l2_lambda} \
 --ssim_lambda ${ssim_lambda} \
 --depth_smoothness_lambda 1e-1 \
 --use_conf_map False \
 --use_lmdb True

#  --trainer_name nv_rec_patch \
#  --cfg shapenet_tuneray_aug_resolution_64_64_nearestSR_patch \

#  --use_conf_map True

#  seed=0 fails to converge at the beginning

#  scripts/vit_triplane_train.py \

#  --rec_cvD_lambda 0.05 \
#  --nvs_cvD_lambda 0.2 \