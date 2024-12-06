set -x 

lpips_lambda=0.8

image_size=64
image_size_encoder=224

patch_size=14

# batch_size=32 # 4*32
batch_size=12 # 4*16=64
microbatch=${batch_size}
# batch_size=1 # for debug

cfg_dropout_prob=0.15

# dataset_size=10000
dataset_name=plane

eval_data_dir=/mnt/lustre/yslan/3D_Dataset/get3d/${dataset_name}_test
DATASET_FLAGS="
 --data_dir /mnt/cache/yslan/get3d/lmdb_debug/${dataset_name} \
"
#  --dataset_size ${dataset_size} \
# resume_checkpoint_EG3D=/mnt/lustre/yslan/Repo/Research/SIGA22/BaseModels/eg3d/checkpoints/ffhq.ckpt

lr=2e-5 # for improved-diffusion unet
kl_lambda=0
vit_lr=1e-5 # for improved-diffusion unet
ce_lambda=0.5 # ?
conv_lr=5e-5
alpha_lambda=1
scale_clip_encoding=18.4

triplane_scaling_divider=1

# * above the best lr config

LR_FLAGS="--encoder_lr $vit_lr \
 --vit_decoder_lr $vit_lr \
 --lpips_lambda $lpips_lambda \
 --triplane_decoder_lr $conv_lr \
 --super_resolution_lr $conv_lr \
 --lr $lr \
 --kl_lambda ${kl_lambda} \
 --bg_lamdba 0.01 \
 --alpha_lambda ${alpha_lambda} \
"

TRAIN_FLAGS="--iterations 10001 --anneal_lr False \
 --batch_size $batch_size --save_interval 10000 \
 --microbatch ${microbatch} \
 --image_size_encoder $image_size_encoder \
 --image_size $image_size \
 --dino_version v2 \
 --sr_training False \
 --encoder_cls_token False \
 --decoder_cls_token False \
 --cls_token False \
 --weight_decay 0.05 \
 --resume_checkpoint /mnt/lustre/yslan/logs/nips23/Reconstruction/final/plane/vae/get3d/RodinSR_256_fusionv5_ConvQuant/upper_new/mv/vitb-kl1.0e-06-64+64-rec_patch_newarch_oldSRModel-128-ctd2-nocompile/model_rec0860000.pt \
 --no_dim_up_mlp True \
 --uvit_skip_encoder True \
 --decoder_load_pretrained True \
 --fg_mse False \
 --vae_p 2 \
 --lpips_delay_iter 100 \
 --sr_delay_iter 25000 \
 --kl_anneal True \
 "

DDPM_MODEL_FLAGS="
--learn_sigma False \
--num_heads 8 \
--num_res_blocks 2 \
--num_channels 320 \
--attention_resolutions "4,2,1" \
--use_spatial_transformer True \
--transformer_depth 1 \
--context_dim 768 \
"
# --pred_type x0 \
# --iw_sample_p drop_all_uniform \
# --loss_type x0 \

# ! diffusion steps and noise schedule not used, since the continuous diffusion is adopted.
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear \
--use_kl False \
--use_amp False \
--triplane_scaling_divider ${triplane_scaling_divider} \
--trainer_name vpsde_crossattn \
--mixed_prediction True \
--train_vae False \
--denoise_in_channels 12 \
--denoise_out_channels 12 \
--diffusion_input_size 32 \
--diffusion_ce_anneal True \
--create_controlnet False \
--p_rendering_loss False \
--pred_type v \
--predict_v True \
"
# --trainer_name vpsde_TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_iterativeED \

# --predict_xstart True \

# logdir=/mnt/lustre/yslan/logs/nips23/LSGM/cldm/${dataset_name}/crossattn/v1-lmdb
# logdir=/mnt/lustre/yslan/logs/nips23/LSGM/cldm/${dataset_name}/crossattn/v3-cfgDrop@0.1
# logdir=/mnt/lustre/yslan/logs/nips23/LSGM/cldm/${dataset_name}/crossattn/v3-cfgDrop0-nonorm-normalize
# logdir=/mnt/lustre/yslan/logs/nips23/LSGM/cldm/${dataset_name}/crossattn/v3-cfgDrop0-normalize-scaling${scale_clip_encoding}
logdir=/mnt/lustre/yslan/logs/nips23/LSGM/cldm/${dataset_name}/crossattn/v3-cfgDrop${cfg_dropout_prob}-normalize-scaling${scale_clip_encoding}

SR_TRAIN_FLAGS_v1_2XC="
--decoder_in_chans 32 \
--out_chans 96 \
--ae_classname vit.vit_triplane.RodinSR_256_fusionv5_ConvQuant_liteSR_dinoInit3DAttn \
--logdir $logdir \
--arch_encoder vits \
--arch_decoder vitb \
--vit_decoder_wd 0.001 \
--encoder_weight_decay 0.001 \
--color_criterion mse \
--triplane_in_chans 32 \
--decoder_output_dim 32 \
"
# --resume_checkpoint /mnt/lustre/yslan/logs/nips23/LSGM/ssd/chair/scaling/entropy/kl0_ema0.9999_vpsde_TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_weightingv0_lsgm_unfreezeD_0.01_gradclip_nocesquare_clipH@0_noallAMP_dataset500/model_joint_denoise_rec_model0910000.pt \



SR_TRAIN_FLAGS=${SR_TRAIN_FLAGS_v1_2XC}

# NUM_GPUS=4
NUM_GPUS=8
# NUM_GPUS=1

rm -rf "$logdir"/runs
mkdir -p "$logdir"/
cp "$0" "$logdir"/

export OMP_NUM_THREADS=12
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_GID_INDEX=3 # https://github.com/huggingface/accelerate/issues/314#issuecomment-1821973930

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=7
# export CUDA_VISIBLE_DEVICES=3,7
# export CUDA_VISIBLE_DEVICES=4,5
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=6,7
# export CUDA_VISIBLE_DEVICES=7

torchrun --nproc_per_node=$NUM_GPUS \
  --nnodes 1 \
  --rdzv-endpoint=localhost:23355 \
 scripts/vit_triplane_diffusion_train.py \
 --num_workers 4 \
 --eval_data_dir $eval_data_dir \
 --depth_lambda 0 \
 ${TRAIN_FLAGS}  \
 ${SR_TRAIN_FLAGS} \
 ${DATASET_FLAGS} \
 ${DIFFUSION_FLAGS} \
 ${DDPM_MODEL_FLAGS} \
 --overfitting False \
 --load_pretrain_encoder True \
 --iterations 5000001 \
 --save_interval 10000 \
 --eval_interval 5000 \
 --decomposed True \
 --logdir $logdir \
 --cfg shapenet_tuneray_aug_resolution_64_64_nearestSR \
 --patch_size ${patch_size} \
 --eval_batch_size 1 \
 ${LR_FLAGS} \
 --ce_lambda ${ce_lambda} \
 --negative_entropy_lambda ${ce_lambda} \
 --triplane_fg_bg False \
 --grad_clip True \
 --interval 5 \
 --normalize_clip_encoding True \
 --scale_clip_encoding ${scale_clip_encoding} \
 --mixing_logit_init -3 \
 --cfg_dropout_prob ${cfg_dropout_prob} \
 --use_lmdb True \