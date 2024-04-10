# mv_latent_noMixing_75K_sgm_legacy

set -x 

lpips_lambda=0.8

image_size=128 # final rendered resolution
image_size_encoder=256

patch_size=14

# batch_size=32 # 4*32
# ! 29GB -> 37GB
# batch_size=8 # 128 when 3?
# batch_size=1 # for debug

# batch_size=48
# batch_size=96 # BS=480 on 5GPU

# batch_size=36 # BS=256 is enough
# batch_size=16 # BS=256 is enough

# batch_size=80 # BS=480 on 5GPU
# batch_size=18 # 126 in total
# microbatch=72

# batch_size=48 # 
# batch_size=40 # 
batch_size=36 # 8GPU here
# batch_size=85 # 
# batch_size=96 # 
# batch_size=24 # 128 in total
# batch_size=36 # 128 in total
# batch_size=40 # 128 in total
# batch_size=96 # 128 in total
# batch_size=64 # 128 in total
# batch_size=80 # 128 in total
microbatch=${batch_size}

cfg_dropout_prob=0.1 # SD config

# dataset_size=10000
# dataset_name=Ani-Trans-Furni
dataset_name="75K"
# num_workers=12
# num_workers=7
# num_workers=12
num_workers=0


# NUM_GPUS=4
# NUM_GPUS=7
# NUM_GPUS=3
# NUM_GPUS=2
NUM_GPUS=8
# NUM_GPUS=7

# shards_lst=/cpfs01/user/lanyushi.p/Repo/diffusion-3d/shell_scripts/baselines/reconstruction/sr/final_mv/diff_shards_lst_3w.txt
# shards_lst="/cpfs01/user/lanyushi.p/Repo/diffusion-3d/shell_scripts/shards_list/diff_mv_latent_132w_3cls.txt"
# eval_shards_lst=/cpfs01/user/lanyushi.p/Repo/diffusion-3d/shell_scripts/baselines/reconstruction/sr/final_mv/shards_lst.txt
# DATASET_FLAGS="
#  --data_dir "NONE" \
#  --shards_lst ${shards_lst} \
#  --eval_data_dir "NONE" \
#  --eval_shards_lst ${eval_shards_lst}  \
# "

# shards_lst=/cpfs01/user/lanyushi.p/Repo/diffusion-3d/shell_scripts/baselines/reconstruction/sr/final_mv/diff_shards_lst_3w.txt
# shards_lst="shell_scripts/baselines/reconstruction/sr/final_mv/diff_shards_lst_3w_shuffle.txt"
shards_lst="shell_scripts/shards_list/diff_singleview_shards_75K.txt"
eval_shards_lst=/cpfs01/user/lanyushi.p/Repo/diffusion-3d/shell_scripts/baselines/reconstruction/sr/final_mv/shards_lst.txt
DATASET_FLAGS="
 --data_dir "NONE" \
 --shards_lst ${shards_lst} \
 --eval_data_dir "NONE" \
 --eval_shards_lst ${eval_shards_lst}  \
"





# eval_data_dir=/cpfs01/shared/V2V/V2V_hdd/yslan/aigc3d/download_unzipped/Furnitures # to update later
# eval_data_dir=${data_dir}

#  --dataset_size ${dataset_size} \

# lr=2e-5 # for official DiT, lr=1e-4 for BS=256
# lr=4e-5 # for official DiT, lr=1e-4 for BS=256
lr=1e-4 # for LDM base learning rate
# lr=3e-5 # for official DiT, lr=1e-4 for BS=256
kl_lambda=0
vit_lr=1e-5 # for improved-diffusion unet
ce_lambda=0.5 # ?
conv_lr=5e-5
alpha_lambda=1
scale_clip_encoding=1

# triplane_scaling_divider=0.8918
# triplane_scaling_divider=0.857916
# triplane_scaling_divider=0.883637
# triplane_scaling_divider=0.89247337
# triplane_scaling_divider=0.82
# triplane_scaling_divider=0.88
# triplane_scaling_divider=0.89
triplane_scaling_divider=0.90

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
 --dino_version mv-sd-dit \
 --sr_training False \
 --encoder_cls_token False \
 --decoder_cls_token False \
 --cls_token False \
 --weight_decay 0.05 \
 --no_dim_up_mlp True \
 --uvit_skip_encoder True \
 --decoder_load_pretrained True \
 --fg_mse False \
 --vae_p 2 \
 --plucker_embedding True \
 --encoder_in_channels 10 \
 --arch_dit_decoder DiT2-B/2 \
 --sd_E_ch 64 \
 --sd_E_num_res_blocks 1 \
 --lrm_decoder False \
 "

#  --resume_checkpoint /cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/Ani-Trans-Furni-rollout-12-lr3e-5-divide0.82/model_joint_denoise_rec_model0960000.pt \

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
--trainer_name sgm_legacy \
--mixed_prediction False \
--train_vae False \
--denoise_in_channels 4 \
--denoise_out_channels 4 \
--diffusion_input_size 32 \
--diffusion_ce_anneal True \
--create_controlnet False \
--p_rendering_loss False \
--pred_type x_start \
--predict_v False \
--create_dit False \
--train_vae False \
--use_eos_feature False \
--roll_out True \
"

# --dit_model_arch DiT-L/2 \

# --trainer_name vpsde_TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_iterativeED \

# --predict_xstart True \

# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-dit/${dataset_name}/cond_abla-rollout
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/${dataset_name}-rollout-${batch_size}-lr${lr}/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/${dataset_name}-rollout-${batch_size}-lr${lr}-ctd-smallBS/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/${dataset_name}-rollout-${batch_size}-lr${lr}-ctd-smallBS/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/${dataset_name}-rollout-${batch_size}-lr${lr}-ctd-smallBS-divide${triplane_scaling_divider}/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/${dataset_name}-rollout-${batch_size}-lr${lr}-ctd-smallBS-divide${triplane_scaling_divider}-mv/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/${dataset_name}-rollout-${batch_size}-lr${lr}-divide${triplane_scaling_divider}/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/${dataset_name}-rollout-${batch_size}-lr${lr}-divide${triplane_scaling_divider}-ctd/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/${dataset_name}-rollout-${batch_size}-lr${lr}-divide${triplane_scaling_divider}-ctd/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/t23d/3cls-138wLatent-gpu${NUM_GPUS}-batch${batch_size}/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/t23d/75K-168wLatent-gpu${NUM_GPUS}-batch${batch_size}/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/t23d/75K-168wLatent-gpu${NUM_GPUS}-batch${batch_size}-load3clsPT/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/t23d/75K-168wLatent-gpu${NUM_GPUS}-batch${batch_size}-noPT/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/t23d/75K-168wLatent-gpu${NUM_GPUS}-batch${batch_size}-noPT-lr${lr}/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/t23d/75K-168wLatent-gpu${NUM_GPUS}-batch${batch_size}-noPT-lr${lr}-newview/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/t23d/75K-168wLatent-gpu${NUM_GPUS}-batch${batch_size}-noPT-lr${lr}-newview-fixingbug/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/t23d/75K-168wLatent-gpu${NUM_GPUS}-batch${batch_size}-noPT-lr${lr}-newview-removeImgCond-clipgrad0.5/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/t23d/75K-168wLatent-gpu${NUM_GPUS}-batch${batch_size}-noPT-lr${lr}-newview-sgm_legacy/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/t23d/75K-168wLatent-gpu${NUM_GPUS}-batch${batch_size}-noPT-lr${lr}-newview-sgm_legacy-newview-addTop-clipgrad0.4/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/t23d/75K-168wLatent-gpu${NUM_GPUS}-batch${batch_size}-noPT-lr${lr}-newview-sgm_legacy-newview-addTop-clipgrad0.4-debug/
# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/t23d/75K-168wLatent-gpu${NUM_GPUS}-batch${batch_size}-noPT-lr${lr}-newview-sgm_legacy-newview-addTop-clipgrad0.4-ctd/
logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/t23d/75K-168wLatent-gpu${NUM_GPUS}-batch${batch_size}-noPT-lr${lr}-newview-sgm_legacy-newview-addTop-clipgrad0.4-ctd-largeLR/
# crossattn/TextEmbed/cfgDrop${cfg_dropout_prob}-gpu${NUM_GPUS}-batch${batch_size}

# logdir=/cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/Reconstruction/final/objav/vae/Furnitures/SD/1000/SD-Encoder-F=8-D128/${batch_size}-gpu${NUM_GPUS}-patch45-32to128-heavy-final-noUpsample-wds-lr${encoder_lr}-lpips2-128-k=4ctd/

SR_TRAIN_FLAGS_v1_2XC="
--decoder_in_chans 32 \
--out_chans 96 \
--ae_classname vit.vit_triplane.RodinSR_256_fusionv6_ConvQuant_liteSR_dinoInit3DAttn_SD_B_3L_C_withrollout_withSD_D_ditDecoder \
--logdir $logdir \
--arch_encoder vits \
--arch_decoder vitb \
--vit_decoder_wd 0.001 \
--encoder_weight_decay 0.001 \
--color_criterion mse \
--triplane_in_chans 32 \
--decoder_output_dim 3 \
--resume_checkpoint /cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/t23d/75K-168wLatent-gpu8-batch36-noPT-lr3e-5-newview-sgm_legacy-newview-addTop-clipgrad0.4-ctd/model_joint_denoise_rec_model2070000.pt \
"
# --resume_checkpoint /cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/t23d/75K-168wLatent-gpu7-batch48-noPT-lr1e-4/model_joint_denoise_rec_model1770000.pt \

# --resume_checkpoint /cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/t23d/3cls-138wLatent-gpu-batch96/model_joint_denoise_rec_model1780000.pt \

# --resume_checkpoint /cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/LSGM/cldm-unet/mv/t23d/75K-168wLatent-gpu-batch64/model_joint_denoise_rec_model1690000.pt \



SR_TRAIN_FLAGS=${SR_TRAIN_FLAGS_v1_2XC}


rm -rf "$logdir"/runs
mkdir -p "$logdir"/
cp "$0" "$logdir"/

export OMP_NUM_THREADS=12
export LC_ALL=en_US.UTF-8 # save caption txt bug
export NCCL_ASYNC_ERROR_HANDLING=1
export OPENCV_IO_ENABLE_OPENEXR=1
export NCCL_IB_GID_INDEX=3 # https://github.com/huggingface/accelerate/issues/314#issuecomment-1821973930
# export CUDA_VISIBLE_DEVICES=0,1,2

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# export CUDA_VISIBLE_DEVICES=7
# export CUDA_VISIBLE_DEVICES=3,7
# export CUDA_VISIBLE_DEVICES=3,4,5
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=1,2
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,3
# export CUDA_VISIBLE_DEVICES=0,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=4,5,6

torchrun --nproc_per_node=$NUM_GPUS \
  --nnodes 1 \
  --rdzv-endpoint=localhost:23371 \
 scripts/vit_triplane_diffusion_train.py \
 --num_workers ${num_workers} \
 --depth_lambda 0 \
 ${TRAIN_FLAGS}  \
 ${SR_TRAIN_FLAGS} \
 ${DATASET_FLAGS} \
 ${DIFFUSION_FLAGS} \
 ${DDPM_MODEL_FLAGS} \
 --overfitting False \
 --load_pretrain_encoder False \
 --iterations 5000001 \
 --save_interval 10000 \
 --eval_interval 5000000 \
 --decomposed True \
 --logdir $logdir \
 --cfg objverse_tuneray_aug_resolution_64_64_auto \
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
 --mixing_logit_init 10000 \
 --objv_dataset True \
 --cfg_dropout_prob ${cfg_dropout_prob} \
 --cond_key caption \
 --use_lmdb_compressed False \
 --use_lmdb False \
 --load_wds_diff True \
 --load_wds_latent False \
 --compile False \
 --split_chunk_input True \
 --append_depth True \
 --mv_input True \
 --duplicate_sample False \
 --enable_mixing_normal False \
 --use_wds True \
 --clip_grad_throld 0.4 \
 --mv_latent_dir /cpfs01/user/lanyushi.p/data/latent_dir/168w-3class-withImg-newview-addTop/latent_dir \
#  --mv_latent_dir /cpfs01/shared/V2V/V2V_hdd/yslan/logs/nips23/Reconstruction/final/objav/vae/MV/75K/infer-latents/168w-3class-withImg/latent_dir \

 