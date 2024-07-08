set -x 

lpips_lambda=0.8

image_size=192
image_size_encoder=256

patch_size=14

batch_size=1
microbatch=${batch_size}

# num_samples=$((50/${batch_size})) # follow ssdnerf and functa
num_samples=4 # how many samples per prompt
cfg_dropout_prob=0.1 # SD config
num_workers=0

# data_dir='./assets/stage1_vae_reconstruction/Objaverse'
eval_path=/cpfs01/user/lanyushi.p/Repo/eccv24/open-source/InstantMesh/examples
# eval_path='logs/examples'

DATASET_FLAGS="
 --data_dir "NONE" \
 --eval_data_dir ${eval_path} \
"

lr=2e-5 # for official DiT, lr=1e-4 for BS=256
kl_lambda=0
vit_lr=1e-5 # for improved-diffusion unet
ce_lambda=0.5 # ?
conv_lr=5e-5
alpha_lambda=1
scale_clip_encoding=1
triplane_scaling_divider=0.96806

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
 --dino_version mv-sd-dit-dynaInp-trilatent \
 --sr_training False \
 --encoder_cls_token False \
 --decoder_cls_token False \
 --cls_token False \
 --weight_decay 0.05 \
 --no_dim_up_mlp True \
 --uvit_skip_encoder True \
 --decoder_load_pretrained False \
 --fg_mse False \
 --vae_p 2 \
 --plucker_embedding True \
 --encoder_in_channels 10 \
 --arch_dit_decoder DiT2-L/2 \
 --sd_E_ch 64 \
 --sd_E_num_res_blocks 1 \
 --lrm_decoder False \
 --resume_checkpoint /nas/shared/V2V/yslan/logs/nips24/LSGM/t23d/FM/9cls/i23d/dit-L2-pixart-lognorm-rmsnorm-layernorm_before_pooled/gpu7-batch40-lr1e-4-bf16-qknorm-ctd3/model_joint_denoise_rec_model2990000.pt \
 "



DDPM_MODEL_FLAGS="
--learn_sigma False \
--num_heads 8 \
--num_res_blocks 2 \
--num_channels 320 \
--attention_resolutions "4,2,1" \
--use_spatial_transformer True \
--transformer_depth 1 \
--context_dim 1024 \
"

# ! diffusion steps and noise schedule not used, since the continuous diffusion is adopted.
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear \
--use_kl False \
--use_amp False \
--triplane_scaling_divider ${triplane_scaling_divider} \
--trainer_name flow_matching \
--mixed_prediction False \
--train_vae False \
--denoise_in_channels 4 \
--denoise_out_channels 4 \
--diffusion_input_size 32 \
--diffusion_ce_anneal True \
--create_controlnet False \
--p_rendering_loss False \
--pred_type v \
--predict_v True \
--create_dit True \
--dit_model_arch DiT-PixArt-L/2 \
--train_vae False \
--use_eos_feature False \
--roll_out True \
"



logdir=./logs/LSGM/inference/Objaverse/i23d/dit-L2/

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
"
# --resume_checkpoint /mnt/lustre/yslan/logs/nips23/LSGM/ssd/chair/scaling/entropy/kl0_ema0.9999_vpsde_TrainLoop3DDiffusionLSGM_cvD_scaling_lsgm_unfreezeD_weightingv0_lsgm_unfreezeD_0.01_gradclip_nocesquare_clipH@0_noallAMP_dataset500/model_joint_denoise_rec_model0910000.pt \


unconditional_guidance_scale=4.0

# for unconditional_guidance_scale in 1.0 2.0 3.0 4.0 5.0 6.0 6.5 7

# do

DDIM_FLAGS="
--unconditional_guidance_scale ${unconditional_guidance_scale} \
"


SR_TRAIN_FLAGS=${SR_TRAIN_FLAGS_v1_2XC}

NUM_GPUS=1

rm -rf "$logdir"/runs
mkdir -p "$logdir"/
cp "$0" "$logdir"/

export OMP_NUM_THREADS=12
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_IB_GID_INDEX=3 # https://github.com/huggingface/accelerate/issues/314#issuecomment-1821973930
export OPENCV_IO_ENABLE_OPENEXR=1
export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=$NUM_GPUS \
  --nnodes 1 \
  --rdzv-endpoint=localhost:24379 \
 scripts/vit_triplane_diffusion_sample_objaverse.py \
 --num_workers ${num_workers} \
 --depth_lambda 0 \
 ${TRAIN_FLAGS}  \
 ${SR_TRAIN_FLAGS} \
 ${DATASET_FLAGS} \
 ${DIFFUSION_FLAGS} \
 ${DDPM_MODEL_FLAGS} \
 ${DDIM_FLAGS} \
 --overfitting False \
 --load_pretrain_encoder False \
 --iterations 5000001 \
 --save_interval 10000 \
 --eval_interval 5000 \
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
 --objv_dataset True \
 --cfg_dropout_prob ${cfg_dropout_prob} \
 --cond_key img \
 --enable_mixing_normal False \
 --use_lmdb_compressed False \
 --use_lmdb False \
 --use_amp True \
 --allow_tf32 True \
 --load_wds_diff True \
 --mv_input True \
 --compile False \
 --num_frames 6 \
 --num_samples ${num_samples} \
 --i23d True \
 --shuffle_across_cls True \
 --snr-type lognorm \
 --use_eos_feature False \
 --load_real True \
 --use_wds False \

#  done