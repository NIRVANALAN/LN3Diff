import torch
import torchvision
from torchvision import transforms
import numpy as np

import os
from omegaconf import OmegaConf
from PIL import Image 

import gradio as gr

import rembg

from huggingface_hub import hf_hub_download


"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import json
import sys
import os

sys.path.append('.')

from pdb import set_trace as st
import imageio
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    continuous_diffusion_defaults,
    control_net_defaults,
)

th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.allow_tf32 = True
th.backends.cudnn.enabled = True

from pathlib import Path

from tqdm import tqdm, trange
import dnnlib
from nsr.train_util_diffusion import TrainLoop3DDiffusion as TrainLoop
from guided_diffusion.continuous_diffusion import make_diffusion as make_sde_diffusion
import nsr
import nsr.lsgm
from nsr.script_util import create_3DAE_model, encoder_and_nsr_defaults, loss_defaults, AE_with_Diffusion, rendering_options_defaults, eg3d_options_default, dataset_defaults

from datasets.shapenet import load_eval_data
from torch.utils.data import Subset
from datasets.eg3d_dataset import init_dataset_kwargs

from transport.train_utils import parse_transport_args

from utils.infer_utils import remove_background, resize_foreground

SEED = 0

def resize_to_224(img):
    img = transforms.functional.resize(img, 224,
        interpolation=transforms.InterpolationMode.LANCZOS)
    return img


def set_white_background(image):
    image = np.array(image).astype(np.float32) / 255.0
    mask = image[:, :, 3:4]
    image = image[:, :, :3] * mask + (1 - mask)
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    return image


def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")



def main(args):

    # args.rendering_kwargs = rendering_options_defaults(args)

    dist_util.setup_dist(args)
    logger.configure(dir=args.logdir)

    th.cuda.empty_cache()

    th.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    # * set denoise model args
    logger.log("creating model and diffusion...")
    args.img_size = [args.image_size_encoder]
    # ! no longer required for LDM
    # args.denoise_in_channels = args.out_chans
    # args.denoise_out_channels = args.out_chans
    args.image_size = args.image_size_encoder  # 224, follow the triplane size

    denoise_model, diffusion = create_model_and_diffusion(
        **args_to_dict(args,
                       model_and_diffusion_defaults().keys()))

    # if 'cldm' in args.trainer_name:
    #     assert isinstance(denoise_model, tuple)
    #     denoise_model, controlNet = denoise_model

    #     controlNet.to(dist_util.dev())
    #     controlNet.train()
    # else:
        # controlNet = None

    opts = eg3d_options_default()
    if args.sr_training:
        args.sr_kwargs = dnnlib.EasyDict(
            channel_base=opts.cbase,
            channel_max=opts.cmax,
            fused_modconv_default='inference_only',
            use_noise=True
        )  # ! close noise injection? since noise_mode='none' in eg3d

    # denoise_model.load_state_dict(
    #     dist_util.load_state_dict(args.ddpm_model_path, map_location="cpu"))
    denoise_model.to(dist_util.dev())
    if args.use_fp16:
        denoise_model.convert_to_fp16()
    denoise_model.eval()

    # * auto-encoder reconstruction model
    logger.log("creating 3DAE...")
    auto_encoder = create_3DAE_model(
        **args_to_dict(args,
                       encoder_and_nsr_defaults().keys()))

    auto_encoder.to(dist_util.dev())
    auto_encoder.eval()

    # TODO, how to set the scale?
    logger.log("create dataset")

    if args.objv_dataset:
        from datasets.g_buffer_objaverse import load_data, load_eval_data, load_memory_data, load_wds_data
    else:  # shapenet
        from datasets.shapenet import load_data, load_eval_data, load_memory_data
    
    # load data if i23d
    if args.i23d:
        data = load_eval_data(
            file_path=args.eval_data_dir,
            batch_size=args.eval_batch_size,
            reso=args.image_size,
            reso_encoder=args.image_size_encoder,  # 224 -> 128
            num_workers=args.num_workers,
            load_depth=True,  # for evaluation
            preprocess=auto_encoder.preprocess,
            **args_to_dict(args,
                            dataset_defaults().keys()))
    else:
        data = None # t23d sampling, only caption required


    TrainLoop = {
        'sgm_legacy':
        nsr.lsgm.sgm_DiffusionEngine.DiffusionEngineLSGM,
        'flow_matching':
        nsr.lsgm.flow_matching_trainer.FlowMatchingEngine,
    }[args.trainer_name]

    # continuous
    sde_diffusion = None

    auto_encoder.decoder.rendering_kwargs = args.rendering_kwargs

    training_loop_class = TrainLoop(rec_model=auto_encoder,
                                    denoise_model=denoise_model,
                                    control_model=None, # to remove
                                    diffusion=diffusion,
                                    sde_diffusion=sde_diffusion,
                                    loss_class=None,
                                    data=data,
                                    eval_data=None,
                                    **vars(args))

    # logger.log("sampling...")
    # dist_util.synchronize()



    # all_prompts_available = [
    #     # prompts used in the paper:
    #     'The Eiffel tower.',  # 0-3
    #     'a stone water well with a wooden shed.', # 7 9 15 19 23 24 28
    #     'A wooden chest with golden trim', # 3 5 6 7
    #     'A plate of sushi.',  # 0 3 7 11 16
    #     'A blue platic chair', # 0 1 2 3
    # ]

    # prompts_and_seed_to_render = {
    #     'The Eiffel tower.': np.array([20, 19, 18, 31, 26, 25, 22]),  # 0-3
    #     'a stone water well with a wooden shed.': np.array([7, 9, 15, 19, 23, 24, 28]),
    #     'A wooden chest with golden trim': np.array([3, 5, 6, 7]), 
    #     'A plate of sushi.': np.array([0, 3, 7, 11, 16]), 
    #     'A blue platic chair': np.array([0, 1, 2, 3]),
    # }

    # for prompt, seeds in prompts_and_seed_to_render.items():

    css = """
    h1 {
        text-align: center;
        display:block;
    }
    """


    def preprocess(input_image, preprocess_background=True, foreground_ratio=0.85):
        if preprocess_background:
            rembg_session = rembg.new_session()
            image = input_image.convert("RGB")
            image = remove_background(image, rembg_session)
            image = resize_foreground(image, foreground_ratio)
            image = set_white_background(image)
        else:
            image = input_image
            if image.mode == "RGBA":
                image = set_white_background(image)
        image = resize_to_224(image)
        return image


    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # LN3Diff (Scalable Latent Neural Fields Diffusion for Speedy 3D Generation)

            **LN3Diff (ECCV 2024)** [[code](https://github.com/NIRVANALAN/LN3Diff), [project page](https://nirvanalan.github.io/projects/ln3diff/)] is a scalable 3D latent diffusion model that supports speedy 3D assets generation. 
            It first trains a 3D VAE on **Objaverse**, which compress each 3D asset into a compact 3D-aware latent. After that, a image/text-conditioned diffusion model is trained following LDM paradigm.
            The model used in the demo adopts DiT-L/2 architecture and flow-matching framework, and supports single-image condition.
            It is trained on 8 A100 GPUs for 1M iterations with batch size 256.
            Locally, on an NVIDIA A100/A10 GPU, each image-conditioned diffusion generation can be done in 10~20 seconds (time varies due to the adaptive-step ODE solver used in flow-mathcing.)
            Upload an image of an object or click on one of the provided examples to see how the LN3Diff works.
            The 3D viewer will render a .obj object exported from the triplane, where the mesh resolution and iso-surface can be set manually.
            For best results run the demo locally and render locally - to do so, clone the [main repository](https://github.com/NIRVANALAN/LN3Diff).
            """
            )
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(
                        label="Input Image",
                        image_mode="RGBA",
                        sources="upload",
                        type="pil",
                        elem_id="content_image",
                    )
                    processed_image = gr.Image(label="Processed Image", interactive=False)

                # params
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            # with gr.Group():

                            unconditional_guidance_scale = gr.Number(
                                label="CFG-scale", value=4.0, interactive=True,
                            )
                            seed = gr.Number(
                                label="Seed", value=42, interactive=True,
                            )

                            num_steps = gr.Number(
                                label="ODE Sampling Steps", value=250, interactive=True,
                            )

                        # with gr.Column():
                        with gr.Row():
                                mesh_size = gr.Number(
                                    label="Mesh Resolution", value=192, interactive=True,
                            )

                                mesh_thres = gr.Number(
                                    label="Mesh Iso-surface", value=10, interactive=True,
                                )

                with gr.Row():
                    with gr.Group():
                        preprocess_background = gr.Checkbox(
                            label="Remove Background", value=True
                        )
                with gr.Row():
                    submit = gr.Button("Generate", elem_id="generate", variant="primary")

                with gr.Row(variant="panel"): 
                    gr.Examples(
                        examples=[
                            str(path) for path in sorted(Path('./assets/i23d_examples').glob('**/*.png'))
                        ],
                        inputs=[input_image],
                        cache_examples=False,
                        label="Examples",
                        examples_per_page=20,
                    )

            with gr.Column():
                with gr.Row():
                    with gr.Tab("Reconstruction"):
                        with gr.Column():
                            output_video = gr.Video(value=None, width=384, label="Rendered Video", autoplay=True, loop=True)
                            output_model = gr.Model3D(
                                height=384,
                                clear_color=(1,1,1,1),
                                label="Output Model",
                                interactive=False
                            )

        gr.Markdown(
            """
            ## Comments:
            1. The sampling time varies since ODE-based sampling method (dopri5 by default) has adaptive internal step, and reducing sampling steps may not reduce the overal sampling time. Sampling steps=250 is the emperical value that works well in most cases.
            2. The 3D viewer shows a colored .glb mesh extracted from volumetric tri-plane, and may differ slightly with the volume rendering result.
            3. If you find your result unsatisfying, tune the CFG scale and change the random seed. Usually slightly increase the CFG value can lead to better performance.
            3. Known limitations include:
            - Texture details missing: since our VAE is trained on 192x192 resolution due the the resource constraints, the texture details generated by the final 3D-LDM may be blurry. We will keep improving the performance in the future.
            4. Regarding reconstruction performance, our model is slightly inferior to state-of-the-art multi-view LRM-based method (e.g. InstantMesh), but offers much better diversity, flexibility and editing potential due to the intrinsic nature of diffusion model.

            ## How does it work?

            LN3Diff is a feedforward 3D Latent Diffusion Model that supports direct 3D asset generation via diffusion sampling. 
            Compared to SDS-based ([DreamFusion](https://dreamfusion3d.github.io/)), mulit-view generation-based ([MVDream](https://arxiv.org/abs/2308.16512), [Zero123++](https://github.com/SUDO-AI-3D/zero123plus), [Instant3D](https://instant-3d.github.io/)) and feedforward 3D reconstruction-based ([LRM](https://yiconghong.me/LRM/), [InstantMesh](https://github.com/TencentARC/InstantMesh), [LGM](https://github.com/3DTopia/LGM)), 
            LN3Diff supports feedforward 3D generation with a unified framework.
            Like 2D/Video AIGC pipeline, LN3Diff first trains a 3D-VAE and then conduct LDM training (text/image conditioned) on the learned latent space. Some related methods from the industry ([Shape-E](https://github.com/openai/shap-e), [CLAY](https://github.com/CLAY-3D/OpenCLAY), [Meta 3D Gen](https://arxiv.org/abs/2303.05371)) also follow the same paradigm.
            Though currently the performance of the origin 3D LDM's works are overall inferior to reconstruction-based methods, we believe the proposed method has much potential and scales better with more data and compute resources, and may yield better 3D editing performance due to its compatability with diffusion model.
            For more results see the [project page](https://szymanowiczs.github.io/splatter-image) and the [ECCV article](https://arxiv.org/pdf/2403.12019).
            """
        )

        submit.click(fn=check_input_image, inputs=[input_image]).success(
            fn=preprocess,
            inputs=[input_image, preprocess_background],
            outputs=[processed_image],
        ).success(
            # fn=reconstruct_and_export,
            # inputs=[processed_image],
            # outputs=[output_model, output_video],
            fn=training_loop_class.eval_i23d_and_export,
            inputs=[processed_image, num_steps, seed, mesh_size, mesh_thres, unconditional_guidance_scale],
            outputs=[output_video, output_model],
        )

    demo.queue(max_size=1)
    demo.launch(share=True)

# training_loop_class.eval_i23d_and_export(
#         # prompt=args.prompt,
#         # prompt=prompt,
#         unconditional_guidance_scale=args.
#         unconditional_guidance_scale,
#         # unconditional_guidance_scale=unconditional_guidance_scale,
#         # use_ddim=args.use_ddim,
#         # save_img=args.save_img,
#         # use_train_trajectory=args.use_train_trajectory,
#         camera=camera,
#         num_instances=args.num_instances,
#         num_samples=args.num_samples,
#         export_mesh=True, 
#         idx_to_render=seeds,
#     )




def create_argparser():
    defaults = dict(
        image_size_encoder=224,
        triplane_scaling_divider=1.0,  # divide by this value
        diffusion_input_size=-1,
        trainer_name='adm',
        use_amp=False,
        # triplane_scaling_divider=1.0, # divide by this value

        # * sampling flags
        clip_denoised=False,
        num_samples=10,
        num_instances=10, # for i23d, loop different condition
        use_ddim=False,
        ddpm_model_path="",
        cldm_model_path="",
        rec_model_path="",

        # * eval logging flags
        logdir="/mnt/lustre/yslan/logs/nips23/",
        data_dir="",
        eval_data_dir="",
        eval_batch_size=1,
        num_workers=1,

        # * training flags for loading TrainingLoop class
        overfitting=False,
        image_size=128,
        iterations=150000,
        schedule_sampler="uniform",
        anneal_lr=False,
        lr=5e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        eval_interval=2500,
        save_interval=10000,
        resume_checkpoint="",
        resume_cldm_checkpoint="",
        resume_checkpoint_EG3D="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        load_submodule_name='',  # for loading pretrained auto_encoder model
        ignore_resume_opt=False,
        freeze_ae=False,
        denoised_ae=True,
        # inference prompt
        prompt="a red chair",
        interval=1,
        save_img=False,
        use_train_trajectory=
        False,  # use train trajectory to sample images for fid calculation
        unconditional_guidance_scale=1.0,
        use_eos_feature=False,
        export_mesh=False,
        cond_key='caption',
        allow_tf32=True,
    )

    defaults.update(model_and_diffusion_defaults())
    defaults.update(encoder_and_nsr_defaults())  # type: ignore
    defaults.update(loss_defaults())
    defaults.update(continuous_diffusion_defaults())
    defaults.update(control_net_defaults())
    defaults.update(dataset_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    parse_transport_args(parser)

    return parser


if __name__ == "__main__":

    # os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    # os.environ["NCCL_DEBUG"] = "INFO"

    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.

    args = create_argparser().parse_args()

    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.gpus = th.cuda.device_count()

    args.rendering_kwargs = rendering_options_defaults(args)

    main(args)
