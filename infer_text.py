import numpy as np
import os
import rembg
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import tyro
from kiui.op import recenter
from safetensors.torch import load_file

from core.models import LGM
from core.options import AllConfigs, Options
from mvdream.pipeline_mvdream import MVDreamPipeline

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

opt = tyro.cli(AllConfigs)

# model
model = LGM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    print(f'[WARN] model randomly initialized, are you sure?')

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().to(device)
model.eval()

rays_embeddings = model.prepare_default_rays(device)

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

pipe_text = MVDreamPipeline.from_pretrained(
    'ashawkey/mvdream-sd2.1-diffusers',  # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # local_files_only=True,
)
pipe_text = pipe_text.to(device)

# load rembg
bg_remover = rembg.new_session()


# process function
def process(opt: Options, prompt, prompt_neg=opt.prompt_neg, input_elevation=0, input_num_steps=30):
    print(f'[INFO] Processing --> {prompt}')
    os.makedirs(opt.workspace, exist_ok=True)

    mv_image_uint8 = pipe_text(prompt, negative_prompt=prompt_neg, num_inference_steps=input_num_steps,
                               guidance_scale=7.5, elevation=input_elevation)
    mv_image_uint8 = (mv_image_uint8 * 255).astype(np.uint8)
    # bg removal
    mv_image = []
    for i in range(4):
        image = rembg.remove(mv_image_uint8[i], session=bg_remover)  # [H, W, 4]
        # to white bg
        image = image.astype(np.float32) / 255
        image = recenter(image, image[..., 0] > 0, border_ratio=0.2)
        image = image[..., :3] * image[..., -1:] + (1 - image[..., -1:])
        mv_image.append(image)

    # generate gaussians
    # generate gaussians
    input_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32
    input_image = torch.from_numpy(input_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    rays_embeddings = model.prepare_default_rays(device, elevation=input_elevation)
    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)

        # save gaussians
        model.gs.save_ply(gaussians, os.path.join(opt.workspace, 'object.ply'))


assert opt.prompt is not None
process(opt, opt.prompt)
