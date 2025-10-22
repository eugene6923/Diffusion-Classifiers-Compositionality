import torch
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image
import json
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.utils import save_image
import random
import copy
from copy import deepcopy
ARNAS_USES = True

if ARNAS_USES:
    sys.path.insert(0, '/mnt/lustre/work/oh/owl661/compositional-vaes/src/vqvae/_post/self_bench/diffusers/src')
    sys.path.insert(0, '/mnt/lustre/work/oh/owl661/compositional-vaes/src/vqvae/_post/self_bench')
else:
    sys.path.append('./diffusers/src')

from diffusers import StableDiffusionImg2ImgPipelineORG, EulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img_real import StableDiffusion3Img2ImgPipelineReal, retrieve_timesteps

if ARNAS_USES:
    from vqvae._post.self_bench.datasets_loading import get_dataset
else:
    from vqvae._post.self_bench.datasets_loading import get_dataset

def setup_model(version="2.0", model_precision="float16"):
    """Setup the Stable Diffusion model."""
    model_dtype = torch.float16 if model_precision == 'float16' else torch.float32
    
    if version == "2.0":
        model_id = "stabilityai/stable-diffusion-2-base"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        model = StableDiffusionImg2ImgPipelineORG.from_pretrained(
            model_id, 
            scheduler=scheduler, 
            torch_dtype=model_dtype
        )
        org_scheduler_timesteps = copy.deepcopy(model.scheduler.timesteps)
    elif version == "1.5":
        model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        model = StableDiffusionImg2ImgPipelineORG.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=model_dtype
        )
        org_scheduler_timesteps = copy.deepcopy(model.scheduler.timesteps)
    elif version == "3-m":
        model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        model = StableDiffusion3Img2ImgPipelineReal.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
        )
        org_scheduler_timesteps = copy.deepcopy(model.scheduler.timesteps)
        # save weights here as pickle
        import pickle
        with open('timesteps.pkl', 'wb') as f:
            pickle.dump(org_scheduler_timesteps, f)
    else:
        raise ValueError(f"Version {version} not supported yet.")
    return model.to("cuda"), org_scheduler_timesteps

def calculate_l2_error(orig_img, recon_img):
    """Calculate L2 error between original and reconstructed images.
    Resizes images to the larger dimensions before calculating error.
    """
    # Get dimensions
    orig_size = orig_img.size
    recon_size = recon_img.size
    
    # Determine target size (use the larger dimensions)
    target_width = max(orig_size[0], recon_size[0])
    target_height = max(orig_size[1], recon_size[1])
    
    # Create transform pipeline
    transform = transforms.Compose([
        transforms.Resize((target_height, target_width), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    
    # Convert both images to tensors at the target size
    orig_tensor = transform(orig_img)
    recon_tensor = transform(recon_img)
    
    # Calculate L2 error
    l2_error = torch.nn.functional.mse_loss(orig_tensor, recon_tensor)
    return l2_error.item()

def create_pdf_visualization(save_dir, prompts, prompt_indices, correct_idx, timesteps):
    """Create a PDF visualization of reconstructions.
    
    Args:
        save_dir: Directory containing reconstructions
        prompts: List of prompts used
        prompt_indices: List of indices of the prompts
        correct_idx: Index of the correct prompt
        timesteps: List of timesteps used
    """
    n_prompts = len(prompts)
    n_timesteps = len(timesteps)
    
    # Create figure
    fig = plt.figure(figsize=(3*n_timesteps + 2, 3*n_prompts + 1))
    gs = GridSpec(n_prompts, n_timesteps + 1, width_ratios=[2] + [1]*n_timesteps)
    
    # Load original image
    orig_img = Image.open(os.path.join(save_dir, "original.png"))
    
    # For each prompt
    for i, (prompt, orig_idx) in enumerate(zip(prompts, prompt_indices)):
        is_correct = orig_idx == correct_idx
        prompt_dir = os.path.join(save_dir, f"prompt_{orig_idx}_{'correct' if is_correct else 'incorrect'}")
        
        # Add prompt text (with correct/incorrect indicator)
        ax = fig.add_subplot(gs[i, 0])
        ax.text(0.5, 0.5, f"{'✓ ' if is_correct else '✗ '}{prompt}", 
                wrap=True, ha='center', va='center', fontsize=12)
        ax.axis('off')
        
        # Add reconstructions for each timestep
        for j, t in enumerate(timesteps):
            ax = fig.add_subplot(gs[i, j + 1])
            recon_img = Image.open(os.path.join(prompt_dir, f"t{t:04d}.png"))
            
            # Calculate L2 error
            l2_error = calculate_l2_error(orig_img, recon_img)
            
            # Display image
            ax.imshow(recon_img)
            
            # Add L2 error text with white background
            text = ax.text(0.02, 0.98, f'L2: {l2_error:.3f}', 
                         transform=ax.transAxes,
                         verticalalignment='top',
                         color='red',
                         fontsize=14)
            text.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            if i == 0:  # Only show timestep on top row
                ax.set_title(f't={t}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reconstruction_summary.pdf'), bbox_inches='tight')
    plt.close()

def reconstruct_from_timesteps(model, image, prompts, prompt_indices, correct_idx, timesteps, save_dir, guidance_scale, num_inference_steps, model_name, org_scheduler_timesteps):
    """Reconstruct image from different timesteps with multiple prompts.
    Args:
        prompts: List of prompts to use
        prompt_indices: List of indices of the prompts in the original prompt list
        correct_idx: Index of the correct prompt
        guidance_scale: Guidance scale for classifier-free guidance
        num_inference_steps: Number of denoising steps to use for each reconstruction
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set the number of inference steps for the scheduler
    # model.scheduler.set_timesteps(num_inference_steps)
    
    # Save original image
    if isinstance(image, torch.Tensor):
        # Denormalize the tensor from [-1,1] to [0,1] range
        denorm_image = (image + 1.0) / 2.0
        denorm_image = torch.clamp(denorm_image, 0.0, 1.0)
        save_image(denorm_image, os.path.join(save_dir, "original.png"))
    else:
        image.save(os.path.join(save_dir, "original.png"))
    
    # Save reconstructions for each prompt and timestep
    for prompt_idx, (prompt, orig_idx) in enumerate(zip(prompts, prompt_indices)):
        is_correct = orig_idx == correct_idx
        prompt_dir = os.path.join(save_dir, f"prompt_{orig_idx}_{'correct' if is_correct else 'incorrect'}")
        os.makedirs(prompt_dir, exist_ok=True)
        
        # Save the prompt text
        with open(os.path.join(prompt_dir, "prompt.txt"), "w") as f:
            f.write(prompt)
        
        # Save reconstruction from each timestep
        for t in timesteps:
            # Generate reconstruction
            with torch.no_grad():
                if model_name == "3-m":
                    # find closest timestep in org_scheduler
                    # Create evenly spaced timesteps from t down to 0
                    # custom_timesteps = torch.linspace(t, 0, num_inference_steps)
                    # Find closest timesteps in original scheduler for each custom timestep
                    # use_timesteps = []
                    # for custom_t in custom_timesteps:
                    #     closest_t = min(org_scheduler_timesteps, key=lambda x: abs(x - custom_t))
                    #     use_timesteps.append(closest_t.item())
                    # timesteps_default, num_steps = model.get_timesteps(num_inference_steps, t, "cuda")
                    
                    # ensure last is 0 
                    # use_timesteps[-1] = 0
                    # use_timestep = use_timesteps[0]  # Use first timestep for initial denoising
                    # timesteps_default = retrieve_timesteps(model.scheduler, num_inference_steps, "cuda", sigmas=None, strength=t/1000.0)
                    # timesteps_default, num_steps = model.get_timesteps(num_inference_steps, 0.5, "cuda")
                    output = model(
                        prompt=prompt,
                        image=image,
                        # strength=t/1000.0,  # Convert timestep to strength
                        strength=t/1000.0,
                        # custom_timesteps=timesteps_default,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,  # Use the fixed number of inference steps
                    ).images[0]
                else:
                    output = model(
                        prompt=prompt,
                        image=image,
                        strength=t/1000.0,  # Convert timestep to strength
                        # strength=1.0,
                        # timesteps=[t],
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,  # Use the fixed number of inference steps
                    ).images[0]
            
            # Save the reconstruction
            output.save(os.path.join(prompt_dir, f"t{t:04d}.png"))
            print(f'saved at {os.path.join(prompt_dir, f"t{t:04d}.png")}')

    # Create PDF visualization
    create_pdf_visualization(save_dir, prompts, prompt_indices, correct_idx, timesteps)

def main(args):
    # Setup model
    model, org_scheduler_timesteps = setup_model(version=args.model_version, model_precision=args.model_precision)
    
    # Set root directory based on ARNAS_USES
    if ARNAS_USES:
        root_dir = '/mnt/lustre/work/oh/owl661/sd-datasets/'
    else:
        root_dir = args.root_dir
    
    # Load datasets
    datasets = {}
    # "geneval_color_attr"]:
    # # for dataset_name in ['clevr_spatial']:#["geneval_single", "geneval_position", "geneval_two", "geneval_counting", "geneval_color_attr"]:
    for dataset_name in [ 'whatsup_A', 'clevr_binding_color', 'spec_count', 'COCO_QA_one', 'sugar_att', 'sugar_obj', 'VG_QA_one', 'geneval_position', 'geneval_two', 'geneval_counting','whatsup_B', 'clevr_spatial']:#["geneval_single", "geneval_position", "geneval_two", "geneval_counting", 
        datasets[dataset_name] = get_dataset(
            dataset_name=dataset_name,
            root_dir=root_dir,
            transform=None,
            resize=512,
            version=args.geneval_version,  # Use geneval_version for dataset loading
            domain='photo',
            cfg=9.0,
            filter=True
        )
    
    # Process specified indices for each dataset
    for dataset_name, dataset in datasets.items():
        print(f"\nProcessing {dataset_name}...")
        print(f"Dataset size: {len(dataset)}")
        
        # Process each specified index
        # randomize idices
        # set random seed
        random.seed(42)
        # Generate random indices within dataset size
        num_samples = min(10, len(dataset))  # Take up to 10 samples
        args.indices = random.sample(range(len(dataset)), num_samples)
        args.indices.sort()  # Sort indices for consistent ordering
        print(f"Selected random indices: {args.indices}")
        
        for idx in args.indices:
            if idx >= len(dataset):
                print(f"Warning: Index {idx} is out of range for dataset {dataset_name}")
                continue
                
            print(f"Processing index {idx}")
            
            # Get data
            if dataset_name == 'clevr_binding_color' or dataset_name == 'clevr_spatial':
                data = dataset[idx]
                img_path = None
                img_tensor = data[0][1][0]
                text = data[1]
                correct_idx = data[3]
            else:
                (img_path, [img_tensor]), text, correct_idx = dataset[idx]
            
            # Select prompts to use (0,1,2 + correct if not already included)
            base_indices = np.arange(min(4, len(text))).astype(int).tolist()
            prompt_indices = base_indices.copy()
            if correct_idx not in base_indices:
                prompt_indices.append(correct_idx)
            selected_prompts = [text[i] for i in prompt_indices]
            
            # Create save directory for this sample
            save_dir = os.path.join(
                args.output_dir,
                dataset_name,
                f"geneval_{args.geneval_version}",  # Include geneval version in path
                f"model_{args.model_version}",      # Include model version in path
                f"cfg_{args.guidance_scale}_steps_{args.num_inference_steps}",  # Add guidance and steps info
                f"sample_{idx:04d}"
            )
            os.makedirs(save_dir, exist_ok=True)
            
            # Perform reconstruction from each timestep
            reconstruct_from_timesteps(
                model=model,
                image=img_tensor,
                prompts=selected_prompts,
                prompt_indices=prompt_indices,
                correct_idx=correct_idx,    
                timesteps=args.timesteps,
                save_dir=save_dir,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                model_name=args.model_version,
                org_scheduler_timesteps=org_scheduler_timesteps
            )
            
            # Save metadata
            metadata = {
                "image_path": img_path,
                "all_prompts": text,
                "correct_idx": correct_idx,
                "selected_prompt_indices": prompt_indices,
                "timesteps": args.timesteps,
                "ARNAS_USES": ARNAS_USES,
                "root_dir": root_dir,
                "dataset": dataset_name,
                "model_version": args.model_version,
                "geneval_version": args.geneval_version,
                "guidance_scale": args.guidance_scale,
                "num_inference_steps": args.num_inference_steps
            }
            with open(os.path.join(save_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", type=str, default="3-m", help="Model version to use for reconstruction")
    parser.add_argument("--geneval_version", type=str, default="3-m", help="Version of geneval dataset to load")
    parser.add_argument("--model_precision", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--root_dir", type=str, default="../geneval/outputs", help="Only used if ARNAS_USES is False")
    parser.add_argument("--output_dir", type=str, default="./reconstructions")
    parser.add_argument("--indices", type=int, nargs="+", default=[10, 11, 12, 13], help="Indices to process")
    parser.add_argument("--timesteps", type=int, nargs="+", default=[100, 400, 500, 533, 566, 600, 633, 666, 700, 733, 766, 800, 833, 866, 900, 933, 966, 999], help="Timesteps for reconstruction")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of denoising steps")
    
    args = parser.parse_args()
    main(args)
