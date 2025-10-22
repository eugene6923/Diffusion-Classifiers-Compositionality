import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import numpy as np
import warnings

import sys
sys.path.append('./diffusers/src')

try:
    from kind_of_globals import ARNAS_USES
except:
    try:
        from vqvae._post.self_bench.kind_of_globals import ARNAS_USES
    except:
        ARNAS_USES = False

import os
# Only import wandb and logger-related stuff if ARNAS_USES is true
if ARNAS_USES:
    sys.path.insert(0, '/mnt/lustre/work/oh/owl661/compositional-vaes/src/vqvae/_post/self_bench/diffusers/src')
    sys.path.insert(0, '/mnt/lustre/work/oh/owl661/compositional-vaes/src/vqvae/_post/self_bench')

    import wandb
    from stuned.utility.logger import log_to_sheet_in_batch, RedneckLogger
    def get_default_param(params, key, default, logger: RedneckLogger):
        if key in params:
            return params[key]
        else:
            # Make sure to report this default value to GSheets too!
            if logger is not None:
                log_to_sheet_in_batch(logger, {f"delta:{key}": default}, sync=False)

        warnings.warn(f"Key {key} not found in params, returning default value {default}")
        return default
    WANDB_PROJECT = "diffusion-itm"
    WANDB_ENTITY = "oshapio"
    
    def login_wandb():
        api_key = "9876773f72d210923a9694ae701f8d71c9d30381"
        os.environ["WANDB_API_KEY"] = api_key
    login_wandb()

from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, StableDiffusionImg2ImgPipeline, StableDiffusion3Img2ImgPipeline, StableDiffusionImg2ImgPipeline_test
import glob
import os
from PIL import Image
from tqdm import tqdm
import argparse
import json   
import random

try:
    from datasets_loading import get_dataset
except:
    from vqvae._post.self_bench.datasets_loading import get_dataset

from torch.utils.data import DataLoader
try:
    from utils import evaluate_scores, save_bias_scores, save_bias_results
except:
    from vqvae._post.self_bench.utils import evaluate_scores, save_bias_scores, save_bias_results
import csv
from accelerate import Accelerator
import cProfile

def convert_to_json_serializable(obj):
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)  # Convert to native Python float
    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)  # Convert to native Python int
    return str(obj)  # Default conversion (if needed)

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError(f"Boolean value expected. Got '{value}' instead.")


def score_batch(i, args, batch, model):
    """
        Takes a batch of images and captions and returns a score for each image-caption pair.
    """

    imgs, texts = batch[0], batch[1]
    imgs, imgs_resize = imgs[0], imgs[1]

    imgs_resize = [img.cuda().to(model.dtype) for img in imgs_resize]
    transform = transforms.ToTensor()
    scores = []
    
    with torch.no_grad():
        presaved_latents = []
        for k in range(len(imgs_resize)):
            latents_this = []
            for i in range(0, len(imgs_resize[k]), imgs_resize[k].shape[0]):
                batch = imgs_resize[k][i:i+imgs_resize[k].shape[0]]
                batch_processed = model.image_processor.preprocess(batch)
                latents_this.append(model.presave_latents(batch_processed))
            presaved_latent = torch.cat(latents_this, dim=0)
            presaved_latents.append(presaved_latent)
        
    cond_noises_per_text = []
    uncond_noises_per_text = []
    target_gaussian_noises_per_text = None
    
    for txt_idx, text in enumerate(texts):
        for img_idx, resized_img in enumerate(imgs_resize):
            if len(resized_img.shape) == 3:
                resized_img = resized_img.unsqueeze(0)
            
            print(f'Batch {i}, Text {txt_idx}, Image {img_idx}')
            if args.save_noise == True:
                save_noise = args
            else: save_noise = None

            import time
            start_time = time.time()
            dists, cond_noises_per_timestep, uncond_noises_per_timestep, target_gaussian_noises_per_timestep = model(prompt=list(text), image=resized_img, guidance_scale=args.guidance_scale, 
                         sampling_steps=args.sampling_steps, unconditional=args.img_retrieval, 
                         middle_step=args.middle_step, time_weighting=args.time_weighting, 
                         imgs_visulalize=None, save_noise=save_noise, presaved_latents=presaved_latents[img_idx], save_noises=True, use_normed_classifier=args.use_normed_classifier)
            dists = dists.to(torch.float32)
            if cond_noises_per_timestep is not None:
                cond_noises_per_text.append(cond_noises_per_timestep)
                uncond_noises_per_text.append(uncond_noises_per_timestep)
                target_gaussian_noises_per_text = target_gaussian_noises_per_timestep
            
            print(f'took {time.time() - start_time} seconds to get dists')

            dists = dists.mean(dim=1)
            dists = -dists
            scores.append(dists)

    scores = torch.stack(scores).permute(1, 0) if args.batchsize > 1 else torch.stack(scores).unsqueeze(0)
    if args.arnas_save_noise and cond_noises_per_text != [] and len(cond_noises_per_text[0]) > 0:
        timesteps_used = [x for x in list(cond_noises_per_text[0].keys())]
            
        # stack up the cond and uncond noises --> num_prompts x num_timesteps x batch_size x -1
        cond_noises = torch.stack([torch.stack([x[time] for time in timesteps_used]) for x in cond_noises_per_text]).cpu()
        uncond_noises = torch.stack([torch.stack([x[time] for time in timesteps_used]) for x in uncond_noises_per_text]).cpu()
        # num_steps x 1 x -1 
        target_gaussian_noises = torch.stack([target_gaussian_noises_per_text[time] for time in timesteps_used]).cpu()
    else:
        cond_noises = None
        uncond_noises = None
        target_gaussian_noises = None
    # Calculate statistics after stacking
    probs = torch.softmax(scores, dim=-1)
    
    stats = {
        'top1_probs': torch.max(probs, dim=-1)[0].tolist(),  # Maximum probability for each prediction
        'entropy': (-probs * torch.log(probs + 1e-10)).sum(dim=-1).tolist(),  # Distribution entropy
        'predictions': torch.argmax(probs, dim=-1).tolist(),  # Predicted indices
        'all_probs': probs.tolist()  # Full probability distribution
    }
    
    return scores, stats, cond_noises, uncond_noises, target_gaussian_noises
        

def main(params, logger=None, local_run=True):
    """Main function for diffusion ITM training.
    
    Args:
        params: Dictionary or Namespace of parameters
        logger: Optional logger instance for parameter logging
        local_run: Whether this is a local run
    """
    # Convert params to args, handling both dict and Namespace cases
    if isinstance(params, dict):
        args = argparse.Namespace(**params)
    else:
        args = params  # Already a Namespace

    # Create params_dict for get_default_param
    params_dict = vars(args) if isinstance(args, argparse.Namespace) else params

    # If we have ARNAS_USES and logger, report all parameters including defaults
    if ARNAS_USES:
        # Get and report all parameters through get_default_param
        args.task = get_default_param(params_dict, "task", "geneval_counting", logger)
        args.seed = get_default_param(params_dict, "seed", 0, logger)
        args.skip = get_default_param(params_dict, "skip", 0, logger)
        args.batchsize = get_default_param(params_dict, "batchsize", 64, logger)
        args.subset = get_default_param(params_dict, "subset", False, logger)
        args.middle_step = get_default_param(params_dict, "middle_step", False, logger)
        args.encoder_drop = get_default_param(params_dict, "encoder_drop", False, logger)
        args.index_subset = get_default_param(params_dict, "index_subset", 0, logger)
        args.sampling_steps = get_default_param(params_dict, "sampling_steps", 30, logger)
        args.img_retrieval = get_default_param(params_dict, "img_retrieval", False, logger)
        args.gray_baseline = get_default_param(params_dict, "gray_baseline", False, logger)
        args.version = get_default_param(params_dict, "version", "2.0", logger)
        args.lora_dir = get_default_param(params_dict, "lora_dir", "", logger)
        args.guidance_scale = get_default_param(params_dict, "guidance_scale", 0.0, logger)
        args.targets = get_default_param(params_dict, "targets", "", logger)
        args.comp_subset = get_default_param(params_dict, "comp_subset", None, logger)
        args.only_big = get_default_param(params_dict, "only_big", False, logger)
        args.time_weighting = get_default_param(params_dict, "time_weighting", None, logger)
        args.domain = get_default_param(params_dict, "domain", "photo", logger)
        args.geneval_cfg = get_default_param(params_dict, "geneval_cfg", 9.0, logger)
        args.sd3_resize = get_default_param(params_dict, "sd3_resize", False, logger)
        args.save_noise = get_default_param(params_dict, "save_noise", False, logger)
        args.arnas_save_noise = get_default_param(params_dict, "arnas_save_noise", False, logger),
        args.wandb = get_default_param(params_dict, "wandb", True, logger)
        args.use_normed_classifier = get_default_param(params_dict, "use_normed_classifier", False, logger)
        
        # Initialize wandb if ARNAS_USES is true
        cwd = os.path.abspath(os.getcwd())
        noise_save_dir = os.path.abspath(f'{cwd}/noise_results/{args.version}')
        
        
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=args.run_id if hasattr(args, 'run_id') else None,
            config=args
        )
        
        # get run id, save in GSheet
        args.run_id = wandb.run.id
        run_id = args.run_id
        if logger is not None:
            log_to_sheet_in_batch(logger, {"run_id": run_id}, sync=True)
        
            # Report wandb url to logger
            try:
                wandb_url = wandb.run.get_url()
                log_to_sheet_in_batch(logger, {"WandB url": wandb_url}, sync=True)
            except:
                log_to_sheet_in_batch(logger, {"WandB url": "Not available"}, sync=True)
        # log to wandb noise_save_dir
        wandb.log({"noise_save_dir": noise_save_dir})
    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if hasattr(args, "cuda_device"):
        torch.cuda.set_device(args.cuda_device)

    # Initialize lists to accumulate statistics
    all_top1_probs = []
    all_entropies = []
    all_predictions = []
    
    accelerator = Accelerator()

    # Model initialization based on version
    if args.version == '2.1' or args.version == 2.1:
        model_id = "stabilityai/stable-diffusion-2-1-base"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    elif args.version == '2-new':
        model_id = "stabilityai/stable-diffusion-2-1-base"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    elif args.version == '2.0' or args.version == 2.0:
        model_id = "stabilityai/stable-diffusion-2-base"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    elif args.version =='1.4':
        model_id = "CompVis/stable-diffusion-v1-4"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, scheduler=scheduler)
    elif args.version == '1.5' or args.version == 1.5:
        model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, scheduler=scheduler)
    elif args.version == 'xl':
        from diffusers import StableDiffusionXLImg2ImgPipeline
        model_id == "stabilityai/stable-diffusion-xl-base-1.0"
        scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        model = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    elif args.version == '3-m':
        model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        if not args.sd3_resize:
            
            # scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            if args.encoder_drop:
                model = StableDiffusion3Img2ImgPipeline.from_pretrained(
                    model_id, 
                    # scheduler=scheduler, 
                    text_encoder_3=None,
                    tokenizer_3=None,
                    torch_dtype=torch.float16,
                    )
                    
            else:
                if args.use_euler:
                    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
                    model = StableDiffusion3Img2ImgPipeline.from_pretrained(
                        model_id,
                        scheduler=scheduler, 
                        torch_dtype=torch.float16,
                        # torch_dtype=torch.float32,
                        )
                else:
                    model = StableDiffusion3Img2ImgPipeline.from_pretrained(
                        model_id,
                        # scheduler=scheduler,
                        torch_dtype=torch.float16,
                        # torch_dtype=torch.float32,
                        )
        else: 
            if args.encoder_drop:
                model = StableDiffusion3Img2ImgPipeline.from_pretrained(
                    model_id, 
                    # scheduler=scheduler, 
                    text_encoder_3=None,
                    tokenizer_3=None,
                    torch_dtype=torch.float16,
                    resize=512
                    )
                    
            else:
                model = StableDiffusion3Img2ImgPipeline.from_pretrained(
                    model_id,
                    # scheduler=scheduler, 
                    torch_dtype=torch.float16,
                    # torch_dtype=torch.float32,
                    resize=512
                    )
            
    elif args.version == 'compdiff':
        if args.comp_subset == None:
            raise ValueError("Subset is needed for this version")
        elif args.comp_subset in ['shape', 'color','texture']:
            pt_folder = f'CompBench/GORS_finetune/checkpoint/{args.comp_subset}/'
            pt_file = next(f'{pt_folder}{i}' for i in os.listdir(pt_folder) if 'text_encder' not in i)
            model_id = "stabilityai/stable-diffusion-2-base"
            # Use the Euler scheduler here instead
            scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            # scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
            model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
            patch_pipe(model, pt_file , patch_text=True, patch_unet=True, patch_ti=False,
                    unet_target_replace_module=["CrossAttention"],
                    text_target_replace_module=["CLIPAttention"])
        else:
            model_id = "stabilityai/stable-diffusion-2-base"
            # Use the Euler scheduler here instead
            scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            # scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
            model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
            patch_pipe(model, f'CompBench/GORS_finetune/checkpoint/{args.comp_subset}/{args.comp_subset}.pt' , patch_text=True, patch_unet=True, patch_ti=False,
                    unet_target_replace_module=["CrossAttention"],
                    text_target_replace_module=["CLIPAttention"])

    elif args.version == '3-L':
        model_id = "stabilityai/stable-diffusion-3.5-large"
        model = StableDiffusion3Img2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

    else:
        raise ValueError(f"Version {args.version} not supported.")
    
    model = model.to(accelerator.device)

    if args.lora_dir != '':
        model.unet.load_attn_procs(args.lora_dir)
        
    if ARNAS_USES:
        root_dir = '/mnt/lustre/work/oh/owl661/sd-datasets/'
    else:
        root_dir = '../../../data/raw'

    if str(args.version).startswith('3'):
        if not args.sd3_resize:
            print("resizing images to 1024")
            dataset = get_dataset(args.task, root_dir = root_dir, transform=None, targets=args.targets, mode = args.img_retrieval, index_subset = args.index_subset, resize= 1024, version = args.geneval_version, domain = args.domain, cfg = args.geneval_cfg, filter = args.geneval_filter)
        else:
            print("resizing images to 512")
            dataset = get_dataset(args.task, root_dir = root_dir, transform=None, targets=args.targets, mode = args.img_retrieval, index_subset = args.index_subset, version = args.geneval_version, domain = args.domain, cfg = args.geneval_cfg, filter = args.geneval_filter)
    else:
        print("resizing images to 512")
        dataset = get_dataset(args.task, root_dir = root_dir, transform=None, targets=args.targets, mode = args.img_retrieval, index_subset = args.index_subset, version = args.geneval_version, domain = args.domain, cfg = args.geneval_cfg, filter = args.geneval_filter)

    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)

    model, dataloader = accelerator.prepare(model, dataloader)

    SKIP_NUMB = 9 if args.task == 'coco_order' else 3

    r1s = []
    r5s = []
    max_more_than_onces = 0
    metrics = []
    ids = []
    clevr_dict = {}
    bias_scores = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
    gender_bias_scores = {'male_clothes': [], 'female_clothes': [], 'male_bags': [], 'female_bags': [], 'male_drinks': [], 'female_drinks': []}
    print(args)

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if i < args.skip:
            continue
        if args.subset and i % SKIP_NUMB != 0:
            continue
        scores, prediction_stats, cond_noises, uncond_noises, target_gaussian_noises = score_batch(i, args, batch, model)

        scores = scores.contiguous()
        accelerator.wait_for_everyone()
        # print(scores)
        scores = accelerator.gather(scores)
        batch[-1] = accelerator.gather(batch[-1])
        if accelerator.is_main_process:
            if 'winoground' in args.task or args.task == 'cola_multi' or 'vismin' in args.task or 'eqbench' in args.task or 'mmvp' in args.task:
                text_scores, img_scores, group_scores = evaluate_scores(args, scores, batch, i)
                metrics += list(zip(text_scores, img_scores, group_scores))
                text_score = sum([m[0] for m in metrics]) / len(metrics)
                img_score = sum([m[1] for m in metrics]) / len(metrics)
                group_score = sum([m[2] for m in metrics]) / len(metrics)
                print(f'Text score: {text_score}')
                print(f'Image score: {img_score}')
                print(f'Group score: {group_score}')
                print(len(metrics))
                if ARNAS_USES:
                    wandb.log({
                        "text_score": text_score,
                        "image_score": img_score,
                        "group_score": group_score,
                        "num_samples": len(metrics),
                        "step": i
                    })
                if args.save:
                    with open(f'{args.outdir}/{args.run_id}.txt', 'w') as f:
                        f.write(f'Text score: {text_score}\n')
                        f.write(f'Image score: {img_score}\n')
                        f.write(f'Group score: {group_score}\n')
                        f.write(f"Sample size {len(metrics)}\n")

            elif args.task in ['flickr30k', 'imagecode', 'imagenet', 'flickr30k_text']:
                scores = evaluate_scores(args, scores, batch, i)
                r1,r5, max_more_than_once = scores
                r1s += r1
                r5s += r5
                max_more_than_onces += max_more_than_once
                r1 = sum(r1s) / len(r1s)
                r5 = sum(r5s) / len(r5s)
                print(f'R@1: {r1}')
                print(f'R@5: {r5}')
                print(f'Max more than once: {max_more_than_onces}')
                if ARNAS_USES:
                    wandb.log({
                        "R@1": r1,
                        "R@5": r5,
                        "max_more_than_once": max_more_than_onces,
                        "num_samples": len(r1s),
                        "step": i
                    })
                with open(f'{args.outdir}/{args.run_id}.txt', 'w') as f:
                    f.write(f'R@1: {r1}\n')
                    f.write(f'R@5: {r5}\n')
                    f.write(f'Max more than once: {max_more_than_onces}\n')
                    f.write(f"Sample size {len(r1s)}\n")
            elif args.task == 'clevr':
                scores = evaluate_scores(args, scores, batch, i)
                acc_list, max_more_than_once = scores
                metrics += acc_list
                acc = sum(metrics) / len(metrics)
                max_more_than_onces += max_more_than_once
                print(f'Accuracy: {acc}')
                print(f'Max more than once: {max_more_than_onces}')
                if  ARNAS_USES:
                    wandb.log({
                        "accuracy": acc,
                        "max_more_than_once": max_more_than_onces,
                        "num_samples": len(metrics),
                        "step": i
                    })
                with open(f'{args.outdir}/{args.run_id}.txt', 'w') as f:
                    f.write(f'Accuracy: {acc}\n')
                    f.write(f'Max more than once: {max_more_than_onces}\n')
                    f.write(f"Sample size {len(metrics)}\n")

                # now do the same but for every subtask of CLEVR
                subtasks = batch[-2]
                for i, subtask in enumerate(subtasks):
                    if subtask not in clevr_dict:
                        clevr_dict[subtask] = []
                    clevr_dict[subtask].append(acc_list[i])
                for subtask in clevr_dict:
                    subtask_acc = sum(clevr_dict[subtask]) / len(clevr_dict[subtask])
                    print(f'{subtask} accuracy: {subtask_acc}')
                    if  ARNAS_USES:
                        wandb.log({
                            f"accuracy_{subtask}": subtask_acc,
                            "num_samples": len(clevr_dict[subtask]),
                            "step": i
                        })
                    with open(f'{args.outdir}/{args.run_id}.txt', 'a') as f:
                        f.write(f'{subtask} accuracy: {subtask_acc}\n')
            elif args.task == 'mmbias':                
                phis = evaluate_scores(args,scores,batch,i)
                for class_idx, phi_list in phis.items():
                    if type(phi_list[0]) != float: # convert from numpy to regular float for json purposes
                        phi_list = [a.item() for a in phi_list]
                    bias_scores[class_idx].extend(phi_list)
                if (i+1)%5==0:
                    print(bias_scores)
                    save_bias_scores(f'./results/{args.run_id}_interim_results{i}.json',bias_scores)
                    if  ARNAS_USES:
                        # Log each class's bias scores
                        for class_idx, scores in bias_scores.items():
                            if scores:  # Only log if there are scores
                                wandb.log({
                                    f"bias_score_class_{class_idx}": sum(scores) / len(scores),
                                    f"bias_score_class_{class_idx}_samples": len(scores),
                                    "step": i
                                })
            elif args.task == 'genderbias':                
                phis = evaluate_scores(args,scores,batch,i)
                for class_id, phi_list in phis.items():
                    if type(phi_list[0]) != float: # convert from numpy to regular float for json purposes
                        phi_list = [a.item() for a in phi_list]
                    gender_bias_scores[class_id].extend(phi_list)
                if (i+1)%5==0:
                    print(gender_bias_scores)
                    save_bias_scores(f'./results/{args.run_id}_interim_results{i}.json',gender_bias_scores)
                    if  ARNAS_USES:
                        # Log each gender bias category
                        for category, scores in gender_bias_scores.items():
                            if scores:  # Only log if there are scores
                                wandb.log({
                                    f"gender_bias_{category}": sum(scores) / len(scores),
                                    f"gender_bias_{category}_samples": len(scores),
                                    "step": i
                                })

            else:
                acc, max_more_than_once = evaluate_scores(args, scores, batch, i)
                metrics += acc
                acc = sum(metrics) / len(metrics)
                max_more_than_onces += max_more_than_once
                print(f'Accuracy: {acc}')
                print(f'Max more than once: {max_more_than_onces}')
                
                # Accumulate statistics
                all_top1_probs.extend(prediction_stats['top1_probs'])
                all_entropies.extend(prediction_stats['entropy'])
                all_predictions.extend(prediction_stats['predictions'])
                
                if ARNAS_USES:
                    # Log basic metrics
                    wandb.log({
                        "accuracy": acc,
                        "max_more_than_once": max_more_than_onces,
                        "num_samples": len(metrics),
                        "step": i,
                        # Log current batch statistics
                        "batch/mean_top1_prob": np.mean(prediction_stats['top1_probs']),
                        "batch/mean_entropy": np.mean(prediction_stats['entropy']),
                        # Log accumulated statistics
                        "total/mean_top1_prob": np.mean(all_top1_probs),
                        "total/std_top1_prob": np.std(all_top1_probs),
                        "total/mean_entropy": np.mean(all_entropies),
                        "total/std_entropy": np.std(all_entropies)
                    })
                    
                    # Log histograms periodically (every 10 batches)
                    if i % 10 == 0:
                        wandb.log({
                            "distributions/top1_probs": wandb.Histogram(all_top1_probs),
                            "distributions/entropy": wandb.Histogram(all_entropies),
                            "step": i
                        })
                        
                if args.save:
                    with open(f'{args.outdir}/{args.run_id}.txt', 'w') as f:
                        f.write(f'Accuracy: {acc}\n')
                        f.write(f'Max more than once: {max_more_than_onces}\n')
                        f.write(f"Sample size {len(metrics)}\n")
                        f.write(f"Mean top1 probability: {np.mean(all_top1_probs):.4f} ± {np.std(all_top1_probs):.4f}\n")
                        f.write(f"Mean entropy: {np.mean(all_entropies):.4f} ± {np.std(all_entropies):.4f}\n")
                        
                # If we're saving noises, we ensure to not save all of them but just a limited number. 
                if args.arnas_save_noise and cond_noises is not None:
                    # Get number of prompts from the noise tensor shape
                    num_prompts = cond_noises.shape[0]
                    
                    # Determine how many noises to save (N)
                    N = min(4, num_prompts)  
                    
                    # Get correct indices from batch[2]
                    correct_indices = batch[2]
                    
                    selected_indices = []
                    filtered_cond_noises = []
                    filtered_uncond_noises = []
                    # Process each item in the batch
                    for batch_idx, correct_idx in enumerate(correct_indices):
                        # Always include the correct prompt index
                        this_selected = []
                        this_selected.append(correct_idx.item())
                        
                        # Create list of incorrect prompt indices (all indices except the correct one)
                        incorrect_indices = [idx for idx in range(num_prompts) if idx != correct_idx]
                        
                        # Randomly select N-1 indices or all remaining if N >= num_prompts
                        if N > 1:
                            num_to_select = min(N - 1, len(incorrect_indices))
                            this_selected.extend(random.sample(incorrect_indices, num_to_select))
                        selected_indices.append(this_selected)
                        
                        filtered_cond_noises.append(cond_noises[this_selected, :, batch_idx])
                        filtered_uncond_noises.append(uncond_noises[0:1, :, batch_idx])
                    
                    filtered_cond_noises = torch.stack(filtered_cond_noises, dim=2)
                    filtered_uncond_noises = torch.stack(filtered_uncond_noises, dim=2)
                    
                    # Ensure we're using float16 for storage efficiency
                    filtered_cond_noises = filtered_cond_noises.to(torch.float16)
                    filtered_uncond_noises = filtered_uncond_noises.to(torch.float16)
                    target_gaussian_noises = target_gaussian_noises.to(torch.float16)
                    
                    # Get current working directory and wandb ID
                    cwd = os.path.abspath(os.getcwd())
                    wandb_id = wandb.run.id if ARNAS_USES else "no_wandb"
                    
                    # Create directory structure with wandb ID using absolute paths
                    save_dir = os.path.abspath(f'{cwd}/noise_results/{args.version}/{wandb_id}')
                    os.makedirs(save_dir, exist_ok=True)
                    
                    target_noise_path = f'{save_dir}/target_gaussian_noises_batch{i}.pt'
                    # if not os.path.exists(target_noise_path):
                    torch.save({'target_gaussian_noises': target_gaussian_noises}, target_noise_path)
                    print(f"Target gaussian noises size (MB): {os.path.getsize(target_noise_path) / 1024/1024:.2f}")
                    
                    # Save the filtered noises without target_gaussian_noises
                    noise_data = {
                        'conditional_noises': filtered_cond_noises,
                        'unconditional_noises': filtered_uncond_noises,
                        'selected_indices': selected_indices,
                        'correct_indices': correct_indices.tolist(),  # Save the original correct indices
                        'num_prompts': num_prompts,
                        'working_directory': cwd,
                        'wandb_id': wandb_id
                    }
                    
                    # Save the noise data with absolute path
                    save_path = f'{save_dir}/{args.run_id}_batch{i}_noises.pt'
                    torch.save(noise_data, save_path)
                    
                    # Print memory usage for debugging
                    print(f"Memory usage per tensor (MB):")
                    print(f"Conditional noises: {filtered_cond_noises.nelement() * filtered_cond_noises.element_size() / 1024/1024:.2f}")
                    print(f"Unconditional noises: {filtered_uncond_noises.nelement() * filtered_uncond_noises.element_size() / 1024/1024:.2f}")
                    print(f"Batch file size (MB): {os.path.getsize(save_path) / 1024/1024:.2f}")
                    
                    if ARNAS_USES:
                        # Log memory usage to wandb
                        wandb.log({
                            "noise_storage/conditional_mb": filtered_cond_noises.nelement() * filtered_cond_noises.element_size() / 1024/1024,
                            "noise_storage/unconditional_mb": filtered_uncond_noises.nelement() * filtered_uncond_noises.element_size() / 1024/1024,
                            "noise_storage/batch_file_mb": os.path.getsize(save_path) / 1024/1024
                        })
                        
                        # Log the save directory to wandb config with absolute path
                        wandb.config.update({
                            "noise_save_directory": save_dir,
                            "working_directory": cwd,
                            "target_noise_path": target_noise_path
                        }, allow_val_change=True)


    if args.save:
        with open(f'{args.outdir}/{args.run_id}.txt', 'a') as f:
            f.write('Done!\n')
            # Write final statistics
            f.write("\nFinal Statistics:\n")
            f.write(f"Number of samples: {len(all_top1_probs)}\n")
            f.write(f"Mean top1 probability: {np.mean(all_top1_probs):.4f} ± {np.std(all_top1_probs):.4f}\n")
            f.write(f"Mean entropy: {np.mean(all_entropies):.4f} ± {np.std(all_entropies):.4f}\n")
            
    print(args)
    print("\nFinal Statistics:")
    print(f"Number of samples: {len(all_top1_probs)}")
    print(f"Mean top1 probability: {np.mean(all_top1_probs):.4f} ± {np.std(all_top1_probs):.4f}")
    print(f"Mean entropy: {np.mean(all_entropies):.4f} ± {np.std(all_entropies):.4f}")
    
    if  ARNAS_USES:
        # Log final statistics
        wandb.run.summary["final_mean_top1_prob"] = np.mean(all_top1_probs)
        wandb.run.summary["final_std_top1_prob"] = np.std(all_top1_probs)
        wandb.run.summary["final_mean_entropy"] = np.mean(all_entropies)
        wandb.run.summary["final_std_entropy"] = np.std(all_entropies)
        wandb.finish()
    

    if args.task == 'mmbias':
        print("\n\n-------------------------We're done!-------------------------\nBias Scores:")
        print(bias_scores)
        if os.path.exists(f'{args.outdir}/{args.run_id}.json'):
            with open(f'{args.outdir}/{args.run_id}.json', 'r') as f:
                existing_bias_scores = json.load(f)
                # add previously calculated ones
                for class_idx, scores in bias_scores.items():
                    if scores == []: # only overwrite if didn't recalculate this time
                        if str(class_idx) in existing_bias_scores:
                            bias_scores[class_idx] = existing_bias_scores[str(class_idx)]
            f.close()
        # now write new contents
        save_bias_scores(f'{args.outdir}/{args.run_id}.json', bias_scores)
        save_bias_results(f'{args.outdir}/{args.run_id}.txt', bias_scores, 'mmbias')
    elif args.task == 'genderbias':
        print("\n\n-------------------------We're done!-------------------------\nGender Bias Scores:")
        print(gender_bias_scores)
        if os.path.exists(f'{args.outdir}/{args.run_id}.json'):
            with open(f'{args.outdir}/{args.run_id}.json', 'r') as f:
                existing_bias_scores = json.load(f)
                # add previously calculated ones
                for class_idx, scores in gender_bias_scores.items():
                    if scores == []: # only overwrite if didn't recalculate this time
                        if str(class_idx) in existing_bias_scores:
                            gender_bias_scores[class_idx] = existing_bias_scores[str(class_idx)]
            f.close()
        # now write new contents
        save_bias_scores(f'{args.outdir}/{args.run_id}.json', gender_bias_scores)
        save_bias_results(f'{args.outdir}/{args.run_id}.txt', gender_bias_scores, 'genderbias')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="geneval_counting")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip', type=int, default=0, help='number of batches to skip\nuse: skip if i < args.skip\ni.e. put 49 if you mean 50')
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--subset', action='store_true')
    parser.add_argument('--middle_step', action='store_true')
    parser.add_argument('--encoder_drop', action='store_true')
    parser.add_argument('--index_subset', type=int, default=0)
    parser.add_argument('--sampling_steps', type=int, default=30) # 10
    parser.add_argument('--img_retrieval', action='store_true')
    parser.add_argument('--gray_baseline', action='store_true')
    parser.add_argument('--version', type=str, default='2.0')
    parser.add_argument('--lora_dir', type=str, default='')
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--targets', type=str, nargs='*', help="which target groups for mmbias",default='')
    parser.add_argument('--comp_subset', type=str, default=None, choices=['color','complex','non_spatial','shape','spatial','texture'], help="only needed when version is compdiff")
    parser.add_argument('--only_big', action='store_true')
    parser.add_argument('--time_weighting', type=str, default=None)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--test_mode', type=str, default=None)
    parser.add_argument('--domain', type=str, default='photo')
    parser.add_argument('--save_noise', action='store_true')
    parser.add_argument('--geneval_cfg', type=float, default = 9.0)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--sd3_resize', action='store_true')
    parser.add_argument('--arnas_save_noise', action='store_true')
    parser.add_argument('--use_normed_classifier', action='store_true')
    parser.add_argument('--geneval_version', type=str, default=None, choices=['1.5', '2.0', '3-m'])
    parser.add_argument('--geneval_filter', type=str, default=None, choices=['True', 'False'])
    parser.add_argument('--use_euler', action='store_true')

    args, unknown = parser.parse_known_args()

    if 'geneval' in args.task:
        # print("here?")
        # args = parser.parse_args()
        if args.geneval_version == None or args.geneval_filter == None:
            print(args)
            raise ValueError("geneval version and filter must be specified")
        args.geneval_filter = str2bool(args.geneval_filter)

    else:
        args.geneval_cfg = None
        args.geneval_version = None
        args.geneval_filter = None

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(args.cuda_device)

    if '3' not in args.version and args.sd3_resize:
        raise ValueError("Only version 3 can be resized")

    if args.lora_dir:
        if 'mixed' in args.lora_dir:
            lora_type = 'mixed'
        elif 'LONGER' in args.lora_dir:
            lora_type = 'vanilla_LONGER'
        elif 'randneg' in args.lora_dir:
            lora_type = 'randneg'
        elif 'hardimgneg' in args.lora_dir:
            lora_type = 'hardimgneg'
        elif 'hardneg1.0' in args.lora_dir:
            lora_type = "hard_neg1.0"
        elif 'vanilla_coco' in args.lora_dir:
            lora_type = "vanilla_coco"
        elif "unhinged" in args.lora_dir:
            lora_type = "unhinged_hard_neg"
        elif "vanilla" in args.lora_dir:
            lora_type = "vanilla"
        elif "relativistic" in args.lora_dir:
            lora_type = "relativistic"
        elif "inferencelike" in args.lora_dir:
            lora_type = "inferencelike"

    # args.run_id = f'{args.task}_diffusion_itm_{args.version}{'_'+args.comp_subset if args.version == 'compdiff' and args.comp_subset != None else ''}_seed{args.seed}_steps{args.sampling_steps}_subset{args.subset}{args.targets}_img_retrieval{args.img_retrieval}_{"lora_" + lora_type if args.lora_dir else ""}_gray{args.gray_baseline}{"_no_t5" if args.encoder_drop else ""}'
    args.outdir = f'./results/{args.version}/{args.seed}'
    os.makedirs(args.outdir, exist_ok=True)
    args.run_id = f'{"use_euler_" if "3" in args.version and args.use_euler else ""}{"img_retrievalTrue_" if args.img_retrieval else ""}{f"test_mode_{args.test_mode}_" if args.test_mode != None else ""}{args.task if "geneval" not in args.task else f"{args.task}_sdversion{args.geneval_version}_cfg{args.geneval_cfg}_filter{args.geneval_filter}"}_version_{"sd3resize512_" if args.sd3_resize else ""}{args.version}{"_"+args.comp_subset if args.version == "compdiff" and args.comp_subset != None else ""}_batchsize{args.batchsize}_seed{args.seed}_guidance_scale{args.guidance_scale}_steps{args.sampling_steps}{"_subset" if args.subset else ""}{args.targets}{"_lora_" + lora_type if args.lora_dir else ""}{"_gray" if args.gray_baseline else ""}{"_no_t5" if args.encoder_drop else ""}{"_big" if args.only_big else ""}{f"_time_weighting_{args.time_weighting}" if args.time_weighting != None else ""}{f"_domain_{args.domain}" if "geneval" in args.task and args.domain != "photo" else ""}{"used_normed_classfier" if args.use_normed_classifier else "square_root_classifier"}'

    main(args)