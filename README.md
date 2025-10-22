# Self-Bench


This project provides a framework for evaluating diffusion models on a wide range of image-text matching tasks. It supports multiple Stable Diffusion versions and various benchmark datasets.

## Supported Models

The framework supports the following model versions:
- `1.5`: Stable Diffusion v1.5
- `2.0`: Stable Diffusion v2.0
- `3-m`: Stable Diffusion 3 Medium
- `3-lt`: Stable Diffusion 3 Large Turbo (distilled model)
- `flux`: Flux model

## Available Tasks

The framework supports numerous benchmark tasks including:

### Compositional Tasks
- `cola_multi`: Compositional Language tasks
- `vg_relation`: Visual Genome relations
- `vg_attribution`: Visual Genome attributes
- `coco_order`: COCO ordering tasks
- `winoground`: Winoground benchmark
- `flickr30k`: Flickr30k dataset
- `flickr30k_text`: Flickr30k text-only evaluation
- `imagenet`: ImageNet evaluation
- `clevr`: CLEVR dataset
- `pets`: Oxford Pets dataset

### Specialized Tasks
- `mmbias`: Multimodal bias evaluation
- `genderbias`: Gender bias evaluation
- `eqbench`: Equality benchmark
- `vismin`: Visual minority evaluation

### Geneval Tasks
- **Color Tasks**
  - `geneval_color`: Basic color understanding
  - `geneval_color_attr`: Color attribution
- **Position Tasks**
  - `geneval_position`: Spatial understanding
- **Counting Tasks**
  - `geneval_counting`: Object counting
- **Object Tasks**
  - `geneval_single`: Single object understanding
  - `geneval_two`: Two object understanding
  - `geneval_two_subset`: Subset of two object tasks

## Usage

### Basic Usage

```bash
python diffusion_itm.py --task TASK_NAME --version MODEL_VERSION
```

### Common Parameters

- `--task`: Specify the benchmark task (required)
- `--version`: Model version to use (required)
- `--batchsize`: Batch size (default: 64)
- `--sampling_steps`: Number of sampling steps (default: 30)
- `--guidance_scale`: Guidance scale for generation (default: 0.0)
- `--img_retrieval`: Enable image retrieval mode
- `--encoder_drop`: Drop encoder for certain tasks
- `--save`: Save results
- `--wandb`: Enable Weights & Biases logging

### Examples

1. Basic evaluation with SD 3 Medium:
```bash
python diffusion_itm.py --task winoground --version 3-m
```

2. Compositional evaluation with specific subset:
```bash
python diffusion_itm.py --task cola_multi --version compdiff --comp_subset color
```

3. Image retrieval mode:
```bash
python diffusion_itm.py --task flickr30k --version 2.0 --img_retrieval
```

4. Full evaluation with saving:
```bash
python diffusion_itm.py --task clevr --version 3-m --save --save_results --wandb --batchsize 16
```

### Dataset Configuration

When running experiments with Self-bench datasets, you need to specify:
1. The model version (`--geneval_version`): Choose from "1.5", "2.0", "3-m", or "flux"
2. The CFG value (`--geneval_cfg`): Default is 9.0
3. The filter flag (`--geneval_filter`): Set to "True" or "False"

Example command for running a Geneval task:
```bash
python diffusion_itm.py --task geneval_color --version 2.0 --geneval_version 2.0 --geneval_cfg 9.0 --geneval_filter True
```

### Dataset Structure
The dataset should be organized as follows:
```
dataset_root/
├── 9.0/  # CFG value
│   ├── stable-diffusion-v1-5/
│   ├── stable-diffusion-2-base/
│   └── stable-diffusion-3-medium-diffusers/
├── prompts/
│   └── zero_shot_prompts.json
└── filter/
    └── SD-{version}-CFG={cfg}.json
```

## Project Structure

```
.
├── diffusion_itm.py          # Main evaluation script
├── datasets_loading.py       # Dataset loading utilities
├── utils.py                  # Utility functions
├── results/                  # Output directory for results
└── diffusers/               # Modified diffusers library
    └── src/
        └── diffusers/
            └── schedulers/   # Custom schedulers
```

## Requirements

- Python 3.8+
- PyTorch
- diffusers
- transformers
- wandb (optional, for logging)
- accelerate

## Notes

- For SD3 models, you can use `--sd3_resize` to enable 512x512 resizing
- Use `--use_normed_classifier` for normalized classifier evaluation
- For compositional tasks, specify the subset using `--comp_subset` (color, shape, texture, complex, spatial, non_spatial)

## Citation

If you use this code in your research, please cite the original self-bench repository and this work.

## License

This project is licensed under the same terms as the original McGill-NLP/diffusion-itm repository, upon which it builds. Please refer to their license for more details.

## Original Work Attribution

This project builds upon the work from the McGill-NLP/diffusion-itm repository. We gratefully acknowledge their contributions to the field.

Please find the original repository here: [https://github.com/McGill-NLP/diffusion-itm](https://github.com/McGill-NLP/diffusion-itm)
