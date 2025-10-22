import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patheffects as path_effects
import random 
import itertools

df_geneval_short = pd.DataFrame({
    "SD 1.5": [0.41],
    "SD 2.0": [0.46],
    "SD 3-m": [0.70],
})
df_gen_Accruacy = pd.DataFrame({
    "SD 1.5": [(271 + 105 + 219 + 18 + 6 + 98) / (320 + 396 + 376 + 400 + 400 + 320)],
    "SD 2.0": [(271 + 129 + 263 + 36 + 19 + 111) / (320 + 396 + 376 + 400 + 400 + 320)],
    "SD 3-m": [(314 + 306 + 314 + 252 + 113 + 230) / (320 + 396 + 376 + 400 + 400 + 320)],
})

attribute_scores_aro = {
    "CLIP RN50x64": 0.62,
    "CLIP ViT-B/32": 0.61,
    "CLIP ViT-L/14": 0.61,
    "openCLIP ViT-H/14": 0.63,
    "openCLIP ViT-G/14": 0.64,
    "SD 1.5": 0.62,
    # "SD 1.5 (discffusion)": 0.80,
    "SD 2.0": 0.63,
    # "SD 2.0 (discffusion)": 0.86,
    "SD 3-m": 0.56
    # "SD 3-m (discffusion)": 0.68
}

color_scores_clevr = {
    "CLIP RN50x64": 0.96,
    "CLIP ViT-B/32": 0.94,
    "CLIP ViT-L/14": 0.95,
    "openCLIP ViT-H/14": 0.98,
    "openCLIP ViT-G/14": 0.99,
    "SD 1.5": 0.84,
    # "SD 1.5 (discffusion)": 0.67,
    "SD 2.0": 0.85,
    # "SD 2.0 (discffusion)": 0.67,
    "SD 3-m": 0.89,
    # "SD 3-m (discffusion)": 0.67
}

binding_color_scores_clevr = {
    "CLIP RN50x64": 0.53,
    "CLIP ViT-B/32": 0.53,
    "CLIP ViT-L/14": 0.50,
    "openCLIP ViT-H/14": 0.52,
    "openCLIP ViT-G/14": 0.49,
    "SD 1.5": 0.63,
    # "SD 1.5 (discffusion)": 0.67,
    "SD 2.0": 0.81,
    # "SD 2.0 (discffusion)": 0.67,
    "SD 3-m": 0.57,
    # "SD 3-m (discffusion)": 0.67
}


attribute_scores_sugarcrepe = {
    "CLIP RN50x64": 0.69,
    "CLIP ViT-B/32": 0.66,
    "CLIP ViT-L/14": 0.66,
    "openCLIP ViT-H/14": 0.75,
    "openCLIP ViT-G/14": 0.73,
    "SD 1.5": 0.70,
    # "SD 1.5 (discffusion)": 0.80,
    "SD 2.0": 0.75,
    # "SD 2.0 (discffusion)": 0.86,
    "SD 3-m": 0.67,
    # "SD 3-m (discffusion)": 0.68
}

attribute_scores_vismin = {
    "CLIP RN50x64": 0.79,
    "CLIP ViT-B/32": 0.72,
    "CLIP ViT-L/14": 0.74,
    "openCLIP ViT-H/14": 0.82,
    "openCLIP ViT-G/14": 0.85,
    "SD 1.5": 0.73,
    # "SD 1.5 (discffusion)": 0.58,
    "SD 2.0": 0.70,
    # "SD 2.0 (discffusion)": 0.71,
    "SD 3-m": 0.57,
    # "SD 3-m (discffusion)": 0.57
}

attribute_scores_eqbench = {
    "CLIP RN50x64": 0.25,
    "CLIP ViT-B/32": 0.4,
    "CLIP ViT-L/14": 0.35,
    "openCLIP ViT-H/14": 0.4,
    "openCLIP ViT-G/14": 0.4,
    "SD 1.5": 0.4,
    # "SD 1.5 (discffusion)": 0.2,
    "SD 2.0": 0.4,
    # "SD 2.0 (discffusion)": 0.4,
    "SD 3-m": 0.3,
    # "SD 3-m (discffusion)": 0.3
}

attribute_scores_mmvp = {
    "CLIP RN50x64": 0.67,
    "CLIP ViT-B/32": 0.53,
    "CLIP ViT-L/14": 0.33,
    "openCLIP ViT-H/14": 0.6,
    "openCLIP ViT-G/14": 0.8,
    "SD 1.5": 0.47,
    # "SD 1.5 (discffusion)": 0.4,
    "SD 2.0": 0.73,
    # "SD 2.0 (discffusion)": 0.67,
    "SD 3-m": 0.73,
    # "SD 3-m (discffusion)": 0.73
}

color_correct_scores_first_block = {
    "CLIP RN50x64": 0.94,
    "CLIP ViT-B/32": 0.94,
    "CLIP ViT-L/14": 0.94,
    "openCLIP ViT-H/14": 0.97,
    "openCLIP ViT-G/14": 0.97,
    "SD 1.5": 0.98,
    # "SD 1.5 (discffusion)": 0.77,
    "SD 2.0": 0.91,
    # "SD 2.0 (discffusion)": 0.91,
    "SD 3-m": 0.88,
    # "SD 3-m (discffusion)": 0.88
}
color_correct_scores_second_block = {
    "CLIP RN50x64": 0.95,
    "CLIP ViT-B/32": 0.94,
    "CLIP ViT-L/14": 0.93,
    "openCLIP ViT-H/14": 0.97,
    "openCLIP ViT-G/14": 0.98,
    "SD 2.0": 0.98,
    # "SD 2.0 (discffusion)": 0.97,
    "SD 1.5": 0.89,
    # "SD 1.5 (discffusion)": 0.61,
    "SD 3-m": 0.90,
    # "SD 3-m (discffusion)": 0.90
}
color_correct_scores_third_block = {
    "CLIP RN50x64": 0.92,
    "CLIP ViT-B/32": 0.91,
    "CLIP ViT-L/14": 0.91,
    "openCLIP ViT-H/14": 0.96,
    "openCLIP ViT-G/14": 0.95,
    "SD 3-m": 0.98,
    "SD 1.5": 0.82,
    "SD 2.0": 0.87,
}
color_attribute_scores_first_block = {
    "CLIP RN50x64": 0.28,
    "CLIP ViT-B/32": 0.22,
    "CLIP ViT-L/14": 0.17,
    "openCLIP ViT-H/14": 0.5,
    "openCLIP ViT-G/14": 0.28,
    "SD 1.5": 0.83,
    "SD 2.0": 0.5,
    "SD 3-m": 0.56,
}
color_attribute_scores_second_block = {
    "CLIP RN50x64": 0.47,
    "CLIP ViT-B/32": 0.47,
    "CLIP ViT-L/14": 0.44,
    "openCLIP ViT-H/14": 0.53,
    "openCLIP ViT-G/14": 0.69,
    "SD 2.0": 0.88,
    "SD 1.5": 0.42,
    "SD 3-m": 0.53,
}
color_attribute_scores_third_block = {
    "CLIP RN50x64": 0.4,
    "CLIP ViT-B/32": 0.43,
    "CLIP ViT-L/14": 0.36,
    "openCLIP ViT-H/14": 0.49,
    "openCLIP ViT-G/14": 0.55,
    "SD 3-m": 0.95,
    "SD 1.5": 0.58,
    "SD 2.0": 0.63,
}

object_scores_winoground = {
    "CLIP RN50x64": 0.32,
    "CLIP ViT-B/32": 0.36,
    "CLIP ViT-L/14": 0.28,
    "openCLIP ViT-H/14": 0.40,
    "openCLIP ViT-G/14": 0.36,
    "SD 1.5": 0.33,
    "SD 2.0": 0.39,
    "SD 3-m": 0.33,}

single_object_correct_first_block = {
    "CLIP RN50x64": 0.99,
    "CLIP ViT-B/32": 0.99,
    "CLIP ViT-L/14": 0.99,
    "openCLIP ViT-H/14": 1.0,
    "openCLIP ViT-G/14": 0.99,
    "SD 1.5": 1.0,
    "SD 2.0": 0.99,
    "SD 3-m": 0.81,
}
single_object_correct_second_block = {
    "CLIP RN50x64": 1.0,
    "CLIP ViT-B/32": 1.0,
    "CLIP ViT-L/14": 1.0,
    "openCLIP ViT-H/14": 1.0,
    "openCLIP ViT-G/14": 1.0,
    "SD 2.0": 1.0,
    "SD 1.5": 0.99,
    "SD 3-m": 0.87,
}
single_object_correct_third_block = {
    "CLIP RN50x64": 0.99,
    "CLIP ViT-B/32": 1.0,
    "CLIP ViT-L/14": 0.99,
    "openCLIP ViT-H/14": 1.0,
    "openCLIP ViT-G/14": 1.0,
    "SD 3-m": 1.0,
    "SD 1.5": 1.0,
    "SD 2.0": 0.99,
}
two_objects_correct_first_block = {
    "CLIP RN50x64": 0.85,
    "CLIP ViT-B/32": 0.87,
    "CLIP ViT-L/14": 0.95,
    "openCLIP ViT-H/14": 0.95,
    "openCLIP ViT-G/14": 0.94,
    "SD 1.5": 0.90,
    "SD 2.0": 0.85,
    "SD 3-m": 0.57,
}

# object
two_objects_correct_second_block = {
    "CLIP RN50x64": 0.91,
    "CLIP ViT-B/32": 0.85,
    "CLIP ViT-L/14": 0.93,
    "openCLIP ViT-H/14": 0.99,
    "openCLIP ViT-G/14": 0.98,
    "SD 2.0": 0.98,
    "SD 1.5": 0.90,
    "SD 3-m": 0.63,
}
two_objects_correct_third_block = {
    "CLIP RN50x64": 0.91,
    "CLIP ViT-B/32": 0.89,
    "CLIP ViT-L/14": 0.98,
    "openCLIP ViT-H/14": 0.97,
    "openCLIP ViT-G/14": 0.98,
    "SD 3-m": 0.98,
    "SD 1.5": 0.88,
    "SD 2.0": 0.92,
}
object_scores_vismin = {
    "CLIP RN50x64": 0.89,
    "CLIP ViT-B/32": 0.80,
    "CLIP ViT-L/14": 0.87,
    "openCLIP ViT-H/14": 0.91,
    "openCLIP ViT-G/14": 0.90,
    "SD 1.5": 0.79,
    "SD 2.0": 0.80,
    "SD 3-m": 0.46,
}
object_scores_sugarcrepe = {
    "CLIP RN50x64": 0.88,
    "CLIP ViT-B/32": 0.84,
    "CLIP ViT-L/14": 0.86,
    "openCLIP ViT-H/14": 0.92,
    "openCLIP ViT-G/14": 0.92,
    "SD 1.5": 0.85,
    "SD 2.0": 0.87,
    "SD 3-m": 0.72,
}
# spatial 
whatsupA_scores = {
    "CLIP RN50x64": 0.34,
    "CLIP ViT-B/32": 0.31,
    "CLIP ViT-L/14": 0.27,
    "openCLIP ViT-H/14": 0.26,
    "openCLIP ViT-G/14": 0.30,
    "SD 1.5": 0.28,
    "SD 2.0": 0.27,
    "SD 3-m": 0.28,
}
whatsupB_scores = {
    "CLIP RN50x64": 0.24,
    "CLIP ViT-B/32": 0.31,
    "CLIP ViT-L/14": 0.26,
    "openCLIP ViT-H/14": 0.27,
    "openCLIP ViT-G/14": 0.26,
    "SD 1.5": 0.27,
    "SD 2.0": 0.28,
    "SD 3-m": 0.37,
}
coco_spatial_one_scores = {
    "CLIP RN50x64": 0.45,
    "CLIP ViT-B/32": 0.44,
    "CLIP ViT-L/14": 0.49,
    "openCLIP ViT-H/14": 0.45,
    "openCLIP ViT-G/14": 0.48,
    "SD 1.5": 0.48,
    "SD 2.0": 0.42,
    "SD 3-m": 0.54,
}
gqa_spatial_one_scores = {
    "CLIP RN50x64": 0.46,
    "CLIP ViT-B/32": 0.47,
    "CLIP ViT-L/14": 0.46,
    "openCLIP ViT-H/14": 0.46,
    "openCLIP ViT-G/14": 0.48,
    "SD 1.5": 0.49,
    "SD 2.0": 0.49,
    "SD 3-m": 0.54,
}
coco_spatial_two_scores = {
    "CLIP RN50x64": 0.50,
    "CLIP ViT-B/32": 0.51,
    "CLIP ViT-L/14": 0.50,
    "openCLIP ViT-H/14": 0.53,
    "openCLIP ViT-G/14": 0.45,
    "SD 1.5": 0.52,
    "SD 2.0": 0.47,
    "SD 3-m": 0.55,
}
gqa_spatial_two_scores = {
    "CLIP RN50x64": 0.53,
    "CLIP ViT-B/32": 0.48,
    "CLIP ViT-L/14": 0.48,
    "openCLIP ViT-H/14": 0.55,
    "openCLIP ViT-G/14": 0.47,
    "SD 1.5": 0.47,
    "SD 2.0": 0.53,
    "SD 3-m": 0.56,
}

spec_absolute_spatial_scores = {
    "CLIP RN50x64": 0.12,
    "CLIP ViT-B/32": 0.13,
    "CLIP ViT-L/14": 0.12,
    "openCLIP ViT-H/14": 0.13,
    "openCLIP ViT-G/14": 0.14,
    "SD 1.5": 0.15,
    "SD 2.0": 0.12,
    "SD 3-m": 0.24,
}
spec_relative_spatial_scores = {
    "CLIP RN50x64": 0.29,
    "CLIP ViT-B/32": 0.28,
    "CLIP ViT-L/14": 0.29,
    "openCLIP ViT-H/14": 0.28,
    "openCLIP ViT-G/14": 0.30,
    "SD 1.5": 0.30,
    "SD 2.0": 0.29,
    "SD 3-m": 0.43,
}
eq_kubric_location_scores = {
    "CLIP RN50x64": 0.0,
    "CLIP ViT-B/32": 0.0,
    "CLIP ViT-L/14": 0.0,
    "openCLIP ViT-H/14": 0.0,
    "openCLIP ViT-G/14": 0.0,
    "SD 1.5": 0.15,
    "SD 2.0": 0.15,
    "SD 3-m": 0.05,
}
relation_scores_vismin = {
    "CLIP RN50x64": 0.09,
    "CLIP ViT-B/32": 0.09,
    "CLIP ViT-L/14": 0.09,
    "openCLIP ViT-H/14": 0.09,
    "openCLIP ViT-G/14": 0.08,
    "SD 1.5": 0.19,
    "SD 2.0": 0.13,
    "SD 3-m": 0.44,
}
spatial_scores_mmvp_vlm = {
    "CLIP RN50x64": 0.07,
    "CLIP ViT-B/32": 0.07,
    "CLIP ViT-L/14": 0.2,
    "openCLIP ViT-H/14": 0.2,
    "openCLIP ViT-G/14": 0.2,
    "SD 1.5": 0.27,
    "SD 2.0": 0.33,
    "SD 3-m": 0.47,
}
mmvp_spatial_scores = {
    "CLIP RN50x64": 0.07,
    "CLIP ViT-B/32": 0.07,
    "CLIP ViT-L/14": 0.2,
    "openCLIP ViT-H/14": 0.2,
    "openCLIP ViT-G/14": 0.2,
    "SD 1.5": 0.27,
    "SD 2.0": 0.33,
    "SD 3-m": 0.47,
}

spatial_scores_clevr = {
    "CLIP RN50x64": 0.51,  
    "CLIP ViT-B/32": 0.54,
    "CLIP ViT-L/14": 0.51,
    "openCLIP ViT-H/14": 0.52,
    "openCLIP ViT-G/14": 0.50,
    "SD 1.5": 0.49,
    "SD 2.0": 0.51,
    "SD 3-m": 0.59,
}
# counting
counting_scores_vismin = {
    "CLIP RN50x64": 0.37,
    "CLIP ViT-B/32": 0.31,
    "CLIP ViT-L/14": 0.37,
    "openCLIP ViT-H/14": 0.65,
    "openCLIP ViT-G/14": 0.65,
    "SD 1.5": 0.36,
    "SD 2.0": 0.39,
    "SD 3-m": 0.26,
}
count_scores_spec = {
    "CLIP RN50x64": 0.30,
    "CLIP ViT-B/32": 0.25,
    "CLIP ViT-L/14": 0.29,
    "openCLIP ViT-H/14": 0.42,
    "openCLIP ViT-G/14": 0.47,
    "SD 1.5": 0.20,
    "SD 2.0": 0.23,
    "SD 3-m": 0.18,
}
eq_kubric_counting_scores = {
    "CLIP RN50x64": 0.25,
    "CLIP ViT-B/32": 0.3,
    "CLIP ViT-L/14": 0.35,
    "openCLIP ViT-H/14": 0.4,
    "openCLIP ViT-G/14": 0.5,
    "SD 1.5": 0.25,
    "SD 2.0": 0.15,
    "SD 3-m": 0.05,
}
counting_scores_vismin = {
    "CLIP RN50x64": 0.37,
    "CLIP ViT-B/32": 0.31,
    "CLIP ViT-L/14": 0.37,
    "openCLIP ViT-H/14": 0.65,
    "openCLIP ViT-G/14": 0.65,
    "SD 1.5": 0.36,
    "SD 2.0": 0.39,
    "SD 3-m": 0.26,
}
quantity_scores_mmvp_vlm = {
    "CLIP RN50x64": 0.07,
    "CLIP ViT-B/32": 0.13,
    "CLIP ViT-L/14": 0.0,
    "openCLIP ViT-H/14": 0.4,
    "openCLIP ViT-G/14": 0.6,
    "SD 1.5": 0.33,
    "SD 2.0": 0.13,
    "SD 3-m": 0.13,
}

position_correct_scores_first_block = {
    "CLIP RN50x64": 0.17,
    "CLIP ViT-B/32": 0.67,
    "CLIP ViT-L/14": 0.5,
    "openCLIP ViT-H/14": 0.33,
    "openCLIP ViT-G/14": 0.67,
    "SD 1.5": 0.67,
    "SD 2.0": 0.67,
    "SD 3-m": 0.33,
}
position_correct_scores_second_block = {
    "CLIP RN50x64": 0.26,
    "CLIP ViT-B/32": 0.16,
    "CLIP ViT-L/14": 0.26,
    "openCLIP ViT-H/14": 0.37,
    "openCLIP ViT-G/14": 0.42,
    "SD 2.0": 0.84,
    "SD 1.5": 0.26,
    "SD 3-m": 0.42,
}
position_correct_scores_third_block = {
    "CLIP RN50x64": 0.27,
    "CLIP ViT-B/32": 0.31,
    "CLIP ViT-L/14": 0.30,
    "openCLIP ViT-H/14": 0.33,
    "openCLIP ViT-G/14": 0.35,
    "SD 3-m": 0.88,
    "SD 1.5": 0.30,
    "SD 2.0": 0.34,
}


counting_correct_scores_first_block = {
    "CLIP RN50x64": 0.67,
    "CLIP ViT-B/32": 0.63,
    "CLIP ViT-L/14": 0.63,
    "openCLIP ViT-H/14": 0.85,
    "openCLIP ViT-G/14": 0.87,
    "SD 1.5": 0.76,
    "SD 2.0": 0.49,
    "SD 3-m": 0.53,
}
counting_correct_scores_second_block = {
    "CLIP RN50x64": 0.77,
    "CLIP ViT-B/32": 0.60,
    "CLIP ViT-L/14": 0.69,
    "openCLIP ViT-H/14": 0.93,
    "openCLIP ViT-G/14": 0.95,
    "SD 2.0": 0.95,
    "SD 1.5": 0.59,
    "SD 3-m": 0.57,
}
counting_correct_scores_third_block = {
    "CLIP RN50x64": 0.70,
    "CLIP ViT-B/32": 0.65,
    "CLIP ViT-L/14": 0.75,
    "openCLIP ViT-H/14": 0.97,
    "openCLIP ViT-G/14": 0.96,
    "SD 3-m": 0.91,
    "SD 1.5": 0.60,
    "SD 2.0": 0.61,
}

size_correct_scores_first_block = {}

object_scores_cola = {
    "CLIP RN50x64": 0.35,
    "CLIP ViT-B/32": 0.34,
    "CLIP ViT-L/14": 0.38,
    "openCLIP ViT-H/14": 0.44,
    "openCLIP ViT-G/14": 0.43,
    "SD 1.5": 0.47,
    "SD 2.0": 0.50,
    "SD 3-m": 0.43,
}

binding_shapes_scores_clevr = {
    "CLIP RN50x64": 0.79,
    "CLIP ViT-B/32": 0.94,
    "CLIP ViT-L/14": 0.86,
    "openCLIP ViT-H/14": 1.0,
    "openCLIP ViT-G/14": 1.0,
    "SD 1.5": 0.84,
    "SD 2.0": 0.85,
    "SD 3-m": 0.63,
}

binding_color_shape_scores_clevr = {
    "CLIP RN50x64": 0.53,
    "CLIP ViT-B/32": 0.50,
    "CLIP ViT-L/14": 0.49,
    "openCLIP ViT-H/14": 0.50,
    "openCLIP ViT-G/14": 0.49,
    "SD 1.5": 0.55,
    "SD 2.0": 0.57,
    "SD 3-m": 0.51,
}

binding_shape_color_scores_clevr = {
"CLIP RN50x64": 0.51,
"CLIP ViT-B/32": 0.51,
"CLIP ViT-L/14": 0.49,
"openCLIP ViT-H/14": 0.50,
"openCLIP ViT-G/14": 0.49,
"SD 1.5": 0.57,
"SD 2.0": 0.59,
"SD 3-m": 0.51,
}

df_attribute = pd.DataFrame({  
    'ARO': attribute_scores_aro,
    'Sugarcrepe': attribute_scores_sugarcrepe,
    'Vismin': attribute_scores_vismin,
    'EQBench': attribute_scores_eqbench,
    'MMVP': attribute_scores_mmvp,
    'Clevr (Color)': color_scores_clevr,
    'Clevr (Color Binding)': binding_color_scores_clevr,
    'Clevr (Shape)': binding_shapes_scores_clevr,
    'Clevr (Col&Sha)': binding_color_shape_scores_clevr,
    'Clevr (Sha&Col)': binding_shape_color_scores_clevr,
    # 'Colors (Block 1)': color_correct_scores_first_block,
    # 'Colors (Block 2)': color_correct_scores_second_block,
    # 'Colors (Block 3)': color_correct_scores_third_block,
    # 'Color_Attribution (Block 1)': color_attribute_scores_first_block,
    # 'Color_Attribution (Block 2)': color_attribute_scores_second_block,
    # 'Color_Attribution (Block 3)': color_attribute_scores_third_block,
})

# Counting Scores
df_counting = pd.DataFrame({
    'Vismin': counting_scores_vismin,
    'Spec': count_scores_spec,
    'EqBench': eq_kubric_counting_scores,
    'MMVP': quantity_scores_mmvp_vlm,
    # 'Counting (Block 1)': counting_correct_scores_first_block,
    # 'Counting (Block 2)': counting_correct_scores_second_block,
    # 'Counting (Block 3)': counting_correct_scores_third_block
})


# Object Scores
df_object = pd.DataFrame({
    'Vismin': object_scores_vismin,
    'Sugarcrepe': object_scores_sugarcrepe,
    'COLA': object_scores_cola,
    'Winoground': object_scores_winoground,
    # 'Single Object (Block 1)': single_object_correct_first_block,
    # 'Single Object (Block 2)': single_object_correct_second_block,
    # 'Single Object (Block 3)': single_object_correct_third_block,
    # 'Two Objects (Block 1)': two_objects_correct_first_block,
    # 'Two Objects (Block 2)': two_objects_correct_second_block,
    # 'Two Objects (Block 3)': two_objects_correct_third_block
})

orientation_scores_mmvp = {
    "CLIP RN50x64": 0.2,
    "CLIP ViT-B/32": 0.07,
    "CLIP ViT-L/14": 0.0,   
    "openCLIP ViT-H/14": 0.27,
    "openCLIP ViT-G/14": 0.33,
    "SD 1.5": 0.07,
    "SD 2.0": 0.13,
    "SD 3-m": 0.33,
}

perspective_scores_mmvp = {
    "CLIP RN50x64": 0.27,
    "CLIP ViT-B/32": 0.2,
    "CLIP ViT-L/14": 0.13,
    "openCLIP ViT-H/14": 0.4,
    "openCLIP ViT-G/14": 0.27,
    "SD 1.5": 0.53,
    "SD 2.0": 0.6,
    "SD 3-m": 0.47,
}

# Spatial Scores
df_spatial = pd.DataFrame({
    'Vismin': relation_scores_vismin,
    'EqBench': eq_kubric_location_scores,
    'MMVP (Orientation)': orientation_scores_mmvp,
    'MMVP (Spatial)': spatial_scores_mmvp_vlm,
    'MMVP (Perspective)': perspective_scores_mmvp,
    'Clevr': spatial_scores_clevr,
    'WhatsupA': whatsupA_scores,
    'WhatsupB': whatsupB_scores,
    'COCO (One)': coco_spatial_one_scores,
    'COCO (Two)': coco_spatial_two_scores,
    'GQA (One)': gqa_spatial_one_scores,
    'GQA (Two)': gqa_spatial_two_scores,
    'Spec (Absolute)': spec_absolute_spatial_scores,
    'Spec (Relative)': spec_relative_spatial_scores,


})

relation_scores_aro = {
    "CLIP RN50x64": 0.51,
    "CLIP ViT-B/32": 0.51,
    "CLIP ViT-L/14": 0.53,
    "openCLIP ViT-H/14": 0.50,
    "openCLIP ViT-G/14": 0.51,
    "SD 1.5": 0.52,
    "SD 2.0": 0.50,
    "SD 3-m": 0.48,
}

order_scores_coco = {
    "CLIP RN50x64": 0.52,
    "CLIP ViT-B/32": 0.48,
    "CLIP ViT-L/14": 0.47,
    "openCLIP ViT-H/14": 0.50,
    "openCLIP ViT-G/14": 0.51,
    "SD 1.5": 0.23,
    "SD 2.0": 0.25,
    "SD 3-m": 0,
}

order_scores_flickr = {
    "CLIP RN50x64": 0.59,
    "CLIP ViT-B/32": 0.59,
    "CLIP ViT-L/14": 0.56,
    "openCLIP ViT-H/14": 0.40,
    "openCLIP ViT-G/14": 0.38,
    "SD 1.5": 0.32,
    "SD 2.0": 0.34,
    "SD 3-m": 0.18,
}

relation_scores_sugarcrepe = {
    "CLIP RN50x64": 0.71,
    "CLIP ViT-B/32": 0.69,
    "CLIP ViT-L/14": 0.65,
    "openCLIP ViT-H/14": 0.72,
    "openCLIP ViT-G/14": 0.73,
    "SD 1.5": 0.66,
    "SD 2.0": 0.68,
    "SD 3-m": 0.58,
}

relation_scores_winoground = {
    "CLIP RN50x64": 0.21,
    "CLIP ViT-B/32": 0.22,
    "CLIP ViT-L/14": 0.25,
    "openCLIP ViT-H/14": 0.27,
    "openCLIP ViT-G/14": 0.27,
    "SD 1.5": 0.33,
    "SD 2.0": 0.30,
    "SD 3-m": 0.33,
}

relation_scores_winoground_both = {
    "CLIP RN50x64": 0.46,
    "CLIP ViT-B/32": 0.81,
    "CLIP ViT-L/14": 0.58,
    "openCLIP ViT-H/14": 0.58,
    "openCLIP ViT-G/14": 0.54,
    "SD 1.5": 0.54,
    "SD 2.0": 0.54,
    "SD 3-m": 0.42,
}

state_scores_mmvp = {
    "CLIP RN50x64": 0.27,
    "CLIP ViT-B/32": 0.27,
    "CLIP ViT-L/14": 0.33,
    "openCLIP ViT-H/14": 0.27,
    "openCLIP ViT-G/14": 0.6,
    "SD 1.5": 0.4,
    "SD 2.0": 0.47,
    "SD 3-m": 0.53,
}

structural_scores_mmvp = {
    "CLIP RN50x64": 0.07,
    "CLIP ViT-B/32": 0.4,
    "CLIP ViT-L/14": 0.27,
    "openCLIP ViT-H/14": 0.53,
    "openCLIP ViT-G/14": 0.6,
    "SD 1.5": 0.47,
    "SD 2.0": 0.27,
    "SD 3-m": 0.27,
}

df_complex = pd.DataFrame({
    "Aro (Relation)": relation_scores_aro,
    "COCO Order": order_scores_coco,
    "Flickr Order": order_scores_flickr,
    "Sugarcrepe": relation_scores_sugarcrepe,
    "Winoground (Relation)": relation_scores_winoground,
    "Winoground (Both)" : relation_scores_winoground_both,
    "MMVP (State)": state_scores_mmvp,
    "MMVP (Structural Character)": structural_scores_mmvp,
})

youcook2_scores_eqbench = {
    "CLIP RN50x64": 0.6,
    "CLIP ViT-B/32": 0.55,
    "CLIP ViT-L/14": 0.4,
    "openCLIP ViT-H/14": 0.7,
    "openCLIP ViT-G/14": 0.8,
    "SD 1.5": 0.5,
    "SD 2.0": 0.55,
    "SD 3-m": 0.4,
}
gebc_scores_eqbench = {
    "CLIP RN50x64": 0.1,
    "CLIP ViT-B/32": 0.25,
    "CLIP ViT-L/14": 0.15,
    "openCLIP ViT-H/14": 0.2,
    "openCLIP ViT-G/14": 0.25,
    "SD 1.5": 0.1,
    "SD 2.0": 0.15,
    "SD 3-m": 0.1,
}
ag_scores_eqbench = {
    "CLIP RN50x64": 0.15,
    "CLIP ViT-B/32": 0.1,
    "CLIP ViT-L/14": 0.15,
    "openCLIP ViT-H/14": 0.2,
    "openCLIP ViT-G/14": 0.25,
    "SD 1.5": 0.3,
    "SD 2.0": 0.15,
    "SD 3-m": 0.05,
}
df_action = pd.DataFrame({
    
    "EQBench (GEBC)": gebc_scores_eqbench,
    "EQBench (AG)": ag_scores_eqbench,
    "EQBench (YouCook2)": youcook2_scores_eqbench,
})


spec_absolute_size_scores = {
    "CLIP RN50x64": 0.35,
    "CLIP ViT-B/32": 0.42,
    "CLIP ViT-L/14": 0.37,
    "openCLIP ViT-H/14": 0.41,
    "openCLIP ViT-G/14": 0.37,
    "SD 1.5": 0.39,
    "SD 2.0": 0.43,
    "SD 3-m": 0.37,
}
spec_relative_size_scores = {
    "CLIP RN50x64": 0.31,
    "CLIP ViT-B/32": 0.34,
    "CLIP ViT-L/14": 0.32,
    "openCLIP ViT-H/14": 0.33,
    "openCLIP ViT-G/14": 0.32,
    "SD 1.5": 0.34,
    "SD 2.0": 0.33,
    "SD 3-m": 0.34,
}

size_scores_clevr = {
    "CLIP RN50x64": 0.49,
    "CLIP ViT-B/32": 0.53,
    "CLIP ViT-L/14": 0.51,
    "openCLIP ViT-H/14": 0.48,
    "openCLIP ViT-G/14": 0.50,
    "SD 1.5": 0.53,
    "SD 2.0": 0.49,
    "SD 3-m": 0.41,}

df_size = pd.DataFrame({
    "SPEC (Absolute)": spec_absolute_size_scores,
    "SPEC (Relative)": spec_relative_size_scores,
    "Clevr": size_scores_clevr,
})

presence_scores_spec = {
    "CLIP RN50x64": 0.57,
    "CLIP ViT-B/32": 0.58,
    "CLIP ViT-L/14": 0.58,
    "openCLIP ViT-H/14": 0.57,
    "openCLIP ViT-G/14": 0.55,
    "SD 1.5": 0.56,
    "SD 2.0": 0.55,
    "SD 3-m": 0.52,
}

presence_scores_mmvp = {
    "CLIP RN50x64": 0.27,
    "CLIP ViT-B/32": 0.07,
    "CLIP ViT-L/14": 0.07,
    "openCLIP ViT-H/14": 0.27,
    "openCLIP ViT-G/14": 0.13,
    "SD 1.5": 0.2,
    "SD 2.0": 0.2,
    "SD 3-m": 0.0,
}

df_presence = pd.DataFrame({
    "SPEC": presence_scores_spec,
    "MMVP": presence_scores_mmvp,
})

text_scores_mmvp = {
    "CLIP RN50x64": 0.4,
    "CLIP ViT-B/32": 0.33,
    "CLIP ViT-L/14": 0.27,
    "openCLIP ViT-H/14": 0.13,
    "openCLIP ViT-G/14": 0.27,
    "SD 1.5": 0.33,
    "SD 2.0": 0.27,
    "SD 3-m": 0.0,

}

df_text = pd.DataFrame({
    "MMVP": text_scores_mmvp,
})

# df_discrimination_Accuracy = pd.concat(df_object, df_attribute,df_spatial,df_counting)
df_discrimination_Accuracy = pd.concat([df_object, df_attribute,df_spatial,df_counting,df_complex,df_action,df_size,df_presence,df_text], axis=1)
df_discrimination_Accuracy = df_discrimination_Accuracy.mean(axis=1)
df_discrimination_Accuracy = pd.DataFrame({"Discrimination Accuracy": df_discrimination_Accuracy})
df_discrimination_Accuracy = df_discrimination_Accuracy.T
# print(df_discrimination_Accuracy)
# exit(0)

# Define x-axis labels
x = ["SD 1.5", "SD 2.0", "SD 3-m"]

# Extract the values (using .iloc[0] because each DataFrame has one row)
geneval_values = [df_geneval_short[col].iloc[0] for col in x]
ours_values = [df_gen_Accruacy[col].iloc[0] for col in x]
from matplotlib.ticker import FormatStrFormatter

fig, ax = plt.subplots(figsize=(3, 3))

bar_width = 0.5

 # color blind pallete
color_pallete = sns.color_palette("colorblind", 3)
# x = 
# Create the bars for each category:
bars_geneval = ax.bar(np.arange(len(x)), geneval_values, bar_width,
                      color=[color_pallete[i] for i in range(3)],
                      label='Geneval', zorder=10)

x_positions = np.arange(3)  # For three categories: 0, 1, 2

# Set these as tick positions and then assign your category names as labels
ax.set_xticks(x_positions)
ax.set_xticklabels(["SD 1.5", "SD 2.0", "SD 3-m"], fontsize=15)
plt.tick_params(axis='x', which='both', bottom=False, top=False)
# plt.yticks(np.linspace(0, 0.8, 5),fontsize=15)  # Change the Y-axis tick label size
plt.yticks([])
# Y-axis formatting and labels
ax.set_ylabel('Accuracy', fontsize=15)
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# plt.yticks(np.linspace(0, 1, 4), fontsize=8)

# Customize spines (if needed)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(1)
ax.spines['bottom'].set_zorder(1000)

# Legend and grid
# plt.legend(fontsize='xx-small', loc='upper right')
plt.grid(False)

plt.tight_layout()
outdir = 'plots/teaser_geneval'
os.makedirs(outdir, exist_ok=True)
plt.savefig(f'{outdir}/gen_accuracy_bar.png', dpi=300, bbox_inches='tight')
# plt.show()


disc_values = [df_discrimination_Accuracy[col].iloc[0] for col in x]

fig, ax = plt.subplots(figsize=(3, 3))

bar_width = 0.5

 # color blind pallete
color_pallete = sns.color_palette("colorblind", 3)
# x = 
# Create the bars for each category:
bars_geneval = ax.bar(np.arange(len(x)), ours_values, bar_width,
                      color=[color_pallete[i] for i in range(3)],
                      label='Geneval', zorder=10)

x_positions = np.arange(3)  # For three categories: 0, 1, 2

# Set these as tick positions and then assign your category names as labels
ax.set_xticks(x_positions)
ax.set_xticklabels(["SD 1.5", "SD 2.0", "SD 3-m"], fontsize=15)
plt.tick_params(axis='x', which='both', bottom=False, top=False)
# plt.yticks(np.linspace(0, 0.8, 5),fontsize=15)  # Change the Y-axis tick label size
plt.yticks([])
ax.set_ylabel('Accuracy', fontsize=15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(1)
ax.spines['bottom'].set_zorder(1000)

# Legend and grid
# plt.legend(fontsize='xx-small', loc='upper right')
plt.grid(False)

plt.tight_layout()

plt.savefig(f'{outdir}/human_accuracy_bar.png', dpi=300, bbox_inches='tight')


fig, ax = plt.subplots(figsize=(3, 3))

bar_width = 0.5

 # color blind pallete
color_pallete = sns.color_palette("colorblind", 3)
# x = 
# Create the bars for each category:
bars_geneval = ax.bar(np.arange(len(x)), disc_values, bar_width,
                      color=[color_pallete[i] for i in range(3)],
                      label='Geneval', zorder=10)

x_positions = np.arange(3)  # For three categories: 0, 1, 2

# Set these as tick positions and then assign your category names as labels
ax.set_xticks(x_positions)
ax.set_xticklabels(["SD 1.5", "SD 2.0", "SD 3-m"], fontsize=15)
plt.tick_params(axis='x', which='both', bottom=False, top=False)
# plt.yticks(np.linspace(0, 0.8, 5),fontsize=15)  # Change the Y-axis tick label size
plt.yticks([])
ax.set_ylabel('Accuracy', fontsize=15)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['left'].set_color('black')
ax.spines['left'].set_linewidth(1)
ax.spines['bottom'].set_visible(True)
ax.spines['bottom'].set_color('black')
ax.spines['bottom'].set_linewidth(1)
ax.spines['bottom'].set_zorder(1000)

# Legend and grid
# plt.legend(fontsize='xx-small', loc='upper right')
plt.grid(False)

plt.tight_layout()

plt.savefig(f'{outdir}/disc_accuracy_bar.png', dpi=300, bbox_inches='tight')
