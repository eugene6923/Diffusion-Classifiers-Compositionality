import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patheffects as path_effects
import random 

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
    "CLIP RN50x64": 0.49,  
    "CLIP ViT-B/32": 0.53,
    "CLIP ViT-L/14": 0.51,
    "openCLIP ViT-H/14": 0.48,
    "openCLIP ViT-G/14": 0.50,
    "SD 1.5": 0.53,
    "SD 2.0": 0.49,
    "SD 3-m": 0.41,
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
    'EqBench': attribute_scores_eqbench,
    'MMVP': attribute_scores_mmvp,
    'Clevr (Color)': color_scores_clevr,
    'Clevr (Color Binding)': binding_color_scores_clevr,
    'Clevr (Shape)': binding_shapes_scores_clevr,
    'Clevr (Color&Shape)': binding_color_shape_scores_clevr,
    'Clevr (Shape&Color)': binding_shape_color_scores_clevr,
    # 'Colors (Block 1)': color_correct_scores_first_block,
    # 'Colors (Block 2)': color_correct_scores_second_block,
    # 'Colors (Block 3)': color_correct_scores_third_block,
    # 'Color_Attribution (Block 1)': color_attribute_scores_first_block,
    # 'Color_Attribution (Block 2)': color_attribute_scores_second_block,
    # 'Color_Attribution (Block 3)': color_attribute_scores_third_block,
})

df_attribute["Attribute"] = df_attribute.mean(axis=1)
df_attribute["Attribute Std"] = df_attribute.std(axis=1)
df_attribute = df_attribute[["Attribute", "Attribute Std"]]  # double brackets keep it as a DataFrame



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

df_counting["Counting"] = df_counting.mean(axis=1)
df_counting["Counting Std"] = df_counting.std(axis=1)
df_counting = df_counting[["Counting", "Counting Std"]]  # double brackets keep it as a DataFrame



# Object Scores
df_object = pd.DataFrame({
    'Vismin': object_scores_vismin,
    'Sugarcrepe': object_scores_sugarcrepe,
    'Winoground': object_scores_winoground,
    'COLA': object_scores_cola,
    # 'Single Object (Block 1)': single_object_correct_first_block,
    # 'Single Object (Block 2)': single_object_correct_second_block,
    # 'Single Object (Block 3)': single_object_correct_third_block,
    # 'Two Objects (Block 1)': two_objects_correct_first_block,
    # 'Two Objects (Block 2)': two_objects_correct_second_block,
    # 'Two Objects (Block 3)': two_objects_correct_third_block
})

df_object["Object"] = df_object.mean(axis=1)
df_object["Object Std"] = df_object.std(axis=1)

df_object = df_object[["Object", "Object Std"]]  # double brackets keep it as a DataFrame


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
    'MMVP (Spatial)': spatial_scores_mmvp_vlm,
    'MMVP (Orientation)': orientation_scores_mmvp,
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
    # 'Spatial (Block 1)': position_correct_scores_first_block,
    # 'Spatial (Block 2)': position_correct_scores_second_block,
    # 'Spatial (Block 3)': position_correct_scores_third_block,

})

df_spatial["Position"] = df_spatial.mean(axis=1)
df_spatial["Position Std"] = df_spatial.std(axis=1)
df_spatial = df_spatial[["Position", "Position Std"]]  # double brackets keep it as a DataFrame


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

df_all = pd.concat([df_object, df_attribute, df_spatial, df_counting], axis=1)



single_self = {
    "SD 1.5": single_object_correct_first_block["SD 1.5"],
    "SD 2.0": single_object_correct_second_block["SD 2.0"],
    "SD 3-m": single_object_correct_third_block["SD 3-m"],
    
}

two_self = {
"SD 1.5": two_objects_correct_first_block["SD 1.5"],   
    "SD 2.0": two_objects_correct_second_block["SD 2.0"],
    "SD 3-m": two_objects_correct_third_block["SD 3-m"],

}

df_object_ours = pd.DataFrame({
    "Single Object": single_self,
    "Two Objects": two_self,
})

df_object_ours["Object"] = df_object_ours.mean(axis=1)
df_object_ours["Object Std"] = df_object_ours.std(axis=1)
df_object_ours = df_object_ours[["Object", "Object Std"]]  # double brackets keep it as a DataFrame

colors_self = {
     "SD 1.5": color_correct_scores_first_block["SD 1.5"],
    "SD 2.0": color_correct_scores_second_block["SD 2.0"],
   "SD 3-m":color_correct_scores_third_block["SD 3-m"]
   }

color_attr_self = {
    "SD 1.5": color_attribute_scores_first_block["SD 1.5"],
    "SD 2.0": color_attribute_scores_second_block["SD 2.0"],
    "SD 3-m": color_attribute_scores_third_block["SD 3-m"],
}

df_attribute_ours = pd.DataFrame({
    "Colors": colors_self,
    "Color Attribution": color_attr_self,
})    

df_attribute_ours["Attribute"] = df_attribute_ours.mean(axis=1)
df_attribute_ours["Attribute Std"] = df_attribute_ours.std(axis=1)
df_attribute_ours = df_attribute_ours[["Attribute", "Attribute Std"]]  # double brackets keep it as a DataFrame


spatial_self ={
    "SD 1.5": position_correct_scores_first_block["SD 1.5"],
    "SD 2.0": position_correct_scores_second_block["SD 2.0"],
    "SD 3-m": position_correct_scores_third_block["SD 3-m"],
}
df_spatial_ours = pd.DataFrame({
    "Position": spatial_self,
    "Position Std": 0,
})

counting_self = {
    "SD 1.5": counting_correct_scores_first_block["SD 1.5"],
    "SD 2.0": counting_correct_scores_second_block["SD 2.0"],
    "SD 3-m": counting_correct_scores_third_block["SD 3-m"],
}
df_counting_ours = pd.DataFrame({
    "Counting": counting_self,
    "Counting Std": 0,
})

df_object_ours["Object Std"] = df_object_ours.std(axis=1)
df_object_ours = df_object_ours[["Object", "Object Std"]]  # double brackets keep it as a DataFrame

df_attribute_ours["Attribute Std"] = df_attribute_ours.std(axis=1)
df_attribute_ours = df_attribute_ours[["Attribute", "Attribute Std"]]  # double brackets keep it as a DataFrame

single_3m = {
    "SD 1.5": single_object_correct_third_block["SD 1.5"],
    "SD 2.0": single_object_correct_third_block["SD 2.0"],
    "SD 3-m": single_object_correct_third_block["SD 3-m"],
}

two_3m = {
    "SD 1.5": two_objects_correct_third_block["SD 1.5"],
    "SD 2.0": two_objects_correct_third_block["SD 2.0"],
    "SD 3-m": two_objects_correct_third_block["SD 3-m"],
}

df_object_ours_3m = pd.DataFrame({
    "Single Object": single_3m,
    "Two Objects": two_3m,
})

df_object_ours_3m["Object"] = df_object_ours_3m.mean(axis=1)
df_object_ours_3m["Object Std"] = df_object_ours_3m.std(axis=1)
df_object_ours_3m = df_object_ours_3m[["Object", "Object Std"]]  # double brackets keep it as a DataFrame

colors_self_3m = {
    "SD 1.5": color_correct_scores_third_block["SD 1.5"],
    "SD 2.0": color_correct_scores_third_block["SD 2.0"],
    "SD 3-m": color_correct_scores_third_block["SD 3-m"],
}

color_attr_self_3m = {
    "SD 1.5": color_attribute_scores_third_block["SD 1.5"],
    "SD 2.0": color_attribute_scores_third_block["SD 2.0"],
    "SD 3-m": color_attribute_scores_third_block["SD 3-m"],
}
df_attribute_ours_3m = pd.DataFrame({
    "Colors": colors_self_3m,
    "Color Attribution": color_attr_self_3m,
})

df_attribute_ours_3m["Attribute"] = df_attribute_ours_3m.mean(axis=1)
df_attribute_ours_3m["Attribute Std"] = df_attribute_ours_3m.std(axis=1)
df_attribute_ours_3m = df_attribute_ours_3m[["Attribute", "Attribute Std"]]  # double brackets keep it as a DataFrame

spatial_self_3m = {
    "SD 1.5": position_correct_scores_third_block["SD 1.5"],
    "SD 2.0": position_correct_scores_third_block["SD 2.0"],
    "SD 3-m": position_correct_scores_third_block["SD 3-m"],
}

df_spatial_ours_3m = pd.DataFrame({
    "Position": spatial_self_3m,
    "Position Std": 0,
})

counting_self_3m = {
    "SD 1.5": counting_correct_scores_third_block["SD 1.5"],
    "SD 2.0": counting_correct_scores_third_block["SD 2.0"],
    "SD 3-m": counting_correct_scores_third_block["SD 3-m"],
}

df_counting_ours_3m = pd.DataFrame({
    "Counting": counting_self_3m,
    "Counting Std": 0,
})

df_all_ours_3m = pd.concat([df_object_ours_3m, df_attribute_ours_3m, df_spatial_ours_3m, df_counting_ours_3m], axis=1)
df_all_ours_3m = df_all_ours_3m.loc[["SD 1.5", "SD 2.0", "SD 3-m"]]



df_all_ours = pd.concat([df_object_ours, df_attribute_ours, df_spatial_ours, df_counting_ours], axis=1)
# df_all_ours = "SD" in df_all_ours.indx
df_all_ours = df_all_ours.loc[["SD 1.5", "SD 2.0", "SD 3-m"]]
df_all= df_all.loc[["SD 1.5", "SD 2.0", "SD 3-m"]]

# print("ours")
# print(df_all_ours.head())
# print("others")
# print(df_all.head())


cols_to_normalize = [col for col in df_all_ours.columns if 'Std' not in col]

# Normalize each selected column individually
for col in cols_to_normalize:
    max_val = df_all_ours[col].max()
    # Avoid division by zero if the column is constant
    if max_val != 0:
        df_all_ours[col] = df_all_ours[col] / max_val

    max_val = df_all[col].max()
    # Avoid division by zero if the column is constant
    if max_val != 0:
        df_all[col] = df_all[col] / max_val

    max_val = df_all_ours_3m[col].max()

    if max_val != 0:
        df_all_ours_3m[col] = df_all_ours_3m[col] / max_val

print("ours")
print(df_all_ours.head())
print("others")
print(df_all.head())
# exit(0)

# Function to create bar plots
def create_bar_plot(df1, df2, category_name):

    df1 = df1[[col for col in df1.columns if 'Std' not in col]]
    df2 = df2[[col for col in df2.columns if 'Std' not in col]]
    
    df1_transposed = df1.T
    df2_transposed = df2.T

    sd_models = list(df1_transposed.columns)  # now only SD models
    sd_colors = sns.color_palette("colorblind", len(sd_models))
    # sd_colors.pop(-2)
    # sd_colors.pop(-2)

    bar_width = 0.3
    n_models = len(sd_models)
    intra_group_spacing = 0.02
    group_spacing = 0.2
    n_bars_per_group = 3
    n_groups = 4
    group_width = n_bars_per_group * (bar_width + intra_group_spacing) + group_spacing

    # index = np.arange(len(df1_transposed.index)) * (len(sd_models) * bar_width + group_spacing)
    index = np.arange(n_groups) * group_width
    fig, ax = plt.subplots(figsize=(3, 0.9))

    for j, model in enumerate(sd_models):
        x_positions = index + j * (bar_width + intra_group_spacing)
        bars1 = plt.bar(x_positions, df1_transposed[model], bar_width, alpha = 1.0 ,label=model, color=sd_colors[j], zorder=10)
        bars2 = plt.bar(x_positions, df2_transposed[model], bar_width, alpha = 0.3 ,label=model, color=sd_colors[j], zorder=1) # , align='edge'
        # put the text on top of the bars of the difference between two with plus and minus
        for i, bar in enumerate(bars1):
            height1 = bar.get_height()
            height2 = bars2[i].get_height()
            
            possible_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
            value = random.choice(possible_values)
            # height = sum([height1, height2])/2 + value
            height = max(height1, height2) + 0.15
            difference =  df2_transposed[model].iloc[i] - df1_transposed[model].iloc[i]
            # ax.text(bar.get_x() + bar.get_width() / 2, height
            #         , f'{df1_transposed[model].iloc[i]:.2f}', ha='center', va='bottom', fontsize=12)
            if difference >=0:
                difference = f'+{difference*100:.0f}%'
            else:
                difference = f'{difference*100:.0f}%'
            if '+' in difference and difference != '+0%':
                txt = ax.text(bar.get_x() + bar.get_width() / 2, height
                            , f'{difference}', ha='center', va='top', fontsize=3.5,zorder=13, color="green", fontweight='bold')
            else:
                txt = ax.text(bar.get_x() + bar.get_width() / 2, height
                            , f'{difference}', ha='center', va='top', fontsize=4,zorder=13, color="#CC3311", fontweight='bold')
            # if "3-m" in model:
            #     txt.set_path_effects([
            #         path_effects.Stroke(linewidth=2, foreground='red'),
            #         path_effects.Normal()
            #     ])
            # else: 
            #     txt.set_path_effects([
            #         path_effects.Stroke(linewidth=2, foreground='black'),
            #         path_effects.Normal()
            #     ])
    
    plt.ylabel('Normed Acc', fontsize=4)

    plt.xticks(index + (n_bars_per_group - 1)*(bar_width + intra_group_spacing)/2, 
        df1_transposed.index, 
            # rotation=18, 
        ha='center', 

        fontsize=5)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.yticks(np.linspace(0, 1, 4),fontsize=4)  # Change the Y-axis tick label size
    plt.tick_params(axis='y', which='both', width=0.5, length=2)
    y_max = plt.gca().get_ylim()[1]
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.legend(title='Models', loc='upper right', bbox_to_anchor=(0.98, 0.98))
    # Remove plot borders (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    # Remove grid lines
    plt.grid(False)
    # plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    # Save Plot
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')   # Set color of the spline
    ax.spines['left'].set_linewidth(0.5)   # Set thickness of the spline
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')   # Set color of the spline
    ax.spines['bottom'].set_linewidth(0.5)   # Set thickness of the spline
    ax.spines['bottom'].set_zorder(1000)


    plt.tight_layout()
    outdir = 'plots/teaser_average_all'
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f'{outdir}/{category_name}_teaser_ours.png', dpi=300, bbox_inches='tight')
    # plt.show()

# Create plots for each category
# create_bar_plot(df_all, "All")
# create_bar_plot(df_all_ours, "All_Ours")
# create_bar_plot(df_all_ours_3m, "All_Ours_3m")
# create_bar_plot(df_all, df_all_ours, "Combined")
create_bar_plot(df_all, df_all_ours_3m, "Combined_3m")
# create_bar_plot(df_attribute, "Attributes")
# create_bar_plot(df_counting, "Counting")
# create_bar_plot(df_object, "Object")
# create_bar_plot(df_spatial, "Spatial")
# create_bar_plot(df_complex, "Complex")
# create_bar_plot(df_size, "Size")
# create_bar_plot(df_presence, "Presence")
# create_bar_plot(df_action, "Action")
# create_bar_plot(df_text, "Text")

