# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
# Get the colorblind palette with 10 colors
colors = sns.color_palette("colorblind", 10)
# print(colors)

# Attribute
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
    'Winoground': object_scores_winoground,
    'COLA': object_scores_cola,
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

spatial_scores_ours = {
    "CLIP RN50x64": 0.27,
    "CLIP ViT-B/32": 0.27,
    "CLIP ViT-L/14": 0.33,
    "openCLIP ViT-H/14": 0.27,
    "openCLIP ViT-G/14": 0.6,
    "SD 1.5": 0.4,
    "SD 2.0": 0.47,
    "SD 3-m": 0.53,
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
    'Ours': spatial_scores_ours,
    # 'Spatial (Block 1)': position_correct_scores_first_block,
    # 'Spatial (Block 2)': position_correct_scores_second_block,
    # 'Spatial (Block 3)': position_correct_scores_third_block,

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


# Function to create bar plots
def create_bar_plot(df, category_name):
    df_transposed = df.T

    # Color schemes
    # colorblind_palette = sns.color_palette("colorblind", len(df_transposed.columns))
    # clip_colors = colorblind_palette[:5]
    # sd_colors = colorblind_palette[5:]
    clip_colors = ['#fff9c4', '#fff176', '#ffeb3b', '#fdd835', '#fbc02d']
    # clip_colors = ['#ffe066', '#ffd54f', '#ffca28', '#ffb300', '#ffa000']
    # clip_colors = ['#ffe066', '#ffd54f', '#ffca28', '#ffb300', '#ffa000']
        # Manually define colorblind-friendly yellows and blues
    # Adjusted color palettes for better visibility and colorblind-friendliness
    # clip_colors = [
    #     '#FFECB3',  # Softer pastel yellow
    #     '#FFD54F',  # Mellow yellow
    #     '#FFC107',  # Amber
    #     '#FFA000',  # Deep amber
    #     '#FF6F00'   # Burnt orange
    # ]

    sd_colors = [
        '#BBDEFB',  # Light sky blue
        '#64B5F6',  # Muted blue
        '#2196F3',  # Vibrant blue
        '#1976D2',  # Deep blue
        '#0D47A1'   # Navy blue
    ]




    # sd_colors = ['#bbdefb', '#90caf9', '#64b5f6', '#42a5f5', '#1e88e5', '#0d47a1']
    # sd_colors = ['#bbdefb', '#64b5f6', '#0d47a1']

    # Group models
    clip_models = ["CLIP RN50x64", "CLIP ViT-B/32", "CLIP ViT-L/14", "openCLIP ViT-H/14", "openCLIP ViT-G/14"]
    sd_models = [model for model in df_transposed.columns if model not in clip_models]

    # Plotting
    if category_name == "Spatial" or category_name == "Attributes" or category_name == "Complex":
        bar_width = 0.2
    else: bar_width = 0.1
    if category_name == "Spatial":
        group_spacing = 0.8
        # Compute the left and right boundaries for your bars
    elif category_name == "Action":
        group_spacing = 0.5
    else:
        group_spacing = 0.3
        
    index = np.arange(len(df_transposed.index)) * (len(df_transposed.columns) * bar_width + group_spacing)
    
        
    if category_name == "Spatial" or category_name == "Attributes" or category_name == "Complex":
        print("Spatial or Attributes")
        fig, ax = plt.subplots(figsize=(30, 5))
        leftmost = index[0] - bar_width -0.2  # Or maybe 0 - bar_width
        rightmost = index[-1] + len(df_transposed.columns)*bar_width +0.2

        # Adjust the x-axis limits to those boundaries
        ax.set_xlim(leftmost, rightmost)
    elif category_name == "Size" or category_name == "Presence" or category_name == "Action":
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig, ax = plt.subplots(figsize=(15, 5))
    for i, model in enumerate(clip_models):
        bars = plt.bar(index + i * bar_width, df_transposed[model], bar_width, label=model, color=clip_colors[i], zorder=10)
        # if category_name == "Spatial":
        # for bar in bars:
        #     yval = bar.get_height()
        #     plt.text(bar.get_x() + bar.get_width() / 2, 
        #             yval + 0.01,  # Position slightly above the bar
        #             f'{yval:.2f}',  # Display with 2 decimal places
        #             ha='center', 
        #             va='bottom',
        #             fontsize=15, 
        #                 rotation=80,
        #             color='black')


    for j, model in enumerate(sd_models):
        bars = plt.bar(index + (len(clip_models) + j) * bar_width, df_transposed[model], bar_width, label=model, color=sd_colors[j], zorder=10)
        # if category_name == "Spatial":
        # for bar in bars:
        #     yval = bar.get_height()
        #     plt.text(bar.get_x() + bar.get_width() / 2, 
        #             yval + 0.01,  # Position slightly above the bar
        #             f'{yval:.2f}',  # Display with 2 decimal places
        #             ha='center', 
        #             va='bottom',
        #             fontsize=12, 
        #                 rotation=60,
        #             color='black')
    # Get x-axis positions for the specific benchmarks
    vismin_pos = index[df_transposed.index.get_loc('Vismin')] if 'Vismin' in df_transposed.index else None
    # eqbench_pos = index[df_transposed.index.get_loc('EqBench')] if 'EqBench' in df_transposed.index else None
    # mmvp_pos = index[df_transposed.index.get_loc('MMVP')] if 'MMVP' in df_transposed.index else None
    
    # Get x-axis position for Eq-Kubric if it exists
    eq_kubric_pos = index[df_transposed.index.get_loc('Eq-Kubric')] if 'Eq-Kubric' in df_transposed.index else None
    # winoground_pos = index[df_transposed.index.get_loc('Winoground')] if 'Winoground' in df_transposed.index else None
    # clevr_pos = index[df_transposed.index.get_loc('Clevr')] if 'Clevr' in df_transposed.index else None
    clevr_color_pos = index[df_transposed.index.get_loc('Clevr (Color)')] if 'Clevr (Color)' in df_transposed.index else None
    clevr_color_binding_pos = index[df_transposed.index.get_loc('Clevr (Color Binding)')] if 'Clevr (Color Binding)' in df_transposed.index else None
    # Generalized approach for detecting all Clevr-related labels
    clevr_positions = [index[i] for i, label in enumerate(df_transposed.index) if 'Clevr' in label]
    mmvp_positions = [index[i] for i, label in enumerate(df_transposed.index) if 'MMVP' in label]
    winoground_positions = [index[i] for i, label in enumerate(df_transposed.index) if 'Winoground' in label]
    eqbench_positions = [(index[i], label) for i, label in enumerate(df_transposed.index) if 'EQBench' in label]
    # Generalized Clevr axhline
    spec_positions = [
        (index[i], label) 
        for i, label in enumerate(df_transposed.index) 
        if 'Spec' in label or "SPEC" in label
    ]
    for winoground_pos in winoground_positions:
        if category_name == "Object":
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(winoground_pos/plt.xlim()[1]), 
                        xmax=((winoground_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.01), 
                        zorder=10)
        elif category_name == "Complex":
            plt.axhline(y=0.25, color='red', linestyle='--', 
                    xmin=(winoground_pos / plt.xlim()[1]), 
                    xmax=((winoground_pos + bar_width * len(df_transposed.columns)) / plt.xlim()[1] + 0.01),
                    zorder=10)
        else:
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(winoground_pos/plt.xlim()[1]+0.02), 
                        xmax=((winoground_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]),
                        zorder=10)
        

    # Then draw lines depending on "Absolute" or "Relative"
    for spec_pos, spec_label in spec_positions:
        # Decide which random-chance value to use
        if category_name=="Spatial" and 'Relative ' in spec_label:
            random_chance = 1/4
            plt.axhline(
                y=random_chance,
                color='red',
                linestyle='--',
                xmin=(spec_pos / plt.xlim()[1]-0.005),
                xmax=((spec_pos + bar_width * len(df_transposed.columns)) / plt.xlim()[1] ),
                zorder=10
            )
        elif category_name =="Presence":
            random_chance = 1/2
            plt.axhline(
                y=random_chance,
                color='red',
                linestyle='--',
                xmin=(spec_pos / plt.xlim()[1]+0.02),
                xmax=((spec_pos + bar_width * len(df_transposed.columns)) / plt.xlim()[1] +0.03),
                zorder=10
            )
        elif category_name == "Size" :
            random_chance = 1/3
            plt.axhline(
                y=random_chance,
                color='red',
                linestyle='--',
                xmin=(spec_pos / plt.xlim()[1]+0.02),
                xmax=((spec_pos + bar_width * len(df_transposed.columns)) / plt.xlim()[1]+0.04),
                zorder=10)
        else:
            # Fallback if needed, or continue
            random_chance = 1/9

        # Now plot the horizontal line for that spec benchmark
            plt.axhline(
                y=random_chance,
                color='red',
                linestyle='--',
                xmin=(spec_pos / plt.xlim()[1]-0.005),
                xmax=((spec_pos + bar_width * len(df_transposed.columns)) / plt.xlim()[1] ),
                zorder=10
            )

    for clevr_pos in clevr_positions:
        if category_name == "Spatial":
            plt.axhline(y=0.5, color='red', linestyle='--', 
                        xmin=(clevr_pos / plt.xlim()[1] ),
                        xmax=((clevr_pos + bar_width * len(df_transposed.columns)) / plt.xlim()[1] + 0.01),
                        zorder=10)
        elif category_name == "Attributes":
            plt.axhline(y=0.5, color='red', linestyle='--', 
                        xmin=(clevr_pos / plt.xlim()[1]),
                        xmax=((clevr_pos + bar_width * len(df_transposed.columns)) / plt.xlim()[1] + 0.011),
                        zorder=10)
        elif category_name == "Size":
            plt.axhline(y=0.5, color='red', linestyle='--', 
                        xmin=(clevr_pos / plt.xlim()[1]-0.01),
                        xmax=((clevr_pos + bar_width * len(df_transposed.columns)) / plt.xlim()[1] ),
                        zorder=10)
        else:
            plt.axhline(y=0.5, color='red', linestyle='--', 
                        xmin=(clevr_pos / plt.xlim()[1] + 0.03),
                        xmax=((clevr_pos + bar_width * len(df_transposed.columns)) / plt.xlim()[1] + 0.03),
                        zorder=10)
                               
    for mmvp_pos in mmvp_positions:
        if category_name == "Counting":
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(mmvp_pos/plt.xlim()[1]), 
                        xmax=((mmvp_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]),
                        zorder=10)
        elif category_name == "Attributes":
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(mmvp_pos/plt.xlim()[1]+0.01), 
                        xmax=((mmvp_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.01),
                        zorder=10)
        elif category_name == "Spatial":
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(mmvp_pos/plt.xlim()[1]+0.005), 
                        xmax=((mmvp_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.01),
                        zorder=10)
        elif category_name == "Complex":
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(mmvp_pos/plt.xlim()[1]), 
                        xmax=((mmvp_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.01),
                        zorder=10)
        elif category_name == "Presence":
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(mmvp_pos/plt.xlim()[1]), 
                        xmax=((mmvp_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]-0.03),
                        zorder=10)
        else:
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(mmvp_pos/plt.xlim()[1]+0.03), 
                        xmax=((mmvp_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+ 0.04), 
                        zorder=10)

    
    cola_pos = index[df_transposed.index.get_loc('COLA')] if 'COLA' in df_transposed.index else None
    if cola_pos is not None:
        if category_name == "Object":
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(cola_pos/plt.xlim()[1]-0.02), 
                        xmax=((cola_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]),
                        zorder=10)
        else:
            plt.axhline(y=0.25, color='red', linestyle='--', 
                    xmin=(cola_pos/plt.xlim()[1]+0.03), 
                    xmax=((cola_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.03),
                    zorder =10)
    # Add red line for random chance at 0.25 for Eq-Kubric
    if eq_kubric_pos is not None:
        plt.axhline(y=0.25, color='red', linestyle='--', 
                    xmin=(eq_kubric_pos/plt.xlim()[1]+ 0.03), 
                    xmax=((eq_kubric_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+ 0.02), 
                    zorder=10)

    # Add red lines for random chance at 0.25
    if vismin_pos is not None:
        if category_name == "Attributes":
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(vismin_pos/plt.xlim()[1]+0.01), 
                        xmax=((vismin_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.015),
                        zorder=10)
            
        elif category_name == "Spatial":
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(vismin_pos/plt.xlim()[1]+0.01), 
                        xmax=((vismin_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.01),
                        zorder=10)
        else:
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(vismin_pos/plt.xlim()[1]+0.035), 
                        xmax=((vismin_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.04), 
                        zorder=10)

    for eqbench_pos, label in eqbench_positions:
        if eqbench_pos is not None:
            if category_name == "Counting":
                plt.axhline(y=0.25, color='red', linestyle='--', 
                            xmin=(eqbench_pos/plt.xlim()[1]+ 0.01), 
                            xmax=((eqbench_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+ 0.02), 
                            zorder=10)
            elif category_name == "Attributes":
                plt.axhline(y=0.25, color='red', linestyle='--', 
                            xmin=(eqbench_pos/plt.xlim()[1]+0.01), 
                            xmax=((eqbench_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+ 0.01),
                            zorder=10)
            elif category_name == "Spatial":
                plt.axhline(y=0.25, color='red', linestyle='--', 
                            xmin=(eqbench_pos/plt.xlim()[1]), 
                            xmax=((eqbench_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.015),
                            zorder=10)
            elif category_name == "Action" and 'GEBC' in label:
                plt.axhline(y=0.25, color='red', linestyle='--', 
                            xmin=(eqbench_pos/plt.xlim()[1]+0.02), 
                            xmax=((eqbench_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.03),
                            zorder=10)
            elif category_name == "Action" and 'AG' in label:
                plt.axhline(y=0.25, color='red', linestyle='--', 
                            xmin=(eqbench_pos/plt.xlim()[1]), 
                            xmax=((eqbench_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]),
                            zorder=10)
            elif category_name == "Action" and 'YouCook2' in label:
                plt.axhline(y=0.25, color='red', linestyle='--', 
                            xmin=(eqbench_pos/plt.xlim()[1]), 
                            xmax=((eqbench_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]),
                            zorder=10)
            else:
                plt.axhline(y=0.25, color='red', linestyle='--', 
                            xmin=(eqbench_pos/plt.xlim()[1]+ 0.04), 
                            xmax=((eqbench_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+ 0.03), 
                            zorder=10)
        
    # if winoground_pos is not None:
        
    # if clevr_pos is not None:
    #     plt.axhline(y=0.5, color='red', linestyle='--', 
    #                 xmin=(clevr_pos/plt.xlim()[1]+0.03), 
    #                 xmax=((clevr_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.03),
    #                 zorder =10)
    # if  clevr_color_pos is not None or clevr_color_binding_pos is not None:
    #     plt.axhline(y=0.5, color='red', linestyle='--', 
    #                 xmin=(clevr_color_pos/plt.xlim()[1]), 
    #                 xmax=((clevr_color_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.01),
    #                 zorder =10)
    #     plt.axhline(y=0.5, color='red', linestyle='--', 
    #                 xmin=(clevr_color_binding_pos/plt.xlim()[1]), 
    #                 xmax=((clevr_color_binding_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.01),
    #                 zorder =10)
    

    

    if category_name == "Object":
        # Get x-axis position for Spec if it exists
        sugar_pos = index[df_transposed.index.get_loc('Sugarcrepe')] if 'Sugarcrepe' in df_transposed.index else None

        # Add red line for random chance at 1/9 (0.11) for Spec
        if sugar_pos is not None:
            plt.axhline(y=1/2, color='red', linestyle='--', 
                        xmin=(sugar_pos/plt.xlim()[1]+0.01 ), 
                        xmax=((sugar_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1] +0.03), 
                        zorder=10)
    if category_name == "Spatial":
        # Get x-axis position for Spec if it exists
        whatsA_pos = index[df_transposed.index.get_loc('WhatsupA')] if 'WhatsupA' in df_transposed.index else None
        if whatsA_pos is not None:
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(whatsA_pos/plt.xlim()[1] ), 
                        xmax=((whatsA_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1] + 0.01),
                        zorder=10)
            
        whatsupB_pos = index[df_transposed.index.get_loc('WhatsupB')] if 'WhatsupB' in df_transposed.index else None
        if whatsupB_pos is not None:
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(whatsupB_pos/plt.xlim()[1] ), 
                        xmax=((whatsupB_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1] + 0.01),
                        zorder=10)
        cocoone_pos = index[df_transposed.index.get_loc('COCO (One)')] if 'COCO (One)' in df_transposed.index else None
        if cocoone_pos is not None:
            plt.axhline(y=0.5, color='red', linestyle='--', 
                        xmin=(cocoone_pos/plt.xlim()[1]), 
                        xmax=((cocoone_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1] + 0.015),
                        zorder=10)
        cocotwo_pos = index[df_transposed.index.get_loc('COCO (Two)')] if 'COCO (Two)' in df_transposed.index else None
        if cocotwo_pos is not None:
            plt.axhline(y=0.5, color='red', linestyle='--', 
                        xmin=(cocotwo_pos/plt.xlim()[1] ), 
                        xmax=((cocotwo_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1] + 0.01),
                        zorder=10)
        gqaone_pos = index[df_transposed.index.get_loc('GQA (One)')] if 'GQA (One)' in df_transposed.index else None
        if gqaone_pos is not None:
            plt.axhline(y=0.5, color='red', linestyle='--', 
                        xmin=(gqaone_pos/plt.xlim()[1] ), 
                        xmax=((gqaone_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.015 ),
                        zorder=10)
        gqatwo_pos = index[df_transposed.index.get_loc('GQA (Two)')] if 'GQA (Two)' in df_transposed.index else None
        if gqatwo_pos is not None:
            plt.axhline(y=0.5, color='red', linestyle='--', 
                        xmin=(gqatwo_pos/plt.xlim()[1] ), 
                        xmax=((gqatwo_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1] +0.01),
                        zorder=10)
    sugar_pos = index[df_transposed.index.get_loc('Sugarcrepe')] if 'Sugarcrepe' in df_transposed.index else None
    flickr_pos = index[df_transposed.index.get_loc('Flickr Order')] if 'Flickr Order' in df_transposed.index else None  
    if flickr_pos is not None:
        plt.axhline(y=1/5, color='red', linestyle='--', 
                    xmin=(flickr_pos/plt.xlim()[1]), 
                    xmax=((flickr_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.01),
                    zorder=10)
    coco_order_pos = index[df_transposed.index.get_loc('COCO Order')] if 'COCO Order' in df_transposed.index else None
    if coco_order_pos is not None:
        plt.axhline(y=1/5, color='red', linestyle='--', 
                    xmin=(coco_order_pos/plt.xlim()[1]+0.01), 
                    xmax=((coco_order_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.01),
                    zorder=10)
    aro_relation_pos = index[df_transposed.index.get_loc('Aro (Relation)')] if 'Aro (Relation)' in df_transposed.index else None
    if aro_relation_pos is not None:
        plt.axhline(y=1/2, color='red', linestyle='--', 
                    xmin=(aro_relation_pos/plt.xlim()[1]+0.015), 
                    xmax=((aro_relation_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.02),
                    zorder=10)
    if sugar_pos is not None:
        if category_name == "Attributes":
        # Add red line for random chance at 1/9 (0.11) for Spec

            plt.axhline(y=1/2, color='red', linestyle='--', 
                        xmin=(sugar_pos/plt.xlim()[1] + 0.01 ), 
                        xmax=((sugar_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1] + 0.015 ),
                        zorder=10) 
        elif category_name == "Complex":
            plt.axhline(y=1/2, color='red', linestyle='--',
                        xmin=(sugar_pos / plt.xlim()[1] + 0.005),
                        xmax=((sugar_pos + bar_width * len(df_transposed.columns)) / plt.xlim()[1] + 0.015),
                        zorder=10)
    if category_name == "Attributes":
        # Get x-axis position for Spec if it exists
        
        aro_pos = index[df_transposed.index.get_loc('ARO')] if 'ARO' in df_transposed.index else None
        if aro_pos is not None:
            plt.axhline(y=1/2, color='red', linestyle='--', 
                        xmin=(aro_pos/plt.xlim()[1]+0.01), 
                        xmax=((aro_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.015),
                        zorder=10)
    # Minimalistic Design
    # plt.yticks([])
    # plt.xticks([])
    # Add x-axis labels (Benchmark names)
    # Turn on the left spine
    # aro_relation 0.5
    
    plt.ylabel('Accuracy', fontsize=25)
    # plt.xticks(ha='center')
    # Modify x-tick labels if "Clevr" is present
    xtick_labels = [
        label.replace('Clevr ', 'Clevr\n') if 'Clevr' in label else label
        for label in df_transposed.index
    ]
    xtick_labels = [
        label.replace('MMVP ', 'MMVP\n') if 'MMVP ' in label else label
        for label in xtick_labels
    ]
    xtick_labels = [
        label.replace('COCO ', 'COCO\n') if 'COCO ' in label and label != "COCO Order" else label
        for label in xtick_labels
    ]
    xtick_labels = [
        label.replace('GQA ', 'GQA\n') if 'GQA ' in label else label
        for label in xtick_labels
    ]
    xtick_labels = [
        label.replace('Spec ', 'SPEC\n') if 'Spec ' in label else label
        for label in xtick_labels
    ]
    xtick_labels = [
        label.replace('Aro ', 'ARO\n') if 'Aro ' in label else label
        for label in xtick_labels
    ]
    xtick_labels = [
        label.replace('Winoground ', 'Winoground\n') if 'Winoground ' in label else label
        for label in xtick_labels
    ]
    if category_name != "Action":
        xtick_labels = [
            label.replace('EQBench ', 'EQBench\n') if 'EQBench ' in label else label
            for label in xtick_labels
        ]
    if category_name == "Spatial" or category_name == "Action":
            plt.xticks(index + bar_width * (len(df_transposed.columns) - 1) / 2, 
            xtick_labels, 
            #    rotation=20, 
            ha='center', 

            fontsize=23)
    else:   
        plt.xticks(index + bar_width * (len(df_transposed.columns) - 1) / 2, 
            xtick_labels, 
            #    rotation=20, 
            ha='center', 

            fontsize=25)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.yticks(fontsize=25)  # Change the Y-axis tick label size
    y_max = plt.gca().get_ylim()[1]


    # plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    # Remove plot borders (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    # Remove grid lines
    # plt.grid(False)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    # Save Plot
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color('black')   # Set color of the spline
    ax.spines['left'].set_linewidth(1.5)   # Set thickness of the spline
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')   # Set color of the spline
    ax.spines['bottom'].set_linewidth(1.5)   # Set thickness of the spline

    plt.tight_layout()
    outdir = 'plots/ours_others'
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f'{outdir}/{category_name}_scores_plot.png', dpi=300, bbox_inches='tight')
    # plt.show()

# Create plots for each category
create_bar_plot(df_attribute, "Attributes")
create_bar_plot(df_counting, "Counting")
create_bar_plot(df_object, "Object")
create_bar_plot(df_spatial, "Spatial")
# create_bar_plot(df_complex, "Complex")
# create_bar_plot(df_size, "Size")
# create_bar_plot(df_presence, "Presence")
# create_bar_plot(df_action, "Action")