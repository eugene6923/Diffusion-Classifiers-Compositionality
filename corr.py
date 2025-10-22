import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patheffects as path_effects
import matplotlib.patches as mpatches


position_full_scores_first_block = {
    # "CLIP RN50x64": 0.33,
    # "CLIP ViT-B/32": 0.24,
    # "CLIP ViT-L/14": 0.31,
    # "openCLIP ViT-H/14": 0.30,
    # "openCLIP ViT-G/14": 0.34,
    "SD 1.5": 0.49,
    # "SD 2.0": 0.36,
    # "SD 3-m": 0.29,
}

position_correct_scores_first_block = {
    # "CLIP RN50x64": 0.17,
    # "CLIP ViT-B/32": 0.67,
    # "CLIP ViT-L/14": 0.5,
    # "openCLIP ViT-H/14": 0.33,
    # "openCLIP ViT-G/14": 0.67,
    "SD 1.5": 0.67,
    # "SD 2.0": 0.67,
    # "SD 3-m": 0.33,
}

position_full_scores_second_block = {
    # "CLIP RN50x64": 0.30,
    # "CLIP ViT-B/32": 0.22,
    # "CLIP ViT-L/14": 0.26,
    # "openCLIP ViT-H/14": 0.44,
    # "openCLIP ViT-G/14": 0.38,
    "SD 2.0": 0.63,
    # "SD 1.5": 0.28,
    # "SD 3-m": 0.30,
}

position_correct_scores_second_block = {
    # "CLIP RN50x64": 0.26,
    # "CLIP ViT-B/32": 0.16,
    # "CLIP ViT-L/14": 0.26,
    # "openCLIP ViT-H/14": 0.37,
    # "openCLIP ViT-G/14": 0.42,
    "SD 2.0": 0.84,
    # "SD 1.5": 0.26,
    # "SD 3-m": 0.42,
}

position_full_scores_third_block = {
    # "CLIP RN50x64": 0.26,
    # "CLIP ViT-B/32": 0.28,
    # "CLIP ViT-L/14": 0.32,
    # "openCLIP ViT-H/14": 0.36,
    # "openCLIP ViT-G/14": 0.37,
    "SD 3-m": 0.72,
    # "SD 1.5": 0.32,
    # "SD 2.0": 0.33,
}
position_correct_scores_third_block = {
    # "CLIP RN50x64": 0.27,
    # "CLIP ViT-B/32": 0.31,
    # "CLIP ViT-L/14": 0.30,
    # "openCLIP ViT-H/14": 0.33,
    # "openCLIP ViT-G/14": 0.35,
    "SD 3-m": 0.89,
    # "SD 1.5": 0.30,
    # "SD 2.0": 0.34,
}

counting_full_scores_first_block = {
    # "CLIP RN50x64": 0.47,
    # "CLIP ViT-B/32": 0.52,
    # "CLIP ViT-L/14": 0.49,
    # "openCLIP ViT-H/14": 0.48,
    # "openCLIP ViT-G/14": 0.49,
    "SD 1.5": 0.65,
    # "SD 2.0": 0.46,
    # "SD 3-m": 0.37,
}
counting_correct_scores_first_block = {
    # "CLIP RN50x64": 0.67,
    # "CLIP ViT-B/32": 0.63,
    # "CLIP ViT-L/14": 0.63,
    # "openCLIP ViT-H/14": 0.85,
    # "openCLIP ViT-G/14": 0.87,
    "SD 1.5": 0.76,
    # "SD 2.0": 0.49,
    # "SD 3-m": 0.53,
}

counting_full_scores_second_block = {
    # "CLIP RN50x64": 0.50,
    # "CLIP ViT-B/32": 0.50,
    # "CLIP ViT-L/14": 0.52,
    # "openCLIP ViT-H/14": 0.53,
    # "openCLIP ViT-G/14": 0.53,
    "SD 2.0": 0.78,
    # "SD 1.5": 0.49,
    # "SD 3-m": 0.45,
}

counting_correct_scores_second_block = {
    # "CLIP RN50x64": 0.77,
    # "CLIP ViT-B/32": 0.60,
    # "CLIP ViT-L/14": 0.69,
    # "openCLIP ViT-H/14": 0.93,
    # "openCLIP ViT-G/14": 0.95,
    "SD 2.0": 0.95,
    # "SD 1.5": 0.59,
    # "SD 3-m": 0.57,
}
counting_full_scores_third_block = {
    # "CLIP RN50x64": 0.66,
    # "CLIP ViT-B/32": 0.61,
    # "CLIP ViT-L/14": 0.68,
    # "openCLIP ViT-H/14": 0.84,
    # "openCLIP ViT-G/14": 0.84,
    "SD 3-m": 0.85,
    # "SD 1.5": 0.57,
    # "SD 2.0": 0.57,
}

counting_correct_scores_third_block = {
    # "CLIP RN50x64": 0.70,
    # "CLIP ViT-B/32": 0.65,
    # "CLIP ViT-L/14": 0.75,
    # "openCLIP ViT-H/14": 0.97,
    # "openCLIP ViT-G/14": 0.96,
    "SD 3-m": 0.91,
    # "SD 1.5": 0.60,
    # "SD 2.0": 0.61,
}

two_objects_full_first_block = {
    # "CLIP RN50x64": 0.46,
    # "CLIP ViT-B/32": 0.46,
    # "CLIP ViT-L/14": 0.54,
    # "openCLIP ViT-H/14": 0.52,
    # "openCLIP ViT-G/14": 0.51,
    "SD 1.5": 0.69,
    # "SD 2.0": 0.48,
    # "SD 3-m": 0.29,
}

two_objects_correct_first_block = {
    # "CLIP RN50x64": 0.85,
    # "CLIP ViT-B/32": 0.87,
    # "CLIP ViT-L/14": 0.95,
    # "openCLIP ViT-H/14": 0.95,
    # "openCLIP ViT-G/14": 0.94,
    "SD 1.5": 0.90,
    # "SD 2.0": 0.85,
    # "SD 3-m": 0.57,
}

# object
two_objects_full_second_block = {
    # "CLIP RN50x64": 0.61,
    # "CLIP ViT-B/32": 0.54,
    # "CLIP ViT-L/14": 0.64,
    # "openCLIP ViT-H/14": 0.69,
    # "openCLIP ViT-G/14": 0.65,
    "SD 2.0": 0.82,
    # "SD 1.5": 0.61,
    # "SD 3-m": 0.40,
}

two_objects_correct_second_block = {
    # "CLIP RN50x64": 0.91,
    # "CLIP ViT-B/32": 0.85,
    # "CLIP ViT-L/14": 0.93,
    # "openCLIP ViT-H/14": 0.99,
    # "openCLIP ViT-G/14": 0.98,
    "SD 2.0": 0.98,
    # "SD 1.5": 0.90,
    # "SD 3-m": 0.63,
}
two_objects_full_third_block = {
    # "CLIP RN50x64": 0.90,
    # "CLIP ViT-B/32": 0.86,
    # "CLIP ViT-L/14": 0.95,
    # "openCLIP ViT-H/14": 0.95,
    # "openCLIP ViT-G/14": 0.95,
    "SD 3-m": 0.98,
    # "SD 1.5": 0.87,
    # "SD 2.0": 0.91,
}

two_objects_correct_third_block = {
    # "CLIP RN50x64": 0.91,
    # "CLIP ViT-B/32": 0.89,
    # "CLIP ViT-L/14": 0.98,
    # "openCLIP ViT-H/14": 0.97,
    # "openCLIP ViT-G/14": 0.98,
    "SD 3-m": 0.98,
    # "SD 1.5": 0.88,
    # "SD 2.0": 0.92,
}

color_full_scores_first_block = {
    # "CLIP RN50x64": 0.86,
    # "CLIP ViT-B/32": 0.87,
    # "CLIP ViT-L/14": 0.87,
    # "openCLIP ViT-H/14": 0.89,
    # "openCLIP ViT-G/14": 0.87,
    "SD 1.5": 0.93,
    # "SD 1.5 (discffusion)": 0.77,
    # "SD 2.0": 0.82,
    # "SD 2.0 (discffusion)": 0.91,
    # "SD 3-m": 0.88,
    # "SD 3-m (discffusion)": 0.88
}

color_correct_scores_first_block = {
    # "CLIP RN50x64": 0.94,
    # "CLIP ViT-B/32": 0.94,
    # "CLIP ViT-L/14": 0.94,
    # "openCLIP ViT-H/14": 0.97,
    # "openCLIP ViT-G/14": 0.97,
    "SD 1.5": 0.98,
    # "SD 1.5 (discffusion)": 0.77,
    # "SD 2.0": 0.91,
    # "SD 2.0 (discffusion)": 0.91,
    # "SD 3-m": 0.88,
    # "SD 3-m (discffusion)": 0.88
}

color_full_scores_second_block = {
    # "CLIP RN50x64": 0.93,
    # "CLIP ViT-B/32": 0.92,
    # "CLIP ViT-L/14": 0.91,
    # "openCLIP ViT-H/14": 0.94,
    # "openCLIP ViT-G/14": 0.94,
    "SD 2.0": 0.97,
    # "SD 2.0 (discffusion)": 0.97,
    # "SD 1.5": 0.85,
    # "SD 1.5 (discffusion)": 0.61,
    # "SD 3-m": 0.87,
    # "SD 3-m (discffusion)": 0.90
}

color_correct_scores_second_block = {
    # "CLIP RN50x64": 0.95,
    # "CLIP ViT-B/32": 0.94,
    # "CLIP ViT-L/14": 0.93,
    # "openCLIP ViT-H/14": 0.97,
    # "openCLIP ViT-G/14": 0.98,
    "SD 2.0": 0.98,
    # "SD 2.0 (discffusion)": 0.97,
    # "SD 1.5": 0.89,
    # "SD 1.5 (discffusion)": 0.61,
    # "SD 3-m": 0.90,
    # "SD 3-m (discffusion)": 0.90
}
color_full_scores_third_block = {
    # "CLIP RN50x64": 0.89,
    # "CLIP ViT-B/32": 0.88,
    # "CLIP ViT-L/14": 0.89,
    # "openCLIP ViT-H/14": 0.91,
    # "openCLIP ViT-G/14": 0.91,
    "SD 3-m": 0.97,
    # "SD 1.5": 0.78,
    # "SD 2.0": 0.82,
}

color_correct_scores_third_block = {
    # "CLIP RN50x64": 0.92,
    # "CLIP ViT-B/32": 0.91,
    # "CLIP ViT-L/14": 0.91,
    # "openCLIP ViT-H/14": 0.96,
    # "openCLIP ViT-G/14": 0.95,
    "SD 3-m": 0.98,
    # "SD 1.5": 0.82,
    # "SD 2.0": 0.87,
}

color_attribute_full_scores_first_block = {
    # "CLIP RN50x64": 0.28,
    # "CLIP ViT-B/32": 0.25,
    # "CLIP ViT-L/14": 0.29,
    # "openCLIP ViT-H/14": 0.31,
    # "openCLIP ViT-G/14": 0.35,
    "SD 1.5": 0.56,
    # "SD 2.0": 0.26,
    # "SD 3-m": 0.26,
}

color_attribute_correct_scores_first_block = {
    # "CLIP RN50x64": 0.28,
    # "CLIP ViT-B/32": 0.22,
    # "CLIP ViT-L/14": 0.17,
    # "openCLIP ViT-H/14": 0.5,
    # "openCLIP ViT-G/14": 0.28,
    "SD 1.5": 0.83,
    # "SD 2.0": 0.5,
    # "SD 3-m": 0.56,
}

color_attribute_full_scores_second_block = {
    # "CLIP RN50x64": 0.30,
    # "CLIP ViT-B/32": 0.35,
    # "CLIP ViT-L/14": 0.28,
    # "openCLIP ViT-H/14": 0.44,
    # "openCLIP ViT-G/14": 0.45,
    "SD 2.0": 0.70,
    # "SD 1.5": 0.37,
    # "SD 3-m": 0.28,
}

color_attribute_correct_scores_second_block = {
    # "CLIP RN50x64": 0.47,
    # "CLIP ViT-B/32": 0.47,
    # "CLIP ViT-L/14": 0.44,
    # "openCLIP ViT-H/14": 0.53,
    # "openCLIP ViT-G/14": 0.69,
    "SD 2.0": 0.88,
    # "SD 1.5": 0.42,
    # "SD 3-m": 0.53,
}

color_attribute_full_scores_third_block = {
    # "CLIP RN50x64": 0.38,
    # "CLIP ViT-B/32": 0.43,
    # "CLIP ViT-L/14": 0.34,
    # "openCLIP ViT-H/14": 0.47,
    # "openCLIP ViT-G/14": 0.51,
    "SD 3-m": 0.91,
    # "SD 1.5": 0.53,
    # "SD 2.0": 0.56,
}

color_attribute_correct_scores_third_block = {
    # "CLIP RN50x64": 0.4,
    # "CLIP ViT-B/32": 0.43,
    # "CLIP ViT-L/14": 0.36,
    # "openCLIP ViT-H/14": 0.49,
    # "openCLIP ViT-G/14": 0.55,
    "SD 3-m": 0.95,
    # "SD 1.5": 0.58,
    # "SD 2.0": 0.63,
}

single_object_full_first_block = {
    # "CLIP RN50x64": 0.97,
    # "CLIP ViT-B/32": 0.96,
    # "CLIP ViT-L/14": 0.97,
    # "openCLIP ViT-H/14": 0.97,
    # "openCLIP ViT-G/14": 0.97,
    "SD 1.5": 0.98,
    # "SD 2.0": 0.96,
    # "SD 3-m": 0.78,
}

single_object_correct_first_block = {
    # "CLIP RN50x64": 0.99,
    # "CLIP ViT-B/32": 0.99,
    # "CLIP ViT-L/14": 0.99,
    # "openCLIP ViT-H/14": 1.0,
    # "openCLIP ViT-G/14": 0.99,
    "SD 1.5": 1.0,
    # "SD 2.0": 0.99,
    # "SD 3-m": 0.81,
}

single_object_full_second_block = {
    # "CLIP RN50x64": 0.99,
    # "CLIP ViT-B/32": 1.0,
    # "CLIP ViT-L/14": 0.99,
    # "openCLIP ViT-H/14": 1.0,
    # "openCLIP ViT-G/14": 1.0,
    "SD 2.0": 1.0,
    # "SD 1.5": 0.99,
    # "SD 3-m": 0.86,
}

single_object_correct_second_block = {
    # "CLIP RN50x64": 1.0,
    # "CLIP ViT-B/32": 1.0,
    # "CLIP ViT-L/14": 1.0,
    # "openCLIP ViT-H/14": 1.0,
    # "openCLIP ViT-G/14": 1.0,
    "SD 2.0": 1.0,
    # "SD 1.5": 0.99,
    # "SD 3-m": 0.87,
}

single_object_full_third_block = {
    # "CLIP RN50x64": 0.99,
    # "CLIP ViT-B/32": 1.0,
    # "CLIP ViT-L/14": 0.99,
    # "openCLIP ViT-H/14": 1.0,
    # "openCLIP ViT-G/14": 1.0,
    "SD 3-m": 1.0,
    # "SD 1.5": 1.0,
    # "SD 2.0": 0.99,
}

single_object_correct_third_block = {
    # "CLIP RN50x64": 0.99,
    # "CLIP ViT-B/32": 1.0,
    # "CLIP ViT-L/14": 0.99,
    # "openCLIP ViT-H/14": 1.0,
    # "openCLIP ViT-G/14": 1.0,
    "SD 3-m": 1.0,
    # "SD 1.5": 1.0,
    # "SD 2.0": 0.99,
}

df_1_5_colors = pd.DataFrame({
      "Generation_colors": {"SD 1.5": 219/376},
    "Full_colors": color_full_scores_first_block,
    "Correct_colors": color_correct_scores_first_block,
})

df_1_5_single = pd.DataFrame({
    "Generation_single": {"SD 1.5": 271/320},
    "Full_single": single_object_full_first_block,
    "Correct_single": single_object_correct_first_block,
})

df_1_5_two_objects = pd.DataFrame({
    "Generation_two": {"SD 1.5": 105/396},
    "Full_two": two_objects_full_first_block,
    "Correct_two": two_objects_correct_first_block,
})

df_1_5_colors_attr = pd.DataFrame({
    "Generation_color_attr":{"SD 1.5": 18/400},
    "Full_color_attr": color_attribute_full_scores_first_block,
    "Correct_color_attr": color_attribute_correct_scores_first_block,
})

df_1_5_position = pd.DataFrame({
    "Generation_position": {"SD 1.5": 6/400},
    "Full_position": position_full_scores_first_block,
    "Correct_position": position_correct_scores_first_block,
})

df_1_5_counting = pd.DataFrame({
    "Generation_counting": {"SD 1.5": 98/320},
    "Full_counting": counting_full_scores_first_block,
    "Correct_counting": counting_correct_scores_first_block,
})

df_1_5 = pd.concat([df_1_5_single, df_1_5_two_objects, df_1_5_colors, df_1_5_colors_attr, df_1_5_position, df_1_5_counting], axis=1)

df_2_0_colors = pd.DataFrame({
     "Generation_colors": {"SD 2.0": 263/376},
    "Full_colors": color_full_scores_second_block,
    "Correct_colors": color_correct_scores_second_block,
})

df_2_0_single = pd.DataFrame({
    "Generation_single": {"SD 2.0": 271/320},
    "Full_single": single_object_full_second_block,
    "Correct_single": single_object_correct_second_block,
})

df_2_0_two_objects = pd.DataFrame({
    "Generation_two": {"SD 2.0": 129/376},
    "Full_two": two_objects_full_second_block,
    "Correct_two": two_objects_correct_second_block,
})

df_2_0_colors_attr = pd.DataFrame({
    "Generation_color_attr": {"SD 2.0": 36/400},
    "Full_color_attr": color_attribute_full_scores_second_block,
    "Correct_color_attr": color_attribute_correct_scores_second_block,
})

df_2_0_position = pd.DataFrame({
    "Generation_position": {"SD 2.0": 19/400},
    "Full_position": position_full_scores_second_block,
    "Correct_position": position_correct_scores_second_block,
})

df_2_0_counting = pd.DataFrame({
    "Generation_counting": {"SD 2.0": 111/320},
    "Full_counting": counting_full_scores_second_block,
    "Correct_counting": counting_correct_scores_second_block,
})

df_2_0 = pd.concat([df_2_0_single, df_2_0_two_objects, df_2_0_colors, df_2_0_colors_attr, df_2_0_position, df_2_0_counting], axis=1)


df_3_m_colors = pd.DataFrame({
    "Generation_colors": {"SD 3-m": 314/376},
    "Full_colors": color_full_scores_third_block,
    "Correct_colors": color_correct_scores_third_block,
})

df_3_m_single = pd.DataFrame({
    "Generation_single": {"SD 3-m": 314/320},
    "Full_single": single_object_full_third_block,
    "Correct_single": single_object_correct_third_block,
})

df_3_m_two_objects = pd.DataFrame({
    "Generation_two": {"SD 3-m": 306/396},
    "Full_two": two_objects_full_third_block,
    "Correct_two": two_objects_correct_third_block,
})

df_3_m_colors_attr = pd.DataFrame({
    "Generation_color_attr": {"SD 3-m": 252/400},
    "Full_color_attr": color_attribute_full_scores_third_block,
    "Correct_color_attr": color_attribute_correct_scores_third_block,
})

df_3_m_position = pd.DataFrame({
    "Generation_position": {"SD 3-m": 113/400},
    "Full_position": position_full_scores_third_block,
    "Correct_position": position_correct_scores_third_block,
})

df_3_m_counting = pd.DataFrame({
    "Generation_counting": {"SD 3-m": 230/320},
    "Full_counting": counting_full_scores_third_block,
    "Correct_counting": counting_correct_scores_third_block,
})

df_3_m = pd.concat([df_3_m_single, df_3_m_two_objects, df_3_m_colors, df_3_m_colors_attr, df_3_m_position, df_3_m_counting], axis=1)

import pandas as pd

# Combine all data into a single DataFrame for correlation calculation
df_combined = pd.concat([df_1_5, df_2_0, df_3_m], axis=0)

# Calculate the correlation between Generation and Full Discrimination accuracy
generation_full_corr = df_combined[['Generation_single', 'Generation_two', 'Generation_colors', 'Generation_color_attr', 'Generation_position', 'Generation_counting']].values.flatten()
full_corr = df_combined[['Full_single', 'Full_two', 'Full_colors', 'Full_color_attr', 'Full_position', 'Full_counting']].values.flatten()
generation_full_correlation = pd.Series(generation_full_corr).corr(pd.Series(full_corr))

# Calculate the correlation between Generation and Correct Discrimination accuracy
correct_corr = df_combined[['Correct_single', 'Correct_two', 'Correct_colors', 'Correct_color_attr', 'Correct_position', 'Correct_counting']].values.flatten()
generation_correct_correlation = pd.Series(generation_full_corr).corr(pd.Series(correct_corr))

print(f"Correlation between Generation and Full Discrimination accuracy: {generation_full_correlation}")
print(f"Correlation between Generation and Correct Discrimination accuracy: {generation_correct_correlation}")

categories = ["Single", "Two", "Colors", "Color_Attr", "Position", "Counting"]

palette = sns.color_palette("colorblind", 10)
import matplotlib.colors as mcolors

# Convert each RGB color to HSV
hsv_palette = [mcolors.rgb_to_hsv(color) for color in palette]

# Filter the palette to remove certain hues
filtered_palette = [
    color for color, hsv in zip(palette, hsv_palette)
    if not (hsv[0] < 0.1 or hsv[0] > 0.9 or (0.55 < hsv[0] < 0.75))
]

# Plotting
# models = ['SD1.5', 'SD2.0', 'SD3-m']
df_combined.index = df_combined.index.str.replace(' ', '')
df_combined.index = df_combined.index.str.replace('1.5', '1')
df_combined.index = df_combined.index.str.replace('2.0', '2')
df_combined.index = df_combined.index.str.replace('3-m', '3')
models = df_combined.index

import matplotlib.ticker as mticker
fig, axes = plt.subplots(1, 6, figsize=(12, 2))
axes = axes.flatten()
color_blind_pallete = sns.color_palette("colorblind", 10)
for i, category in enumerate(categories):
    ax = axes[i]
    ax2 = ax.twinx()
    # ax.plot(df_combined.index, df_combined[f'Full_{category.lower()}'], label='Full', marker='o',color = color_blind_pallete[0], linewidth=5)
    ax.plot(df_combined.index, df_combined[f'Correct_{category.lower()}'], label='Correct', marker='o', color=filtered_palette[4], linewidth=5)
    ax2.plot(df_combined.index, df_combined[f'Generation_{category.lower()}'], label='Generation', marker='o', color=color_blind_pallete[4], linewidth=5)
    ax.set_title(category, fontsize=18)
    # ax.set_yticks(np.linspace(0.4, 1, 3))
    # ax2.set_yticks(np.linspace(0.4, 1, 3))
    # ax.set_yticklabels(np.linspace(0.4, 1, 3), fontsize=11, rotation=90)
    ax.set_xticks(models)
    ax.set_ylim(top=1.0)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.set_xticklabels(models, fontsize=15)
    # ax.set_xlabel('Model')
    # ax.set_ylabel('Accuracy')
    # plt.yticks([])
    # ax.legend()

plt.tight_layout()
outdir ='plots'
os.makedirs(outdir, exist_ok=True)
plt.savefig(f'{outdir}/correlation.pdf')