import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patheffects as path_effects
import matplotlib.patches as mpatches


position_discffusion_scores_first_block = {
    # "CLIP RN50x64": 0.33,
    # "CLIP ViT-B/32": 0.24,
    # "CLIP ViT-L/14": 0.31,
    # "openCLIP ViT-H/14": 0.30,
    # "openCLIP ViT-G/14": 0.34,
    "SD 1.5": 0.33,
    # "SD 2.0": 0.36,
    # "SD 3-m": 0.29,
}

position_zero_scores_first_block = {
    # "CLIP RN50x64": 0.17,
    # "CLIP ViT-B/32": 0.67,
    # "CLIP ViT-L/14": 0.5,
    # "openCLIP ViT-H/14": 0.33,
    # "openCLIP ViT-G/14": 0.67,
    "SD 1.5": 0.67,
    # "SD 2.0": 0.67,
    # "SD 3-m": 0.33,
}

position_discffusion_scores_second_block = {
    # "CLIP RN50x64": 0.30,
    # "CLIP ViT-B/32": 0.22,
    # "CLIP ViT-L/14": 0.26,
    # "openCLIP ViT-H/14": 0.44,
    # "openCLIP ViT-G/14": 0.38,
    "SD 2.0": 0.89,
    # "SD 1.5": 0.28,
    # "SD 3-m": 0.30,
}

position_zero_scores_second_block = {
    # "CLIP RN50x64": 0.26,
    # "CLIP ViT-B/32": 0.16,
    # "CLIP ViT-L/14": 0.26,
    # "openCLIP ViT-H/14": 0.37,
    # "openCLIP ViT-G/14": 0.42,
    "SD 2.0": 0.84,
    # "SD 1.5": 0.26,
    # "SD 3-m": 0.42,
}

position_discffusion_scores_third_block = {
    # "CLIP RN50x64": 0.26,
    # "CLIP ViT-B/32": 0.28,
    # "CLIP ViT-L/14": 0.32,
    # "openCLIP ViT-H/14": 0.36,
    # "openCLIP ViT-G/14": 0.37,
    "SD 3-m": 0.91,
    # "SD 1.5": 0.32,
    # "SD 2.0": 0.33,
}
position_zero_scores_third_block = {
    # "CLIP RN50x64": 0.27,
    # "CLIP ViT-B/32": 0.31,
    # "CLIP ViT-L/14": 0.30,
    # "openCLIP ViT-H/14": 0.33,
    # "openCLIP ViT-G/14": 0.35,
    "SD 3-m": 0.89,
    # "SD 1.5": 0.30,
    # "SD 2.0": 0.34,
}
counting_zero_scores_first_block = {
    # "CLIP RN50x64": 0.47,
    # "CLIP ViT-B/32": 0.52,
    # "CLIP ViT-L/14": 0.49,
    # "openCLIP ViT-H/14": 0.48,
    # "openCLIP ViT-G/14": 0.49,
    "SD 1.5": 0.76,
    # "SD 2.0": 0.46,
    # "SD 3-m": 0.37,
}
counting_discffusion_scores_first_block = {
    # "CLIP RN50x64": 0.47,
    # "CLIP ViT-B/32": 0.52,
    # "CLIP ViT-L/14": 0.49,
    # "openCLIP ViT-H/14": 0.48,
    # "openCLIP ViT-G/14": 0.49,
    "SD 1.5": 0.46,
    # "SD 2.0": 0.46,
    # "SD 3-m": 0.37,
}
counting_zero_scores_first_block = {
    # "CLIP RN50x64": 0.67,
    # "CLIP ViT-B/32": 0.63,
    # "CLIP ViT-L/14": 0.63,
    # "openCLIP ViT-H/14": 0.85,
    # "openCLIP ViT-G/14": 0.87,
    "SD 1.5": 0.76,
    # "SD 2.0": 0.49,
    # "SD 3-m": 0.53,
}

counting_discffusion_scores_second_block = {
    # "CLIP RN50x64": 0.50,
    # "CLIP ViT-B/32": 0.50,
    # "CLIP ViT-L/14": 0.52,
    # "openCLIP ViT-H/14": 0.53,
    # "openCLIP ViT-G/14": 0.53,
    "SD 2.0": 0.89,
    # "SD 1.5": 0.49,
    # "SD 3-m": 0.45,
}

counting_zero_scores_second_block = {
    # "CLIP RN50x64": 0.77,
    # "CLIP ViT-B/32": 0.60,
    # "CLIP ViT-L/14": 0.69,
    # "openCLIP ViT-H/14": 0.93,
    # "openCLIP ViT-G/14": 0.95,
    "SD 2.0": 0.95,
    # "SD 1.5": 0.59,
    # "SD 3-m": 0.57,
}
counting_discffusion_scores_third_block = {
    # "CLIP RN50x64": 0.66,
    # "CLIP ViT-B/32": 0.61,
    # "CLIP ViT-L/14": 0.68,
    # "openCLIP ViT-H/14": 0.84,
    # "openCLIP ViT-G/14": 0.84,
    "SD 3-m": 0.92,
    # "SD 1.5": 0.57,
    # "SD 2.0": 0.57,
}

counting_zero_scores_third_block = {
    # "CLIP RN50x64": 0.70,
    # "CLIP ViT-B/32": 0.65,
    # "CLIP ViT-L/14": 0.75,
    # "openCLIP ViT-H/14": 0.97,
    # "openCLIP ViT-G/14": 0.96,
    "SD 3-m": 0.91,
    # "SD 1.5": 0.60,
    # "SD 2.0": 0.61,
}

two_objects_discffusion_first_block = {
    # "CLIP RN50x64": 0.46,
    # "CLIP ViT-B/32": 0.46,
    # "CLIP ViT-L/14": 0.54,
    # "openCLIP ViT-H/14": 0.52,
    # "openCLIP ViT-G/14": 0.51,
    "SD 1.5": 0.59,
    # "SD 2.0": 0.48,
    # "SD 3-m": 0.29,
}

two_objects_zero_first_block = {
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
two_objects_discffusion_second_block = {
    # "CLIP RN50x64": 0.61,
    # "CLIP ViT-B/32": 0.54,
    # "CLIP ViT-L/14": 0.64,
    # "openCLIP ViT-H/14": 0.69,
    # "openCLIP ViT-G/14": 0.65,
    "SD 2.0": 0.98,
    # "SD 1.5": 0.61,
    # "SD 3-m": 0.40,
}

two_objects_zero_second_block = {
    # "CLIP RN50x64": 0.91,
    # "CLIP ViT-B/32": 0.85,
    # "CLIP ViT-L/14": 0.93,
    # "openCLIP ViT-H/14": 0.99,
    # "openCLIP ViT-G/14": 0.98,
    "SD 2.0": 0.98,
    # "SD 1.5": 0.90,
    # "SD 3-m": 0.63,
}
two_objects_discffusion_third_block = {
    # "CLIP RN50x64": 0.90,
    # "CLIP ViT-B/32": 0.86,
    # "CLIP ViT-L/14": 0.95,
    # "openCLIP ViT-H/14": 0.95,
    # "openCLIP ViT-G/14": 0.95,
    "SD 3-m": 0.99,
    # "SD 1.5": 0.87,
    # "SD 2.0": 0.91,
}

two_objects_zero_third_block = {
    # "CLIP RN50x64": 0.91,
    # "CLIP ViT-B/32": 0.89,
    # "CLIP ViT-L/14": 0.98,
    # "openCLIP ViT-H/14": 0.97,
    # "openCLIP ViT-G/14": 0.98,
    "SD 3-m": 0.98,
    # "SD 1.5": 0.88,
    # "SD 2.0": 0.92,
}

color_discffusion_scores_first_block = {
    # "CLIP RN50x64": 0.86,
    # "CLIP ViT-B/32": 0.87,
    # "CLIP ViT-L/14": 0.87,
    # "openCLIP ViT-H/14": 0.89,
    # "openCLIP ViT-G/14": 0.87,
    "SD 1.5": 0.77,
    # "SD 1.5 (discffusion)": 0.77,
    # "SD 2.0": 0.82,
    # "SD 2.0 (discffusion)": 0.91,
    # "SD 3-m": 0.88,
    # "SD 3-m (discffusion)": 0.88
}

color_zero_scores_first_block = {
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

color_discffusion_scores_second_block = {
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

color_zero_scores_second_block = {
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
color_discffusion_scores_third_block = {
    # "CLIP RN50x64": 0.89,
    # "CLIP ViT-B/32": 0.88,
    # "CLIP ViT-L/14": 0.89,
    # "openCLIP ViT-H/14": 0.91,
    # "openCLIP ViT-G/14": 0.91,
    "SD 3-m": 0.98,
    # "SD 1.5": 0.78,
    # "SD 2.0": 0.82,
}

color_zero_scores_third_block = {
    # "CLIP RN50x64": 0.92,
    # "CLIP ViT-B/32": 0.91,
    # "CLIP ViT-L/14": 0.91,
    # "openCLIP ViT-H/14": 0.96,
    # "openCLIP ViT-G/14": 0.95,
    "SD 3-m": 0.98,
    # "SD 1.5": 0.82,
    # "SD 2.0": 0.87,
}

color_attribute_discffusion_scores_first_block = {
    # "CLIP RN50x64": 0.28,
    # "CLIP ViT-B/32": 0.25,
    # "CLIP ViT-L/14": 0.29,
    # "openCLIP ViT-H/14": 0.31,
    # "openCLIP ViT-G/14": 0.35,
    "SD 1.5": 0.5,
    # "SD 2.0": 0.26,
    # "SD 3-m": 0.26,
}

color_attribute_zero_scores_first_block = {
    # "CLIP RN50x64": 0.28,
    # "CLIP ViT-B/32": 0.22,
    # "CLIP ViT-L/14": 0.17,
    # "openCLIP ViT-H/14": 0.5,
    # "openCLIP ViT-G/14": 0.28,
    "SD 1.5": 0.83,
    # "SD 2.0": 0.5,
    # "SD 3-m": 0.56,
}

color_attribute_discffusion_scores_second_block = {
    # "CLIP RN50x64": 0.30,
    # "CLIP ViT-B/32": 0.35,
    # "CLIP ViT-L/14": 0.28,
    # "openCLIP ViT-H/14": 0.44,
    # "openCLIP ViT-G/14": 0.45,
    "SD 2.0": 0.83,
    # "SD 1.5": 0.37,
    # "SD 3-m": 0.28,
}

color_attribute_zero_scores_second_block = {
    # "CLIP RN50x64": 0.47,
    # "CLIP ViT-B/32": 0.47,
    # "CLIP ViT-L/14": 0.44,
    # "openCLIP ViT-H/14": 0.53,
    # "openCLIP ViT-G/14": 0.69,
    "SD 2.0": 0.88,
    # "SD 1.5": 0.42,
    # "SD 3-m": 0.53,
}

color_attribute_discffusion_scores_third_block = {
    # "CLIP RN50x64": 0.38,
    # "CLIP ViT-B/32": 0.43,
    # "CLIP ViT-L/14": 0.34,
    # "openCLIP ViT-H/14": 0.47,
    # "openCLIP ViT-G/14": 0.51,
    "SD 3-m": 0.95,
    # "SD 1.5": 0.53,
    # "SD 2.0": 0.56,
}

color_attribute_zero_scores_third_block = {
    # "CLIP RN50x64": 0.4,
    # "CLIP ViT-B/32": 0.43,
    # "CLIP ViT-L/14": 0.36,
    # "openCLIP ViT-H/14": 0.49,
    # "openCLIP ViT-G/14": 0.55,
    "SD 3-m": 0.95,
    # "SD 1.5": 0.58,
    # "SD 2.0": 0.63,
}

single_object_discffusion_first_block = {
    # "CLIP RN50x64": 0.97,
    # "CLIP ViT-B/32": 0.96,
    # "CLIP ViT-L/14": 0.97,
    # "openCLIP ViT-H/14": 0.97,
    # "openCLIP ViT-G/14": 0.97,
    "SD 1.5": 0.86,
    # "SD 2.0": 0.96,
    # "SD 3-m": 0.78,
}

single_object_zero_first_block = {
    # "CLIP RN50x64": 0.99,
    # "CLIP ViT-B/32": 0.99,
    # "CLIP ViT-L/14": 0.99,
    # "openCLIP ViT-H/14": 1.0,
    # "openCLIP ViT-G/14": 0.99,
    "SD 1.5": 1.0,
    # "SD 2.0": 0.99,
    # "SD 3-m": 0.81,
}

single_object_discffusion_second_block = {
    # "CLIP RN50x64": 0.99,
    # "CLIP ViT-B/32": 1.0,
    # "CLIP ViT-L/14": 0.99,
    # "openCLIP ViT-H/14": 1.0,
    # "openCLIP ViT-G/14": 1.0,
    "SD 2.0": 1.0,
    # "SD 1.5": 0.99,
    # "SD 3-m": 0.86,
}

single_object_zero_second_block = {
    # "CLIP RN50x64": 1.0,
    # "CLIP ViT-B/32": 1.0,
    # "CLIP ViT-L/14": 1.0,
    # "openCLIP ViT-H/14": 1.0,
    # "openCLIP ViT-G/14": 1.0,
    "SD 2.0": 1.0,
    # "SD 1.5": 0.99,
    # "SD 3-m": 0.87,
}

single_object_discffusion_third_block = {
    # "CLIP RN50x64": 0.99,
    # "CLIP ViT-B/32": 1.0,
    # "CLIP ViT-L/14": 0.99,
    # "openCLIP ViT-H/14": 1.0,
    # "openCLIP ViT-G/14": 1.0,
    "SD 3-m": 1.0,
    # "SD 1.5": 1.0,
    # "SD 2.0": 0.99,
}

single_object_zero_third_block = {
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
    "discffusion_colors": color_discffusion_scores_first_block,
    "zero_colors": color_zero_scores_first_block,
})

df_1_5_single = pd.DataFrame({
    "discffusion_single": single_object_discffusion_first_block,
    "zero_single": single_object_zero_first_block,
})

df_1_5_two_objects = pd.DataFrame({
    "discffusion_two": two_objects_discffusion_first_block,
    "zero_two": two_objects_zero_first_block,
})

df_1_5_colors_attr = pd.DataFrame({
    "discffusion_color_attr": color_attribute_discffusion_scores_first_block,
    "zero_color_attr": color_attribute_zero_scores_first_block,
})

df_1_5_position = pd.DataFrame({
    "discffusion_position": position_discffusion_scores_first_block,
    "zero_position": position_zero_scores_first_block,
})

df_1_5_counting = pd.DataFrame({
    "discffusion_counting": counting_discffusion_scores_first_block,
    "zero_counting": counting_zero_scores_first_block,
})

df_1_5 = pd.concat([df_1_5_single, df_1_5_two_objects, df_1_5_colors, df_1_5_colors_attr, df_1_5_position, df_1_5_counting], axis=1)

df_2_0_colors = pd.DataFrame({
    "discffusion_colors": color_discffusion_scores_second_block,
    "zero_colors": color_zero_scores_second_block,
})

df_2_0_single = pd.DataFrame({
    "discffusion_single": single_object_discffusion_second_block,
    "zero_single": single_object_zero_second_block,
})

df_2_0_two_objects = pd.DataFrame({
    "discffusion_two": two_objects_discffusion_second_block,
    "zero_two": two_objects_zero_second_block,
})

df_2_0_colors_attr = pd.DataFrame({
    "discffusion_color_attr": color_attribute_discffusion_scores_second_block,
    "zero_color_attr": color_attribute_zero_scores_second_block,
})

df_2_0_position = pd.DataFrame({
    "discffusion_position": position_discffusion_scores_second_block,
    "zero_position": position_zero_scores_second_block,
})

df_2_0_counting = pd.DataFrame({
    "discffusion_counting": counting_discffusion_scores_second_block,
    "zero_counting": counting_zero_scores_second_block,
})

df_2_0 = pd.concat([df_2_0_single, df_2_0_two_objects, df_2_0_colors, df_2_0_colors_attr, df_2_0_position, df_2_0_counting], axis=1)


df_3_m_colors = pd.DataFrame({
    "discffusion_colors": color_discffusion_scores_third_block,
    "zero_colors": color_zero_scores_third_block,
})

df_3_m_single = pd.DataFrame({
    "discffusion_single": single_object_discffusion_third_block,
    "zero_single": single_object_zero_third_block,
})

df_3_m_two_objects = pd.DataFrame({
    "discffusion_two": two_objects_discffusion_third_block,
    "zero_two": two_objects_zero_third_block,
})

df_3_m_colors_attr = pd.DataFrame({
    "discffusion_color_attr": color_attribute_discffusion_scores_third_block,
    "zero_color_attr": color_attribute_zero_scores_third_block,
})

df_3_m_position = pd.DataFrame({
    "discffusion_position": position_discffusion_scores_third_block,
    "zero_position": position_zero_scores_third_block,
})

df_3_m_counting = pd.DataFrame({
    "discffusion_counting": counting_discffusion_scores_third_block,
    "zero_counting": counting_zero_scores_third_block,
})

df_3_m = pd.concat([df_3_m_single, df_3_m_two_objects, df_3_m_colors, df_3_m_colors_attr, df_3_m_position, df_3_m_counting], axis=1)

# Function to create bar plots
def create_bar_plot(df_transposed, category_name):
    # df_transposed = df.T
    # print(df_transposed)
    # exit(0)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 2))
    # bar plot with matplotlib
    bar_width = 0.4
    group_spacing = 0.1 
    color_blind_pallete = sns.color_palette("colorblind", 10)
    # Example: manually select one column per category
    # (Adjust these column names as needed)
    df_transposed_discffusion = df_transposed[['discffusion_single', 'discffusion_two', 'discffusion_colors', 'discffusion_color_attr', 'discffusion_position', 'discffusion_counting']]
    df_transposed_zero = df_transposed[['zero_single', 'zero_two', 'zero_colors', 'zero_color_attr', 'zero_position', 'zero_counting']]


    num_groups = len(df_transposed_discffusion.columns)
    x = np.arange(num_groups)  # positions for each category

    categories = ["Single", "Two", "Colors", "Color Attr", "Position", "Counting"]

    discffusion_values = df_transposed_discffusion.iloc[0].values.astype(float)  # shape (6,)
    zero_values = df_transposed_zero.iloc[0].values.astype(float)  # shape (6,)
    palette = sns.color_palette("colorblind", 20)
    import matplotlib.colors as mcolors
    hsv_palette = [mcolors.rgb_to_hsv(color) for color in palette]

    # Filter the palette to remove certain hues
    filtered_palette = [
        color for color, hsv in zip(palette, hsv_palette)
        if not (hsv[0] < 0.1 or hsv[0] > 0.9 or (0.55 < hsv[0] < 0.75))
    ]

    bars_zero = ax.bar(x - bar_width/2, zero_values, bar_width, label="zero", color=filtered_palette[4], zorder=10)

    bars_discffusion = ax.bar(x + bar_width/2, discffusion_values, bar_width, label="discffusion", color=color_blind_pallete[0], zorder= 10)
    
    height1 = []
    for bar in  bars_zero :
        height = bar.get_height()
        height1.append(height)
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height*100:.0f}',
                ha='center', va='bottom', fontsize=18)

    # Annotate the bars for "zero"
    for idx, bar in  enumerate(bars_discffusion):

        height = bar.get_height()
        if height1[idx] < height:
        
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{height*100:.0f}',
                    ha='center', va='bottom', fontsize=18, color=color_blind_pallete[2])
        elif height1[idx] > height:
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{height*100:.0f}',
                    ha='center', va='bottom', fontsize=18, color=color_blind_pallete[3])
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{height*100:.0f}',
                    ha='center', va='bottom', fontsize=18)

    # Make the plot
    # for i, model in enumerate(df_transposed_discffusion.columns):
    #     print(model)
    #     plt.bar(index[i] + (i+ group_spacing) * barWidth , df_transposed_discffusion[model].values.flatten(),
    #             color='skyblue', width=barWidth, edgecolor='grey', label=model, zorder=10)
    #     break

    # for i, model in enumerate(df_transposed_zero.columns):
    #     plt.bar(index + (i*1 ) * barWidth, df_transposed_zero[model].values.flatten(),
    #             color='orange', width=barWidth, edgecolor='grey', label=model, zorder=10)

    plt.ylabel('Accuracy (%)', fontsize=17)

    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.yticks(fontsize=12)  # Change the Y-axis tick label size
    y_max = plt.gca().get_ylim()[1]
    plt.yticks(np.linspace(0, 1, 6),fontsize=12)  # Change the Y-axis tick label size
    y_max = plt.gca().get_ylim()[1]
    import matplotlib.ticker as mticker

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    plt.yticks([])


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
    ax.spines['bottom'].set_zorder(100000)
    ax.set_xticks(np.arange(len(categories)))  # Set tick positions for 6 categories
    ax.set_xticklabels(categories, fontsize=18)
    


    plt.tight_layout()
    outdir = 'plots/discffusion_zero'
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f'{outdir}/{category_name}.pdf', dpi=300, bbox_inches='tight')
    # plt.show()

create_bar_plot(df_1_5, "First_Block")
create_bar_plot(df_2_0, "Second_Block")
create_bar_plot(df_3_m, "Third_Block")



