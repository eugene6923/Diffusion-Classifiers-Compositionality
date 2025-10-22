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
    "Full_colors": color_full_scores_first_block,
    "Correct_colors": color_correct_scores_first_block,
})

df_1_5_single = pd.DataFrame({
    "Full_single": single_object_full_first_block,
    "Correct_single": single_object_correct_first_block,
})

df_1_5_two_objects = pd.DataFrame({
    "Full_two": two_objects_full_first_block,
    "Correct_two": two_objects_correct_first_block,
})

df_1_5_colors_attr = pd.DataFrame({
    "Full_color_attr": color_attribute_full_scores_first_block,
    "Correct_color_attr": color_attribute_correct_scores_first_block,
})

df_1_5_position = pd.DataFrame({
    "Full_position": position_full_scores_first_block,
    "Correct_position": position_correct_scores_first_block,
})

df_1_5_counting = pd.DataFrame({
    "Full_counting": counting_full_scores_first_block,
    "Correct_counting": counting_correct_scores_first_block,
})

df_1_5 = pd.concat([df_1_5_single, df_1_5_two_objects, df_1_5_colors, df_1_5_colors_attr, df_1_5_position, df_1_5_counting], axis=1)

df_2_0_colors = pd.DataFrame({
    "Full_colors": color_full_scores_second_block,
    "Correct_colors": color_correct_scores_second_block,
})

df_2_0_single = pd.DataFrame({
    "Full_single": single_object_full_second_block,
    "Correct_single": single_object_correct_second_block,
})

df_2_0_two_objects = pd.DataFrame({
    "Full_two": two_objects_full_second_block,
    "Correct_two": two_objects_correct_second_block,
})

df_2_0_colors_attr = pd.DataFrame({
    "Full_color_attr": color_attribute_full_scores_second_block,
    "Correct_color_attr": color_attribute_correct_scores_second_block,
})

df_2_0_position = pd.DataFrame({
    "Full_position": position_full_scores_second_block,
    "Correct_position": position_correct_scores_second_block,
})

df_2_0_counting = pd.DataFrame({
    "Full_counting": counting_full_scores_second_block,
    "Correct_counting": counting_correct_scores_second_block,
})

df_2_0 = pd.concat([df_2_0_single, df_2_0_two_objects, df_2_0_colors, df_2_0_colors_attr, df_2_0_position, df_2_0_counting], axis=1)


df_3_m_colors = pd.DataFrame({
    "Full_colors": color_full_scores_third_block,
    "Correct_colors": color_correct_scores_third_block,
})

df_3_m_single = pd.DataFrame({
    "Full_single": single_object_full_third_block,
    "Correct_single": single_object_correct_third_block,
})

df_3_m_two_objects = pd.DataFrame({
    "Full_two": two_objects_full_third_block,
    "Correct_two": two_objects_correct_third_block,
})

df_3_m_colors_attr = pd.DataFrame({
    "Full_color_attr": color_attribute_full_scores_third_block,
    "Correct_color_attr": color_attribute_correct_scores_third_block,
})

df_3_m_position = pd.DataFrame({
    "Full_position": position_full_scores_third_block,
    "Correct_position": position_correct_scores_third_block,
})

df_3_m_counting = pd.DataFrame({
    "Full_counting": counting_full_scores_third_block,
    "Correct_counting": counting_correct_scores_third_block,
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
    df_transposed_full = df_transposed[['Full_single', 'Full_two', 'Full_colors', 'Full_color_attr', 'Full_position', 'Full_counting']]
    df_transposed_correct = df_transposed[['Correct_single', 'Correct_two', 'Correct_colors', 'Correct_color_attr', 'Correct_position', 'Correct_counting']]


    num_groups = len(df_transposed_full.columns)
    x = np.arange(num_groups)  # positions for each category

    categories = ["Single", "Two", "Colors", "Color Attr", "Position", "Counting"]

    full_values = df_transposed_full.iloc[0].values.astype(float)  # shape (6,)
    correct_values = df_transposed_correct.iloc[0].values.astype(float)  # shape (6,)

    palette = sns.color_palette("colorblind", 10)
    import matplotlib.colors as mcolors

    # Convert each RGB color to HSV
    hsv_palette = [mcolors.rgb_to_hsv(color) for color in palette]

    # Filter the palette to remove certain hues
    filtered_palette = [
        color for color, hsv in zip(palette, hsv_palette)
        if not (hsv[0] < 0.1 or hsv[0] > 0.9 or (0.55 < hsv[0] < 0.75))
    ]


    bars_full = ax.bar(x - bar_width/2, full_values, bar_width, label="Full", color=color_blind_pallete[0], zorder= 10)
    bars_correct = ax.bar(x + bar_width/2, correct_values, bar_width, label="Correct", color=filtered_palette[4], zorder=10)



    for bar in bars_full:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height*100:.0f}',
                ha='center', va='bottom', fontsize=18)

    # Annotate the bars for "Correct"
    for bar in bars_correct:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height*100:.0f}',
                ha='center', va='bottom', fontsize=18, color=color_blind_pallete[2])

    # Make the plot
    # for i, model in enumerate(df_transposed_full.columns):
    #     print(model)
    #     plt.bar(index[i] + (i+ group_spacing) * barWidth , df_transposed_full[model].values.flatten(),
    #             color='skyblue', width=barWidth, edgecolor='grey', label=model, zorder=10)
    #     break

    # for i, model in enumerate(df_transposed_correct.columns):
    #     plt.bar(index + (i*1 ) * barWidth, df_transposed_correct[model].values.flatten(),
    #             color='orange', width=barWidth, edgecolor='grey', label=model, zorder=10)

    plt.ylabel('Accuracy (%)', fontsize=18)

    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.yticks(fontsize=18)  # Change the Y-axis tick label size
    y_max = plt.gca().get_ylim()[1]
    plt.yticks(np.linspace(0, 1, 6),fontsize=18)  # Change the Y-axis tick label size
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
    outdir = 'plots/full_correct'
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f'{outdir}/{category_name}.png', dpi=300, bbox_inches='tight')
    # plt.show()

create_bar_plot(df_1_5, "First_Block")
create_bar_plot(df_2_0, "Second_Block")
create_bar_plot(df_3_m, "Third_Block")



