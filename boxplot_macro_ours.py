import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd

# Data for Generation Accuracy

colors_generation_accuracy = {
    "SD1.5_Generation": 219 / 376,
    "SD2.0_Generation": 263 / 376,
    "SD3-m_Generation": 314 / 376
}
color_attr_generation_accuracy = {
    "SD1.5_Generation": 18 / 400,
    "SD2.0_Generation": 36 / 400,
    "SD3-m_Generation": 252 / 400
}

single_object_generation_accuracy = {
    "SD1.5_Generation": 271 / 320,
    "SD2.0_Generation": 271 / 320,
    "SD3-m_Generation": 314 / 320
}
two_objects_generation_accuracy = {
    "SD1.5_Generation": 105 / 396,
    "SD2.0_Generation": 129 / 396,
    "SD3-m_Generation": 306 / 396
}

position_generation_accruacy = {
    "SD1.5_Generation": 6 / 400,
    "SD2.0_Generation": 19 / 400,
    "SD3-m_Generation": 113 / 400
}
counting_generation_accruacy = {
    "SD1.5_Generation": 98 / 376,
    "SD2.0_Generation": 111 / 376,
    "SD3-m_Generation": 230 / 376
}


# Macro Accuracy Data

color_macro_first_block = {
    "Generation": colors_generation_accuracy["SD1.5_Generation"],
    "CLIP RN50x64": 0.92,
    "CLIP ViT-B/32": 0.90,
    "CLIP ViT-L/14": 0.92,
    "openCLIP ViT-H/14": 0.97,
    "openCLIP ViT-G/14": 0.98,
    "SD1.5": 0.97,
    "SD2.0": 0.91,
    "SD3-m": 0.88,
}

color_macro_second_block = {
    "Generation": colors_generation_accuracy["SD2.0_Generation"],
    "CLIP RN50x64": 0.94,
    "CLIP ViT-B/32": 0.94,
    "CLIP ViT-L/14": 0.93,
    "openCLIP ViT-H/14": 0.97,
    "openCLIP ViT-G/14": 0.98,
   "SD2.0" : 0.99,
    "SD1.5": 0.90,
    "SD3-m": 0.91,
}
color_macro_third_block = {
    "Generation": colors_generation_accuracy["SD3-m_Generation"],
    "CLIP RN50x64": 0.91,
    "CLIP ViT-B/32": 0.90,
    "CLIP ViT-L/14": 0.91,
    "openCLIP ViT-H/14": 0.96,
    "openCLIP ViT-G/14": 0.95,
    "SD3-m": 0.98,
    "SD1.5": 0.83,
    "SD2.0": 0.86,
}

single_macro_first_block = {
    "Generation": single_object_generation_accuracy["SD1.5_Generation"],
    "CLIP RN50x64": 0.99,
    "CLIP ViT-B/32": 0.99,
    "CLIP ViT-L/14": 0.99,
    "openCLIP ViT-H/14": 1.0,
    "openCLIP ViT-G/14": 0.99,
    "SD1.5": 1.0,
    "SD2.0": 0.99,
    "SD3-m": 0.80,
}

single_macro_second_block = {
    "Generation": single_object_generation_accuracy["SD2.0_Generation"],
    "CLIP RN50x64": 1.0,
    "CLIP ViT-B/32": 1.0,
    "CLIP ViT-L/14": 1.0,
    "openCLIP ViT-H/14": 1.0,
    "openCLIP ViT-G/14": 1.0,
    "SD1.5": 0.99,
    "SD2.0": 1.0,
    "SD3-m": 0.88,
}

single_macro_third_block = {
    "Generation": single_object_generation_accuracy["SD3-m_Generation"],

    "CLIP RN50x64": 0.99,
    "CLIP ViT-B/32": 1.00,
    "CLIP ViT-L/14": 0.99,
    "openCLIP ViT-H/14": 1.00,
    "openCLIP ViT-G/14": 1.00,
    "SD1.5": 1.0,
    "SD2.0": 0.99,
    "SD3-m": 1.0,
}

two_macro_first_block = {
    "Generation": two_objects_generation_accuracy["SD1.5_Generation"],
    "CLIP RN50x64": 0.85,
    "CLIP ViT-B/32": 0.85,
    "CLIP ViT-L/14": 0.96,
    "openCLIP ViT-H/14": 0.96,
    "openCLIP ViT-G/14": 0.95,
    "SD1.5": 0.92,
    "SD2.0": 0.85,
    "SD3-m": 0.59,
}

two_macro_second_block = {
    "Generation": two_objects_generation_accuracy["SD2.0_Generation"],
    "CLIP RN50x64": 0.97,
    "CLIP ViT-B/32": 0.87,
    "CLIP ViT-L/14": 0.94,
    "openCLIP ViT-H/14": 0.99,
    "openCLIP ViT-G/14": 0.98,
    "SD1.5": 0.92,
    "SD2.0": 0.99,
    "SD3-m": 0.58,
}

two_macro_third_block = {
    "Generation": two_objects_generation_accuracy["SD3-m_Generation"],
    "CLIP RN50x64": 0.93,
    "CLIP ViT-B/32": 0.90,
    "CLIP ViT-L/14": 0.98,
    "openCLIP ViT-H/14": 0.98,
    "openCLIP ViT-G/14": 0.99,
    "SD1.5": 0.92,
    "SD2.0": 0.94,
    "SD3-m": 0.99,
}

color_attr_macro_first_block = {
    "Generation": color_attr_generation_accuracy["SD1.5_Generation"],
    "CLIP RN50x64": 0.35,
    "CLIP ViT-B/32": 0.21,
    "CLIP ViT-L/14": 0.15,
    "openCLIP ViT-H/14": 0.65,
    "openCLIP ViT-G/14": 0.31,
    "SD1.5": 0.87,
    "SD2.0": 0.65,
    "SD3-m": 0.58,
}

color_attr_macro_second_block = {
    "Generation": color_attr_generation_accuracy["SD2.0_Generation"],
    "CLIP RN50x64": 0.43,
    "CLIP ViT-B/32": 0.50,
    "CLIP ViT-L/14": 0.44,
    "openCLIP ViT-H/14": 0.51,
    "openCLIP ViT-G/14": 0.71,
    "SD1.5": 0.38,
    "SD2.0": 0.85,
    "SD3-m": 0.49,
}

color_attr_macro_third_block = {
    "Generation": color_attr_generation_accuracy["SD3-m_Generation"],
    "CLIP RN50x64": 0.40,
    "CLIP ViT-B/32": 0.43,
    "CLIP ViT-L/14": 0.36,
    "openCLIP ViT-H/14": 0.49,
    "openCLIP ViT-G/14": 0.55,
    "SD1.5": 0.58,
    "SD2.0": 0.63,
    "SD3-m": 0.95,
}

counting_macro_first_block = {
    "Generation": counting_generation_accruacy["SD1.5_Generation"],
    "CLIP RN50x64": 0.47,
    "CLIP ViT-B/32": 0.50,
    "CLIP ViT-L/14": 0.45,
    "openCLIP ViT-H/14": 0.66,
    "openCLIP ViT-G/14": 0.68,
    "SD1.5": 0.86,
    "SD2.0": 0.40,
    "SD3-m": 0.41,
}

counting_macro_second_block = {
    "Generation": counting_generation_accruacy["SD2.0_Generation"],
    "CLIP RN50x64": 0.53,
    "CLIP ViT-B/32": 0.48,
    "CLIP ViT-L/14": 0.54,
    "openCLIP ViT-H/14": 0.70,
    "openCLIP ViT-G/14": 0.73,
    "SD2.0": 0.78,
    "SD1.5": 0.47,
    "SD3-m": 0.65,
}

counting_macro_third_block = {
    "Generation": counting_generation_accruacy["SD3-m_Generation"],
    "CLIP RN50x64": 0.53,
    "CLIP ViT-B/32": 0.50,
    "CLIP ViT-L/14": 0.56,
    "openCLIP ViT-H/14": 0.73,
    "openCLIP ViT-G/14": 0.72,
    "SD3-m": 0.86,
    "SD1.5": 0.63,
    "SD2.0": 0.65,
}

position_macro_first_block = {
    "Generation": position_generation_accruacy["SD1.5_Generation"],
    "CLIP RN50x64": 0.12,
    "CLIP ViT-B/32": 0.38,
    "CLIP ViT-L/14": 0.29,
    "openCLIP ViT-H/14": 0.21,
    "openCLIP ViT-G/14": 0.38,
    "SD1.5": 0.42,
    "SD2.0": 0.38,
    "SD3-m": 0.21,
}

position_macro_second_block = {
    "Generation": position_generation_accruacy["SD2.0_Generation"],
    "CLIP RN50x64": 0.16,
    "CLIP ViT-B/32": 0.10,
    "CLIP ViT-L/14": 0.17,
    "openCLIP ViT-H/14": 0.49,
    "openCLIP ViT-G/14": 0.58,
    "SD2.0": 0.64,
    "SD1.5": 0.21,
    "SD3-m": 0.58,
}

position_macro_third_block = {
    "Generation": position_generation_accruacy["SD3-m_Generation"],
    "CLIP RN50x64": 0.29,
    "CLIP ViT-B/32": 0.22,
    "CLIP ViT-L/14": 0.29,
    "openCLIP ViT-H/14": 0.35,
    "openCLIP ViT-G/14": 0.36,
    "SD3-m": 0.90,
    "SD1.5": 0.27,
    "SD2.0": 0.30,
}


df_colors = {
    "Colors (Block1)": color_macro_first_block,
    "Colors (Block2)": color_macro_second_block,
    "Colors (Block3)": color_macro_third_block,
}

df_single = {
    "Single Object (Block1)": single_macro_first_block,
    "Single Object (Block2)": single_macro_second_block,
    "Single Object (Block3)": single_macro_third_block,
}

df_two = {
    "Two Objects (Block1)": two_macro_first_block,
    "Two Objects (Block2)": two_macro_second_block,
    "Two Objects (Block3)": two_macro_third_block,
}

df_color_attr = {
    "Color_attr (Block1)": color_attr_macro_first_block,
    "Color_attr (Block2)": color_attr_macro_second_block,
    "Color_attr (Block3)": color_attr_macro_third_block,
}

df_counting = {
    "Counting (Block1)": counting_macro_first_block,
    "Counting (Block2)": counting_macro_second_block,
    "Counting (Block3)": counting_macro_third_block,
}

df_position = {
    "Position (Block1)": position_macro_first_block,
    "Position (Block2)": position_macro_second_block,
    "Position (Block3)": position_macro_third_block,
}

df_colors = pd.DataFrame(df_colors)
df_single = pd.DataFrame(df_single)
df_two = pd.DataFrame(df_two)
df_color_attr = pd.DataFrame(df_color_attr)
df_counting = pd.DataFrame(df_counting)
df_position = pd.DataFrame(df_position)


def create_bar_plot(df, category_name):
    df_transposed = df.T

    # Color schemes
    # colorblind_palette = sns.color_palette("colorblind", len(df_transposed.columns))
    # clip_colors = colorblind_palette[:5]
    # sd_colors = colorblind_palette[5:]
    # generation_color ='#9C27B0'  # Purple
    # Alternative Option:
    generation_color = '#E64A19'  # Orange
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
    sd_models = [model for model in df_transposed.columns  if (model not in clip_models and model != "Generation")]
    

    # Plotting
    # if category_name == "Spatial" or category_name == "Attributes":
    #     bar_width = 0.2
    # else: 
    bar_width = 0.1
    group_spacing = 0.2
    index = np.arange(len(df_transposed.index)) * (len(df_transposed.columns) * bar_width + group_spacing)

        
    # if category_name == "Spatial" or category_name == "Attributes":
    #     print("Spatial or Attributes")
    #     fig, ax = plt.subplots(figsize=(30, 5))

    # else:
    if "Object" in category_name:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig, ax = plt.subplots(figsize=(5, 5))
    gen_offset = 0  # Generation Accuracy first

    # Optional: Add text labels above Generation bars
    # Adjust the positions of bars to avoid overlapping
    

    # Increase the spacing between groups
    
    # Define offsets
    clip_offset = 0
    gap = 0.4  # Small gap between clip and generation
    gen_offset = len(clip_models) + gap  # Generation after Clip with a gap
    sd_offset = gen_offset + 1  # SD Models right after Generation

    # Plot Clip Models
    generation_plotted = False

# Plot Clip Models
    for i, model in enumerate(clip_models):
        plt.bar(index + (clip_offset + i) * bar_width, 
                df_transposed[model], 
                bar_width, 
                label=model, 
                color=clip_colors[i], 
                zorder=10)

    # Plot Generation with a small gap after Clip
    # if 'Generation' in df_transposed.columns and not generation_plotted:
    if 'Generation' in df_transposed.columns:
        if df_transposed['Generation'].notna().all():  # Check if it has values
            plt.bar(index + (gen_offset * bar_width), 
                    df_transposed['Generation'], 
                    bar_width, 
                    label='Generation', 
                    color=generation_color, 
                    zorder=10)

    # Plot SD Models
    for j, model in enumerate(sd_models):
        plt.bar(index + (sd_offset + j) * bar_width, 
                df_transposed[model], 
                bar_width, 
                label=model, 
                color=sd_colors[j], 
                zorder=10)

    
    # for i, model in enumerate(clip_models):
    #     plt.bar(index + (clip_offset + i) * bar_spacing, 
    #         df_transposed[model], 
    #         bar_width, 
    #         label=model, 
    #         color=clip_colors[i], 
    #         zorder=10)
    #     # bars = plt.bar(index + i * bar_width, df_transposed[model], bar_width, label=model, color=clip_colors[i], zorder=10)
    #     # if category_name == "Spatial":
    #     # for bar in bars:
    #     #     yval = bar.get_height()
    #     #     plt.text(bar.get_x() + bar.get_width() / 2, 
    #     #             yval + 0.01,  # Position slightly above the bar
    #     #             f'{yval:.2f}',  # Display with 2 decimal places
    #     #             ha='center', 
    #     #             va='bottom',
    #     #             fontsize=15, 
    #     #                 rotation=80,
    #     #             color='black')
    # bars = plt.bar(index + gen_offset * bar_spacing, 
    #                df_transposed['Generation'], 
    #                bar_width, 
    #                label='Generation', 
    #                color=generation_color, 
    #                zorder=10)

    # for j, model in enumerate(sd_models):
    #     plt.bar(index + (sd_offset + j) * bar_spacing, 
    #         df_transposed[model], 
    #         bar_width, 
    #         label=model, 
    #         color=sd_colors[j], 
    #         zorder=10)
        # bars = plt.bar(index + (len(clip_models) + j) * bar_width, df_transposed[model], bar_width, label=model, color=sd_colors[j], zorder=10)
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
    eqbench_pos = index[df_transposed.index.get_loc('EqBench')] if 'EqBench' in df_transposed.index else None
    mmvp_pos = index[df_transposed.index.get_loc('MMVP')] if 'MMVP' in df_transposed.index else None
    # Get x-axis position for Eq-Kubric if it exists
    eq_kubric_pos = index[df_transposed.index.get_loc('Eq-Kubric')] if 'Eq-Kubric' in df_transposed.index else None
    winoground_pos = index[df_transposed.index.get_loc('Winoground')] if 'Winoground' in df_transposed.index else None
    clevr_pos = index[df_transposed.index.get_loc('Clevr')] if 'Clevr' in df_transposed.index else None
    clevr_color_pos = index[df_transposed.index.get_loc('Clevr (Color)')] if 'Clevr (Color)' in df_transposed.index else None
    clevr_color_binding_pos = index[df_transposed.index.get_loc('Clevr (Color Binding)')] if 'Clevr (Color Binding)' in df_transposed.index else None
    # Add red line for random chance at 0.25 for Eq-Kubric
    
    if category_name == "Spatial":
        plt.axhline(y=1/4, color='red', linestyle='--', zorder=10)
    elif category_name == "Counting":
        plt.axhline(y=1/4, color='red', linestyle='--', zorder=10)
    elif category_name == "Single Object":
        plt.axhline(y=1/80, color='red', linestyle='--', zorder=10)
    elif category_name == "Color_attr":
        plt.axhline(y=1/100, color='red', linestyle='--', zorder=10)
    elif category_name == "Colors":
        plt.axhline(y=0.1, color='red', linestyle='--', zorder=10)
    elif category_name == "Two Objects":
        plt.axhline(y=1/101, color='red', linestyle='--', zorder=10)
    if eq_kubric_pos is not None:
        plt.axhline(y=0.25, color='red', linestyle='--', 
                    xmin=(eq_kubric_pos/plt.xlim()[1]+ 0.03), 
                    xmax=((eq_kubric_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+ 0.02), 
                    zorder=10)

    # Add red lines for random chance at 0.25
    if vismin_pos is not None:
        if category_name != "Attributes":
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(vismin_pos/plt.xlim()[1]+0.035), 
                        xmax=((vismin_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.04), 
                        zorder=10)
        else:
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(vismin_pos/plt.xlim()[1]+0.02), 
                        xmax=((vismin_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.03),
                        zorder=10)

    if eqbench_pos is not None:
        if category_name == "Counting":
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(eqbench_pos/plt.xlim()[1]+ 0.01), 
                        xmax=((eqbench_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+ 0.02), 
                        zorder=10)
        elif category_name == "Attributes":
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(eqbench_pos/plt.xlim()[1]+0.015), 
                        xmax=((eqbench_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+ 0.02),
                        zorder=10)
        else:
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(eqbench_pos/plt.xlim()[1]+ 0.04), 
                        xmax=((eqbench_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+ 0.03), 
                        zorder=10)

    if mmvp_pos is not None:
        if category_name == "Counting":
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(mmvp_pos/plt.xlim()[1]), 
                        xmax=((mmvp_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]),
                        zorder=10)
        elif category_name == "Attributes":
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(mmvp_pos/plt.xlim()[1]+0.01), 
                        xmax=((mmvp_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.02),
                        zorder=10)
        else:
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(mmvp_pos/plt.xlim()[1]+0.03), 
                        xmax=((mmvp_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+ 0.04), 
                        zorder=10)
    if winoground_pos is not None:
        plt.axhline(y=0.25, color='red', linestyle='--', 
                    xmin=(winoground_pos/plt.xlim()[1]), 
                    xmax=((winoground_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]), 
                    zorder=10)
    if clevr_pos is not None:
        plt.axhline(y=0.5, color='red', linestyle='--', 
                    xmin=(clevr_pos/plt.xlim()[1]+0.03), 
                    xmax=((clevr_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.03),
                    zorder =10)
    if  clevr_color_pos is not None or clevr_color_binding_pos is not None:
        plt.axhline(y=0.5, color='red', linestyle='--', 
                    xmin=(clevr_color_pos/plt.xlim()[1]), 
                    xmax=((clevr_color_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.01),
                    zorder =10)
        plt.axhline(y=0.5, color='red', linestyle='--', 
                    xmin=(clevr_color_binding_pos/plt.xlim()[1]), 
                    xmax=((clevr_color_binding_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.01),
                    zorder =10)
    if category_name == "Counting":
        # Get x-axis position for Spec if it exists
        spec_pos = index[df_transposed.index.get_loc('Spec')] if 'Spec' in df_transposed.index else None

        # Add red line for random chance at 1/9 (0.11) for Spec
        if spec_pos is not None:
            plt.axhline(y=1/9, color='red', linestyle='--', 
                        xmin=(spec_pos/plt.xlim()[1] +0.02), 
                        xmax=((spec_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1] + 0.03), 
                        zorder=10)


    if category_name == "Object":
        # Get x-axis position for Spec if it exists
        sugar_pos = index[df_transposed.index.get_loc('Sugarcrepe')] if 'Sugarcrepe' in df_transposed.index else None

        # Add red line for random chance at 1/9 (0.11) for Spec
        if sugar_pos is not None:
            plt.axhline(y=1/2, color='red', linestyle='--', 
                        xmin=(sugar_pos/plt.xlim()[1] ), 
                        xmax=((sugar_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1] +0.02), 
                        zorder=10)
    if category_name == "Spatial":
        # Get x-axis position for Spec if it exists
        whatsA_pos = index[df_transposed.index.get_loc('WhatsupA')] if 'WhatsupA' in df_transposed.index else None
        if whatsA_pos is not None:
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(whatsA_pos/plt.xlim()[1] + 0.02), 
                        xmax=((whatsA_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1] + 0.03),
                        zorder=10)
            
        whatsupB_pos = index[df_transposed.index.get_loc('WhatsupB')] if 'WhatsupB' in df_transposed.index else None
        if whatsupB_pos is not None:
            plt.axhline(y=0.25, color='red', linestyle='--', 
                        xmin=(whatsupB_pos/plt.xlim()[1] + 0.02), 
                        xmax=((whatsupB_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1] + 0.03),
                        zorder=10)
        cocoone_pos = index[df_transposed.index.get_loc('COCO (One)')] if 'COCO (One)' in df_transposed.index else None
        if cocoone_pos is not None:
            plt.axhline(y=0.5, color='red', linestyle='--', 
                        xmin=(cocoone_pos/plt.xlim()[1] + 0.02), 
                        xmax=((cocoone_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1] + 0.03),
                        zorder=10)
        cocotwo_pos = index[df_transposed.index.get_loc('COCO (Two)')] if 'COCO (Two)' in df_transposed.index else None
        if cocotwo_pos is not None:
            plt.axhline(y=0.5, color='red', linestyle='--', 
                        xmin=(cocotwo_pos/plt.xlim()[1] + 0.02), 
                        xmax=((cocotwo_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1] + 0.02),
                        zorder=10)
        gqaone_pos = index[df_transposed.index.get_loc('GQA (One)')] if 'GQA (One)' in df_transposed.index else None
        if gqaone_pos is not None:
            plt.axhline(y=0.5, color='red', linestyle='--', 
                        xmin=(gqaone_pos/plt.xlim()[1] + 0.02), 
                        xmax=((gqaone_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1]+0.01 ),
                        zorder=10)
        gqatwo_pos = index[df_transposed.index.get_loc('GQA (Two)')] if 'GQA (Two)' in df_transposed.index else None
        if gqatwo_pos is not None:
            plt.axhline(y=0.5, color='red', linestyle='--', 
                        xmin=(gqatwo_pos/plt.xlim()[1] + 0.02), 
                        xmax=((gqatwo_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1] ),
                        zorder=10)
            
    if category_name == "Attributes":
        # Get x-axis position for Spec if it exists
        sugar_pos = index[df_transposed.index.get_loc('Sugarcrepe')] if 'Sugarcrepe' in df_transposed.index else None

        # Add red line for random chance at 1/9 (0.11) for Spec
        if sugar_pos is not None:
            plt.axhline(y=1/2, color='red', linestyle='--', 
                        xmin=(sugar_pos/plt.xlim()[1] + 0.035 ), 
                        xmax=((sugar_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1] + 0.045 ),
                        zorder=10)
        aro_pos = index[df_transposed.index.get_loc('ARO')] if 'ARO' in df_transposed.index else None
        if aro_pos is not None:
            plt.axhline(y=1/2, color='red', linestyle='--', 
                        xmin=(aro_pos/plt.xlim()[1] + 0.035 ), 
                        xmax=((aro_pos + bar_width * len(df_transposed.columns))/plt.xlim()[1] + 0.045 ),
                        zorder=10)
    # Minimalistic Design
    # plt.yticks([])
    # plt.xticks([])
    # Add x-axis labels (Benchmark names)
    # Turn on the left spine
    
    # plt.ylabel('Accuracy', fontsize=25)
    # plt.xticks(ha='center')
    xtick_label = ['SD1.5', 'SD2.0', 'SD3-m']
    
    # xtick_label = [
    #     label.replace('SD', 'SD\n')
    #     for label in xtick_labels
    # ]
    if "Object" in category_name or "Objects" in category_name or "Counting" in category_name:
        # print("here?")
        xtick_label = ['SD1.5', 'SD2.0', 'SD3-m']
    if "Object" in category_name or "Counting" in category_name:
        plt.xticks(index + bar_width * (len(df_transposed.columns) - 1) / 2, 
        #    df_transposed.index, 
            xtick_label,
        #    rotation=20, 

           ha='center', 

           fontsize=25)
        plt.yticks(fontsize=23)
    else:
        plt.xticks(index + bar_width * (len(df_transposed.columns) - 1) / 2, 
        #    df_transposed.index, 
            xtick_label,
        #    rotation=20, 

           ha='center', 

           fontsize=25)
        plt.yticks(fontsize=25)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
      # Change the Y-axis tick label size
    y_max = plt.gca().get_ylim()[1]
    plt.yticks(np.linspace(0, 1, 6),fontsize=25)  # Change the Y-axis tick label size
    y_max = plt.gca().get_ylim()[1]
    import matplotlib.ticker as mticker
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    

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
    outdir = 'plots/macro_ours/'
    os.makedirs(f'{outdir}', exist_ok=True)

    plt.savefig(f'{outdir}/{category_name}_scores_plot.png', dpi=300, bbox_inches='tight')
    # plt.show()

# Create plots for each category
create_bar_plot(df_colors, "Colors")
create_bar_plot(df_color_attr, "Color_attr")
create_bar_plot(df_counting, "Counting")
create_bar_plot(df_single, "Single Object")
create_bar_plot(df_position, "Spatial")
create_bar_plot(df_two, "Two Objects")
