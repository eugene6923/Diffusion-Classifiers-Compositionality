import re
import ast

image_list = []  # Stores image paths
text_list = None  # Will store the full text list (only one occurrence expected)
guess_list = []  # Stores guessed values
gt_list = []  # Stores ground truth values
top5_list = []  # Stores top5 scores
top5_idx_list = []  # Stores top5 index values
top5_text_list = []  # Stores the extracted top 5 text descriptions

file_path = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/confusion/ours_1_5_color_attr/ours_1_5_color_attr_1.5_results.txt"
