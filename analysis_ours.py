import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

version = ["1.5", "2.0", "3-m"]

for ver in version:
    file = pd.read_csv(f"results/analysis/csv/geneval_full_{ver}_labeled.csv")
    tag_list = set(file["tag"].values)

    dict_class = {}
    for tag in tag_list:
        # Filter the file for the current tag
        file_tag = file[file["tag"] == tag]

        # Define `choice` based on the current tag
        if tag in ["color_attr", "colors"]:
            choice = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
        elif tag == "position":
            choice = ["left of", "right of", "above", "below"]
        elif tag == "counting":
            choice = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        else:
            choice = []  # Default to an empty list for unknown tags

        # Skip plotting if `choice` is empty
        if not choice:
            continue

        # Populate `dict_class` for the current choice
        for c in choice:
            count_c = len(file[file["prompt"].str.contains(c, na=False)])
            count_c_changed = len(file[file["real_prompt"].str.contains(c, na=False)])
            count_c_not_changed = len(file[file["prompt"].str.contains(c, na=False)])

            dict_class[f"{c}"] = [count_c, count_c_changed, count_c_not_changed]

        # Plotting
        plt.figure()
        x = np.arange(len(choice))  # Numeric positions for the bars
        bar_width = 0.4

        # "Changed" bars
        plt.bar(x - bar_width / 2, [dict_class[c][1] for c in choice], width=bar_width, label="Changed")

        # "Not changed" bars
        plt.bar(x + bar_width / 2, [dict_class[c][2] for c in choice], width=bar_width, label="Not Changed")

        # Set x-ticks and labels
        plt.xticks(x, choice, rotation=90)  # Use `choice` for x-axis labels
        plt.xlabel(f"{tag}")
        plt.ylabel("Count")
        plt.title(f"{ver} {tag}")
        plt.legend()

        # Save the figure
        os.makedirs(f"results/analysis/graph/class", exist_ok=True)
        plt.savefig(f"results/analysis/graph/class/{ver}_{tag}.png")
        plt.close()
