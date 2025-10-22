import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

geneval = {"color_attr": {"sd1.5": 6.25, "sd2.0": 12.50, "sd2.1":10.75, "sd3-m": 58.75, "sd3-L": 55.50},
           "colors": {"sd1.5": 74.47, "sd2.0": 84.04, "sd2.1":84.57, "sd3-m": 82.98, "sd3-L": 84.04},
           "counting": {"sd1.5": 38.12, "sd2.0": 37.81, "sd2.1":44.38, "sd3-m": 70.62, "sd3-L": 73.44},
           "position": {"sd1.5": 2.5, "sd2.0": 7.25, "sd2.1":7.00, "sd3-m": 28.75, "sd3-L": 24.25},
           "single_object": {"sd1.5": 95.94, "sd2.0": 99.69, "sd2.1": 98.44, "sd3-m": 99.69, "sd3-L": 99.06},
           "two_object":{"sd1.5":38.89, "sd2.0": 47.47, "sd2.1": 48.74, "sd3-m": 86.87, "sd3-L": 93.18},
}

zero_shot = {"color_attr": {"sd1.5": 43.25, "sd2.0": 52, "sd2.1": 47.75, "sd3-m": 71, "sd3-L": 66},
                "colors": {"sd1.5": 87.77, "sd2.0": 87.23, "sd2.1": 90.96, "sd3-m": 72.07, "sd3-L":76.86170212765957 },
                "counting": {"sd1.5": 58.125, "sd2.0": 70.9375, "sd2.1":68.125, "sd3-m": 70, "sd3-L": 82.1875},
                "position": {"sd1.5": 50.25, "sd2.0": 58, "sd2.1": 57.75, "sd3-m": 82, "sd3-L": 76.75},
                "single_object": {"sd1.5": 96.86, "sd2.0": 100.0, "sd2.1": 99.69, "sd3-m": 98.13, "sd3-L": 99.375},
                "two_object":{"sd1.5": 42.17171717171717, "sd2.0": 56.81818181818182, "sd2.1": 56.56565656565656, "sd3-m": 79.16666666666666, "sd3-L": 0.0},
    }

compbench = {"color_attr": {"sd1.5": 24.803250000000016, "sd2.0": 33.72797499999997, "sd2.1": 29.43057499999998, "sd3-m": 80.1306999999999, "sd3-L": 78.63632499999997},
             "colors": {"sd1.5": 82.14069148936173, "sd2.0": 88.01930851063823, "sd2.1": 88.67210106382983, "sd3-m": 92.17422872340415, "sd3-L": 88.72821808510633},
                "counting": {"sd1.5": 62.5, "sd2.0": 67.34375, "sd2.1": 67.96875, "sd3-m": 79.6875, "sd3-L": 80.78125},
                "position": {"sd1.5": 1.823315868860109, "sd2.0": 3.290177465127203, "sd2.1": 3.02202055416161, "sd3-m": 9.747808753467084, "sd3-L": 9.586638347128083},
                "single_object": {"sd1.5": 94.692625, "sd2.0": 96.24478124999998, "sd2.1": 95.37618750000003, "sd3-m": 96.75540625, "sd3-L": 96.75540625},
                "two_object":{"sd1.5": 45.254343434343414, "sd2.0": 55.04224747474747, "sd2.1": 53.87785353535353, "sd3-m": 85.5005303030304, "sd3-L": 87.55323232323231},
                }

tasks = ["color_attr", "colors", "counting", "position", "single_object", "two_object"]
versions = ["sd1.5", "sd2.0", "sd2.1", "sd3-m", "sd3-L"]


def filter_nonzero(data, versions):
    filtered_versions = []
    filtered_values = []
    for v in versions:
        if data[v] != 0.0:
            filtered_versions.append(v)
            filtered_values.append(data[v])
    return filtered_versions, filtered_values

# Plot each task separately
for task in tasks:
    plt.figure(figsize=(8, 5))
    
    # Extract non-zero data for plotting
    geneval_versions, geneval_values = filter_nonzero(geneval[task], versions)
    zero_shot_versions, zero_shot_values = filter_nonzero(zero_shot[task], versions)
    compbench_versions, compbench_values = filter_nonzero(compbench[task], versions)

    # Plot lines
    if geneval_values:
        plt.plot(geneval_versions, geneval_values, marker='o', linestyle='-', label="Geneval")
    if zero_shot_values:
        plt.plot(zero_shot_versions, zero_shot_values, marker='s', linestyle='--', label="Zero-Shot")
    if compbench_values:
        plt.plot(compbench_versions, compbench_values, marker='^', linestyle='-.', label="Compbench")

    # Labels and title
    plt.xlabel("Version")
    plt.ylabel("Accuracy")
    plt.title(f"Performance on {task.replace('_', ' ').title()}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"accuracy_{task}.png", dpi=300, bbox_inches="tight")

# # exit(0)

geneval = pd.DataFrame(geneval)
zero_shot = pd.DataFrame(zero_shot)
compbench = pd.DataFrame(compbench)

geneval = geneval.drop(index="sd3-L")
zero_shot = zero_shot.drop(index="sd3-L")
compbench = compbench.drop(index="sd3-L")

# # remove colum "two_object"
# geneval = geneval.drop(columns="two_object")
# zero_shot = zero_shot.drop(columns="two_object")
# compbench = compbench.drop(columns="two_object")

corr_dict = {}
for i in geneval.columns:
    corr_dict[i] = {
        "geneval_zero_shot": geneval[i].corr(zero_shot[i]),
        "geneval_compbench": geneval[i].corr(compbench[i]),
        "compbench_zero_shot": compbench[i].corr(zero_shot[i])
    }
# Convert dictionary to DataFrame
corr_df = pd.DataFrame.from_dict(corr_dict, orient="index")

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".5f")
plt.title("Correlation Heatmap")
# Save the figure
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")  # Save as PNG


# correlations between geneval, zero_shot, compbench
# with open("geneval_corr.txt", "w") as f:
#     f.write("geneval_columns:\n")
#     for i in geneval.columns:
#         gen_zero_corr = geneval[i].corr(zero_shot[i])
#         print(f"geneval_zero_shot_corr_{i}_per_colum: {gen_zero_corr}")
#         f.write(f"geneval_zero_shot_corr_{i}_per_colum: {gen_zero_corr}\n")
#         gen_comp_corr = geneval[i].corr(compbench[i])
#         print(f"geneval_compbench_corr_{i}_per_colum: {gen_comp_corr}")
#         f.write(f"geneval_compbench_corr_{i}_per_colum: {gen_comp_corr}\n")
#         comp_zero_corr = compbench[i].corr(zero_shot[i])
#         print(f"compbench_zero_shot_corr_{i}_per_colum: {comp_zero_corr}")
#         f.write(f"compbench_zero_shot_corr_{i}_per_colum: {comp_zero_corr}\n\n")
    

    # f.write("geneval_index:\n")
    # for i in geneval.index:
    #     gen_zero_corr = geneval.loc[i].corr(zero_shot.loc[i])
    #     print(f"geneval_zero_shot_corr_{i}_per_index: {gen_zero_corr}")
    #     f.write(f"geneval_zero_shot_corr_{i}_per_index: {gen_zero_corr}\n")
    #     gen_comp_corr = geneval.loc[i].corr(compbench.loc[i])
    #     print(f"geneval_compbench_corr_{i}_per_index: {gen_comp_corr}")
    #     f.write(f"geneval_compbench_corr_{i}_per_index: {gen_comp_corr}\n")
    #     comp_zero_corr = compbench.loc[i].corr(zero_shot.loc[i])
    #     print(f"compbench_zero_shot_corr_{i}_per_index: {comp_zero_corr}")
    #     f.write(f"compbench_zero_shot_corr_{i}_per_index: {comp_zero_corr}\n\n")