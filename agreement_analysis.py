import os
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
model_id = ["2.1", "2.0", "1.5", "3-m"]
os.makedirs("results/analysis/plots", exist_ok=True)
for i in model_id:

    agreement = pd.read_csv(f"results/analysis/csv/geneval_full_{i}.csv")

    # Define tasks
    tasks = agreement["tag"].unique()  # Get unique tasks dynamically

    # Create directory
    os.makedirs("results/analysis/plots", exist_ok=True)

    # Initialize lists for plotting
    task_labels = []
    geneval_agreement_rates = []
    compbench_agreement_rates = []

    # Compute mean agreement for each task
    for j in tasks:
        agreement_task = agreement[agreement["tag"] == j]
        agreement_rate = agreement_task["zero_geneval_agreement"].mean()
        compbench_rate = agreement_task["compbench_answer"].mean()

        task_labels.append(j)
        geneval_agreement_rates.append(agreement_rate)
        compbench_agreement_rates.append(compbench_rate)

    # Plot the results
    plt.figure(figsize=(10, 6))
    x = range(len(task_labels))

    plt.bar(x, geneval_agreement_rates, width=0.4, label="Geneval & Zero-shot Agreement", align="center", alpha=0.7)
    plt.bar([p + 0.4 for p in x], compbench_agreement_rates, width=0.4, label="Compbench Answer Average", align="center", alpha=0.7)

    # Labels & title
    plt.xlabel("Task")
    plt.ylabel("Mean Agreement")
    plt.title(f"SD {i}: Zero Agreement Rates per Task ")
    plt.xticks([p + 0.2 for p in x], task_labels, rotation=45)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Save plot
    plt.tight_layout()
    os.makedirs("results/analysis/plots", exist_ok=True)
    plt.savefig(f"results/analysis/plots/zero_agreement_plot_{i}.png")

# for i in model_id:

#     agreement = pd.read_csv(f"results/analysis/csv/geneval_full_{i}.csv")

#     # Define tasks
#     tasks = agreement["tag"].unique()  # Get unique tasks dynamically

#     # Initialize lists for plotting
#     task_labels = []
#     correct_zero_count = []
#     incorrect_geneval_count = []

#     # Compute disagreement count for each task
#     for j in tasks:
#         agreement_task = agreement[agreement["tag"] == j]
#         disagree = agreement_task[agreement_task["correct_geneval"] != agreement_task["correct_zero"]]

#         task_labels.append(j)
#         correct_zero_count.append(disagree[disagree["correct_zero"] == 1].shape[0])  # Fix: Use .shape[0] for row count
#         incorrect_geneval_count.append(disagree[disagree["correct_geneval"] == 1].shape[0])  # Fix: Use .shape[0]

#     # Plot the results
#     plt.figure(figsize=(10, 6))
#     x = range(len(task_labels))

#     plt.bar(x, correct_zero_count, width=0.4, label="Correct Zero-shot", align="center", alpha=0.7)
#     plt.bar([p + 0.4 for p in x], incorrect_geneval_count, width=0.4, label="Correct Geneval", align="center", alpha=0.7)

#     # Labels & title
#     plt.xlabel("Task")
#     plt.ylabel("Disagreement Count")
#     plt.title(f"SD {i}: Zero Disagreement Counts per Task")
#     plt.xticks([p + 0.2 for p in x], task_labels, rotation=45)
#     plt.legend()
#     plt.grid(axis="y", linestyle="--", alpha=0.6)

#     # Save plot
#     plt.tight_layout()
#     plt.savefig(f"results/analysis/plots/disagree_count_{i}.png")