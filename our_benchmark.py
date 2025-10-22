import pandas as pd

ver1_5 = pd.read_csv("results/analysis/csv/geneval_full_1.5.csv")
ver2_0 = pd.read_csv("results/analysis/csv/geneval_full_2.0.csv")
# ver2_1 = pd.read_csv("results/analysis/csv/geneval_full_2.1.csv")
ver3_m = pd.read_csv("results/analysis/csv/geneval_full_3-m.csv")

unique_tags = set(ver1_5["tag"].tolist())
# Convert unique tags to a list
unique_tags_list = list(unique_tags)

# Filter rows with the first tag and count them
print(unique_tags_list[0])# colors 94
num_rows = len(set(ver1_5[ver1_5["tag"] == unique_tags_list[0]]["prompt"].tolist()))
print(num_rows) 
print(unique_tags_list[1])# counting 80
print(len(set(ver1_5[ver1_5["tag"] == unique_tags_list[1]]["prompt"].tolist())))
print(unique_tags_list[2]) # position 100
print(len(set(ver1_5[ver1_5["tag"] == unique_tags_list[2]]["prompt"].tolist())))
print(unique_tags_list[3]) # position 100
print(len(set(ver1_5[ver1_5["tag"] == unique_tags_list[3]]["prompt"].tolist())))
# exit(0)
# exit(0)
# Initialize the random_selection column with 0
ver1_5["random_selection"] = 0

# Randomly select 25 rows for each tag
for tag in unique_tags:
    # Get rows with the current tag
    tag_rows = ver1_5[ver1_5["tag"] == tag]
    
    # Drop duplicate prompts while keeping the first occurrence
    unique_prompt_rows = tag_rows.drop_duplicates(subset=["prompt"])
    
    # Randomly select 25 rows or less if not enough
    selected_indices = unique_prompt_rows.sample(n=min(25, len(unique_prompt_rows)), random_state=42).index
    
    # Mark selected rows with 1
    ver1_5.loc[selected_indices, "random_selection"] = 1
# Check the updated DataFrame

# print(ver1_5.head(4))
# exit(0)

ver1_5_filtered = ver1_5[ver1_5["random_selection"] == 0]
# save
ver1_5_filtered.to_csv("results/analysis/csv/geneval_full_1.5_not_filtered.csv", index=False)

ver2_0["random_selection"] = 0
ver2_0["random_selection"] = ver2_0.apply(
    lambda row: 1 if any(
        (ver1_5["prompt"] == row["prompt"]) &
        (ver1_5["random_selection"] == 1) &
        (ver1_5["filename"] == row["filename"])
    ) else 0,
    axis=1
)

print(ver2_0.head())

ver2_0_filtered = ver2_0[ver2_0["random_selection"] == 0]
ver2_0_filtered.to_csv("results/analysis/csv/geneval_full_2.0_not_filtered.csv", index=False)

ver3_m["random_selection"] = 0
ver3_m["random_selection"] = ver3_m.apply(
    lambda row: 1 if any(
        (ver1_5["prompt"] == row["prompt"]) &
        (ver1_5["random_selection"] == 1) &
        (ver1_5["filename"] == row["filename"])
    ) else 0,
    axis=1
)
print(ver3_m.head())

ver3_m_filtered = ver3_m[ver3_m["random_selection"] == 0]
ver3_m_filtered.to_csv("results/analysis/csv/geneval_full_3-m_not_filtered.csv", index=False)

# ver1_5_filtered = ver1_5[ver1_5["random_selection"] == 1]
# # save
# ver1_5_filtered.to_csv("results/analysis/csv/geneval_full_1.5_filtered.csv", index=False)

# ver2_0["random_selection"] = 0
# ver2_0["random_selection"] = ver2_0.apply(
#     lambda row: 1 if any(
#         (ver1_5["prompt"] == row["prompt"]) &
#         (ver1_5["random_selection"] == 1) &
#         (ver1_5["filename"] == row["filename"])
#     ) else 0,
#     axis=1
# )

# print(ver2_0.head())

# ver2_0_filtered = ver2_0[ver2_0["random_selection"] == 1]
# ver2_0_filtered.to_csv("results/analysis/csv/geneval_full_2.0_filtered.csv", index=False)

# ver3_m["random_selection"] = 0
# ver3_m["random_selection"] = ver3_m.apply(
#     lambda row: 1 if any(
#         (ver1_5["prompt"] == row["prompt"]) &
#         (ver1_5["random_selection"] == 1) &
#         (ver1_5["filename"] == row["filename"])
#     ) else 0,
#     axis=1
# )
# print(ver3_m.head())

# ver3_m_filtered = ver3_m[ver3_m["random_selection"] == 1]
# ver3_m_filtered.to_csv("results/analysis/csv/geneval_full_3-m_filtered.csv", index=False)