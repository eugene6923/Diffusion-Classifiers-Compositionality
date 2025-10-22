import os

# Set paths and filtering parameters
path = "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/3-m/0/confusion"
subset = 'geneval_position'
version = '3-m'
geneval_version = '3-m'

# Look for a file that includes the version string in its name
data = []
subset_path = os.path.join(path, subset)
for filename in os.listdir(subset_path):
    file = os.path.join(subset_path, filename)
    # if version in filename:
    if f'version_{version}' in filename and 'sd3resize512' not in filename and 'cfg1.0' not in filename and 'filterFalse' in filename  and f'sdversion{geneval_version}' in filename:
        with open(os.path.join(subset_path, filename), 'r') as f:
            data = f.readlines()
        break  # Use the first matching file

# print(data[:10])
# exit(0)

# Define our expected colors
# colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
colors = ["left of", "right of", "above", "below"]
# Initialize lists for each block of data
text_list = []
data_path = []
scores = []
gt_class = []
predict = []

# Parse each line in the file
for line in data:
    if 'text list' in line:
        text_list.append(line)
    elif 'data path' in line:
        data_path.append(line)
    elif 'top5 scores' in line:
        scores.append(line)
    elif 'truth idx' in line:
        # Expected format: "truth idx: <num>, guess idx: <num>"
        parts = line.split(',')
        truth = parts[0].split()[-1].strip()
        prediction = parts[1].split()[-1].strip()
        predict.append(truth == prediction)
    elif 'truth' in line and 'truth idx' not in line:
        parts = line.strip().split()
        
        # Identify spatial relationship in the sentence
        found_relation = None
        for relation in colors:
            if relation in line:
                found_relation = relation
                break
        
        if found_relation:
            gt_class.append(found_relation)  # Store spatial relationship
            
            # Extract object before spatial relation
            relation_idx = line.index(found_relation)
            before_relation = line[:relation_idx].strip().split()
            
            # Extract the object before the spatial relation
            object_name = None
            for i in range(len(before_relation) - 1, -1, -1):  # Reverse iterate to find last noun
                if before_relation[i] not in ["a", "photo", "of"]:
                    object_name = before_relation[i].strip().lower()
                    break

            # Extract the entity after the spatial relation
            after_relation = line[relation_idx + len(found_relation):].strip().split()
            entity_name = after_relation[1] if len(after_relation) > 1 else ""
            
            # print(f"Object: {object_name}, Position: {found_relation}, Reference Object: {entity_name}")
        

# Print lengths to ensure they match (should be 376, for example)
print("Lengths:", len(text_list), len(data_path), len(scores), len(gt_class), len(predict))
# print("Lengths2:", len(list(set(text_list))), len(list(set(data_path))), len(list(set(scores))), len(list(set(gt_class))), len(list(set(predict))))
# Zip the lists into a single list of samples
data_all = list(zip(text_list, data_path, scores, predict, gt_class))

# Initialize accuracy_per_class for each color as [correct_count, total_count]
accuracy_per_class = {color: [0, 0] for color in colors}

# Calculate accuracy per class
for sample in data_all:
    predicted_correctly = sample[3]  # boolean from 'truth idx' line
    ground_truth_color = sample[4]
    
    # Use an exact match to assign the sample to its color category
    for color in colors:
        if ground_truth_color == color:
            if predicted_correctly:
                accuracy_per_class[color][0] += 1
            accuracy_per_class[color][1] += 1
            break  # Stop after the first match

# Display the per-class counts
print("filterFalse")
print("Accuracy per class counts:", accuracy_per_class)

# Compute the total samples counted and per-class percentages
total_samples_counted = sum([v[1] for v in accuracy_per_class.values()])
accuracy_per_class_percentage = {
    k: (v[0] / v[1]) if v[1] > 0 else 0 for k, v in accuracy_per_class.items()
}
print("file name:", file)
print("Total samples counted:", total_samples_counted)
print(f"{subset} {version}")
print("Accuracy percentages per class:", accuracy_per_class_percentage)

data = []
subset_path = os.path.join(path, subset)
for filename in os.listdir(subset_path):
    file = os.path.join(subset_path, filename)
    # if version in filename:
    if f'version_{version}' in filename and 'sd3resize512' not in filename and 'cfg1.0' not in filename and 'filterTrue' in filename  and f'sdversion{geneval_version}' in filename:
        with open(os.path.join(subset_path, filename), 'r') as f:
            data = f.readlines()
        break  # Use the first matching file

# print(data[:10])
# exit(0)

# Define our expected colors
# colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
colors = ["left of", "right of", "above", "below"]
# Initialize lists for each block of data
text_list = []
data_path = []
scores = []
gt_class = []
predict = []

# Parse each line in the file
for line in data:
    if 'text list' in line:
        text_list.append(line)
    elif 'data path' in line:
        data_path.append(line)
    elif 'top5 scores' in line:
        scores.append(line)
    elif 'truth idx' in line:
        # Expected format: "truth idx: <num>, guess idx: <num>"
        parts = line.split(',')
        truth = parts[0].split()[-1].strip()
        prediction = parts[1].split()[-1].strip()
        predict.append(truth == prediction)
    elif 'truth' in line and 'truth idx' not in line:
        parts = line.strip().split()
        
        # Identify spatial relationship in the sentence
        found_relation = None
        for relation in colors:
            if relation in line:
                found_relation = relation
                break
        
        if found_relation:
            gt_class.append(found_relation)  # Store spatial relationship
            
            # Extract object before spatial relation
            relation_idx = line.index(found_relation)
            before_relation = line[:relation_idx].strip().split()
            
            # Extract the object before the spatial relation
            object_name = None
            for i in range(len(before_relation) - 1, -1, -1):  # Reverse iterate to find last noun
                if before_relation[i] not in ["a", "photo", "of"]:
                    object_name = before_relation[i].strip().lower()
                    break

            # Extract the entity after the spatial relation
            after_relation = line[relation_idx + len(found_relation):].strip().split()
            entity_name = after_relation[1] if len(after_relation) > 1 else ""
            
            # print(f"Object: {object_name}, Position: {found_relation}, Reference Object: {entity_name}")
        

# Print lengths to ensure they match (should be 376, for example)
print("Lengths:", len(text_list), len(data_path), len(scores), len(gt_class), len(predict))
# print("Lengths2:", len(list(set(text_list))), len(list(set(data_path))), len(list(set(scores))), len(list(set(gt_class))), len(list(set(predict))))
# Zip the lists into a single list of samples
data_all = list(zip(text_list, data_path, scores, predict, gt_class))

# Initialize accuracy_per_class for each color as [correct_count, total_count]
accuracy_per_class = {color: [0, 0] for color in colors}

# Calculate accuracy per class
for sample in data_all:
    predicted_correctly = sample[3]  # boolean from 'truth idx' line
    ground_truth_color = sample[4]
    
    # Use an exact match to assign the sample to its color category
    for color in colors:
        if ground_truth_color == color:
            if predicted_correctly:
                accuracy_per_class[color][0] += 1
            accuracy_per_class[color][1] += 1
            break  # Stop after the first match

# Display the per-class counts
print("filterTrue")
print("Accuracy per class counts:", accuracy_per_class)

# Compute the total samples counted and per-class percentages
total_samples_counted = sum([v[1] for v in accuracy_per_class.values()])
accuracy_per_class_percentage = {
    k: (v[0] / v[1]) if v[1] > 0 else 0 for k, v in accuracy_per_class.items()
}
print("file name:", file)
print("Total samples counted:", total_samples_counted)
print(f"{subset} {version}")
print("Accuracy percentages per class:", accuracy_per_class_percentage)
