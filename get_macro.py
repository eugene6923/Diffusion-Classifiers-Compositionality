import os
import re 

task = 'position'
version = '2.0'
geneval_version = '3-m'
filter = True
if version == '3-m':
    path = f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/{version}/0/confusion/geneval_{task}'
else:
    
    if geneval_version == version:
        path = f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/confusion/geneval{"_filter" if filter else ""}_{task}'
        # geneval_version = geneval_version.replace('.', '_')
    else:
        geneval_version = geneval_version.replace('.', '_').replace('-','_')
        path = f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/confusion/geneval_{geneval_version}{"_filter" if filter else ""}_{task}'

if 'two' in task :
    subset_list = open('../geneval/prompts/object_names.txt', 'r').read().splitlines()
elif task == 'color_attr':
    subset_list = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'black', 'white']

# print(path)

for i in os.listdir(path):
    if version == '3-m':
        if f'geneval_{task}_sdversion{geneval_version}_cfg9.0_filter{filter}_version_{version}' in i:
            path = os.path.join(path, i)
            print(path)
            with open(path, 'r') as f:
                data = f.readlines()
            break
    else:
        # print(f'geneval_{"filter_" if filter else ""}{task}_{version}')
        # print(i)
        if geneval_version == version:
            # print(path)
            if f'geneval_{"filter_" if filter else ""}{task}_{version}' in i:
                path = os.path.join(path, i)
                print(path)
                with open(path, 'r') as f:
                    data = f.readlines()
                break
        else:
            if f'geneval_{geneval_version}_{"filter_" if filter else ""}{task}_{version}' in i:
                path = os.path.join(path, i)
                print(path)
                with open(path, 'r') as f:
                    data = f.readlines()
                break
macro_accr = {}
correct = []
objects = []

def extract_subsets(sentence, subsets):
    # Create a regex pattern that matches any of the subsets
        pattern = re.compile(r'\b(' + '|'.join(subsets) + r')\b', re.IGNORECASE)
        
        # Find all matches in order
        matches = pattern.findall(sentence)
        
        return matches

for line in data:
    if 'truth idx:' in line:
        parts = line.split(',')
        truth = parts[0].split()[-1].strip()
        prediction = parts[1].split()[-1].strip()
        if truth == prediction:
            correct.append(1)
            # macro_accr[truth].append(1)
        else:
            correct.append(0)
            # macro_accr[truth].append(0)
    if 'truth:' in line and 'truth idx' not in line:
        parts = line.strip().split()
        if task == 'single':
            object_name = ' '.join(parts[5:])
            objects.append(object_name)
        elif task == 'color_attr':
            first,second = extract_subsets(line, subset_list)

            # object_name = ' '.join(parts[6:])
            objects.append(f'{first}_{second}')
        elif 'two' in task :
            first, second = extract_subsets(line, subset_list)
            objects.append(f'{first}_{second}')
            # object_name = ' '.join(parts[5:])
            # objects.append(object_name)
        

for i, obj in enumerate(objects):
    if task == 'single':
        if obj not in macro_accr:
            macro_accr[obj] = []
        macro_accr[obj].append(correct[i])
    else:
        for j in obj.split('_'):
            if j not in macro_accr:
                macro_accr[j] = []
            macro_accr[j].append(correct[i])

# print(macro_accr)
# print(macro_accr.keys())
print(len(macro_accr.keys()))
print(task)
print("version", version)
print("geneval_version", geneval_version)

for key, value in macro_accr.items():
    correct = sum(value)
    total = len(value)
    # try:
    macro_accr[key] = correct / total
    # except:
    #     macro_accr[key] = 0
# print(sum(macro_accr)/len(macro_accr))
print("full_macro", sum(macro_accr.values())/len(macro_accr))
