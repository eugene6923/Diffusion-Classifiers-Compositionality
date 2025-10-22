import json
import os
import yaml

import numpy as np

# numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
numbers = ["one", "two", "three", "four"]

colors = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]

positions = ["left of", "right of", "above", "below"]

domain = ["painting", "photo", "drawing", "sketch", "illustration", "portrait", "image", "picture", "artwork", "photograph"]

with open("object_names.txt") as cls_file:
    classnames = [line.strip() for line in cls_file]

def with_article(name: str):
    if name[0] in "aeiou":
        return f"an {name}"
    return f"a {name}"


def make_plural(name: str):
    if name[-1] in "s":
        return f"{name}es"
    return f"{name}s"

TAGS = ["single_object", "two_object", "counting", "colors", "position", "color_attr"]

# TAG = "single_object"
# prompt=f"a photo of {with_article(classnames[idx])}"

# TAG = "two_object"
# prompt=f"a photo of {with_article(classnames[idx_a])} and {with_article(classnames[idx_b])}"

# TAG = "counting"
# prompt=f"a photo of {numbers[num]} {make_plural(classnames[idx])}

# TAG = "colors"
# prompt=f"a photo of {with_article(color)} {classnames[idx]}"

# TAG = "position"
# prompt=f"a photo of {with_article(classnames[idx_a])} {position} {with_article(classnames[idx_b])}"

# TAG = "color_attr"
# prompt=f"a photo of {with_article(colors[cidx_a])} {classnames[idx_a]} and {with_article(colors[cidx_b])} {classnames[idx_b]}"

prompt_dict = {}
for domain_name in domain:
    prompt_dict[domain_name] = {}
    for tag in TAGS:
        prompt_dict[domain_name][tag] = {}
        if tag == "single_object":
            prompt_dict[domain_name][tag] = [f"{with_article(domain_name)} of {with_article(class_name)}" for class_name in classnames]
        elif tag == "two_object":
            prompt_dict[domain_name][tag] = [f"{with_article(domain_name)} of {with_article(class_name_a)} and {with_article(class_name_b)}" for class_name_a in classnames for class_name_b in classnames]
        elif tag == "counting":
            for class_name in classnames:
                prompt_dict[domain_name][tag][class_name] = [f"{with_article(domain_name)} of {num} {make_plural(class_name)}" if num != "one" else f"{with_article(domain_name)} of {num} {class_name}" for num in numbers]
        elif tag == "colors":
            for class_name in classnames:
                prompt_dict[domain_name][tag][class_name] = [f"{with_article(domain_name)} of {with_article(color)} {class_name}" for color in colors]
        elif tag == "position":
            for class_name_a in classnames:
                prompt_dict[domain_name][tag][class_name_a] = {}
                for class_name_b in classnames:
                    prompt_dict[domain_name][tag][class_name_a][class_name_b] = [f"{with_article(domain_name)} of {with_article(class_name_a)} {position} {with_article(class_name_b)}" for position in positions]
        elif tag == "color_attr":
            for class_name_a in classnames:
                prompt_dict[domain_name][tag][class_name_a] = {}
                for class_name_b in classnames:
                    prompt_dict[domain_name][tag][class_name_a][class_name_b] = [f"{with_article(domain_name)} of {with_article(color_a)} {class_name_a} and {with_article(color_b)} {class_name_b}" for color_a in colors for color_b in colors]

with open("zero_shot_prompts.json", "w") as f:
    json.dump(prompt_dict, f, indent=4)