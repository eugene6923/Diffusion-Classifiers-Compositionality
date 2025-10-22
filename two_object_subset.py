import json

with open("../geneval/prompts/evaluation_metadata.jsonl", "r") as f:
    prompt_dict = [json.loads(line) for line in f]

with open("../geneval/prompts/zero_shot_prompts.json", "r") as f:
    zero_dict = json.load(f)

subset_dict = {}


def with_article(name: str):
    if name[0] in "aeiou":
        return f"an {name}"
    return f"a {name}"

text_list = zero_dict["photo"]["two_object"]


for i in prompt_dict:
    if i["tag"] == "two_object":
        first_class = i["include"][0]["class"]
        second_class = i["include"][1]["class"] 
        subset_dict[(first_class,second_class)] = []
        a_count = 0
        b_count = 0
        for j in text_list:
            if j == f"a photo of {with_article(first_class)} and {with_article(second_class)}":  
                subset_dict[first_class,second_class].append(i["prompt"])
            elif a_count <50 and j.startswith(f"a photo of {with_article(first_class)} and") and (second_class not in j):
                a_count +=1
                subset_dict[first_class,second_class].append(j)
            elif b_count < 50 and not j.startswith(f"a photo of {with_article(first_class)} and") and (j.endswith(f"and {with_article(second_class)}")):
                b_count +=1
                subset_dict[first_class,second_class].append(j)

for i in subset_dict.keys():
    if len(subset_dict[i]) != 101:
        print(i, len(subset_dict[i]))
        exit(0)

with open("two_object_subset.json", "w") as f:
   
    json.dump({f"{k[0]}_{k[1]}": v for k, v in subset_dict.items()}, f, indent=4)


