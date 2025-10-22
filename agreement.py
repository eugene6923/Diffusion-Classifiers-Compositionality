import json
import os
import pandas as pd

# import argparse

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--model_id",
#         type=str,
#         default= "2.1", # TODO
#     )
#     parser.add_argument()
#     args = parser.parse_args()
#     return args

if __name__ == "__main__":

    model_id = ["2.1", "2.0", "1.5", "3-m" , "3-L"]
    task = ["color","color_attr","counting","position","single", "two"]

    for i in model_id:
        model_full_dict = {}
        for j in task:
            # if j == 'color_attr':
                zero_shot = os.path.join("results/analysis", f"geneval_{j}_{i}_results.txt")
                with open(zero_shot, "r") as f:
                    zero_shot_data = [line.strip() for line in f]
                zero_shot_text_data = [data for idx, data in enumerate(zero_shot_data) if idx%2 == 0]
                zero_shot_data = [data for idx, data in enumerate(zero_shot_data) if idx%2 != 0]
                zero_shot_path = ["/".join(data.split(":")[0].split("/")[4:]) for data in zero_shot_data]
                zero_shot_guess = [int(data.split(" ")[2]) for data in zero_shot_data]
                
                zero_shot_gt = [int(data.split(" ")[4]) for data in zero_shot_data]
                zero_shot_score = [True if data == zero_shot_gt[idx] else False for idx, data in enumerate(zero_shot_guess)]
                zero_shot_guess = [zero_shot_text_data[idx].split("}")[int(data.split(" ")[2])].replace(", ","").replace(", ","").replace("[","").replace("'","").replace("{","").replace("}","").replace("]","").replace("text: ","") for idx,data in enumerate(zero_shot_data)]

                zero_shot_data = dict(zip(zero_shot_path, zero_shot_score))
                zero_shot_data = pd.DataFrame(zero_shot_data.items(), columns=["filename", "correct_zero"])
                
                zero_shot_data["zero_guess"] = zero_shot_guess

                geneval = "../geneval/results"
                if i == "2.1":
                    full_model_id = "stable-diffusion-2-1-base"
                elif i == "2.0":
                    full_model_id = "stable-diffusion-2-base"
                elif i == "1.5":
                    full_model_id = "stable-diffusion-v1-5"
                elif i == "3-m":
                    full_model_id = "stable-diffusion-3-medium-diffusers"
                elif i =="3-L":
                    full_model_id = "stable-diffusion-3.5-large"
                geneval = os.path.join(geneval, full_model_id)
                geneval = os.path.join(geneval, "results.jsonl")
                with open(geneval, "r") as f:
                    geneval_data = [json.loads(line) for line in f]  # Load each line as a separate JSON object
                
                geneval_data = pd.DataFrame(geneval_data)
                geneval_data["filename"] = geneval_data["filename"].apply(lambda x: "/".join(x.split("/")[2:]))
                

                compbench = os.path.join("./CompBench", "results")
                if 'color' in j or 'object' in j:
                    if j == 'color':
                        j = 'colors'
                    compbench = os.path.join(compbench, "vqa_results","geneval",j,full_model_id)
                    compbench_results = json.load(open(os.path.join(compbench,"annotation_blip","vqa_result.json"), 'r'))
                    compbench_results = pd.DataFrame(compbench_results)
                    compbench_question = [int(comp.replace("annotation","").replace("_blip","")) for comp in os.listdir(compbench) if comp != "annotation_blip"]
                    compbench_question = max(compbench_question)
                    compbench_question = os.path.join(compbench, f"annotation{compbench_question}_blip","vqa_test.json")
                    compbench_question = json.load(open(compbench_question, 'r'))
                    compbench_question = pd.DataFrame(compbench_question)
                    compbench_question["filename"] = compbench_question["image"].apply(lambda x: "/".join(x.split("/")[-3:]))
                    compbench_question = compbench_question[["filename","question_id"]]
                    compbench_results = pd.merge(compbench_results, compbench_question, on="question_id")
                    compbench_results = compbench_results.rename(columns={"answer":"compbench_answer"}).drop(columns=["question_id"])

                    if j =='colors':
                        j = 'color'
                elif "counting" in j:
                    compbench = os.path.join(compbench, "det_results","geneval",j,full_model_id)
                    compbench = json.load(open(os.path.join(compbench,"vqa_result.json"), 'r'))
                    compbench = pd.DataFrame(compbench)
                    compbench["filename"] = compbench["filename"].apply(lambda x: "/".join(x.split("/")[-3:]))
                    compbench_results = compbench.rename(columns={"answer":"compbench_answer"})
                elif "position" in j:
                    compbench = os.path.join(compbench, "det_results","geneval",j,full_model_id,"annotation_obj_detection_2d")
                    compbench = json.load(open(os.path.join(compbench,"vqa_result.json"), 'r'))
                    compbench = pd.DataFrame(compbench)
                    compbench["filename"] = compbench["image"].apply(lambda x: "/".join(x.split("/")[-3:]))
                    compbench = compbench.drop(columns=["image","question_id"])
                    compbench_results = compbench.rename(columns={"answer":"compbench_answer"})


                data = pd.merge(geneval_data, zero_shot_data, on="filename")

                data = pd.merge(data, compbench_results, on="filename")


                data["zero_geneval_agreement"] = data["correct_zero"] == data["correct"]
                data = data.rename(columns={"correct":"correct_geneval"})
                data["version"] = i

                data.to_csv(f"results/analysis/csv/geneval_{j}_{i}_agreement.csv", index=False)
                # print(f"Model {i} Task {j}, Agreement: {data['agreement'].mean()}")
                data = data.to_dict()
                model_full_dict[j] = data

        model_full_dict = pd.concat(pd.DataFrame(model_full_dict[j]).sort_values(by='filename',ascending=True) for j in task)

        # print(model_full_dict.head(3))
        # exit(0)
        model_full_dict.to_csv(f"results/analysis/csv/geneval_full_{i}.csv", index=False)

