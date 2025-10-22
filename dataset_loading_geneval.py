import json
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import PIL
import numpy as np
import torch
from torchvision import datasets
from glob import glob
# from aro.dataset_zoo import VG_Relation, VG_Attribution, COCO_Order, Flickr30k_Order
import pandas as pd
import ast
from datasets import load_dataset
from huggingface_hub import login
# from whatsup_vlms.dataset_zoo import Controlled_Images, COCO_QA, VG_QA
from easydict import EasyDict as edict
import re
from collections import defaultdict
import itertools

from PIL import ImageFile
from kind_of_globals import ARNAS_USES
ImageFile.LOAD_TRUNCATED_IMAGES = True

def diffusers_preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = image.squeeze(0)
    return 2.0 * image - 1.0

class Geneval_final(Dataset):
    def __init__(self, transform, root_dir, subset, version, resize=512, scoring_only=False, cfg = 9.0, filter = False):
        self.version = version

        # root_dir should be the path to the root directory of the dataset (including cfg/version/images)
        if ARNAS_USES:
            # enum all combinations to make it transparent ;) 
            assert cfg == 9.0, "only cfg 9.0 is supported for now"
            if version == "1.5" or version == 1.5:
                self.root_dir = '/mnt/lustre/work/oh/owl661/sd-datasets/stable-diffusion-v1-5'
            elif version == "2.0" or version == 2.0:
                self.root_dir = '/mnt/lustre/work/oh/owl661/sd-datasets/sd2-base/stable-diffusion-2-base'
            elif version == "3-m":
                self.root_dir = '/mnt/lustre/work/oh/owl661/sd-datasets/stable-diffusion-3-medium-diffusers'
            else: raise ValueError('Invalid version')
        else:
            try:
                self.root_dir = f'{root_dir}/{cfg}/{self.version.split("/")[1]}'
            except:
                if version == "1.5": self.version = "stable-diffusion-v1-5"
                elif version == "2.0": self.version = "stable-diffusion-2-base"
                elif version == "3-m": self.version = "stable-diffusion-3-medium-diffusers"
                elif version == "flux" : self.version = "FLUX.1-dev"
                else: raise ValueError('Invalid version')
                self.root_dir = f'{root_dir}/{cfg}/{self.version}'
            
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset= subset
        self.examples = []
        self.prompts = []
        self.include = []
        self.cfg = cfg

        self.filter = filter

        if self.subset == "two_object_subset":
            subset = "two_object"
            text_path= '/mnt/lustre/work/oh/owl661/compositional-vaes/src/vqvae/_post/self_bench/two_object_subset.json' if ARNAS_USES else '/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/two_object_subset.json'
            self.text = json.load(open(text_path, 'r'))

        else:
            if not ARNAS_USES:
                prompt = f'{root_dir}/../prompts/zero_shot_prompts.json' # all possible prompts list
            else:
                prompt = f'/mnt/lustre/work/oh/owl661/compositional-vaes/src/vqvae/_post/self_bench/filter_prompts/prompts/zero_shot_prompts.json'
            self.text = json.load(open(prompt, 'r'))['photo'][self.subset]

        if not self.filter:
            # we will take all possible images that are generated 
            # (unfortunately, we do not have the json file with all path with subset so I just run the for loop 
            # -- we can make but this for loop doesnt take lots of time and will work along with geneval generation results)
            for i in os.listdir(self.root_dir): 
                metadata = os.path.join(self.root_dir, i, 'metadata.jsonl')
                metadata = json.load(open(metadata, 'r'))
                if metadata['tag'] == subset:
                    for j in range(4):
                        self.examples.append(os.path.join(self.root_dir, i, 'samples', f'0000{j}.png'))
                        self.prompts.append(metadata["prompt"])
                        self.include.append(metadata["include"])
            
        else:
            filter_prompt = f'{root_dir}/../filter/SD-{version}-CFG={str(int(self.cfg))}.json' if not ARNAS_USES else f'/mnt/lustre/work/oh/owl661/compositional-vaes/src/vqvae/_post/self_bench/filter_prompts/filter/SD-{version}-CFG={str(int(self.cfg))}.json'
            # please change the path
            
            filter_prompt = json.load(open(filter_prompt, 'r'))
            
            if ARNAS_USES:
                # super hacky xD 
                for i in filter_prompt:
                    try:
                        if version == "3-m" and self.cfg == 9.0: # for sd 3-m that Yujin only labeled
                            if i['tag'] == subset and any(j["label"] == "original_good" for j in i["labels"]) :
                                self.examples.append(f"{self.root_dir}/" + "/".join(i["sample_path"].split("/")[-3:]))
                                self.prompts.append(i["original_prompt"])
                                self.include.append(i["full_metadata"]["include"])
                        else:   # if both agrees that it is good then we include
                            if i['tag'] == subset and all(j["label"] == "original_good" for j in i["labels"]) :
                                self.examples.append(f"{self.root_dir}/" + "/".join(i["sample_path"].split("/")[-3:]))
                                self.prompts.append(i["original_prompt"])
                                self.include.append(i["full_metadata"]["include"])
                    except: # for sd 2.0  (human_label)
                            if i['tag'] == subset and i["human_label"]=="original_good":
                                self.examples.append(f"{self.root_dir}/" + "/".join(i["sample_path"].split("/")[-3:]))
                                self.prompts.append(i["original_prompt"])
                                self.include.append(i["full_metadata"]["include"])

            else:
                for i in filter_prompt:
                    try:
                        if version == "3-m" and self.cfg == 9.0: # for sd 3-m that Yujin only labeled
                            if i['tag'] == subset and any(j["label"] == "original_good" for j in i["labels"]) :
                                self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                                self.prompts.append(i["original_prompt"])
                                self.include.append(i["full_metadata"]["include"])
                        else:   # if both agrees that it is good then we include
                            if i['tag'] == subset and all(j["label"] == "original_good" for j in i["labels"]) :
                                self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                                self.prompts.append(i["original_prompt"])
                                self.include.append(i["full_metadata"]["include"])
                    except: # for sd 2.0  (human_label)
                            if i['tag'] == subset and i["human_label"]=="original_good":
                                self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                                self.prompts.append(i["original_prompt"])
                                self.include.append(i["full_metadata"]["include"])

        if self.__len__() == 0:
            raise ValueError('No examples found for the given subset and version')
        
        print(f"Number of examples: {self.__len__()}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # metadata = os.path.join(self.examples[idx].split('samples')[0],'metadata.jsonl')
        # metadata = json.load(open(metadata, 'r'))
        
        img0 = Image.open(self.examples[idx]).convert('RGB')
        if not self.scoring_only:
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
            
        if self.subset == 'color_attr':
            text = self.text[self.include[idx][0]["class"]][self.include[idx][1]["class"]]
        elif self.subset == 'position':
            text = self.text[self.include[idx][1]["class"]][self.include[idx][0]["class"]]
        elif self.subset in ['single_object','two_object']:
            text = self.text
        elif self.subset == 'two_object_subset':
            first = self.include[idx][0]["class"]
            second = self.include[idx][1]["class"]
            text = self.text[f'{first}_{second}']
        else:
            text = self.text[self.include[idx][0]["class"]]
        
        if self.scoring_only:
            return text, idx
        else:
            return (self.examples[idx], [img0_resize]), text, text.index(self.prompts[idx])

