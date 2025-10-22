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

import sys
import os

# Remove the directory containing your local datasets from Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir in sys.path:
    sys.path.remove(parent_dir)

from torchvision import datasets as torchvision_datasets  # For torchvision datasets
from datasets import load_dataset  # Now should find HuggingFace datasets
from glob import glob
from aro.dataset_zoo import VG_Relation, VG_Attribution, COCO_Order, Flickr30k_Order
import pandas as pd
import ast
from huggingface_hub import login
try:
    from vqvae._post.self_bench.kind_of_globals import ARNAS_USES
except:
    ARNAS_USES = False
if ARNAS_USES:
    from vqvae._post.self_bench.whatsup_vlms2.dataset_zoo import Controlled_Images, COCO_QA, VG_QA
else:
    from whatsup_vlms.dataset_zoo import Controlled_Images, COCO_QA, VG_QA
from easydict import EasyDict as edict
import re
from collections import defaultdict
import itertools
# from dataset_loading_geneval import Geneval_final

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_dataset(dataset_name, root_dir, transform=None, resize=512, scoring_only=False, tokenizer=None, split='val', max_train_samples=None, hard_neg=False, targets=None, neg_img=False, mixed_neg=False, details=False, mode=False, index_subset = 0, version= None, domain='photo', cfg = 9.0, filter= False):
    if dataset_name == 'winoground':
        return WinogroundDataset(root_dir, transform, resize=resize, scoring_only=scoring_only)
    elif dataset_name == 'winoground_relation':
        return WinogroundDataset(root_dir, transform, resize=resize, scoring_only=scoring_only, subset='Relation')
    elif dataset_name == 'winoground_object':
        return WinogroundDataset(root_dir, transform, resize=resize, scoring_only=scoring_only, subset='Object')
    elif dataset_name == 'winoground_both':
        return WinogroundDataset(root_dir, transform, resize=resize, scoring_only=scoring_only, subset='Both')
    elif dataset_name == 'mmbias':
        return BiasDataset(root_dir, resize=resize, transform=transform, targets=targets)
    elif dataset_name == 'genderbias':
        return GenderBiasDataset(root_dir, resize=resize, transform=transform)
    elif dataset_name == 'imagecode':
        return ImageCoDeDataset(root_dir, split, transform, resize=resize, scoring_only=scoring_only)
    elif dataset_name == 'imagecode_video':
        return ImageCoDeDataset(root_dir, split, transform, resize=resize, scoring_only=scoring_only, static=False)
    elif dataset_name == 'flickr30k':
        return Flickr30KDataset(root_dir, transform,  resize=resize, scoring_only=scoring_only, split=split, tokenizer=tokenizer, details=details)
    elif dataset_name == 'flickr30k_text':
        return Flickr30KTextRetrievalDataset(root_dir, transform, resize=resize,  scoring_only=scoring_only, split=split, tokenizer=tokenizer, hard_neg=hard_neg, details=details)
    elif dataset_name == 'flickr30k_neg':
        return Flickr30KNegativesDataset(root_dir, transform, resize=resize,  scoring_only=scoring_only, split=split, tokenizer=tokenizer, hard_neg=hard_neg)
    elif dataset_name == 'lora_flickr30k':
        return LoRaFlickr30KDataset(root_dir, transform, resize=resize,  tokenizer=tokenizer, max_train_samples=max_train_samples)
    elif dataset_name == 'imagenet':
        return ImagenetDataset(root_dir, transform, resize=resize, scoring_only=scoring_only)
    elif dataset_name == 'svo_verb':
        return SVOClassificationDataset(root_dir, transform, resize=resize, scoring_only=scoring_only, neg_type='verb')
    elif dataset_name == 'svo_subj':
        return SVOClassificationDataset(root_dir, transform, resize=resize, scoring_only=scoring_only, neg_type='subj')
    elif dataset_name == 'svo_obj':
        return SVOClassificationDataset(root_dir, transform, resize=resize, scoring_only=scoring_only, neg_type='obj')
    elif dataset_name == 'clevr':
        return CLEVRDataset(root_dir, transform, resize=resize, scoring_only=scoring_only)
    elif dataset_name == 'clevr_spatial':
        return CLEVRDataset(root_dir, transform, resize=resize, scoring_only=scoring_only, subset='spatial')
    elif dataset_name == 'clevr_binding_color':
        return CLEVRDataset(root_dir, transform, resize=resize, scoring_only=scoring_only, subset='pair_binding_color')
    elif dataset_name == 'pets':
        return PetsDataset(root_dir, transform, resize=resize, scoring_only=scoring_only)
    elif dataset_name == 'stl10':
        return STL10(root_dir, transform, resize=resize, scoring_only=scoring_only, split='test')
    elif dataset_name == 'cifar10':
        return Cifar10(root_dir, transform, resize=resize, scoring_only=scoring_only, split=split)
    elif dataset_name == 'vg_relation': # aro
        return VG_Relation(image_preprocess=transform, resize=resize, download=True, root_dir=f'{root_dir}/ARO')
    elif dataset_name == 'vg_attribution':
        return VG_Attribution(image_preprocess=transform, resize=resize, download=True, root_dir=f'{root_dir}/ARO')
    elif dataset_name == 'coco_order':
        return COCO_Order(image_preprocess=transform, resize=resize, download=True, root_dir=f'{root_dir}/coco_order')
    elif dataset_name == 'flickr30k_order':
        return Flickr30k_Order(image_preprocess=transform, resize=resize, download=True, root_dir=f'{root_dir}/flickr-image-dataset/versions/1/flickr30k_images/flickr30k_images')
    elif dataset_name == 'mscoco':  
        return MSCOCODataset(root_dir, transform, resize=resize, split=split, tokenizer=tokenizer, hard_neg=hard_neg, neg_img=neg_img, mixed_neg=mixed_neg)
    elif dataset_name == 'mscoco_val':
        return ValidMSCOCODataset(root_dir, transform, resize=resize, split='val', tokenizer=tokenizer, neg_img=neg_img, hard_neg=hard_neg)
    elif dataset_name == 'whatsup_A':
        return Controlled_Images(image_preprocess = transform, resize=resize, download = True, root_dir =f'{root_dir}/whatsup', subset='A')
    elif dataset_name == 'whatsup_B':
        return Controlled_Images(image_preprocess = transform, resize=resize, download = True, root_dir =f'{root_dir}/whatsup', subset='B')
    elif dataset_name == 'COCO_QA_one':
        return COCO_QA(image_preprocess = transform, resize=resize, download = True, root_dir =f'{root_dir}/coco_qa', subset='one')
    elif dataset_name == 'COCO_QA_two':
        return COCO_QA(image_preprocess = transform, resize=resize, download = True, root_dir =f'{root_dir}/coco_qa', subset='two')
    elif dataset_name == 'VG_QA_one':
        return VG_QA(image_preprocess = transform, resize=resize, download = True, root_dir =f'{root_dir}/VG_QA', subset='one')
    elif dataset_name == 'VG_QA_two':
        return VG_QA(image_preprocess = transform, resize=resize, download = True, root_dir =f'{root_dir}/VG_QA', subset='two')
    elif dataset_name == 'sugar_add_att':
        return SugarCrepe(transform, resize=resize, root_dir =f'{root_dir}/COCO2017/val2017',subset='add_att')
    elif dataset_name == 'sugar_add_obj':
        return SugarCrepe(transform, resize=resize, root_dir =f'{root_dir}/COCO2017/val2017',subset='add_obj')
    elif dataset_name == 'sugar_replace_att':
        return SugarCrepe(transform, resize=resize, root_dir =f'{root_dir}/COCO2017/val2017',subset='replace_att')
    elif dataset_name == 'sugar_replace_obj':
        return SugarCrepe(transform, resize=resize, root_dir =f'{root_dir}/COCO2017/val2017',subset='replace_obj')
    elif dataset_name == 'sugar_replace_rel':
        return SugarCrepe(transform, resize=resize, root_dir =f'{root_dir}/COCO2017/val2017',subset='replace_rel')
    elif dataset_name == 'sugar_swap_att':
        return SugarCrepe(transform, resize=resize, root_dir =f'{root_dir}/COCO2017/val2017',subset='swap_att')
    elif dataset_name == 'sugar_swap_obj':
        return SugarCrepe(transform, resize=resize, root_dir =f'{root_dir}/COCO2017/val2017',subset='swap_obj')
    elif dataset_name == 'sugar_att':
        return SugarCrepe(transform, resize=resize, root_dir =f'{root_dir}/COCO2017/val2017',subset='att')
    elif dataset_name == 'sugar_obj':
        return SugarCrepe(transform, resize=resize, root_dir =f'{root_dir}/COCO2017/val2017',subset='obj')
    elif dataset_name == 'sugar_rel':
        return SugarCrepe(transform, resize=resize, root_dir =f'{root_dir}/COCO2017/val2017',subset='rel')
    elif dataset_name == 'cola_multi':
        return Cola_Multi(transform, resize=resize, root_dir =f'{root_dir}/GQA/images')
    elif dataset_name == 'cola_single_gqa':
        # return cola_single_gqa(transform, root_dir =f'{root_dir}/GQA/images')
        return ValueError('cola_single_gqa cannot be implemented')
    elif dataset_name == 'cola_single_clevr':
        # return cola_single_clevr(transform, root_dir =f'{root_dir}/clevr/CLEVR_CoGenT_v1.0/images')
        return ValueError('cola_single_clevr cannot be implemented')
    elif dataset_name == 'vismin':
        return VisMin(transform, resize=resize, root_dir =f'{root_dir}/VisMin/images/train', split=split)
    elif dataset_name == 'vismin_relation':
        return VisMin(transform, resize=resize, root_dir =f'{root_dir}/VisMin/images/train',subset='relation', split=split)
    elif dataset_name == 'vismin_attribute':
        return VisMin(transform, resize=resize, root_dir =f'{root_dir}/VisMin/images/train',subset='attribute', split=split)
    elif dataset_name == 'vismin_object':
        return VisMin(transform, resize=resize, root_dir =f'{root_dir}/VisMin/images/train',subset='object', split=split)
    elif dataset_name == 'vismin_counting':
        return VisMin(transform, resize=resize, root_dir =f'{root_dir}/VisMin/images/train',subset='counting', split=split)
    elif dataset_name == 'countbench':
        return CountBench(transform, resize=resize, root_dir =f'{root_dir}')
    elif dataset_name == 'valse_action-replacement':
        return VALSE(transform, resize=resize, root_dir =f'{root_dir}/SWiG/images_512', subset='action-replacement')
    elif dataset_name == 'valse_actant-swap':
        return VALSE(transform, resize=resize, root_dir =f'{root_dir}/SWiG/images_512', subset='actant-swap')
    elif dataset_name == 'valse_existence':
        return VALSE(transform, resize=resize, root_dir =f'{root_dir}/visual7w/images', subset='existence')
    elif dataset_name == 'valse_counting-adversarial':
        return VALSE(transform, resize=resize, root_dir =f'{root_dir}/visual7w/images', subset='counting-adversarial')
    elif dataset_name == 'valse_counting-hard':
        return VALSE(transform, resize=resize, root_dir =f'{root_dir}/visual7w/images', subset='counting-hard')
    elif dataset_name == 'valse_counting-small-quant':
        return VALSE(transform, resize=resize, root_dir =f'{root_dir}/visual7w/images', subset='counting-small-quant')
    elif dataset_name == 'valse_relations':
        return VALSE(transform, resize=resize, root_dir =f'{root_dir}/COCO2017/val2017', subset='relations')
    elif dataset_name == 'valse_foil-it':
        return VALSE(transform, resize=resize, root_dir =f'{root_dir}/COCO2014/val2014', subset='foil-it')
    elif dataset_name == 'valse_plurals':
        return VALSE(transform, resize=resize, root_dir =f'{root_dir}/COCO2017/val2017', subset='plurals')
    elif dataset_name == 'spec_absolute_size':
        if mode == False:
            return SPEC_Image2Text(transform, root_dir =f'{root_dir}/spec', resize=resize, scoring_only=scoring_only, subset='absolute_size')
        else:
            return SPEC_Text2Image(transform, root_dir =f'{root_dir}/spec', resize=resize, scoring_only=scoring_only, subset='absolute_size')
    elif dataset_name == 'spec_absolute_spatial':
        if mode == False:
            return SPEC_Image2Text(transform, root_dir =f'{root_dir}/spec', resize=resize, scoring_only=scoring_only, subset='absolute_spatial')
        else:
            return SPEC_Text2Image(transform, root_dir =f'{root_dir}/spec', resize=resize, scoring_only=scoring_only, subset='absolute_spatial')
    elif dataset_name == 'spec_count':
        if mode == False:
            return SPEC_Image2Text(transform,  root_dir =f'{root_dir}/spec', resize=resize, scoring_only=scoring_only, subset='count')
        else:
            return SPEC_Text2Image(transform, root_dir =f'{root_dir}/spec', resize=resize, scoring_only=scoring_only, subset='count')
        # return SPEC(transform, root_dir =f'{root_dir}/spec',  resize=resize, scoring_only=scoring_only, subset='count')
    elif dataset_name == 'spec_existence':
        if mode == False:
            return SPEC_Image2Text(transform, root_dir =f'{root_dir}/spec', resize=resize, scoring_only=scoring_only, subset='existence')
        else:
            return SPEC_Text2Image(transform, root_dir =f'{root_dir}/spec', resize=resize, scoring_only=scoring_only, subset='existence')
        # return SPEC(transform, root_dir =f'{root_dir}/spec', resize=resize, scoring_only=scoring_only, subset='existence')
    elif dataset_name == 'spec_relative_size':
        if mode == False:
            return SPEC_Image2Text(transform,  root_dir =f'{root_dir}/spec', resize=resize, scoring_only=scoring_only, subset='relative_size')
        else:
            return SPEC_Text2Image(transform, root_dir =f'{root_dir}/spec', resize=resize, scoring_only=scoring_only, subset='relative_size')
        # return SPEC(transform, root_dir =f'{root_dir}/spec', resize=resize, scoring_only=scoring_only, subset='relative_size')
    elif dataset_name == 'spec_relative_spatial':
        if mode == False:
            return SPEC_Image2Text(transform,  root_dir =f'{root_dir}/spec', resize=resize, scoring_only=scoring_only, subset='relative_spatial')
        else:
            return SPEC_Text2Image(transform,  root_dir =f'{root_dir}/spec', resize=resize, scoring_only=scoring_only, subset='relative_spatial')
        # return SPEC(transform, root_dir =f'{root_dir}/spec', resize=resize, scoring_only=scoring_only, subset='relative_spatial')
    elif dataset_name == 'vlcheck_action':
        return VLCheck_Attribute(transform, resize=resize, root_dir =f'{root_dir}/VG_100K/image', subset='action')
    elif dataset_name == 'vlcheck_color':
        return VLCheck_Attribute(transform, resize=resize, root_dir =f'{root_dir}/VG_100K/image', subset='color')
    elif dataset_name == 'vlcheck_material':
        return VLCheck_Attribute(transform, resize=resize, root_dir =f'{root_dir}/VG_100K/image', subset='material')
    elif dataset_name == 'vlcheck_size':
        return VLCheck_Attribute(transform, resize=resize, root_dir =f'{root_dir}/VG_100K/image', subset='size')
    elif dataset_name == 'vlcheck_state':
        return VLCheck_Attribute(transform, resize=resize, root_dir =f'{root_dir}/VG_100K/image', subset='state')
    elif dataset_name == 'vlcheck_Object_Location_hake':
        return VLCheck_Object_Location(transform, resize=resize, root_dir =f'{root_dir}/HAKE', subset='hake')
    elif dataset_name == 'vlcheck_Object_Location_swig':
        return VLCheck_Object_Location(transform, resize=resize, root_dir =f'{root_dir}/SWiG/images_512', subset='swig')
    elif dataset_name == 'vlcheck_Object_Location_vg':
        return VLCheck_Object_Location(transform, resize=resize, root_dir =f'{root_dir}/VG_100K/image', subset='vg')
    elif dataset_name == 'vlcheck_Object_Size_hake':
        return VLCheck_Object_Size(transform, resize=resize, root_dir =f'{root_dir}/HAKE', subset='hake')
    elif dataset_name == 'vlcheck_Object_Size_swig':
        return VLCheck_Object_Size(transform, resize=resize, root_dir =f'{root_dir}/SWiG/images_512', subset='swig')
    elif dataset_name == 'vlcheck_Object_Size_vg':
        return VLCheck_Object_Size(transform, resize=resize, root_dir =f'{root_dir}/VG_100K/image', subset='vg')
    elif dataset_name == 'vlcheck_Relation_vg_action':
        return VLCheck_Relation(transform, resize=resize, root_dir =f'{root_dir}/VG_100K/image', subset='vg_action')
    elif dataset_name == 'vlcheck_Relation_vg_spatial':
        return VLCheck_Relation(transform, resize=resize, root_dir =f'{root_dir}/VG_100K/image', subset='vg_spatial')
    elif dataset_name == 'vlcheck_Relation_hake':
        return VLCheck_Relation(transform, resize=resize, root_dir =f'{root_dir}/HAKE', subset='hake')
    elif dataset_name == 'vlcheck_Relation_swig':
        return VLCheck_Relation(transform, resize=resize, root_dir =f'{root_dir}/SWiG/images_512', subset='swig')
    # elif dataset_name == "eqbench_eqbenyoucook2":
    #     return EQBench(transform, resize=resize, root_dir =f'{root_dir}/eqbench', subset='eqbenyoucook2')
    # elif dataset_name == "eqbench_eqbengebc":
    #     return EQBench(transform, resize=resize, root_dir =f'{root_dir}/eqbench', subset='eqbengebc')
    # elif dataset_name == "eqbench_eqbenag":
    #     return EQBench(transform, resize=resize, root_dir =f'{root_dir}/eqbench', subset='eqbenag')
    # elif dataset_name == "eqbench_eqbenkubric_attr":
    #     return EQBench(transform, resize=resize, root_dir =f'{root_dir}/eqbench', subset='eqbenkubric_attr')
    # elif dataset_name == "eqbench_eqbenkubric_cnt":
    #     return EQBench(transform, resize=resize, root_dir =f'{root_dir}/eqbench', subset='eqbenkubric_cnt')
    # elif dataset_name == "eqbench_eqbenkubric_loc":
    #     return EQBench(transform, resize=resize, root_dir =f'{root_dir}/eqbench', subset='eqbenkubric_loc')
    # elif dataset_name == "eqbench_eqbensd":
    #     return EQBench(transform, resize=resize, root_dir =f'{root_dir}/eqbench', subset='eqbensd')
    elif dataset_name == "eqbench_eqbenyoucook2":
        return EQBench_subset(transform, resize=resize, root_dir =f'{root_dir}/eqbench', subset='eqbenyoucook2')
    elif dataset_name == "eqbench_eqbengebc":
        return EQBench_subset(transform, resize=resize, root_dir =f'{root_dir}/eqbench', subset='eqbengebc')
    elif dataset_name == "eqbench_eqbenag":
        return EQBench_subset(transform, resize=resize, root_dir =f'{root_dir}/eqbench', subset='eqbenag')
    elif dataset_name == "eqbench_eqbenkubric_attr":
        return EQBench_subset(transform, resize=resize, root_dir =f'{root_dir}/eqbench', subset='eqbenkubric_attr')
    elif dataset_name == "eqbench_eqbenkubric_cnt":
        return EQBench_subset(transform, resize=resize, root_dir =f'{root_dir}/eqbench', subset='eqbenkubric_cnt')
    elif dataset_name == "eqbench_eqbenkubric_loc":
        return EQBench_subset(transform, resize=resize, root_dir =f'{root_dir}/eqbench', subset='eqbenkubric_loc')
    elif dataset_name == "eqbench_eqbensd":
        return EQBench_subset(transform, resize=resize, root_dir =f'{root_dir}/eqbench', subset='eqbensd')
    elif dataset_name == "naturalbench":
        return NaturalBench(transform, resize=resize, subset=index_subset)  
    # elif dataset_name == "geneval_color":
    #     return Geneval(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "colors", version = version, domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_position":
    #     return Geneval(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "position", version = version, domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_counting":
    #     return Geneval(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "counting", version = version, domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_single":
    #     return Geneval(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "single_object", version = version, domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_two":
    #     return Geneval(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "two_object", version = version, domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_color_attr":
    #     return Geneval(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "color_attr", version = version, domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_two_subset":
    #     return Geneval(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "two_object_subset", version = version, domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_filter_color":
    #     return Geneval_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "colors", version = version, domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_filter_position":
    #     return Geneval_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "position", version = version, domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_filter_counting":
    #     return Geneval_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "counting", version = version, domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_filter_single":
    #     return Geneval_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "single_object", version = version, domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_filter_two":
    #     return Geneval_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "two_object", version = version, domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_filter_color_attr":
    #     return Geneval_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "color_attr", version = version, domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_filter_two_subset":
    #     return Geneval_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "two_object_subset", version = version, domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_1_5_color":
    #     return Geneval_1_5(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "colors", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_1_5_position":
    #     return Geneval_1_5(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "position", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_1_5_counting":
    #     return Geneval_1_5(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "counting", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_1_5_single":
    #     return Geneval_1_5(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "single_object", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_1_5_two":
    #     return Geneval_1_5(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "two_object", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_1_5_color_attr":
    #     return Geneval_1_5(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "color_attr", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_1_5_two_subset":
    #     return Geneval_1_5(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "two_object_subset", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_2_0_color":
    #     return Geneval_2_0(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "colors", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_2_0_position":
    #     return Geneval_2_0(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "position", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_2_0_counting":
    #     return Geneval_2_0(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "counting", domain = domain,  cfg = cfg)
    # elif dataset_name == "geneval_2_0_single":
    #     return Geneval_2_0(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "single_object", domain = domain,  cfg = cfg)
    # elif dataset_name == "geneval_2_0_two":
    #     return Geneval_2_0(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "two_object", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_2_0_color_attr":
    #     return Geneval_2_0(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "color_attr", domain = domain,cfg = cfg)
    # elif dataset_name == "geneval_2_0_two_subset":
    #     return Geneval_2_0(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "two_object_subset", domain = domain, cfg = cfg)
    
    # elif dataset_name == "geneval_3_m_color":
    #     return Geneval_3_m(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "colors", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_3_m_position":
    #     return Geneval_3_m(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "position", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_3_m_counting":
    #     return Geneval_3_m(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "counting", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_3_m_single":
    #     return Geneval_3_m(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "single_object", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_3_m_two":
    #     return Geneval_3_m(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "two_object", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_3_m_color_attr":
    #     return Geneval_3_m(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "color_attr", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_3_m_two_subset":
    #     return Geneval_3_m(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "two_object_subset", domain = domain, cfg = cfg)
    
    
    # elif dataset_name == "geneval_1_5_filter_color":
    #     return Geneval_1_5_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "colors", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_1_5_filter_position":
    #     return Geneval_1_5_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "position", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_1_5_filter_counting":
    #     return Geneval_1_5_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "counting", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_1_5_filter_single":
    #     return Geneval_1_5_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "single_object", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_1_5_filter_two":
    #     return Geneval_1_5_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "two_object", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_1_5_filter_color_attr":
    #     return Geneval_1_5_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "color_attr", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_1_5_filter_two_subset":
    #     return Geneval_1_5_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "two_object_subset", domain = domain, cfg = cfg)
    
    
    # elif dataset_name == "geneval_2_0_filter_color":
    #     return Geneval_2_0_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "colors", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_2_0_filter_position":
    #     return Geneval_2_0_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "position", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_2_0_filter_counting":
    #     return Geneval_2_0_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "counting", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_2_0_filter_single":
    #     return Geneval_2_0_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "single_object", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_2_0_filter_two":
    #     return Geneval_2_0_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "two_object", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_2_0_filter_color_attr":
    #     return Geneval_2_0_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "color_attr", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_2_0_filter_two_subset":
    #     return Geneval_2_0_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "two_object_subset", domain = domain, cfg = cfg)
    
    
    # elif dataset_name == "geneval_3_m_filter_color":
    #     return Geneval_3_m_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "colors", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_3_m_filter_position":
    #     return Geneval_3_m_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "position", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_3_m_filter_counting":
    #     return Geneval_3_m_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "counting", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_3_m_filter_single":
    #     return Geneval_3_m_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "single_object", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_3_m_filter_two":
    #     return Geneval_3_m_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "two_object", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_3_m_filter_color_attr":
    #     return Geneval_3_m_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "color_attr", domain = domain, cfg = cfg)
    # elif dataset_name == "geneval_3_m_filter_two_subset":
    #     return Geneval_3_m_filter(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "two_object_subset", domain = domain, cfg = cfg)

    elif dataset_name == "geneval_colors" or dataset_name == "geneval_color":
        return Geneval_final(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "colors", filter= filter, version = version)
    elif dataset_name == "geneval_color_attr" or dataset_name == "geneval_color_attribution":
        return Geneval_final(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "color_attr", filter= filter, version = version)
    elif dataset_name == "geneval_position":
        return Geneval_final(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "position", filter= filter, version = version)
    elif dataset_name == "geneval_counting":
        return Geneval_final(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "counting", filter= filter, version = version)
    elif dataset_name == "geneval_single" or dataset_name == "geneval_single_object":
        return Geneval_final(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "single_object", filter= filter, version = version)
    elif dataset_name == "geneval_two" or dataset_name == "geneval_two_object":
        return Geneval_final(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "two_object_subset", filter= filter, version = version)

    elif dataset_name == "ours_colors":
        return Ours(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "colors", version = version, domain = domain)
    elif dataset_name == "ours_color_attr":
        return Ours(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "color_attr", version = version, domain = domain)
    elif dataset_name == "ours_position":
        return Ours(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "position", version = version, domain = domain)
    elif dataset_name == "ours_counting":
        return Ours(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "counting", version = version, domain = domain)
    # elif dataset_name == "compbench_color":
    #     return Compbench(transform, resize=resize, root_dir =f'./CompBench/examples',subset= "color", version = version, domain = domain)
    elif dataset_name == "ours_before_colors":
        return Ours(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "colors", version = version, domain = domain, before= True)
    elif dataset_name == "ours_before_color_attr":
        return Ours(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "color_attr", version = version, domain = domain, before = True)
    elif dataset_name == "ours_before_position":
        return Ours(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "position", version = version, domain = domain, before = True)
    elif dataset_name == "ours_before_counting":
        return Ours(transform, resize=resize, root_dir =f'../geneval/outputs',subset= "counting", version = version, domain = domain, before = True)
    elif dataset_name == "ours_1_5_colors":
        return Ours_1_5(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "colors", domain = domain, before = False)
    elif dataset_name == "ours_1_5_color_attr":
        return Ours_1_5(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "color_attr", domain = domain, before = False)
    elif dataset_name == "ours_1_5_position":
        return Ours_1_5(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "position", domain = domain, before = False)
    elif dataset_name == "ours_1_5_counting":
        return Ours_1_5(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "counting", domain = domain, before = False)
    elif dataset_name == "ours_1_5_before_colors":
        return Ours_1_5(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "colors", domain = domain, before = True)
    elif dataset_name == "ours_1_5_before_color_attr":
        return Ours_1_5(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "color_attr", domain = domain, before = True)
    elif dataset_name == "ours_1_5_before_position":
        return Ours_1_5(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "position", domain = domain, before = True)
    elif dataset_name == "ours_1_5_before_counting":
        return Ours_1_5(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "counting", domain = domain, before = True)
    elif dataset_name == "ours_2_0_colors":
        return Ours_2_0(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "colors", domain = domain, before = False)
    elif dataset_name == "ours_2_0_color_attr":
        return Ours_2_0(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "color_attr", domain = domain, before = False)
    elif dataset_name == "ours_2_0_position":
        return Ours_2_0(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "position", domain = domain, before = False)
    elif dataset_name == "ours_2_0_counting":
        return Ours_2_0(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "counting", domain = domain, before = False)
    elif dataset_name == "ours_2_0_before_colors":
        return Ours_2_0(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "colors", domain = domain, before = True)
    elif dataset_name == "ours_2_0_before_color_attr":
        return Ours_2_0(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "color_attr", domain = domain, before = True)
    elif dataset_name == "ours_2_0_before_position":
        return Ours_2_0(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "position", domain = domain, before = True)
    elif dataset_name == "ours_2_0_before_counting":
        return Ours_2_0(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "counting", domain = domain, before = True)
    elif dataset_name == "ours_3_m_colors":
        return Ours_3_m(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "colors", domain = domain, before = False)
    elif dataset_name == "ours_3_m_color_attr":
        return Ours_3_m(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "color_attr", domain = domain, before = False)
    elif dataset_name == "ours_3_m_position":
        return Ours_3_m(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "position", domain = domain, before = False)
    elif dataset_name == "ours_3_m_counting":
        return Ours_3_m(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "counting", domain = domain, before = False)
    elif dataset_name == "ours_3_m_before_colors":
        return Ours_3_m(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "colors", domain = domain, before = True)
    elif dataset_name == "ours_3_m_before_color_attr":
        return Ours_3_m(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "color_attr", domain = domain, before = True)
    elif dataset_name == "ours_3_m_before_position":
        return Ours_3_m(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "position", domain = domain, before = True)
    elif dataset_name == "ours_3_m_before_counting":
        return Ours_3_m(transform, resize=resize, root_dir =f'../geneval/outputs', subset= "counting", domain = domain, before = True)
    elif dataset_name == "mmvp_camera": # perspective
        return MMVP_VLM(transform, resize=resize, root_dir = os.environ["HF_HOME"], subset= "Camera Perspective")
    elif dataset_name == "mmvp_color":
        return MMVP_VLM(transform, resize=resize, root_dir =os.environ["HF_HOME"], subset= "Color")
    elif dataset_name == "mmvp_orientation":
        return MMVP_VLM(transform, resize=resize, root_dir =os.environ["HF_HOME"], subset= "Orientation")
    elif dataset_name == "mmvp_presence":
        return MMVP_VLM(transform, resize=resize, root_dir =os.environ["HF_HOME"], subset= "Presence")
    elif dataset_name == "mmvp_quantity":
        return MMVP_VLM(transform, resize=resize, root_dir =os.environ["HF_HOME"], subset= "Quantity")
    elif dataset_name == "mmvp_spatial":
        return MMVP_VLM(transform, resize=resize, root_dir =os.environ["HF_HOME"], subset= "Spatial")
    elif dataset_name == "mmvp_state":
        return MMVP_VLM(transform, resize=resize, root_dir =os.environ["HF_HOME"], subset= "State")
    elif dataset_name == "mmvp_structural":
        if ARNAS_USES:
            if "HF_HOME" not in os.environ:
                os.environ["HF_HOME"] = '/home/oh/owl661/.cache/huggingface/'
        
        return MMVP_VLM(transform, resize=resize, root_dir =os.environ["HF_HOME"], subset= "Structural Character")
    elif dataset_name == "mmvp_text":
        return MMVP_VLM(transform, resize=resize, root_dir =os.environ["HF_HOME"], subset= "Text")
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

def diffusers_preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = image.squeeze(0)
    return 2.0 * image - 1.0
    

lora_train_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512) if True else transforms.RandomCrop(512),
            transforms.RandomHorizontalFlip() if True else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

class STL10(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False, split = 'test'):
        self.root_dir = f'{root_dir}/stl10_binary'
        self.transform = transform
        self.split = split
        self.data = datasets.STL10(root_dir, split=split, transform=transform,
                                 target_transform=None, download=(not os.path.exists(self.root_dir)))

        self.resize = resize
        self.classes = [f'a photo of a {i}' for i in self.data.classes]
        # print(self.classes)
        self.scoring_only = scoring_only

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.scoring_only:
            img, class_id = self.data[idx]
            # print(class_id)
            # exit(0)
            img = img.convert("RGB")
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
        else:
            class_id = idx // 50

        if self.scoring_only:
            return self.classes, class_id
        else:
            return ([img], [img_resize]), self.classes, class_id


class Cifar10(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False, split = 'test'):
        self.root_dir = f'{root_dir}/cifar-10-batches-py'
        self.transform = transform
        self.split = split
        self.data = datasets.CIFAR10(root_dir, train=False if split == 'val' else True, transform=transform,
                                 target_transform=None, download=(not os.path.exists(self.root_dir)))

        self.resize = resize
        self.classes = [f'a photo of a {i}' for i in self.data.classes]
        self.scoring_only = scoring_only
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.scoring_only:
            img, class_id = self.data[idx]
            img = img.convert("RGB")
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
        else:
            class_id = idx // 50

        if self.scoring_only:
            return self.classes, class_id
        else:
            return (idx, [img_resize]), self.classes, class_id

class ImagenetDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False):
        self.root_dir = root_dir
        self.data = datasets.ImageFolder(root_dir + '/imagenet/val')
        # self.loader = torch.utils.data.DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        self.resize = resize
        self.transform = transform
        self.classes = list(json.load(open(root_dir +f'/imagenet/imagenet_classes.json', 'r')).values())
        if True:
            prompted_classes = []
            for c in self.classes:
                class_text = 'a photo of a ' + c
                prompted_classes.append(class_text)
            self.classes = prompted_classes
        self.scoring_only = scoring_only

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.scoring_only:
            img, class_id = self.data[idx]
            img = img.convert("RGB")
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
        else:
            class_id = idx // 50

        if self.scoring_only:
            return self.classes, class_id
        else:
            return ([img], [img_resize]), self.classes, class_id

class PetsDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False):
        root_dir =  "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/raw/oxford-iiit-pet"
        self.root_dir = root_dir
        # read all imgs in root_dir with glob
        imgs = list(glob(root_dir + '/images/*.jpg'))
        self.resize = resize
        self.transform = transform
        self.classes = list(open(f'data/pets/classes.txt', 'r').read().splitlines())
        self.data = []
        for img_path in imgs:
            filename = img_path.split('/')[-1].split('_')
            class_name = ' '.join(filename[:-1])
            lower_case_class_name = class_name.lower()
            class_id = self.classes.index(lower_case_class_name)
            self.data.append((img_path, class_id))
        prompted_classes = []
        for c in self.classes:
            class_text = 'a photo of a ' + c
            prompted_classes.append(class_text)
        self.classes = prompted_classes
        self.scoring_only = scoring_only

    def __getitem__(self, idx):
        if not self.scoring_only:
            img, class_id = self.data[idx]
            img = Image.open(img)
            img = img.convert("RGB")
            if self.transform:
                img_resize = self.transform(img).unsqueeze(0)
            else:
                img_resize = img.resize((self.resize, self.resize))
                img_resize = diffusers_preprocess(img_resize)
        else:
            class_id = idx // 50
        print(class_id)
        if self.scoring_only:
            return self.classes, class_id
        else:
            return [0, [img_resize]], self.classes, class_id

    def __len__(self):
        return len(self.data)

class GenderBiasDataset(Dataset):
    def __init__(self, root_dir, resize=512, transform=None, targets=None):
        self.root_dir = root_dir #datasets/genderbias/
        self.resize = resize
        self.transform = transform
        self.data = []
        # self.attributes = {'apron':'clothes','suit':'clothes','briefcase':'bags','purse':'bags','beer':'drinks','wine':'drinks'}
        clothes_imgs = list(glob(f'{root_dir}/suit-images/*.jpg')) + list(glob(f'{root_dir}/apron-images/*.jpg'))
        bags_imgs = list(glob(f'{root_dir}/briefcase-images/*.jpg')) + list(glob(f'{root_dir}/purse-images/*.jpg'))
        drinks_imgs = list(glob(f'{root_dir}/beer-images/*.jpg')) + list(glob(f'{root_dir}/wine-images/*.jpg'))
        self.data.extend((img,('clothes',['suit','apron'])) for img in clothes_imgs)
        self.data.extend((img,('bags',['briefcase','purse'])) for img in bags_imgs)
        self.data.extend((img,('drinks',['beer','wine'])) for img in drinks_imgs)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, (target, texts) = self.data[idx]
        # class_id is {male,female}_{clothes,drinks,bags}
        class_id = f'male_{target}' if int(img.split('/')[-1].split('.')[0]) <=6 else f'female_{target}'
        img = Image.open(img)
        img = img.convert("RGB")
        if self.transform:
                img_resize = self.transform(img).unsqueeze(0)
        else:
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)
        return (0, [img_resize]), texts, class_id
    
class BiasDataset(Dataset):
    def __init__(self, root_dir, resize=512, transform=None, targets=None):
        root_dir = f'{root_dir}/mmbias' #datasets/mmbias/
        self.resize = resize
        self.transform = transform
        self.phrases = json.load(open(f'{root_dir}/textual_phrases.txt'))
        self.texts = self.phrases['pleasant_phrases']
        self.texts.extend(self.phrases['unpleasant_phrases']) # texts is ['good','texts',...,'bad','texts']
        self.classes = ['Christian','Muslim','Jewish','Hindu','American','Arab','Heterosexual','LGBT','Buddhist']
        if targets == '': # use all classes by default
            targets=self.classes
        print(targets)
        self.data = []
        
        # Religion
        if 'Christian' in targets:
            self.christian_imgs = list(glob(f'{root_dir}/data/Images/Religion/Christian/*.jpg'))
            self.data.extend([(img_path,0) for img_path in self.christian_imgs])
        if 'Muslim' in targets:
            self.muslim_imgs = list(glob(f'{root_dir}/data/Images/Religion/Muslim/*.jpg'))
            self.data.extend([(img_path,1) for img_path in self.muslim_imgs])
        if 'Jewish' in targets:
            self.jewish_imgs = list(glob(f'{root_dir}/data/Images/Religion/Jewish/*.jpg'))
            self.data.extend([(img_path,2) for img_path in self.jewish_imgs])
        if 'Hindu' in targets:
            self.hindu_imgs = list(glob(f'{root_dir}/data/Images/Religion/Hindu/*.jpg'))
            self.data.extend([(img_path,3) for img_path in self.hindu_imgs])
        if 'Buddhist' in targets:
            self.buddhist_imgs = list(glob(f'{root_dir}/data/Images/Religion/Buddhist/*.jpg'))
            self.data.extend([(img_path,8) for img_path in self.buddhist_imgs])
        # Nationality
        if 'American' in targets:
            self.american_imgs = list(glob(f'{root_dir}/data/Images/Nationality/American/*.jpg'))
            self.data.extend([(img_path,4) for img_path in self.american_imgs])
        if 'Arab' in targets:
            self.arab_imgs = list(glob(f'{root_dir}/data/Images/Nationality/Arab/*.jpg'))
            self.data.extend([(img_path,5) for img_path in self.arab_imgs])
        # Sexuality
        if 'Heterosexual' in targets:
            self.hetero_imgs = list(glob(f'{root_dir}/data/Images/Sexual Orientation/Heterosexual/*.jpg'))
            self.data.extend([(img_path,6) for img_path in self.hetero_imgs])
        if 'LGBT' in targets:
            self.lgbt_imgs = list(glob(f'{root_dir}/data/Images/Sexual Orientation/LGBT/*.jpg'))
            self.data.extend([(img_path,7) for img_path in self.lgbt_imgs])
        # uncommment for just subset
        # self.data = self.data[::5]
        # self.texts = self.texts[::3]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, class_id = self.data[idx]
        img = Image.open(img)
        img = img.convert("RGB")
        if self.transform:
                img_resize = self.transform(img).unsqueeze(0)
        else:
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)
        return (0, [img_resize]), self.texts, class_id


class WinogroundDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False,subset=None):
        # with open("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/hf_token.txt", 'r') as f:
        #     token = f.read().strip()
        # login(token) # datasets==2.14.6 works
        # # download_config = DownloadConfig(use_auth_token=token)
        # os.environ["HUGGINGFACE_TOKEN"] = token
        # self.examples = load_dataset("facebook/winoground",use_auth_token=True)["test"]
        with open("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/hf_token.txt", "r") as f:
            hf_token = f.read().strip()
    
        try:
            self.examples = load_dataset('facebook/winoground', token = hf_token)
        except:
            self.examples = load_dataset('facebook/winoground', use_auth_token=hf_token)
        self.examples = self.examples['test']
        # print(self.examples)
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        if self.subset != None:
            self.examples = self.examples.filter(lambda x: x['collapsed_tag'] == self.subset)
        # print(self.examples)
        # exit(0)
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        cap0 = ex['caption_0']
        cap1 = ex['caption_1']
        img_id = ex['id']
        if not self.scoring_only:
            img0 = ex['image_0'].convert('RGB')
            img1 = ex['image_1'].convert('RGB')
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
                img1_resize = self.transform(img1).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img1_resize = img1.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
                img1_resize = diffusers_preprocess(img1_resize)
        text = [cap0, cap1]
        if self.scoring_only:
            return text, img_id
        else:
            return ((f"{idx}_0", f"{idx}_1"), [img0_resize, img1_resize]), text, img_id

class ImageCoDeDataset(Dataset):
    def __init__(self, root_dir, split, transform, resize=512, scoring_only=False, static=True):
        self.root_dir = f'{root_dir}/imagecode'
        self.resize = resize
        self.dataset = self.load_data(self.root_dir, split, static_only=static)
        self.transform = transform
        self.scoring_only = scoring_only

    @staticmethod
    def load_data(data_dir, split, static_only=True):
        split = 'valid' if split == 'val' else split
        with open(f'{data_dir}/{split}_data.json') as f:
            json_file = json.load(f)
        img_path = f'{data_dir}/image-sets'

        dataset = []
        for img_dir, data in json_file.items():
            img_files = list((Path(f'{img_path}/{img_dir}')).glob('*.jpg'))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            for img_idx, text in data.items():
                static = 'open-images' in img_dir
                if static_only:
                    if static:
                        dataset.append((img_dir, img_files, int(img_idx), text))
                else:
                    dataset.append((img_dir, img_files, int(img_idx), text))

        return dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img_dir, img_files, img_idx, text = self.dataset[idx]
        if not self.scoring_only:
            imgs = [Image.open(img_path).convert("RGB") for img_path in img_files]
            if self.transform:
                imgs_resize = [self.transform(img).unsqueeze(0) for img in imgs]
            else:
                imgs_resize = [img.resize((self.resize, self.resize)) for img in imgs]
                imgs_resize = [diffusers_preprocess(img) for img in imgs_resize]

        if self.scoring_only:
            return text, img_dir, img_idx
        else:
            return ([img_path for img_path in img_files], imgs_resize), [text], img_dir, img_idx



class MSCOCODataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, split='val', tokenizer=None, hard_neg=True, neg_img=False, mixed_neg=False, tsv_path='aro/temp_data/train_neg_clip.tsv'):
        self.root_dir = 'data/mscoco/train2014'
        self.resize = resize
        self.data = pd.read_csv(tsv_path, delimiter='\t')
        self.all_texts = self.data['title'].tolist()
        self.transform = transform
        self.split = split
        self.tokenizer = tokenizer
        self.hard_neg = hard_neg
        self.neg_img = neg_img
        self.mixed_neg = mixed_neg
        self.rand_neg = not self.hard_neg and not self.neg_img


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['filepath']
        # only get filename
        img_path = img_path.split('/')[-1]
        if 'train2014' in img_path:
            img_path = f"{self.root_dir}/{img_path}"
        else:
            img_path = f"data/coco_order/val2014/{img_path}"
        text = row['title']
        neg_captions =  ast.literal_eval(row['neg_caption'])
        neg_caption = neg_captions[np.random.randint(0, len(neg_captions))]

        neg_img_ids = ast.literal_eval(row['neg_image']) # a list of row indices in self.data
        neg_paths = self.data.iloc[neg_img_ids]['filepath'].tolist()
        new_neg_paths = []
        for path in neg_paths:
            path = path.split('/')[-1]
            if 'train2014' in path:
                path = f"{self.root_dir}/{path}"
            else:
                path = f"data/coco_order/val2014/{path}"
            new_neg_paths.append(path)
        neg_paths = new_neg_paths
        
        
        if self.tokenizer:
            text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            text0 = text.input_ids.squeeze(0)
            # text0 = text[0]
            if self.mixed_neg:
                text_neg = self.tokenizer(neg_caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_neg = text_neg.input_ids.squeeze(0)
                text_rand = self.tokenizer(self.all_texts[np.random.randint(0, len(self.all_texts))], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
                text = torch.stack([text0, text_neg, text_rand])
            elif self.hard_neg:
                text_rand = self.tokenizer(neg_caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
                text = torch.stack([text0, text_rand])
            elif self.rand_neg:
                text_rand = self.tokenizer(self.all_texts[np.random.randint(0, len(self.all_texts))], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
                text = torch.stack([text0, text_rand])
            else:
                text = text0
        
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img_resize = self.transform(img).unsqueeze(0)
        else:
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)
        imgs = [img_resize]

        if self.neg_img or self.mixed_neg:
            assert not self.hard_neg
            rand_path = neg_paths[np.random.randint(0, len(neg_paths))]
            rand_img = Image.open(rand_path).convert("RGB")
            if self.transform:
                rand_img = self.transform(rand_img).unsqueeze(0)
            else:
                rand_img = rand_img.resize((self.resize, self.resize))
                rand_img = diffusers_preprocess(rand_img)
            imgs.append(rand_img)

        # if np.random.rand() > 0.99:
        #     print("Img true:", img_path)
        #     print("Neg Img:", rand_path)
        #     print(text)
        
        return [0, imgs], text, 0


class ValidMSCOCODataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, split='val', tokenizer=None, hard_neg=False, tsv_path='aro/temp_data/valid_neg_clip.tsv', neg_img=False):
        self.root_dir = 'data/mscoco/'
        self.resize = resize
        self.data = pd.read_csv(tsv_path, delimiter='\t')
        self.transform = transform
        self.split = split
        self.tokenizer = tokenizer
        self.hard_neg = hard_neg
        self.neg_img = neg_img
        if not self.neg_img:
            self.hard_neg = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['filepath']
        # only get filename
        img_path = img_path.split('/')[-1]
        img_path = f"data/coco_order/val2014/{img_path}"
        text = row['title']
        if self.hard_neg:
            neg_captions =  ast.literal_eval(row['neg_caption'])
            neg_caption = neg_captions[np.random.randint(0, len(neg_captions))]
            text = [text, neg_caption]
        else:
            text = [text]

        neg_img_ids = ast.literal_eval(row['neg_image'])
        neg_paths = self.data.iloc[neg_img_ids]['filepath'].tolist()
        new_neg_paths = []
        for path in neg_paths:
            path = path.split('/')[-1]
            if 'train2014' in path:
                path = f"{self.root_dir}/{path}"
            else:
                path = f"data/coco_order/val2014/{path}"
            new_neg_paths.append(path)
        neg_paths = new_neg_paths

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img_resize = self.transform(img).unsqueeze(0)
        else:
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)
        imgs = [img_resize]

        if self.neg_img:
            assert not self.hard_neg
            rand_path = neg_paths[np.random.randint(0, len(neg_paths))]
            rand_img = Image.open(rand_path).convert("RGB")
            if self.transform:
                rand_img = self.transform(rand_img).unsqueeze(0)
            else:
                rand_img = rand_img.resize((self.resize, self.resize))
                rand_img = diffusers_preprocess(rand_img)
            imgs.append(rand_img)

        # print("Img true:", img_path)
        # print("Neg Img:", rand_path)
        # print(text)

        return [0, imgs], text, 0


class Flickr30KDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False, split='val', tokenizer=None, first_query=True, details=False):
        self.root_dir = root_dir
        self.resize = resize
        self.data = json.load(open(f'{root_dir}/flickr-image-dataset/versions/{split}_top10_RN50x64.json', 'r'))
        self.data = list(self.data.items())
        # get only every 5th example
        if first_query:
            self.data = self.data[::5]
        self.transform = transform
        self.scoring_only = scoring_only
        self.tokenizer = tokenizer
        self.details = details
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        text = ex[0]
        if self.tokenizer:
            text = self.tokenizer([text], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            text = text.input_ids.squeeze(0)
        img_paths = ex[1]
        img_idx = 0

        imgs = [Image.open(f'{img_path.replace("datasets/flickr30k/images",f"{self.root_dir}/flickr-image-dataset/versions/1/flickr30k_images/flickr30k_images")}').convert("RGB") for img_path in img_paths]
        
        if self.transform:
            imgs_resize = [self.transform(img).unsqueeze(0) for img in imgs]
        else:
            imgs_resize = [img.resize((self.resize, self.resize)) for img in imgs]
            imgs_resize = [diffusers_preprocess(img) for img in imgs_resize]

        # imgs_resize = [img.resize((self.resize, self.resize)) for img in imgs]
        
        # imgs_resize = [diffusers_preprocess(img) for img in imgs_resize]

        # if self.transform:
        #     imgs = [self.transform(img) for img in imgs]
        # else:
        #     imgs = [transforms.ToTensor()(img) for img in imgs]
        
        return [img_paths, imgs_resize], [text], img_idx

class Flickr30KTextRetrievalDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False, split='val', tokenizer=None, hard_neg=False, details=False):
        self.root_dir = root_dir
        self.resize = resize
        self.data = json.load(open(f'{self.root_dir}/flickr-image-dataset/versions//{split}_top10_RN50x64_text.json', 'r'))
        if split == 'val':
            self.data = list(self.data.items()) # dictionary from img_path to list of 10 captions
        self.all_captions = []
        for img_path, captions in self.data:
            self.all_captions.extend(captions)
        self.transform = transform
        self.scoring_only = scoring_only
        self.tokenizer = tokenizer
        self.hard_neg = hard_neg
        self.details = details

        # print(len(list(set([c[1] for c in self.data]))))
        # unique_elements = set(tuple(c[1]) if isinstance(c[1], list) else c[1] for c in self.data)
        # print(len(unique_elements)) # 1014
        # exit(0)


    def __len__(self):
        # print(len(self.data))
        # exit(0)
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        img_path = ex[0]
        text = ex[1]
        if self.tokenizer:
            # print("here") # no
            
            text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            text = text.input_ids.squeeze(0)
            text0 = text[0]
            if self.hard_neg:
                text_rand = text[np.random.randint(5, len(text))]
            else:
                # get text from self.all_captions
                text_rand = self.all_captions[np.random.randint(0, len(self.all_captions))]
                text_rand = self.tokenizer(text_rand, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
            text = torch.stack([text0, text_rand])
        # exit(0)
        # img = Image.open(f'{img_path.replace("datasets",self.root_dir)}').convert("RGB")
        
        # print(img_path)
        # input()
        img = Image.open(f'{img_path.replace("datasets/flickr30k/images",f"{self.root_dir}/flickr-image-dataset/versions/1/flickr30k_images/flickr30k_images")}').convert('RGB')
        if self.transform:
            img_resize = self.transform(img).unsqueeze(0)
        else:
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)

            img = transforms.ToTensor()(img.resize((self.resize, self.resize)))

        return [[img], [img_resize]], text, 0

class Flickr30KNegativesDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False, split='val', tokenizer=None, hard_neg=False):
        self.root_dir = 'data/flickr30k'
        self.resize = resize
        self.data = json.load(open(f'{self.root_dir}/{split}_top10_RN50x64_text.json', 'r'))
        if split == 'val':
            self.data = list(self.data.items()) # dictionary from img_path to list of 10 captions
        self.all_captions = []
        for img_path, captions in self.data:
            self.all_captions.extend(captions)

        self.txt2img = json.load(open(f'{self.root_dir}/{split}_top10_RN50x64.json', 'r'))
        self.transform = transform
        self.scoring_only = scoring_only
        self.tokenizer = tokenizer
        self.hard_neg = hard_neg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        img_path = ex[0]
        strings = ex[1]
        if self.tokenizer:
            text = self.tokenizer(strings, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            text = text.input_ids.squeeze(0)
            text0 = text[0]
            if self.hard_neg:
                rand_idx = np.random.randint(5, len(text))
                text_rand = text[rand_idx]
                string_rand = strings[rand_idx]
                img_rand = self.txt2img[string_rand][0]
            else:
                # get text from self.all_captions
                text_rand = self.all_captions[np.random.randint(0, len(self.all_captions))]
                img_rand = self.txt2img[text_rand][0]
                text_rand = self.tokenizer(text_rand, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
            img_rand = Image.open(f'{img_rand}').convert("RGB")
            img_rand_resize = img_rand.resize((self.resize, self.resize))
            img_rand_resize = diffusers_preprocess(img_rand_resize)
            empty_text = ''
            empty_text = self.tokenizer(empty_text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            empty_text = empty_text.input_ids.squeeze(0)
            text = torch.stack([text0, text_rand, empty_text])
        img = Image.open(f'{img_path}').convert("RGB")
        if self.transform:
            img_resize = self.transform(img).unsqueeze(0)
        else:
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)

            return [0, [img_resize, img_rand_resize]], text, 0
 

class LoRaFlickr30KDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, tokenizer=None, max_train_samples=None):
        self.root_dir = root_dir
        self.resize = resize
        self.max_train_samples = max_train_samples
        self.data = json.load(open(f'{root_dir}/train_top10_RN50x64.json', 'r'))
        self.data = list(self.data.items())
        if self.max_train_samples is not None:
            self.data = self.data[:self.max_train_samples]
        self.transform = transform
        self.tokenizer = tokenizer
        self.two_imgs = True
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        text = ex[0]
        img_paths = ex[1]
        img_idx = 0
        if self.two_imgs:
            text = self.tokenizer([text], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            text = text.input_ids.squeeze(0)
            img0 = Image.open(img_paths[0]).convert("RGB")
            img_rand = Image.open(random.choice(img_paths[1:])).convert("RGB")
            imgs = [img0, img_rand]
        else:
            imgs = [Image.open(img_path).convert("RGB") for img_path in img_paths]
            text = [text]
        #convert pillow to numpy array
        # imgs_resize = [np.array(img) for img in imgs]
        imgs_resize = [img.resize((self.resize, self.resize)) for img in imgs]
        imgs_resize = [diffusers_preprocess(img) for img in imgs_resize]

        return imgs_resize, text, img_idx

class SVOClassificationDataset(Dataset):

    def __init__(self, root_dir, transform, resize=512, scoring_only=False, neg_type='verb'):
        self.transform = transform
        self.root_dir = f'{root_dir}/svo'
        self.data = self.load_data(self.root_dir, neg_type=neg_type)
        self.resize = resize
        self.scoring_only = scoring_only

    def load_data(self, data_dir, neg_type='verb'):
        dataset = []
        split_file = os.path.join(data_dir, 'svo.json')
        with open(split_file) as f:
            json_file = json.load(f)

        for i, row in enumerate(json_file):
            if row['neg_type'] != neg_type:
                continue
            pos_id = str(row['pos_id'])
            neg_id = str(row['neg_id'])
            sentence = row['sentence']
            # get two different images
            pos_file = os.path.join(data_dir, "images", pos_id)
            neg_file = os.path.join(data_dir, "images", neg_id)
            dataset.append((pos_file, neg_file, sentence))

        return dataset
    
    def __getitem__(self, idx):
        file0, file1, text = self.data[idx]
        img0 = Image.open(file0).convert("RGB")
        img1 = Image.open(file1).convert("RGB")
        if not self.scoring_only:
            imgs = [img0, img1]
            if self.transform:
                imgs_resize = [self.transform(img).unsqueeze(0) for img in imgs]
            else:
                imgs_resize = [img.resize((self.resize, self.resize)) for img in imgs]
                imgs_resize = [diffusers_preprocess(img) for img in imgs_resize]

        if self.scoring_only:
            return [text], 0
        else:
            return (0, imgs_resize), [text], 0
 
        
    def __len__(self):
        return len(self.data)

class CLEVRDataset(Dataset):
    def __init__(self, root_dir, transform, resize=512, scoring_only=False, subset =None):
        # root_dir = '../clevr/validation'
        # root_dir = "data/clevr"
        self.root_dir = os.path.join(root_dir, 'clevr')
        if subset == None:
            subtasks = ['pair_binding_size', 'pair_binding_color', 'recognition_color', 'recognition_shape', 'spatial', 'binding_color_shape', 'binding_shape_color']
            data_ = []
            for subtask in subtasks:
                self.data = json.load(open(f'{self.root_dir}/captions/{subtask}.json', 'r')).items()
                for k, v in self.data:
                    for i in range(len(v)):
                        if subtask == 'spatial':
                            texts = [v[i][1], v[i][0]]
                        else:
                            texts = [v[i][0], v[i][1]]
                        data_.append((k, texts, subtask))
        else:
            self.data = json.load(open(f'{self.root_dir}/captions/{subset}.json', 'r')).items()
            data_ = []
            for k, v in self.data:
                for i in range(len(v)):
                    if subset == 'spatial':
                        texts = [v[i][1],v[i][0]]
                        data_.append((k, texts, subset))
                    else:
                        texts = [v[i][0], v[i][1]]
                        data_.append((k, texts, subset))
        # print(len(data_))
        # exit(0)
        self.data = data_
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        cap0 = ex[1][0]
        cap1 = ex[1][1]
        img_id = ex[0]
        subtask = ex[2]
        img_path0 = f'{self.root_dir}/images/{img_id}'
        if not self.scoring_only:
            img0 = Image.open(img_path0).convert("RGB")
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)

        text = [cap0, cap1]
        if self.scoring_only:
            return text, 0
        else:
            return (0, [img0_resize]), text, subtask, 0


class SugarCrepe(Dataset):
    def __init__(self, transform, root_dir, resize=512, tokenizer=None, subset = 'add_att'):
        self.root_dir = root_dir
        self.resize = resize
        self.subset = subset
        if self.subset == 'att' or self.subset == 'obj':
            
            if ARNAS_USES:
                data1 = json.load(open(f"/mnt/lustre/work/oh/owl661/sd-datasets/data/sugarcrepe/add_{self.subset}.json", 'r'))
                data2 = json.load(open(f"/mnt/lustre/work/oh/owl661/sd-datasets/data/sugarcrepe/replace_{self.subset}.json", 'r'))
                data3 = json.load(open(f"/mnt/lustre/work/oh/owl661/sd-datasets/data/sugarcrepe/swap_{self.subset}.json", 'r'))
                self.data = {**data1, **data2, **data3}
            else:
                data1 = json.load(open(f"/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/sugar-crepe/data/add_{self.subset}.json", 'r'))
                data2 = json.load(open(f"/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/sugar-crepe/data/replace_{self.subset}.json", 'r'))
                data3 = json.load(open(f"/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/sugar-crepe/data/swap_{self.subset}.json", 'r'))
                self.data = {**data1, **data2, **data3}
        elif self.subset == 'rel':
            self.data = json.load(open(f"/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/sugar-crepe/data/replace_{self.subset}.json", 'r'))
        else:
            if ARNAS_USES:
                self.data = json.load(open(f"/mnt/lustre/work/oh/owl661/sd-datasets/data/sugarcrepe/{self.subset}.json", 'r'))
            else:
                self.data = json.load(open(f"/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/sugar-crepe/data/{self.subset}.json", 'r'))
        
        # replace_data = json.load(open(f"./sugar-crepe/replace_data/{subset2}.json", 'r'))
        # if self.subset == 'swap':
        #     swap_data = json.load(open(f"./sugar-crepe/data/{subset3}.json", 'r'))
        #     self.data = {add_data, replace_data, swap_data}
        # elif self.subset != 'add':
        # add_data = json.load(open(f"./sugar-crepe/data/add_{self.subset}.json", 'r'))
        # else:
        #     self.data = {add_data, replace_data}
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not (self.subset == 'swap_obj' and idx ==108):
            row = self.data[str(idx)]

            img_path = row['filename']
            # only get filename
            # img_path = img_path.split('/')[-1]
            img_path = f"{self.root_dir}/{img_path}"

            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img_resize = self.transform(img).unsqueeze(0)
            else:
                img_resize = img.resize((self.resize, self.resize))
                img_resize = diffusers_preprocess(img_resize)
            imgs = [img_resize]

            return [img_path, imgs], [row['caption'],row['negative_caption']], 0
        else:
            return self.__getitem__(idx+1)

class COCOref(Dataset):
    def __init__(self, root_dir, transform, resize=512, split='val', tokenizer=None, hard_neg=True, neg_img=False, mixed_neg=False, tsv_path='aro/temp_data/train_neg_clip.tsv'):
        self.root_dir = 'data/mscoco/train2014'
        self.resize = resize
        self.data = pd.read_csv(tsv_path, delimiter='\t')
        self.all_texts = self.data['title'].tolist()
        self.transform = transform
        self.split = split
        self.tokenizer = tokenizer
        self.hard_neg = hard_neg
        self.neg_img = neg_img
        self.mixed_neg = mixed_neg
        self.rand_neg = not self.hard_neg and not self.neg_img


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['filepath']
        # only get filename
        img_path = img_path.split('/')[-1]
        if 'train2014' in img_path:
            img_path = f"{self.root_dir}/{img_path}"
        else:
            img_path = f"data/coco_order/val2014/{img_path}"
        text = row['title']
        neg_captions =  ast.literal_eval(row['neg_caption'])
        neg_caption = neg_captions[np.random.randint(0, len(neg_captions))]

        neg_img_ids = ast.literal_eval(row['neg_image']) # a list of row indices in self.data
        neg_paths = self.data.iloc[neg_img_ids]['filepath'].tolist()
        new_neg_paths = []
        for path in neg_paths:
            path = path.split('/')[-1]
            if 'train2014' in path:
                path = f"{self.root_dir}/{path}"
            else:
                path = f"data/coco_order/val2014/{path}"
            new_neg_paths.append(path)
        neg_paths = new_neg_paths
        
        
        if self.tokenizer:
            text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            text0 = text.input_ids.squeeze(0)
            # text0 = text[0]
            if self.mixed_neg:
                text_neg = self.tokenizer(neg_caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_neg = text_neg.input_ids.squeeze(0)
                text_rand = self.tokenizer(self.all_texts[np.random.randint(0, len(self.all_texts))], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
                text = torch.stack([text0, text_neg, text_rand])
            elif self.hard_neg:
                text_rand = self.tokenizer(neg_caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
                text = torch.stack([text0, text_rand])
            elif self.rand_neg:
                text_rand = self.tokenizer(self.all_texts[np.random.randint(0, len(self.all_texts))], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
                text = torch.stack([text0, text_rand])
            else:
                text = text0
        
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img_resize = self.transform(img).unsqueeze(0)
        else:
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)
        imgs = [img_resize]

        if self.neg_img or self.mixed_neg:
            assert not self.hard_neg
            rand_path = neg_paths[np.random.randint(0, len(neg_paths))]
            rand_img = Image.open(rand_path).convert("RGB")
            if self.transform:
                rand_img = self.transform(rand_img).unsqueeze(0)
            else:
                rand_img = rand_img.resize((self.resize, self.resize))
                rand_img = diffusers_preprocess(rand_img)
            imgs.append(rand_img)

        # if np.random.rand() > 0.99:
        #     print("Img true:", img_path)
        #     print("Neg Img:", rand_path)
        #     print(text)
        
        return [0, imgs], text, 0

class cola_single_clevr(Dataset):
    def __init__(self, transform, root_dir, text_perturb_fn=None, image_perturb_fn=None, download=False, subset='CLEVR', resize=512):
        self.root_dir = root_dir
        annotation_file = f"/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/COLA/data/COLA_singleobjects_benchmark_CLEVR.json"

        if not os.path.exists(root_dir):
            print("Image directory for Controlled Images B could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")
        with open(annotation_file, 'r') as f:
            self.dataset = json.load(f)
        self.all_prepositions = []

        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # print(type(self.dataset)) 
        # exit(0)
        test_case = self.dataset["data"][index]
        print(test_case)
        exit(0)
        # label = self.dataset["labels"][index]
        split = test_case[0].split('/')[-2]
        image_path = os.path.join(self.root_dir,split, test_case[0].split('/')[-1])
        # print(image_path)
        # print(os.path.exists(image_path))
        # print(test_case)
        # exit(0)

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.image_preprocess(image)
        else:
            img0_resize = image.resize((self.resize, self.resize))
            img0_resize = diffusers_preprocess(img0_resize)
        
        # item = edict({"image_options": [image], "caption_options": test_case['caption_options']})
        return [image_path, [img0_resize]], test_case["caption_options"], 0

    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "controlled_images.tar.gz")
        subprocess.call(["gdown", "--no-cookies",  "19KGYVQjrV3syb00GgcavB2nZTW5NXX0H", "--output", image_zip_file])
        subprocess.call(["tar", "-xvf", "controlled_images.tar.gz"], cwd=self.root_dir)
        image_zip_file = os.path.join(self.root_dir, "controlled_clevr.tar.gz")
        subprocess.call(["gdown", "--no-cookies",  "13jdBpg8t3NqW3jrL6FK8HO93vwsUjDxG", "--output", image_zip_file])
        subprocess.call(["tar", "-xvf", "controlled_clevr.tar.gz"], cwd=self.root_dir)



    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 4, i.e. first caption is right, next three captions are wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        metrics["Accuracy"] = np.mean(correct_mask)
        print("Individual accuracy: {}".format(metrics['Accuracy']*100))

        prepositions = ['on', 'under', 'front', 'behind', 'left', 'right']
        prep_counts = {p: {p1: 0 for p1 in prepositions} for p in prepositions}
        for i, d in enumerate(self.dataset):
            prep = list(set(prepositions).intersection(set(d['caption_options'][preds[i]].split())))
            gold_prep = list(set(prepositions).intersection(set(d['caption_options'][0].split())))
            #if len(prep) != 1 or len(gold_prep)!=1:
            #    pdb.set_trace()
            #    print("?")
            prep = prep[0]
            gold_prep = gold_prep[0]
            prep_counts[gold_prep][prep] += 1

            self.pred_dict[(d['image_path'].split('/')[-1].split('_')[0], \
                            d['image_path'].split('/')[-1].split('_')[-1][:-5])][d['image_path'].split('/')[-1].split('_')[1]] = prep
        #print(prep_counts)
        for d, correct in zip(self.dataset, correct_mask):
            self.eval_dict[(d['image_path'].split('/')[-1].split('_')[0], \
                            d['image_path'].split('/')[-1].split('_')[-1][:-5])][d['image_path'].split('/')[-1].split('_')[1]] = correct

        
        pair_correct = 0
        set_correct = 0
        for obj_pair, correct_dict in self.eval_dict.items():
            if correct_dict['left'] and correct_dict['right']:
                pair_correct += 1
            if self.subset == 'A':
                if correct_dict['on'] and correct_dict['under']:
                    pair_correct += 1
            else:
                if correct_dict['in-front'] and correct_dict['behind']:
                    pair_correct += 1
            if sum(correct_dict.values()) == 4:
                set_correct += 1
        pair_accuracy = pair_correct*100/(len(self.dataset)/2)
        set_accuracy = set_correct*100/(len(self.dataset)/4)
        print("Pair accuracy: {}".format(pair_accuracy))
        print("Set accuracy: {}".format(set_accuracy))
        all_prepositions = np.array(self.all_prepositions)

        result_records = []
        # Log the accuracy of all prepositions
        for prepositions in np.unique(all_prepositions):
            prepositions_mask = (all_prepositions == prepositions)
            if prepositions_mask.sum() == 0:
                continue
            result_records.append({
                "Preposition": prepositions,
                "Accuracy": correct_mask[prepositions_mask].mean(),
                "Count": prepositions_mask.sum(),
                "Dataset": "Controlled Images - {}".format(self.subset)
            })
        return result_records


class COCOref(Dataset):
    def __init__(self, root_dir, transform, resize=512, split='val', tokenizer=None, hard_neg=True, neg_img=False, mixed_neg=False, tsv_path='aro/temp_data/train_neg_clip.tsv'):
        self.root_dir = 'data/mscoco/train2014'
        self.resize = resize
        self.data = pd.read_csv(tsv_path, delimiter='\t')
        self.all_texts = self.data['title'].tolist()
        self.transform = transform
        self.split = split
        self.tokenizer = tokenizer
        self.hard_neg = hard_neg
        self.neg_img = neg_img
        self.mixed_neg = mixed_neg
        self.rand_neg = not self.hard_neg and not self.neg_img


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['filepath']
        # only get filename
        img_path = img_path.split('/')[-1]
        if 'train2014' in img_path:
            img_path = f"{self.root_dir}/{img_path}"
        else:
            img_path = f"data/coco_order/val2014/{img_path}"
        text = row['title']
        neg_captions =  ast.literal_eval(row['neg_caption'])
        neg_caption = neg_captions[np.random.randint(0, len(neg_captions))]

        neg_img_ids = ast.literal_eval(row['neg_image']) # a list of row indices in self.data
        neg_paths = self.data.iloc[neg_img_ids]['filepath'].tolist()
        new_neg_paths = []
        for path in neg_paths:
            path = path.split('/')[-1]
            if 'train2014' in path:
                path = f"{self.root_dir}/{path}"
            else:
                path = f"data/coco_order/val2014/{path}"
            new_neg_paths.append(path)
        neg_paths = new_neg_paths
        
        
        if self.tokenizer:
            text = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            text0 = text.input_ids.squeeze(0)
            # text0 = text[0]
            if self.mixed_neg:
                text_neg = self.tokenizer(neg_caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_neg = text_neg.input_ids.squeeze(0)
                text_rand = self.tokenizer(self.all_texts[np.random.randint(0, len(self.all_texts))], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
                text = torch.stack([text0, text_neg, text_rand])
            elif self.hard_neg:
                text_rand = self.tokenizer(neg_caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
                text = torch.stack([text0, text_rand])
            elif self.rand_neg:
                text_rand = self.tokenizer(self.all_texts[np.random.randint(0, len(self.all_texts))], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
                text_rand = text_rand.input_ids.squeeze(0)
                text = torch.stack([text0, text_rand])
            else:
                text = text0
        
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img_resize = self.transform(img).unsqueeze(0)
        else:
            img_resize = img.resize((self.resize, self.resize))
            img_resize = diffusers_preprocess(img_resize)
        imgs = [img_resize]

        if self.neg_img or self.mixed_neg:
            assert not self.hard_neg
            rand_path = neg_paths[np.random.randint(0, len(neg_paths))]
            rand_img = Image.open(rand_path).convert("RGB")
            if self.transform:
                rand_img = self.transform(rand_img).unsqueeze(0)
            else:
                rand_img = rand_img.resize((self.resize, self.resize))
                rand_img = diffusers_preprocess(rand_img)
            imgs.append(rand_img)

        # if np.random.rand() > 0.99:
        #     print("Img true:", img_path)
        #     print("Neg Img:", rand_path)
        #     print(text)
        
        return [0, imgs], text, 0

class Cola_Single_Clevr(Dataset):
    def __init__(self, image_preprocess, root_dir, text_perturb_fn=None, image_perturb_fn=None, download=False, subset='CLEVR', resize=512):
        self.root_dir = root_dir
        annotation_file = f"./COLA/data/COLA_singleobjects_benchmark_CLEVR.json"

        if not os.path.exists(root_dir):
            print("Image directory for Controlled Images B could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")
        with open(annotation_file, 'r') as f:
            self.dataset = json.load(f)
        self.all_prepositions = []

        self.image_preprocess = image_preprocess
        self.resize = resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # print(type(self.dataset)) 
        # exit(0)
        test_case = self.dataset["data"][index]
        # test_label = [' '.join(i[0]) + ' ' + i[1] for i in test_case[1]]
        
        # gt_label = self.dataset["labels"][index]
        # gt_label = ' '.join(gt_label[0]) + ' ' + gt_label[1]
        # print(test_label)
        # print(gt_label)
        # print(test_label.index(gt_label))
        print(test_case)
        print(gt_label)
        exit(0)
        split = test_case[0].split('/')[-2]

        image_path = os.path.join(self.root_dir,split, test_case[0].split('/')[-1])

        image = Image.open(image_path).convert('RGB')
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        else:
            img0_resize = image.resize((self.resize, self.resize))
            img0_resize = diffusers_preprocess(img0_resize)
        
        # item = edict({"image_options": [image], "caption_options": test_case['caption_options']})
        return [image_path, [img0_resize]], test_label, test_label.index(gt_label)

    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "controlled_images.tar.gz")
        subprocess.call(["gdown", "--no-cookies",  "19KGYVQjrV3syb00GgcavB2nZTW5NXX0H", "--output", image_zip_file])
        subprocess.call(["tar", "-xvf", "controlled_images.tar.gz"], cwd=self.root_dir)
        image_zip_file = os.path.join(self.root_dir, "controlled_clevr.tar.gz")
        subprocess.call(["gdown", "--no-cookies",  "13jdBpg8t3NqW3jrL6FK8HO93vwsUjDxG", "--output", image_zip_file])
        subprocess.call(["tar", "-xvf", "controlled_clevr.tar.gz"], cwd=self.root_dir)


class Cola_Multi(Dataset):
    def __init__(self, transform, root_dir, resize=512, scoring_only=False):
        self.root_dir = root_dir
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.dataset = json.load(open(f"/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/COLA/data/COLA_multiobjects_matching_benchmark.json", 'r'))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        ex = self.dataset[idx]
        cap0 = ex[1]
        cap1 = ex[3]
        img0_path =  os.path.join(self.root_dir,ex[0].split('/')[-1])
        img1_path = os.path.join(self.root_dir,ex[2].split('/')[-1])
        if not self.scoring_only:
            img0 = Image.open(img0_path).convert('RGB')
            img1 = Image.open(img1_path).convert('RGB')
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
                img1_resize = self.transform(img1).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img1_resize = img1.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
                img1_resize = diffusers_preprocess(img1_resize)
        text = [cap0, cap1]
        if self.scoring_only:
            return text, idx
        else:
            return ((img0_path, img1_path), [img0_resize, img1_resize]), text, idx

class VisMin(Dataset):
    def __init__(self, transform, root_dir, resize=512, scoring_only=False,subset=None,split='test'):
        self.root_dir = root_dir
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        self.split = split
        if self.split == 'train':
            self.dataset =  json.load(open(f"./vismin/vismin.json", 'r'))
            if self.subset is not None:
                # Filter the dataset to include only entries where the 'category' matches the subset
                self.dataset = [item for item in self.dataset if item['category'] == self.subset]
        else:
            with open("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/hf_token.txt", "r") as f:
                    hf_token = f.read().strip()
            try:
                dataset = load_dataset('mair-lab/vismin-bench', use_auth_token=hf_token)
            except:
                dataset = load_dataset('mair-lab/vismin-bench', token=hf_token)
            # dataset = load_dataset("mair-lab/vismin-bench")

            dataset = dataset['test']
            # self.subset
            if self.subset != None:
                print("subset", self.subset)
                self.dataset = dataset.filter(lambda x: x['category'] == self.subset)
            else:
                print("subset", self.subset)
                self.dataset = dataset
            # if self.subset == "counting":
            #     self.dataset = self.dataset.select(range(len(self.dataset) - 100, len(self.dataset)))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.split == 'train':
            img0, img1 = sample["pos_image_path"], sample["neg_image_path"]
            cap0, cap1 = sample["pos_caption"], sample["neg_caption"]
            img0 = Image.open(img0).convert('RGB')
            img1 = Image.open(img1).convert('RGB')
        else:
            img0, img1 = sample["image_0"], sample["image_1"]
            # if img0 is None:
            #     raise ValueError(f"Missing image at index {idx}: img0 is None")
            # if img1 is None:
            #     raise ValueError(f"Missing image at index {idx}: img1 is None")

            cap0, cap1 = sample["text_0"], sample["text_1"]
            img0 = img0.convert('RGB')
            img1 = img1.convert('RGB')
            # print(f"Before Transform IDX {idx}: img0 shape={img0.size}, img1 shape={img1.size}")


            if not isinstance(img0, Image.Image):
                raise TypeError(f"Expected PIL image but got {type(img0)} at index {idx}")
            if not isinstance(img1, Image.Image):
                raise TypeError(f"Expected PIL image but got {type(img1)} at index {idx}")

        if not self.scoring_only:
            
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
                img1_resize = self.transform(img1).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img1_resize = img1.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
                img1_resize = diffusers_preprocess(img1_resize)
        text = [cap0, cap1]
        if self.scoring_only:
            return text, idx
        else:
            return ((f"{str(idx)}_0", f"{str(idx)}_1"), [img0_resize, img1_resize]), text, idx



class CountBench(Dataset):
    DIGIT_TO_WORD = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }

    def __init__(self, img_root_dir: str, annotation_file: str, model_type: str = "clip", scoring=False):
        super(CountBench, self).__init__()
        self.model_type = model_type
        # self.dataset = self.load_dataset(img_root_dir, annotation_file)
        self.get_item = self.get_item_clip if model_type == "clip" else self.get_item_mllm
        self.examples = load_dataset('nielsr/countbench', use_auth_token="/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/hf_token.txt")
        self.digits = list(DIGIT_TO_WORD.keys())
        self.classes = list(DIGIT_TO_WORD.values())
        # if True:
        #     prompted_classes = []
        #     for c in self.classes:
        #         class_text = 'a photo of a ' + c
        #         prompted_classes.append(class_text)
        #     self.classes = prompted_classes
        self.scoring_only = scoring_only


    # def get_item_clip(self, index):
    #     sample = self.dataset[index]
    #     return {"image": sample["image"], "caption": sample["captions"]}

    # def get_item_mllm(self, index):
    #     sample = self.dataset[index]
    #     # we randomly choose the order of the images to avoid any bias in the model
    #     captions = sample["captions"]
    #     captions = captions if index % 2 == 0 else captions[::-1]
    #     label = "A" if index % 2 == 0 else "B"
    #     prompt = prompt_getter.get_text_task_prompt(captions)
    #     return {
    #         "image": sample["image"],
    #         "text": prompt,
    #         "label": label,
    #     }

    def __getitem__(self, index):
        ex = self.examples[idx]
        cap0 = ex['text']
        text = [
        re.sub(r'\bone\b', word, re.sub(r'\b1\b', digit, sentence)) if re.search(r'\b1\b', sentence) 
        else re.sub(r'\bone\b', word, sentence)
        for word, digit in zip(self.classes, self.digits)
    ]
        print(text)
        exit(0)
        img_id = ex['image_url']
        if not self.scoring_only:
            img0 = ex['image'].convert('RGB')
            # img1 = ex['image_1'].convert('RGB')
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
                # img1_resize = self.transform(img1).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                # img1_resize = img1.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
                # img1_resize = diffusers_preprocess(img1_resize)

        if self.scoring_only:
            return text, img_id
        else:
            return (0, [img0_resize, img1_resize]), text, img_id
        # return self.get_item(index)

    # def __getitem__(self, idx):
    #     if not self.scoring_only:
    #         img, class_id = self.data[idx]
    #         img = img.convert("RGB")
    #         img_resize = img.resize((self.resize, self.resize))
    #         img_resize = diffusers_preprocess(img_resize)
    #         if self.transform:
    #             img = self.transform(img)
    #         else:
    #             img = transforms.ToTensor()(img)
    #     else:
    #         class_id = idx // 50

    #     if self.scoring_only:
    #         return self.classes, class_id
    #     else:
    #         return ([img], [img_resize]), self.classes, class_id


    def __len__(self):
        return len(self.dataset)


class VALSE(Dataset):
    def __init__(self, transform, root_dir, resize=512, scoring_only=False, subset='action-replacement'):
        self.root_dir = root_dir
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        # self.dataset = json.load(open(f"./VALSE/data/{self.subset}.json", 'r'))
        with open(f"./VALSE/data/{self.subset}.json", "r") as f:
            self.dataset = json.load(f)
        self.keys = list(self.dataset.keys())

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        ex = self.dataset[key]
        cap0 = ex['caption']
        cap1 = ex['foil']
        img0 =  os.path.join(self.root_dir,ex['image_file'])
        # img1 = os.path.join(self.root_dir,ex[2].split('/')[-1])
        if not self.scoring_only:
            img0 = Image.open(img0).convert('RGB')
            # img1 = Image.open(img1).convert('RGB')
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
                # img1_resize = self.transform(img1).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                # img1_resize = img1.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
                # img1_resize = diffusers_preprocess(img1_resize)
        text = [cap0, cap1]
        if self.scoring_only:
            return text, idx
        else:
            return (idx, [img0_resize]), text, 0


class SPEC_Image2Text(Dataset):
    def __init__(self, transform, root_dir, resize=512, scoring_only=False, subset='action-replacement'):
        self.root_dir = root_dir
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        self.dataset = json.load(open(f"{self.root_dir}/{self.subset}/image2text.json", 'r'))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        ex = self.dataset[idx]
        img_path0 =  os.path.join(self.root_dir,self.subset,ex['query'])
        # img1 = os.path.join(self.root_dir,ex[2].split('/')[-1])
        if not self.scoring_only:
            img0 = Image.open(img_path0).convert('RGB')
            # img1 = Image.open(img1).convert('RGB')
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
                # img1_resize = self.transform(img1).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                # img1_resize = img1.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
                # img1_resize = diffusers_preprocess(img1_resize)
        text = ex["keys"]
        # print("length", len(text)) # count: 9
        # exit(0)
        # print("length", len(text)) # count: 9
        # exit(0)
        if self.scoring_only:
            return text, idx
        else:
            return (0, [img0_resize]), text, ex["label"]

class SPEC_Text2Image(Dataset):
    def __init__(self, transform, root_dir, resize=512, scoring_only=False, subset='action-replacement'):
        self.root_dir = root_dir
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        self.dataset = json.load(open(f"{self.root_dir}/{self.subset}/text2image.json", 'r'))

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        ex = self.dataset[idx]
        text = ex['query']
        img0 =  os.path.join(self.root_dir,self.subset,ex['keys'][0])
        img1 = os.path.join(self.root_dir,self.subset,ex['keys'][1])
        if self.subset != 'existence':
            img2 = os.path.join(self.root_dir,self.subset,ex['keys'][2])
        if not self.scoring_only:
            img0 = Image.open(img0).convert('RGB')
            img1 = Image.open(img1).convert('RGB')
            if self.subset != 'existence':
                img2 = Image.open(img2).convert('RGB')
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
                img1_resize = self.transform(img1).unsqueeze(0)
                if self.subset != 'existence':
                    img2_resize = self.transform(img2).unsqueeze(0)

            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img1_resize = img1.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
                img1_resize = diffusers_preprocess(img1_resize)
                if self.subset != 'existence':
                    img2_resize = img2.resize((self.resize, self.resize))
                    img2_resize = diffusers_preprocess(img2_resize)
    
        if self.scoring_only:
            return text, idx
        else:
            if self.subset == 'existence':
                return (0, [img0_resize,img1_resize]), [text], ex["label"]
            return (0, [img0_resize,img1_resize,img2_resize]), [text], ex["label"]

class VLCheck_Attribute(Dataset):
    def __init__(self, transform, root_dir, resize=512, scoring_only=False, subset='action-replacement'):
        self.root_dir = root_dir
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        self.dataset = json.load(open(f"./VL-CheckList/data/Attribute/vaw/{self.subset}.json", 'r'))
        self.dataset1 = json.load(open(f"./VL-CheckList/data/Attribute/vg/{self.subset}.json", 'r'))
        self.dataset = self.dataset + self.dataset1

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        ex = self.dataset[idx]
        img0 =  os.path.join(self.root_dir,ex[0].split('/')[-1])
        # img1 = os.path.join(self.root_dir,ex[2].split('/')[-1])
        if not self.scoring_only:
            img0 = Image.open(img0).convert('RGB')
            # img1 = Image.open(img1).convert('RGB')
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
                # img1_resize = self.transform(img1).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                # img1_resize = img1.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
                # img1_resize = diffusers_preprocess(img1_resize)
        cap0 = ex[1]["POS"][0]
        cap1 = ex[1]["NEG"][0]
        text = [cap0, cap1]
        if self.scoring_only:
            return text, idx
        else:
            return (0, [img0_resize]), text, 0  

class VLCheck_Object_Location(Dataset):
    def __init__(self, transform, root_dir, resize=512, scoring_only=False, subset='hake'):
        self.root_dir = root_dir
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        if self.subset == 'hake':
            self.dataset = [] 
            for i in os.listdir(f"./VL-CheckList/data/Object/Location/{self.subset}_location"):
                dataset = [
                item for item in json.load(open(f"./VL-CheckList/data/Object/Location/{self.subset}_location/{i}", 'r'))
                if 'pic/' not in item[0]
            ] 
                # dataset = json.load(open(f"./VL-CheckList/data/Object/Location/{self.subset}_location/{i}", 'r'))
                self.dataset += dataset
        elif self.subset == 'swig':
            self.dataset = []
            for i in os.listdir(f"./VL-CheckList/data/Object/Location/{self.subset}_location"):
                for j in os.listdir(f"./VL-CheckList/data/Object/Location/{self.subset}_location/{i}"):
                    dataset = json.load(open(f"./VL-CheckList/data/Object/Location/{self.subset}_location/{i}/{j}", 'r'))
                    self.dataset += dataset
        elif self.subset == 'vg':
            self.dataset = []
            for i in os.listdir(f"./VL-CheckList/data/Object/Location/{self.subset}_location"):
                dataset = json.load(open(f"./VL-CheckList/data/Object/Location/{self.subset}_location/{i}", 'r'))
                self.dataset += dataset
        else:
            RuntimeError("Invalid subset")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            ex = self.dataset[idx]
            if self.subset == 'hake':
                if 'coco' not in ex[0]:
                    img0 =  os.path.join(self.root_dir,ex[0])
                else:
                    img0 = os.path.join(self.root_dir,ex[0])
                    img0 = img0.replace('HAKE/vcoco','COCO2014')
            else:
                img0 =  os.path.join(self.root_dir,ex[0].split('/')[-1])
            # img1 = os.path.join(self.root_dir,ex[2].split('/')[-1])
            if not self.scoring_only:
                img0 = Image.open(img0).convert('RGB')
                # img1 = Image.open(img1).convert('RGB')
                if self.transform:
                    img0_resize = self.transform(img0).unsqueeze(0)
                    # img1_resize = self.transform(img1).unsqueeze(0)
                else:
                    img0_resize = img0.resize((self.resize, self.resize))
                    # img1_resize = img1.resize((self.resize, self.resize))
                    img0_resize = diffusers_preprocess(img0_resize)
                    # img1_resize = diffusers_preprocess(img1_resize)
            cap0 = ex[1]["POS"][0]
            cap1 = ex[1]["NEG"][0]
            text = [cap0, cap1]
            if self.scoring_only:
                return text, idx
            else:
                return (0, [img0_resize]), text, 0  
        except FileNotFoundError:
            print(f"file name: {os.path.join(self.root_dir,ex[0])}")
            # return self.__getitem__(idx+1)
            if idx+1 == len(self.dataset):
                return self.__getitem__(0)
            return self.__getitem__(idx+1)
        

class VLCheck_Object_Size(Dataset):
    def __init__(self, transform, root_dir, resize=512, scoring_only=False, subset='hake'):
        self.root_dir = root_dir
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        if self.subset == 'hake':
            self.dataset = [] 
            for i in os.listdir(f"./VL-CheckList/data/Object/Size/{self.subset}_size"):
                # dataset = json.load(open(f"./VL-CheckList/data/Object/Size/{self.subset}_size/{i}", 'r'))
                dataset = [
                item for item in json.load(open(f"./VL-CheckList/data/Object/Size/{self.subset}_size/{i}", 'r'))
                if 'pic/' not in item[0]
            ] 
                self.dataset += dataset
        elif self.subset == 'swig':
            self.dataset = []
            for i in os.listdir(f"./VL-CheckList/data/Object/Size/{self.subset}_size"):
                for j in os.listdir(f"./VL-CheckList/data/Object/Size/{self.subset}_size/{i}"):
                    dataset = json.load(open(f"./VL-CheckList/data/Object/Size/{self.subset}_size/{i}/{j}", 'r'))
                    self.dataset += dataset
        elif self.subset == 'vg':
            self.dataset = []
            for i in os.listdir(f"./VL-CheckList/data/Object/Size/{self.subset}_size"):
                dataset = json.load(open(f"./VL-CheckList/data/Object/Size/{self.subset}_size/{i}", 'r'))
                self.dataset += dataset
        else:
            RuntimeError("Invalid subset")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            ex = self.dataset[idx]
            if self.subset == 'hake':
                if 'vcoco' not in ex[0]:
                    img0 =  os.path.join(self.root_dir,ex[0])
                else:
                    img0 = os.path.join(self.root_dir,ex[0])
                    img0 = img0.replace('HAKE/vcoco','COCO2014')
            else:
                img0 =  os.path.join(self.root_dir,ex[0].split('/')[-1])
            # print(ex[0])
            # exit(0)
            # img1 = os.path.join(self.root_dir,ex[2].split('/')[-1])
            if not self.scoring_only:
                img0 = Image.open(img0).convert('RGB')
                # img1 = Image.open(img1).convert('RGB')
                if self.transform:
                    img0_resize = self.transform(img0).unsqueeze(0)
                    # img1_resize = self.transform(img1).unsqueeze(0)
                else:
                    img0_resize = img0.resize((self.resize, self.resize))
                    # img1_resize = img1.resize((self.resize, self.resize))
                    img0_resize = diffusers_preprocess(img0_resize)
                    # img1_resize = diffusers_preprocess(img1_resize)
            cap0 = ex[1]["POS"][0]
            cap1 = ex[1]["NEG"][0]
            text = [cap0, cap1]
            if self.scoring_only:
                return text, idx
            else:
                return (0, [img0_resize]), text, 0  
        except FileNotFoundError:
            print(f"file name: {os.path.join(self.root_dir,ex[0])}")
            if idx+1 == len(self.dataset):
                return self.__getitem__(0)
            return self.__getitem__(idx+1)

class VLCheck_Relation(Dataset):
    def __init__(self, transform, root_dir, resize=512, scoring_only=False, subset='hake'):
        self.root_dir = root_dir
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        if self.subset == 'hake':
            self.dataset = [
                item for item in json.load(open(f"./VL-CheckList/data/Relation/{self.subset}_action.json", 'r'))
                if 'pic/' not in item[0]
            ] 
        
        elif self.subset == 'swig':
            self.dataset = json.load(open(f"./VL-CheckList/data/Relation/{self.subset}_action.json", 'r'))
            # I want self.dataset filtering by '/pic/' not in self.dataset[idx][0]

        elif 'vg' in self.subset:
            subset = self.subset.replace('vg_', '')
            self.dataset = json.load(open(f"./VL-CheckList/data/Relation/vg/{subset}.json", 'r'))
        else:
            RuntimeError("Invalid subset")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            ex = self.dataset[idx]
            if self.subset == 'hake':
                if 'coco' not in ex[0]:
                    img0 =  os.path.join(self.root_dir,ex[0])
                else:
                    img0 = os.path.join(self.root_dir,ex[0])
                    img0 = img0.replace('HAKE/vcoco','COCO2014')
            else:
                img0 =  os.path.join(self.root_dir,ex[0].split('/')[-1])
                # img1 = os.path.join(self.root_dir,ex[2].split('/')[-1])
            if not self.scoring_only:
                img0 = Image.open(img0).convert('RGB')
                # img1 = Image.open(img1).convert('RGB')
                if self.transform:
                    img0_resize = self.transform(img0).unsqueeze(0)
                    # img1_resize = self.transform(img1).unsqueeze(0)
                else:
                    img0_resize = img0.resize((self.resize, self.resize))
                    # img1_resize = img1.resize((self.resize, self.resize))
                    img0_resize = diffusers_preprocess(img0_resize)
                    # img1_resize = diffusers_preprocess(img1_resize)
            cap0 = ex[1]["POS"][0]
            cap1 = ex[1]["NEG"][0]
            text = [cap0, cap1]
            if self.scoring_only:
                return text, idx
            else:
                return (0, [img0_resize]), text, 0  
    
        except FileNotFoundError:
            print(f"file name: {os.path.join(self.root_dir,ex[0])}")
            if idx+1 == len(self.dataset):
                return self.__getitem__(0)
            return self.__getitem__(idx+1)

class EQBench(Dataset):
    def __init__(self, transform, root_dir, resize=512, scoring_only=False, subset='eqbenag'):
        self.root_dir = f"{root_dir}/image_jpg"
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset

        with open(f"{root_dir}/ann_json_finegrained_random.json", 'r') as f:
            data = [item for item in json.load(f) if (f'{self.subset}/' in item['image']) and ('train' not in item['image']) and (item['private_info']['name']=='c0i0' or item['private_info']['name']=='c1i1')]

    # Group entries by the common base path
        grouped_data = defaultdict(list)
        for entry in data:
            # Split the path to get the common base (up to the second last element for grouping)
            base_path = "/".join(entry["image"].split("/")[:-1])
            grouped_data[base_path].append(entry)

        # Assemble the new structure
        new_data = []
        
        for base_path, items in grouped_data.items():

            # Create a dictionary for each group
            combined_entry = {}
            for idx, item in enumerate(items[:2], start=1):
                # Add each image and caption with unique keys (image1, image2, caption1, caption2, etc.)
                combined_entry[f"image{idx}"] = item["image"]
                combined_entry[f"caption{idx}"] = item["caption"]
            new_data.append(combined_entry)

        self.dataset = new_data


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        ex = self.dataset[idx]

        img0 =  os.path.join(self.root_dir,ex['image1'])
        
        img1 = os.path.join(self.root_dir,ex['image2'])
        
        if not self.scoring_only:
            if '.npy' in img0:
                img0 = Image.fromarray(np.load(img0)[:, :, [2, 1, 0]], 'RGB')
                img1 = Image.fromarray(np.load(img1)[:, :, [2, 1, 0]], 'RGB')
            else:
                img0 = img0.replace('.png','.jpg')
                img1 = img1.replace('.png','.jpg')

                img0 = Image.open(img0).convert('RGB')
                img1 = Image.open(img1).convert('RGB')
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
                img1_resize = self.transform(img1).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img1_resize = img1.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
                img1_resize = diffusers_preprocess(img1_resize)
        cap0 = ex["caption1"]
        cap1 = ex["caption2"]
        text = [cap0, cap1]
        if self.scoring_only:
            return text, idx
        else:
            return (0, [img0_resize,img1_resize]), text, idx


class EQBench_subset(Dataset):
    def __init__(self, transform, root_dir, resize=512, scoring_only=False, subset='eqbenag'):
        self.root_dir = f"{root_dir}/eqbench_subset/images"
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        dataset = json.load(open(f"{root_dir}/eqbench_subset/all_select.json", 'r'))
        self.dataset = [i for i in dataset if self.subset in i["image0"]]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        ex = self.dataset[idx]

        img0 =  os.path.join(self.root_dir,ex['image0'])
        img1 = os.path.join(self.root_dir,ex['image1'])
        
        if not self.scoring_only:
            if '.npy' in img0:
                img0 = Image.fromarray(np.load(img0)[:, :, [2, 1, 0]], 'RGB')
                img1 = Image.fromarray(np.load(img1)[:, :, [2, 1, 0]], 'RGB')
            else:
                img0 = Image.open(img0).convert('RGB')
                img1 = Image.open(img1).convert('RGB')
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
                img1_resize = self.transform(img1).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img1_resize = img1.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
                img1_resize = diffusers_preprocess(img1_resize)
        cap0 = ex["caption0"]
        cap1 = ex["caption1"]
        text = [cap0, cap1]
        if self.scoring_only:
            return text, idx
        else:
            return ((ex['image0'],ex['image1']), [img0_resize,img1_resize]), text, idx

class NaturalBench(Dataset):
    def __init__(self, transform, subset, resize=512, scoring_only=False):
        self.examples = load_dataset('BaiqiL/NaturalBench', use_auth_token="/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/hf_token.txt")
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        self.examples == self.examples

    def __len__(self):
        # return len(self.examples['train'])
        return 1
    
    def __getitem__(self, idx):
        ex = self.examples['train'][self.subset]
        img0 = ex['Image_1']
        # img1 = ex['Image_1']
        print(ex['Question_0']) 
        cap0 = 'the musician is wearing a scarf'
        cap1 = 'the musician is not wearing a scarf'

        if not self.scoring_only:
            img0 = img0.convert('RGB')
            # img1 = ex['Image_1'].convert('RGB')
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
                # img1_resize = self.transform(img1).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                # img1_resize = img1.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
                # img1_resize = diffusers_preprocess(img1_resize)
        text = [cap0,cap1]
        # text = [cap0]
        if self.scoring_only:
            return text, idx
        else:
            return (0, [img0_resize]), text, 0


class Geneval_final(Dataset):
    def __init__(self, transform, root_dir, subset, version, resize=512, scoring_only=False, cfg = 9.0, filter = False):
        self.version = version

        # root_dir should be the path to the root directory of the dataset (including cfg/version/images)

        try:
            self.root_dir = f'{root_dir}/{cfg}/{self.version.split("/")[1]}'
        except:
            if version == "1.5": self.version = "stable-diffusion-v1-5"
            elif version == "2.0": self.version = "stable-diffusion-2-base"
            elif version == "3-m": self.version = "stable-diffusion-3-medium-diffusers"
            elif version == "flux": self.version = "FLUX.1-dev"
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
            self.text = json.load(open(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/two_object_subset.json', 'r'))

        else:
            prompt = f'{root_dir}/../prompts/zero_shot_prompts.json' # all possible prompts list
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
            filter_prompt = f'{root_dir}/../filter/SD-{version}-CFG={str(int(self.cfg))}.json' # please change the path
            filter_prompt = json.load(open(filter_prompt, 'r'))
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
        metadata = os.path.join(self.examples[idx].split('samples')[0],'metadata.jsonl')
        metadata = json.load(open(metadata, 'r'))
        
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


class Geneval(Dataset):
    def __init__(self, transform, root_dir, subset, version, resize=512, scoring_only=False, domain = 'photo', cfg = 9.0):
        self.version = version
        try:
            self.root_dir = f'{root_dir}/{cfg}/{self.version.split("/")[1]}'
        except:
            if version == "1.5": self.version = "stable-diffusion-v1-5"
            elif version == "2.0": self.version = "stable-diffusion-2-base"
            elif version == "3-m": self.version = "stable-diffusion-3-medium-diffusers"
            else: raise ValueError('Invalid version')
            self.root_dir = f'{root_dir}/{cfg}/{self.version}'
        
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset= subset
        self.domain = domain
        self.examples = []
        if self.subset == "two_object_subset":
            subset = "two_object"
        for i in os.listdir(self.root_dir):
            metadata = os.path.join(self.root_dir, i, 'metadata.jsonl')
            metadata = json.load(open(metadata, 'r'))
            if metadata['tag'] == subset:
                for j in range(4):
                    self.examples.append(os.path.join(self.root_dir, i, 'samples', f'0000{j}.png'))
        
        if self.subset == "two_object_subset":
            self.prompts = json.load(open(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/two_object_subset.json', 'r'))
        else:
            prompt = f'{root_dir}/../prompts/zero_shot_prompts.json'
            self.prompts = json.load(open(prompt, 'r'))[domain][self.subset]
        
    def __len__(self):
        return len(self.examples)

    def with_article(self, name: str):
        if name[0] in "aeiou":
            return f"an {name}"
        return f"a {name}"

    def __getitem__(self, idx):
        metadata = os.path.join(self.examples[idx].split('samples')[0],'metadata.jsonl')
        metadata = json.load(open(metadata, 'r'))
        
        img0 = Image.open(self.examples[idx]).convert('RGB')
        if not self.scoring_only:
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
        if self.subset == 'two_object_subset':
            first = metadata["include"][0]["class"]
            second = metadata["include"][1]["class"]
            text = self.prompts[f'{first}_{second}']
        elif self.subset == 'color_attr':
            text = self.prompts[metadata["include"][0]["class"]][metadata["include"][1]["class"]]
        elif self.subset == 'position':
            text = self.prompts[metadata["include"][1]["class"]][metadata["include"][0]["class"]]
        elif self.subset in ['single_object','two_object']:
            text = self.prompts
        else:
            text = self.prompts[metadata["include"][0]["class"]]

        
        if self.scoring_only:
            return text, idx
        else:
            if self.domain == 'photo':
                return (self.examples[idx], [img0_resize]), text, text.index(metadata['prompt'])
            else:
                return (self.examples[idx], [img0_resize]), text, text.index(metadata['prompt'].replace('a photo',self.with_article(self.domain)))


class Geneval_1_5(Dataset):
    def __init__(self, transform, root_dir, subset, resize=512, scoring_only=False, domain = 'photo', cfg = 9.0):
        self.root_dir = f'{root_dir}/{cfg}/stable-diffusion-v1-5'
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        self.domain = domain
        self.examples = []
        if self.subset == "two_object_subset":
            subset = "two_object"
        for i in os.listdir(self.root_dir):
            metadata = os.path.join(self.root_dir, i, 'metadata.jsonl')
            metadata = json.load(open(metadata, 'r'))
            if metadata['tag'] == subset:
                for j in range(4):
                    self.examples.append(os.path.join(self.root_dir, i, 'samples', f'0000{j}.png'))
        if self.subset == "two_object_subset":
            self.prompts = json.load(open(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/two_object_subset.json', 'r'))
        else:
            prompt = f'{root_dir}/../prompts/zero_shot_prompts.json'
            self.prompts = json.load(open(prompt, 'r'))[domain][self.subset]
        
    def __len__(self):
        return len(self.examples)

    def with_article(self, name: str):
        if name[0] in "aeiou":
            return f"an {name}"
        return f"a {name}"

    def __getitem__(self, idx):
        metadata = os.path.join(self.examples[idx].split('samples')[0],'metadata.jsonl')
        metadata = json.load(open(metadata, 'r'))
        
        img0 = Image.open(self.examples[idx]).convert('RGB')
        if not self.scoring_only:
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
        if self.subset == 'color_attr':
            text = self.prompts[metadata["include"][0]["class"]][metadata["include"][1]["class"]]
        elif self.subset == 'two_object_subset':
            first = metadata["include"][0]["class"]
            second = metadata["include"][1]["class"]
            text = self.prompts[f'{first}_{second}']
        elif self.subset == 'position':
            text = self.prompts[metadata["include"][1]["class"]][metadata["include"][0]["class"]]
        elif self.subset in ['single_object','two_object']:
            text = self.prompts
        else:
            text = self.prompts[metadata["include"][0]["class"]]


        if self.scoring_only:
            return text, idx
        else:
            if self.domain == 'photo':
                return (self.examples[idx], [img0_resize]), text, text.index(metadata['prompt'])
            else:
                return (self.examples[idx], [img0_resize]), text, text.index(metadata['prompt'].replace('a photo',self.with_article(self.domain)))


class Geneval_2_0(Dataset):
    def __init__(self, transform, root_dir, subset, resize=512, scoring_only=False, domain = 'photo', cfg = 9.0):

        
        self.root_dir = f'{root_dir}/{cfg}/stable-diffusion-2-base'
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        self.domain = domain
        self.examples = []
        if self.subset == "two_object_subset":
            subset = "two_object"
        for i in os.listdir(self.root_dir):
            metadata = os.path.join(self.root_dir, i, 'metadata.jsonl')
            metadata = json.load(open(metadata, 'r'))
            if metadata['tag'] == subset:
                for j in range(4):
                    self.examples.append(os.path.join(self.root_dir, i, 'samples', f'0000{j}.png'))

        if self.subset == "two_object_subset":
            self.prompts = json.load(open(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/two_object_subset.json', 'r'))
        else:
            prompt = f'{root_dir}/../prompts/zero_shot_prompts.json'
            self.prompts = json.load(open(prompt, 'r'))[domain][self.subset]
        
    def __len__(self):
        return len(self.examples)

    def with_article(self, name: str):
        if name[0] in "aeiou":
            return f"an {name}"
        return f"a {name}"

    def __getitem__(self, idx):
        metadata = os.path.join(self.examples[idx].split('samples')[0],'metadata.jsonl')
        metadata = json.load(open(metadata, 'r'))
        
        img0 = Image.open(self.examples[idx]).convert('RGB')
        if not self.scoring_only:
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
        if self.subset == 'color_attr':
            text = self.prompts[metadata["include"][0]["class"]][metadata["include"][1]["class"]]
        elif self.subset == 'two_object_subset':
            first = metadata["include"][0]["class"]
            second = metadata["include"][1]["class"]
            text = self.prompts[f'{first}_{second}']
        elif self.subset == 'position':
            text = self.prompts[metadata["include"][1]["class"]][metadata["include"][0]["class"]]
        elif self.subset in ['single_object','two_object']:
            text = self.prompts
        else:
            text = self.prompts[metadata["include"][0]["class"]]


        if self.scoring_only:
            return text, idx
        else:
            if self.domain == 'photo':
                return (self.examples[idx], [img0_resize]), text, text.index(metadata['prompt'])
            else:
                return (self.examples[idx], [img0_resize]), text, text.index(metadata['prompt'].replace('a photo',self.with_article(self.domain)))


class Geneval_3_m(Dataset):
    def __init__(self, transform, root_dir, subset, resize=512, scoring_only=False, domain = 'photo', cfg = 9.0):

        
        self.root_dir = f'{root_dir}/{cfg}/stable-diffusion-3-medium-diffusers'
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        self.domain = domain
        self.examples = []
        if self.subset == "two_object_subset":
            subset = "two_object"
        for i in os.listdir(self.root_dir):
            metadata = os.path.join(self.root_dir, i, 'metadata.jsonl')
            metadata = json.load(open(metadata, 'r'))
            if metadata['tag'] == subset:
                for j in range(4):
                    self.examples.append(os.path.join(self.root_dir, i, 'samples', f'0000{j}.png'))
        if self.subset == "two_object_subset":
            self.prompts = json.load(open(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/two_object_subset.json', 'r'))

        else:
            prompt = f'{root_dir}/../prompts/zero_shot_prompts.json'
            self.prompts = json.load(open(prompt, 'r'))[domain][self.subset]
        
    def __len__(self):
        return len(self.examples)

    def with_article(self, name: str):
        if name[0] in "aeiou":
            return f"an {name}"
        return f"a {name}"

    def __getitem__(self, idx):
        metadata = os.path.join(self.examples[idx].split('samples')[0],'metadata.jsonl')
        metadata = json.load(open(metadata, 'r'))
        
        img0 = Image.open(self.examples[idx]).convert('RGB')
        if not self.scoring_only:
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
        if self.subset == 'color_attr':
            text = self.prompts[metadata["include"][0]["class"]][metadata["include"][1]["class"]]
        elif self.subset == 'two_object_subset':
            first = metadata["include"][0]["class"]
            second = metadata["include"][1]["class"]
            text = self.prompts[f'{first}_{second}']
        elif self.subset == 'position':
            text = self.prompts[metadata["include"][1]["class"]][metadata["include"][0]["class"]]
        elif self.subset in ['single_object','two_object']:
            text = self.prompts
        else:
            text = self.prompts[metadata["include"][0]["class"]]


        if self.scoring_only:
            return text, idx
        else:
            if self.domain == 'photo':
                return (self.examples[idx], [img0_resize]), text, text.index(metadata['prompt'])
            else:
                return (self.examples[idx], [img0_resize]), text, text.index(metadata['prompt'].replace('a photo',self.with_article(self.domain)))


class Geneval_filter(Dataset):
    def __init__(self, transform, root_dir, subset, version, resize=512, scoring_only=False, domain = 'photo', cfg = 9.0):
        self.version = version
        try:
            self.root_dir = f'{root_dir}/{cfg}/{self.version.split("/")[1]}'
            if "1-5" in version: version = "1.5"
            elif "2-base" in version: version = "2.0"
            elif "3-m" in version : version = "3-m" 
            else: raise ValueError('Invalid version')
        except:
            if version == "1.5": self.version = "stable-diffusion-v1-5"
            elif version == "2.0": self.version = "stable-diffusion-2-base"
            elif version == "3-m": self.version = "stable-diffusion-3-medium-diffusers"
            else: raise ValueError('Invalid version')
            self.root_dir = f'{root_dir}/{cfg}/{self.version}'
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        self.domain = domain
        self.examples = []
        self.prompts = []
        self.include = []
        self.cfg = cfg
        prompt = f'{root_dir}/../filter/SD-{version}-CFG={str(int(self.cfg))}.json'
        prompt = json.load(open(prompt, 'r')) # [domain][self.subset]
        
        if self.subset == "two_object_subset":
            self.text = json.load(open(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/two_object_subset.json', 'r'))
            subset = "two_object"
        else: 
            self.text = json.load(open(f'{root_dir}/../prompts/zero_shot_prompts.json', 'r'))[self.domain][self.subset]
        for i in prompt:
            try:
                if version == "3-m" and self.cfg == 9.0:
                    if i['tag'] == subset and any(j["label"] == "original_good" for j in i["labels"]) :
                        self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                        self.prompts.append(i["original_prompt"])
                        self.include.append(i["full_metadata"]["include"])
                else:
                    if i['tag'] == subset and all(j["label"] == "original_good" for j in i["labels"]) :
                        self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                        self.prompts.append(i["original_prompt"])
                        self.include.append(i["full_metadata"]["include"])
            except:
                    if i['tag'] == subset and i["human_label"]=="original_good":
                        self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                        self.prompts.append(i["original_prompt"])
                        self.include.append(i["full_metadata"]["include"])

        if self.__len__() == 0:
            raise ValueError('No examples found for the given subset and version')
        if version == "3-m" and self.cfg == 9.0:
            print(self.__len__())
            if self.__len__() < 100:
                raise ValueError('Not enough examples found for the given subset and version')

    def __len__(self):
        return len(self.examples)

    def with_article(self, name: str):
        if name[0] in "aeiou":
            return f"an {name}"
        return f"a {name}"

    def __getitem__(self, idx):
        
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
            if self.domain == 'photo':
                return (self.examples[idx], [img0_resize]), text, text.index(self.prompts[idx])
            else:
                return (self.examples[idx], [img0_resize]), text, text.index(self.prompts[idx].replace('a photo',self.with_article(self.domain)))

class Geneval_1_5_filter(Dataset):
    def __init__(self, transform, root_dir, subset, resize=512, scoring_only=False, domain = 'photo', cfg = 9.0):
        self.version = "stable-diffusion-v1-5"
        version = "1.5"
        self.root_dir = f'{root_dir}/{cfg}/{self.version}'
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        self.domain = domain
        self.examples = []
        self.prompts = []
        self.include = []
        self.cfg = cfg
        prompt = f'{root_dir}/../filter/SD-{version}-CFG={str(int(self.cfg))}.json'
        prompt = json.load(open(prompt, 'r')) # [domain][self.subset]
        if self.subset == "two_object_subset":
            self.text = json.load(open(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/two_object_subset.json', 'r'))
            subset = "two_object"
        else:
           
            self.text = json.load(open(f'{root_dir}/../prompts/zero_shot_prompts.json', 'r'))[self.domain][self.subset]
        for i in prompt:
            try:
                if version == "3-m" and self.cfg == 9.0:
                    if i['tag'] == subset and any(j["label"] == "original_good" for j in i["labels"]) :
                        self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                        self.prompts.append(i["original_prompt"])
                        self.include.append(i["full_metadata"]["include"])
                else:
                    if i['tag'] == subset and all(j["label"] == "original_good" for j in i["labels"]) :
                        self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                        self.prompts.append(i["original_prompt"])
                        self.include.append(i["full_metadata"]["include"])
            except:
                    if i['tag'] == subset and i["human_label"]=="original_good":
                        self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                        self.prompts.append(i["original_prompt"])
                        self.include.append(i["full_metadata"]["include"])

        if self.__len__() == 0:
            raise ValueError('No examples found for the given subset and version')

    def __len__(self):
        return len(self.examples)

    def with_article(self, name: str):
        if name[0] in "aeiou":
            return f"an {name}"
        return f"a {name}"

    def __getitem__(self, idx):
        
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
            if self.domain == 'photo':
                return (self.examples[idx], [img0_resize]), text, text.index(self.prompts[idx])
            else:
                return (self.examples[idx], [img0_resize]), text, text.index(self.prompts[idx].replace('a photo',self.with_article(self.domain)))

class Geneval_2_0_filter(Dataset):
    def __init__(self, transform, root_dir, subset, resize=512, scoring_only=False, domain = 'photo', cfg = 9.0):
        # try:
        #     self.root_dir = f'{root_dir}/{cfg}/{self.version.split("/")[1]}'
        #     if "1-5" in version: version = "1.5"
        #     elif "2-base" in version: version = "2.0"
        #     elif "3-m" in version : version = "3-m" 
        #     else: raise ValueError('Invalid version')
        # except:
        #     if version == "1.5": self.version = "stable-diffusion-v1-5"
        #     elif version == "2.0": self.version = "stable-diffusion-2-base"
        #     elif version == "3-m": self.version = "stable-diffusion-3-medium-diffusers"
            # else: raise ValueError('Invalid version')
            # self.root_dir = f'{root_dir}/{cfg}/{self.version}'
        self.version = "stable-diffusion-2-base"
        version = "2.0"
        self.root_dir = f'{root_dir}/{cfg}/{self.version}'
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        self.domain = domain
        self.examples = []
        self.prompts = []
        self.include = []
        self.cfg = cfg
        prompt = f'{root_dir}/../filter/SD-{version}-CFG={str(int(self.cfg))}.json'
        prompt = json.load(open(prompt, 'r')) # [domain][self.subset]
        
        if self.subset == "two_object_subset":
            self.text = json.load(open(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/two_object_subset.json', 'r'))
            subset = "two_object"
        else:
            
            self.text = json.load(open(f'{root_dir}/../prompts/zero_shot_prompts.json', 'r'))[self.domain][self.subset]
        for i in prompt:
            try:
                if version == "3-m" and self.cfg == 9.0:
                    if i['tag'] == subset and any(j["label"] == "original_good" for j in i["labels"]) :
                        self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                        self.prompts.append(i["original_prompt"])
                        self.include.append(i["full_metadata"]["include"])
                else:
                    if i['tag'] == subset and all(j["label"] == "original_good" for j in i["labels"]) :
                        self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                        self.prompts.append(i["original_prompt"])
                        self.include.append(i["full_metadata"]["include"])
            except:
                    if i['tag'] == subset and i["human_label"]=="original_good":
                        self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                        self.prompts.append(i["original_prompt"])
                        self.include.append(i["full_metadata"]["include"])

        if self.__len__() == 0:
            raise ValueError('No examples found for the given subset and version')

    def __len__(self):
        return len(self.examples)

    def with_article(self, name: str):
        if name[0] in "aeiou":
            return f"an {name}"
        return f"a {name}"

    def __getitem__(self, idx):
        
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
            if self.domain == 'photo':
                return (self.examples[idx], [img0_resize]), text, text.index(self.prompts[idx])
            else:
                return (self.examples[idx], [img0_resize]), text, text.index(self.prompts[idx].replace('a photo',self.with_article(self.domain)))

class Geneval_3_m_filter(Dataset):
    def __init__(self, transform, root_dir, subset, resize=512, scoring_only=False, domain = 'photo', cfg = 9.0):
        # try:
        #     self.root_dir = f'{root_dir}/{cfg}/{self.version.split("/")[1]}'
        #     if "1-5" in version: version = "1.5"
        #     elif "2-base" in version: version = "2.0"
        #     elif "3-m" in version : version = "3-m" 
        #     else: raise ValueError('Invalid version')
        # except:
        #     if version == "1.5": self.version = "stable-diffusion-v1-5"
        #     elif version == "2.0": self.version = "stable-diffusion-2-base"
        #     elif version == "3-m": self.version = "stable-diffusion-3-medium-diffusers"
            # else: raise ValueError('Invalid version')
            # self.root_dir = f'{root_dir}/{cfg}/{self.version}'
        self.version = "stable-diffusion-3-medium-diffusers"
        version = "3-m"
        self.root_dir = f'{root_dir}/{cfg}/{self.version}'
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        self.domain = domain
        self.examples = []
        self.prompts = []
        self.include = []
        self.cfg = cfg
        prompt = f'{root_dir}/../filter/SD-{version}-CFG={str(int(self.cfg))}.json'
        prompt = json.load(open(prompt, 'r')) # [domain][self.subset]
        if self.subset == "two_object_subset":
            self.text = json.load(open(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/two_object_subset.json', 'r'))
            subset = "two_object"
        else:
            
            self.text = json.load(open(f'{root_dir}/../prompts/zero_shot_prompts.json', 'r'))[self.domain][self.subset]
        for i in prompt:
            try:
                if version == "3-m" and self.cfg == 9.0:
                    if i['tag'] == subset and any(j["label"] == "original_good" for j in i["labels"]) :
                        self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                        self.prompts.append(i["original_prompt"])
                        self.include.append(i["full_metadata"]["include"])
                else:
                    if i['tag'] == subset and all(j["label"] == "original_good" for j in i["labels"]) :
                        self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                        self.prompts.append(i["original_prompt"])
                        self.include.append(i["full_metadata"]["include"])
            except:
                    if i['tag'] == subset and i["human_label"]=="original_good":
                        self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                        self.prompts.append(i["original_prompt"])
                        self.include.append(i["full_metadata"]["include"])

        if self.__len__() == 0:
            raise ValueError('No examples found for the given subset and version')

    def __len__(self):
        return len(self.examples)

    def with_article(self, name: str):
        if name[0] in "aeiou":
            return f"an {name}"
        return f"a {name}"

    def __getitem__(self, idx):
        
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
            if self.domain == 'photo':
                return (self.examples[idx], [img0_resize]), text, text.index(self.prompts[idx])
            else:
                return (self.examples[idx], [img0_resize]), text, text.index(self.prompts[idx].replace('a photo',self.with_article(self.domain)))


# class Compbench(Dataset):
#     def __init__(self, transform, root_dir, subset, version, resize=512, scoring_only=False, domain = 'photo'):
#         self.version = version
#         self.root_dir = f'{root_dir}_val/{self.version.split("/")[1]}/samples'
#         self.resize = resize
#         self.transform = transform
#         self.scoring_only = scoring_only
#         self.subset = subset
#         self.domain = domain
#         self.examples = os.listdir(self.root_dir)
#         self.prompts = [i.split("_")[0] for i in self.examples]
        
#     def __len__(self):
#         return len(self.examples)
    
#     def __getitem__(self, idx):

#         img0 = Image.open(os.path.join(self.root_dir, self.examples[idx])).convert('RGB')
#         if not self.scoring_only:
#             if self.transform:
#                 img0_resize = self.transform(img0).unsqueeze(0)
#             else:
#                 img0_resize = img0.resize((self.resize, self.resize))
#                 img0_resize = diffusers_preprocess(img0_resize)

#         text = self.prompts[idx]


#         if self.scoring_only:
#             return text, idx
#         else:
#             if self.domain == 'photo':
#                 return (self.examples[idx], [img0_resize]), text, text.index(metadata['prompt'])
#             else:
#                 return (self.examples[idx], [img0_resize]), text, text.index(metadata['prompt'].replace('a photo',self.with_article(self.domain)))


class Ours(Dataset):
    def __init__(self, transform, root_dir, subset, version, resize=512, scoring_only=False, domain = 'photo', before=False):
        self.version = version
        
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        self.domain = domain
        self.before = before
        self.examples = []
        self.prompts = []
        
        if '/' in version:
            self.root_dir = f'{root_dir}/{self.version.split("/")[1]}'
            if version == 'stabilityai/stable-diffusion-2-base':
                prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_2.0_labeled.csv')
            elif version == 'stable-diffusion-v1-5/stable-diffusion-v1-5':
                prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_1.5_labeled.csv')
            elif version == 'stabilityai/stable-diffusion-3-medium-diffusers':
                prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_3-m_labeled.csv')
            else:
                raise ValueError("Invalid version")
        else:
            if version == "2.0":
                self.version = 'stabilityai/stable-diffusion-2-base'
            elif version == "1.5":
                self.version = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
            elif version == "3-m":
                self.version = 'stabilityai/stable-diffusion-3-medium-diffusers'
            else:
                raise ValueError("Invalid version")
            self.root_dir = f'{root_dir}/{self.version.split("/")[1]}'
            prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_{version}_labeled.csv')

        self.prompts = prompts[prompts['tag']==self.subset]
        for i in self.prompts['filename']:
            self.examples.append(os.path.join(self.root_dir, i))

        # for i in os.listdir(self.root_dir):
        #     metadata = os.path.join(self.root_dir, i, 'metadata.jsonl')
        #     metadata = json.load(open(metadata, 'r'))
        #     if metadata['tag'] == self.subset:
        #         for j in range(4):
        #             self.examples.append(os.path.join(self.root_dir, i, 'samples', f'0000{j}.png'))
        #             self.prompts.append(metadata['prompt'])
        
        if self.subset == 'color_attr' or self.subset == 'colors':
            choice = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
            self.choice = [self.with_article(i) for i in choice]
            # print(self.choice)

        elif self.subset == 'position':
            self.choice = ["left of", "right of", "above", "below"]
        elif self.subset == 'counting':
            self.choice = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        else:
            raise ValueError("Invalid subset")
    
        color_pattern = re.compile(r'\b(' + '|'.join(self.choice) + r')\b')
        self.all_text = []

        if not self.before:
            for text in self.prompts['real_prompt']:
                self.subset_text = []
                found_colors = color_pattern.findall(text)
                if len(found_colors) == 0:
                    raise ValueError(f"No colors found in text: '{text}'")
                elif self.subset == 'color_attr' and len(found_colors) == 1:
                    raise ValueError(f"Only one color found in text: '{text}'")
                
                # Generate all combinations of replacements for found colors
                if self.subset != "color_attr":
                    for replacements in itertools.product(self.choice, repeat=len(found_colors)):
                            for original_color, new_color in zip(found_colors, replacements):
                                modified_template = text.replace(original_color, new_color, 1)
                            self.subset_text.append(modified_template)
                    assert len(self.subset_text) == len(self.choice)
                else: 

                    # Generate all 10x10 combinations
                    expected_combinations = list(itertools.product(self.choice, repeat=2))
                    for replacements in expected_combinations:
                        modified_template = text
                        temp_found_colors = found_colors.copy()  # Keep original order

                        # Create a unique pattern to prevent duplicate replacements
                        placeholder_text = modified_template.replace(temp_found_colors[0], "COLOR1", 1)
                        placeholder_text = placeholder_text.replace(temp_found_colors[1], "COLOR2", 1)

                        # Replace placeholders correctly
                        final_text = placeholder_text.replace("COLOR1", replacements[0], 1).replace("COLOR2", replacements[1], 1)

                        self.subset_text.append(final_text)

                    # Convert to a set AFTER all replacements to preserve order
                    unique_prompts = list(dict.fromkeys(self.subset_text))  # Removes accidental duplicates while keeping order
                    self.subset_text = unique_prompts  # Store only unique prompts

                    assert len(self.subset_text) == len(self.choice) * len(self.choice)
            
                self.all_text.append(self.subset_text)

        else:
            for text in self.prompts['prompt']:
                self.subset_text = []
                found_colors = color_pattern.findall(text)
                if len(found_colors) == 0:
                    raise ValueError(f"No colors found in text: '{text}'")
                elif self.subset == 'color_attr' and len(found_colors) == 1:
                    raise ValueError(f"Only one color found in text: '{text}'")
                
                # Generate all combinations of replacements for found colors
                if self.split != "color_attr":
                    for replacements in itertools.product(self.choice, repeat=len(found_colors)):
                            for original_color, new_color in zip(found_colors, replacements):
                                modified_template = text.replace(original_color, new_color, 1)
                            self.subset_text.append(modified_template)
                    assert len(self.subset_text) == len(self.choice)
                else: 

                    # Generate all 10x10 combinations
                    expected_combinations = list(itertools.product(self.choice, repeat=2))
                    for replacements in expected_combinations:
                        modified_template = text
                        temp_found_colors = found_colors.copy()  # Keep original order

                        # Create a unique pattern to prevent duplicate replacements
                        placeholder_text = modified_template.replace(temp_found_colors[0], "COLOR1", 1)
                        placeholder_text = placeholder_text.replace(temp_found_colors[1], "COLOR2", 1)

                        # Replace placeholders correctly
                        final_text = placeholder_text.replace("COLOR1", replacements[0], 1).replace("COLOR2", replacements[1], 1)

                        self.subset_text.append(final_text)

                    # Convert to a set AFTER all replacements to preserve order
                    unique_prompts = list(dict.fromkeys(self.subset_text))  # Removes accidental duplicates while keeping order
                    self.subset_text = unique_prompts  # Store only unique prompts

                    assert len(self.subset_text) == len(self.choice) * len(self.choice)

                self.all_text.append(self.subset_text)

    def __len__(self):
        return len(self.examples)

    def with_article(self, name: str):
        if name[0] in "aeiou":
            return f"an {name}"
        return f"a {name}"

    def __getitem__(self, idx):

        img0 = Image.open(self.examples[idx]).convert('RGB')
        if not self.scoring_only:
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
        # print
        if not self.before:
            text = self.prompts[self.prompts['filename']==self.examples[idx].split(self.root_dir)[1][1:]]['real_prompt'].iloc[0]
        else:
            text = self.prompts[self.prompts['filename']==self.examples[idx].split(self.root_dir)[1][1:]]['prompt'].iloc[0]
        
        if self.scoring_only:
            return text, idx
        else:
            if self.domain == 'photo':
                return (self.examples[idx], [img0_resize]), self.all_text[idx], self.all_text[idx].index(text)
            else:
                return (self.examples[idx], [img0_resize]), self.all_text[idx], self.all_text[idx].index(text.replace('a photo',self.with_article(self.domain)))

class Ours_3_m(Dataset):
    def __init__(self, transform, root_dir, subset, resize=512, scoring_only=False, domain = 'photo', before=False):
        
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        self.domain = domain
        self.before = before
        self.examples = []
        prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_3-m_labeled.csv')
        self.root_dir = f'{root_dir}/stable-diffusion-3-medium-diffusers'

        # if '/' in version:
        #     self.root_dir = f'{root_dir}/{self.version.split("/")[1]}'
        #     if version == 'stabilityai/stable-diffusion-2-base':
        #         prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_2.0_labeled.csv')
        #     elif version == 'stable-diffusion-v1-5/stable-diffusion-v1-5':
        #         prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_1.5_labeled.csv')
        #     elif version == 'stabilityai/stable-diffusion-3-medium-diffusers':
        #         prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_3-m_labeled.csv')
        #     else:
        #         raise ValueError("Invalid version")
        # else:
        #     if version == "2.0":
        #         self.version = 'stabilityai/stable-diffusion-2-base'
        #     elif version == "1.5":
        #         self.version = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
        #     elif version == "3-m":
        #         self.version = 'stabilityai/stable-diffusion-3-medium-diffusers'
        #     else:
        #         raise ValueError("Invalid version")
        #     self.root_dir = f'{root_dir}/{self.version.split("/")[1]}'
        #     prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_{version}_labeled.csv')

        self.prompts = prompts[prompts['tag']==self.subset]

        for i in self.prompts['filename']:
            self.examples.append(os.path.join(self.root_dir, i))
        if self.subset == 'color_attr' or self.subset == 'colors':
            choice = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
            self.choice = [self.with_article(i) for i in choice]
        elif self.subset == 'position':
            self.choice = ["left of", "right of", "above", "below"]
        elif self.subset == 'counting':
            self.choice = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        else:
            raise ValueError("Invalid subset")
    
        color_pattern = re.compile(r'\b(' + '|'.join(self.choice) + r')\b')
        self.all_text = []
        if not self.before:
            for text in self.prompts['real_prompt']:
                self.subset_text = []
                found_colors = color_pattern.findall(text)
                if len(found_colors) == 0:
                    raise ValueError(f"No colors found in text: '{text}'")
                elif self.subset == 'color_attr' and len(found_colors) == 1:
                    raise ValueError(f"Only one color found in text: '{text}'")

                # Generate all combinations of replacements for found colors
                if self.subset != "color_attr":
                    for replacements in itertools.product(self.choice, repeat=len(found_colors)):
                            for original_color, new_color in zip(found_colors, replacements):
                                modified_template = text.replace(original_color, new_color, 1)
                            self.subset_text.append(modified_template)
                    assert len(self.subset_text) == len(self.choice)
                else: 

                    # Generate all 10x10 combinations
                    expected_combinations = list(itertools.product(self.choice, repeat=2))
                    for replacements in expected_combinations:
                        modified_template = text
                        temp_found_colors = found_colors.copy()  # Keep original order

                        # Create a unique pattern to prevent duplicate replacements
                        placeholder_text = modified_template.replace(temp_found_colors[0], "COLOR1", 1)
                        placeholder_text = placeholder_text.replace(temp_found_colors[1], "COLOR2", 1)

                        # Replace placeholders correctly
                        final_text = placeholder_text.replace("COLOR1", replacements[0], 1).replace("COLOR2", replacements[1], 1)

                        self.subset_text.append(final_text)

                    # Convert to a set AFTER all replacements to preserve order
                    unique_prompts = list(dict.fromkeys(self.subset_text))  # Removes accidental duplicates while keeping order
                    self.subset_text = unique_prompts  # Store only unique prompts

                    assert len(self.subset_text) == len(self.choice) * len(self.choice)
                self.all_text.append(self.subset_text)

        else:
            for text in self.prompts['prompt']:
                self.subset_text = []
                found_colors = color_pattern.findall(text)
                if len(found_colors) == 0:
                    raise ValueError(f"No colors found in text: '{text}'")
                elif self.subset == 'color_attr' and len(found_colors) == 1:
                    raise ValueError(f"Only one color found in text: '{text}'")

                # Generate all combinations of replacements for found colors
                if self.subset != "color_attr":
                    for replacements in itertools.product(self.choice, repeat=len(found_colors)):
                            for original_color, new_color in zip(found_colors, replacements):
                                modified_template = text.replace(original_color, new_color, 1)
                            self.subset_text.append(modified_template)
                    assert len(self.subset_text) == len(self.choice)
                else: 

                    # Generate all 10x10 combinations
                    expected_combinations = list(itertools.product(self.choice, repeat=2))
                    for replacements in expected_combinations:
                        modified_template = text
                        temp_found_colors = found_colors.copy()  # Keep original order

                        # Create a unique pattern to prevent duplicate replacements
                        placeholder_text = modified_template.replace(temp_found_colors[0], "COLOR1", 1)
                        placeholder_text = placeholder_text.replace(temp_found_colors[1], "COLOR2", 1)

                        # Replace placeholders correctly
                        final_text = placeholder_text.replace("COLOR1", replacements[0], 1).replace("COLOR2", replacements[1], 1)

                        self.subset_text.append(final_text)

                    # Convert to a set AFTER all replacements to preserve order
                    unique_prompts = list(dict.fromkeys(self.subset_text))  # Removes accidental duplicates while keeping order
                    self.subset_text = unique_prompts  # Store only unique prompts

                    assert len(self.subset_text) == len(self.choice) * len(self.choice)
                self.all_text.append(self.subset_text)

    def __len__(self):
        return len(self.examples)

    def with_article(self, name: str):
        if name[0] in "aeiou":
            return f"an {name}"
        return f"a {name}"

    def __getitem__(self, idx):

        img0 = Image.open(self.examples[idx]).convert('RGB')
        if not self.scoring_only:
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
        if not self.before:
            text = self.prompts[self.prompts['filename']==self.examples[idx].split(self.root_dir)[1][1:]]['real_prompt'].iloc[0]
        else:
            text = self.prompts[self.prompts['filename']==self.examples[idx].split(self.root_dir)[1][1:]]['prompt'].iloc[0]
        if self.scoring_only:
            return text, idx
        else:
            if self.domain == 'photo':
                return (self.examples[idx], [img0_resize]), self.all_text[idx], self.all_text[idx].index(text)
            else:
                return (self.examples[idx], [img0_resize]), self.all_text[idx], self.all_text[idx].index(text.replace('a photo',self.with_article(self.domain)))

class Ours_2_0(Dataset):
    def __init__(self, transform, root_dir, subset, resize=512, scoring_only=False, domain = 'photo', before=False):

        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        self.domain = domain
        self.before = before
        self.examples = []
        self.root_dir = f'{root_dir}/stable-diffusion-2-base'
        prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_2.0_labeled.csv')
        # if '/' in version:
        #     self.root_dir = f'{root_dir}/{self.version.split("/")[1]}'
        #     if version == 'stabilityai/stable-diffusion-2-base':
        #         prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_2.0_labeled.csv')
        #     elif version == 'stable-diffusion-v1-5/stable-diffusion-v1-5':
        #         prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_1.5_labeled.csv')
        #     elif version == 'stabilityai/stable-diffusion-3-medium-diffusers':
        #         prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_3-m_labeled.csv')
        #     else:
        #         raise ValueError("Invalid version")
        # else:
        #     if version == "2.0":
        #         self.version = 'stabilityai/stable-diffusion-2-base'
        #     elif version == "1.5":
        #         self.version = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
        #     elif version == "3-m":
        #         self.version = 'stabilityai/stable-diffusion-3-medium-diffusers'
        #     else:
        #         raise ValueError("Invalid version")
        #     self.root_dir = f'{root_dir}/{self.version.split("/")[1]}'
        #     prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_{version}_labeled.csv')

        self.prompts = prompts[prompts['tag']==self.subset]

        for i in self.prompts['filename']:
            self.examples.append(os.path.join(self.root_dir, i))
        if self.subset == 'color_attr' or self.subset == 'colors':
            choice = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
            self.choice = [self.with_article(i) for i in choice]
        elif self.subset == 'position':
            self.choice = ["left of", "right of", "above", "below"]
        elif self.subset == 'counting':
            self.choice = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        else:
            raise ValueError("Invalid subset")
    
        color_pattern = re.compile(r'\b(' + '|'.join(self.choice) + r')\b')
        self.all_text = []
        if not self.before:
            for text in self.prompts['real_prompt']:
                self.subset_text = []
                found_colors = color_pattern.findall(text)
                if len(found_colors) == 0:
                    raise ValueError(f"No colors found in text: '{text}'")
                elif self.subset == 'color_attr' and len(found_colors) == 1:
                    raise ValueError(f"Only one color found in text: '{text}'")

                # Generate all combinations of replacements for found colors
                if self.subset != "color_attr":
                    for replacements in itertools.product(self.choice, repeat=len(found_colors)):
                            for original_color, new_color in zip(found_colors, replacements):
                                modified_template = text.replace(original_color, new_color, 1)
                            self.subset_text.append(modified_template)
                    assert len(self.subset_text) == len(self.choice)
                else: 

                    # Generate all 10x10 combinations
                    expected_combinations = list(itertools.product(self.choice, repeat=2))
                    for replacements in expected_combinations:
                        modified_template = text
                        temp_found_colors = found_colors.copy()  # Keep original order

                        # Create a unique pattern to prevent duplicate replacements
                        placeholder_text = modified_template.replace(temp_found_colors[0], "COLOR1", 1)
                        placeholder_text = placeholder_text.replace(temp_found_colors[1], "COLOR2", 1)

                        # Replace placeholders correctly
                        final_text = placeholder_text.replace("COLOR1", replacements[0], 1).replace("COLOR2", replacements[1], 1)

                        self.subset_text.append(final_text)

                    # Convert to a set AFTER all replacements to preserve order
                    unique_prompts = list(dict.fromkeys(self.subset_text))  # Removes accidental duplicates while keeping order
                    self.subset_text = unique_prompts  # Store only unique prompts

                    assert len(self.subset_text) == len(self.choice) * len(self.choice)
                self.all_text.append(self.subset_text)

        else:
            for text in self.prompts['prompt']:
                self.subset_text = []
                found_colors = color_pattern.findall(text)
                if len(found_colors) == 0:
                    raise ValueError(f"No colors found in text: '{text}'")
                elif self.subset == 'color_attr' and len(found_colors) == 1:
                    raise ValueError(f"Only one color found in text: '{text}'")

                # Generate all combinations of replacements for found colors
                if self.subset != "color_attr":
                    for replacements in itertools.product(self.choice, repeat=len(found_colors)):
                            for original_color, new_color in zip(found_colors, replacements):
                                modified_template = text.replace(original_color, new_color, 1)
                            self.subset_text.append(modified_template)
                    assert len(self.subset_text) == len(self.choice)
                else: 

                    # Generate all 10x10 combinations
                    expected_combinations = list(itertools.product(self.choice, repeat=2))
                    for replacements in expected_combinations:
                        modified_template = text
                        temp_found_colors = found_colors.copy()  # Keep original order

                        # Create a unique pattern to prevent duplicate replacements
                        placeholder_text = modified_template.replace(temp_found_colors[0], "COLOR1", 1)
                        placeholder_text = placeholder_text.replace(temp_found_colors[1], "COLOR2", 1)

                        # Replace placeholders correctly
                        final_text = placeholder_text.replace("COLOR1", replacements[0], 1).replace("COLOR2", replacements[1], 1)

                        self.subset_text.append(final_text)

                    # Convert to a set AFTER all replacements to preserve order
                    unique_prompts = list(dict.fromkeys(self.subset_text))  # Removes accidental duplicates while keeping order
                    self.subset_text = unique_prompts  # Store only unique prompts

                    assert len(self.subset_text) == len(self.choice) * len(self.choice)
                self.all_text.append(self.subset_text)

    def __len__(self):
        return len(self.examples)

    def with_article(self, name: str):
        if name[0] in "aeiou":
            return f"an {name}"
        return f"a {name}"

    def __getitem__(self, idx):

        img0 = Image.open(self.examples[idx]).convert('RGB')
        if not self.scoring_only:
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
        if not self.before:
            text = self.prompts[self.prompts['filename']==self.examples[idx].split(self.root_dir)[1][1:]]['real_prompt'].iloc[0]
        else:
            text = self.prompts[self.prompts['filename']==self.examples[idx].split(self.root_dir)[1][1:]]['prompt'].iloc[0]
        if self.scoring_only:
            return text, idx
        else:
            if self.domain == 'photo':
                return (self.examples[idx], [img0_resize]), self.all_text[idx], self.all_text[idx].index(text)
            else:
                return (self.examples[idx], [img0_resize]), self.all_text[idx], self.all_text[idx].index(text.replace('a photo',self.with_article(self.domain)))

class Ours_1_5(Dataset):
    def __init__(self, transform, root_dir, subset, resize=512, scoring_only=False, domain = 'photo', before=False):

        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        self.domain = domain
        self.before = before
        self.examples = []
        self.root_dir = f'{root_dir}/stable-diffusion-v1-5'
        prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_1.5_labeled.csv')
        # if '/' in version:
        #     self.root_dir = f'{root_dir}/{self.version.split("/")[1]}'
        #     if version == 'stabilityai/stable-diffusion-2-base':
        #         prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_2.0_labeled.csv')
        #     elif version == 'stable-diffusion-v1-5/stable-diffusion-v1-5':
        #         prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_1.5_labeled.csv')
        #     elif version == 'stabilityai/stable-diffusion-3-medium-diffusers':
        #         prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_3-m_labeled.csv')
        #     else:
        #         raise ValueError("Invalid version")
        # else:
        #     if version == "2.0":
        #         self.version = 'stabilityai/stable-diffusion-2-base'
        #     elif version == "1.5":
        #         self.version = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
        #     elif version == "3-m":
        #         self.version = 'stabilityai/stable-diffusion-3-medium-diffusers'
        #     else:
        #         raise ValueError("Invalid version")
        #     self.root_dir = f'{root_dir}/{self.version.split("/")[1]}'
        #     prompts = pd.read_csv(f'/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/results/analysis/csv/geneval_full_{version}_labeled.csv')

        self.prompts = prompts[prompts['tag']==self.subset]

        for i in self.prompts['filename']:
            self.examples.append(os.path.join(self.root_dir, i))
        if self.subset == 'color_attr' or self.subset == 'colors':
            choice = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
            self.choice = [self.with_article(i) for i in choice]
        elif self.subset == 'position':
            self.choice = ["left of", "right of", "above", "below"]
        elif self.subset == 'counting':
            self.choice = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        else:
            raise ValueError("Invalid subset")
    
        color_pattern = re.compile(r'\b(' + '|'.join(self.choice) + r')\b')
        self.all_text = []
        if not self.before:
            for text in self.prompts['real_prompt']:
                self.subset_text = []
                found_colors = color_pattern.findall(text)
                if len(found_colors) == 0:
                    raise ValueError(f"No colors found in text: '{text}'")
                elif self.subset == 'color_attr' and len(found_colors) == 1:
                    raise ValueError(f"Only one color found in text: '{text}'")

                # Generate all combinations of replacements for found colors
                if self.subset != "color_attr":
                    for replacements in itertools.product(self.choice, repeat=len(found_colors)):
                            for original_color, new_color in zip(found_colors, replacements):
                                modified_template = text.replace(original_color, new_color, 1)
                            self.subset_text.append(modified_template)
                    assert len(self.subset_text) == len(self.choice)
                else: 

                    # Generate all 10x10 combinations
                    expected_combinations = list(itertools.product(self.choice, repeat=2))
                    for replacements in expected_combinations:
                        modified_template = text
                        temp_found_colors = found_colors.copy()  # Keep original order

                        # Create a unique pattern to prevent duplicate replacements
                        placeholder_text = modified_template.replace(temp_found_colors[0], "COLOR1", 1)
                        placeholder_text = placeholder_text.replace(temp_found_colors[1], "COLOR2", 1)

                        # Replace placeholders correctly
                        final_text = placeholder_text.replace("COLOR1", replacements[0], 1).replace("COLOR2", replacements[1], 1)

                        self.subset_text.append(final_text)

                    # Convert to a set AFTER all replacements to preserve order
                    unique_prompts = list(dict.fromkeys(self.subset_text))  # Removes accidental duplicates while keeping order
                    self.subset_text = unique_prompts  # Store only unique prompts

                    assert len(self.subset_text) == len(self.choice) * len(self.choice)
                self.all_text.append(self.subset_text)

        else:
            for text in self.prompts['prompt']:
                self.subset_text = []
                found_colors = color_pattern.findall(text)
                if len(found_colors) == 0:
                    raise ValueError(f"No colors found in text: '{text}'")
                elif self.subset == 'color_attr' and len(found_colors) == 1:
                    raise ValueError(f"Only one color found in text: '{text}'")

                # Generate all combinations of replacements for found colors
                if self.subset != "color_attr":
                    for replacements in itertools.product(self.choice, repeat=len(found_colors)):
                            for original_color, new_color in zip(found_colors, replacements):
                                modified_template = text.replace(original_color, new_color, 1)
                            self.subset_text.append(modified_template)
                    assert len(self.subset_text) == len(self.choice)
                else: 
                    # Generate all 10x10 combinations
                    expected_combinations = list(itertools.product(self.choice, repeat=2))
                    for replacements in expected_combinations:
                        modified_template = text
                        temp_found_colors = found_colors.copy()  # Keep original order

                        # Create a unique pattern to prevent duplicate replacements
                        placeholder_text = modified_template.replace(temp_found_colors[0], "COLOR1", 1)
                        placeholder_text = placeholder_text.replace(temp_found_colors[1], "COLOR2", 1)

                        # Replace placeholders correctly
                        final_text = placeholder_text.replace("COLOR1", replacements[0], 1).replace("COLOR2", replacements[1], 1)

                        self.subset_text.append(final_text)

                    # Convert to a set AFTER all replacements to preserve order
                    unique_prompts = list(dict.fromkeys(self.subset_text))  # Removes accidental duplicates while keeping order
                    self.subset_text = unique_prompts  # Store only unique prompts

                    assert len(self.subset_text) == len(self.choice) * len(self.choice)
                self.all_text.append(self.subset_text)

    def __len__(self):
        return len(self.examples)

    def with_article(self, name: str):
        if name[0] in "aeiou":
            return f"an {name}"
        return f"a {name}"

    def __getitem__(self, idx):

        img0 = Image.open(self.examples[idx]).convert('RGB')
        if not self.scoring_only:
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
        if not self.before:
            text = self.prompts[self.prompts['filename']==self.examples[idx].split(self.root_dir)[1][1:]]['real_prompt'].iloc[0]
        else:
            text = self.prompts[self.prompts['filename']==self.examples[idx].split(self.root_dir)[1][1:]]['prompt'].iloc[0]

        if self.scoring_only:
            return text, idx
        else:
            if self.domain == 'photo':
                return (self.examples[idx], [img0_resize]), self.all_text[idx], self.all_text[idx].index(text)
            else:
                return (self.examples[idx], [img0_resize]), self.all_text[idx], self.all_text[idx].index(text.replace('a photo',self.with_article(self.domain)))

class MMVP_VLM(Dataset):
    def __init__(self, transform, root_dir, subset, resize=512, scoring_only=False, domain = 'photo', before=False):
        self.resize = resize
        self.transform = transform
        self.scoring_only = scoring_only
        self.subset = subset
        
        path = os.path.join(os.path.dirname(__file__), "../hf_token.txt")
        # check if exists otherwise use HF_HOME
        if os.path.exists(path):
            with open(path, "r") as f:
                hf_token = f.read().strip()
        else:
            hf_token = os.path.join(os.environ["HF_HOME"], "hf_token.txt")
            
            with open(hf_token, "r") as f:
                hf_token = f.read().strip()
            
        
        # with open("../hf_token.txt", "r") as f:
        #         hf_token = f.read().strip()
        try:
            self.examples = []
            self.all_text = []
            examples = os.path.join(root_dir, 'hub/datasets--MMVP--MMVP_VLM/snapshots/211372d5357398f914d806d07dc305aea1f257d2')
            questions = pd.read_csv(os.path.join(examples,'Questions.csv'))
            self.questions = questions[questions['Type']==self.subset]
            # for i in range(len(self.questions)):
            self.questions = self.questions.sort_values(by='Question ID')
            for i in range(0, len(self.questions), 2):
                example_set = []
                text_set = []
                example_set.append(os.path.join(examples, 'MLLM_VLM Images', self.subset, str(self.questions['Question ID'].iloc[i])+'.jpg'))
                example_set.append(os.path.join(examples, 'MLLM_VLM Images', self.subset, str(self.questions['Question ID'].iloc[i+1])+'.jpg'))
                text_set.append("a photo of "+ str(self.questions['Statement'].iloc[i]))
                text_set.append("a photo of "+ str(self.questions['Statement'].iloc[i+1]))
                self.examples.append(example_set)
                self.all_text.append(text_set)
        except:
            try:
                examples = load_dataset('MMVP/MMVP_VLM', use_auth_token=hf_token, )
            except:
                examples = load_dataset('MMVP/MMVP_VLM', token = hf_token)
            self.examples = examples['train']
            # label = {0:'Camera Perspective', 1:'Color', 2: 'Orientation', 3: }
            raise ValueError("Please complete the code for the subset")
        # examples = load_dataset('MMVP/MMVP_VLM', token=hf_token)
        
        # self.examples = examples['train']['label']
        # print(self.examples['label'])
        # print(self.examples['image'])
        # # print(self.examples)
        # exit(0)

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # print(self.examples[idx])
        try:
            img0 = Image.open(self.examples[idx][0]).convert('RGB')
            img1 = Image.open(self.examples[idx][1]).convert('RGB')
        except:
            img0 = Image.open(self.examples[idx]['image'][0]).convert('RGB')
            img1 = Image.open(self.examples[idx]['image'][0]).convert('RGB')
        if not self.scoring_only:
            if self.transform:
                img0_resize = self.transform(img0).unsqueeze(0)
                img1_resize = self.transform(img1).unsqueeze(0)
            else:
                img0_resize = img0.resize((self.resize, self.resize))
                img0_resize = diffusers_preprocess(img0_resize)
                img1_resize = img1.resize((self.resize, self.resize))
                img1_resize = diffusers_preprocess(img1_resize)


        caption = [self.all_text[idx][0], self.all_text[idx][1]]
        # caption = [self.examples[idx]['label'][0], self.examples[idx]['label'][1]]    

        if self.scoring_only:
            return text, idx
        else:
            return ((self.examples[idx][0],self.examples[idx][1]), [img0_resize,img1_resize]), caption, idx
