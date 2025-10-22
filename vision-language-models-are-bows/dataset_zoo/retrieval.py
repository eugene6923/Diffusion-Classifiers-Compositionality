import os
import re
import json
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision import datasets
from glob import glob
from .constants import COCO_ROOT, FLICKR_ROOT
from .utils import AverageMeter
from easydict import EasyDict as edict
import itertools
import pandas as pd

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    
    return caption


class COCO_Retrieval(Dataset):
    def __init__(self, image_preprocess=None, root_dir=COCO_ROOT, max_words=30, split="test",
                 image_perturb_fn=None, download=False):  
        """
        COCO Retrieval Dataset.
        image_preprocess: image preprocessing function
        root_dir: The directory of the coco dataset. This directory should contain test2014 files.
        max_words: Cropping the caption to max_words.
        split: 'val' or 'test'
        image_perturb_fn: image perturbation function for patch permutation experiments.
        download: Whether to download the dataset if it does not exist.
        """
        self.root_dir = root_dir
        if not os.path.exists(root_dir):
            print("Directory for COCO could not be found!")
            if download:
                print("Downloading COCO now.")
                self.download()
            else:
                raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")
        
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'}
        filenames = {'val':'coco_karpathy_val.json','test':'coco_karpathy_test.json'}
        download_url(urls[split],root_dir)
        
        
        self.annotation = json.load(open(os.path.join(root_dir,filenames[split]),'r'))
        self.image_preprocess = image_preprocess
        self.image_perturb_fn = image_perturb_fn
        self.image_root = root_dir
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        
        if self.image_preprocess is not None: 
            image = self.image_preprocess(image)
          
        if self.image_perturb_fn is not None:
            image = self.image_perturb_fn(image) 
         
        return {"image": image, "idx": index}
    
    def download(self):
        import subprocess
        os.makedirs(self.root_dir, exist_ok=True)
        #subprocess.call(["wget", "http://images.cocodataset.org/zips/train2014.zip"], cwd=self.root_dir)
        #subprocess.call(["unzip", "train2014.zip"], cwd=self.root_dir)
        
        subprocess.call(["wget", "http://images.cocodataset.org/zips/val2014.zip"], cwd=self.root_dir)
        subprocess.call(["unzip", "val2014.zip"], cwd=self.root_dir)
        
        subprocess.call(["wget", "http://images.cocodataset.org/zips/test2014.zip"], cwd=self.root_dir)
        subprocess.call(["unzip", "test2014.zip"], cwd=self.root_dir)
        
    
    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
    
        else:
            scores_t2i = scores
            scores_i2t = scores

        print(f"COCO results across {scores_i2t.shape} samples. ")
        prec_at_1 = AverageMeter()
        prec_at_5 = AverageMeter()

        # Text retrieval
        tqdm_iterator = tqdm(range(len(self.img2txt)))
        for i in tqdm_iterator:
            top5_captions = np.argsort(scores_i2t[i])[-5:]
            true_captions = self.img2txt[i]

            prec_at_1.update(len(set(true_captions) & set(top5_captions[-1:]))>0)
            prec_at_5.update(len(set(true_captions) & set(top5_captions))>0)

            tqdm_iterator.set_description(f"Text Retrieval Prec@1: {prec_at_1.avg:.3f}, Prec@5: {prec_at_5.avg:.3f}")

        # Image Retrieval
        image_prec_at_1 = AverageMeter()
        image_prec_at_5 = AverageMeter()

        tqdm_iterator = tqdm(range(len(self.txt2img)))
        for i in tqdm_iterator:
            top5_images = np.argsort(scores_t2i[:, i])[-5:]
            true_image = self.txt2img[i]

            image_prec_at_1.update(true_image in top5_images[-1:])
            image_prec_at_5.update(true_image in top5_images)

            tqdm_iterator.set_description(f"Image Retrieval Prec@1: {image_prec_at_1.avg:.3f}, Prec@5: {image_prec_at_5.avg:.3f}")

        records = [{"ImagePrec@1": image_prec_at_1.avg, "ImagePrec@5": image_prec_at_5.avg, "TextPrec@1": prec_at_1.avg, "TextPrec@5": prec_at_5.avg}]
        return records



class Flickr30k_Retrieval(Dataset):
    def __init__(self, image_preprocess, split, root_dir=FLICKR_ROOT, max_words=30,
                 image_perturb_fn=None, *args, **kwargs):  
        '''
        Flickr30k dataset for retrieval.
        image_preprocess: image preprocessing function
        root_dir: The directory of the coco dataset. This directory should contain test2014 files.
        max_words: Cropping the caption to max_words.
        split: 'val' or 'test'
        image_perturb_fn: image perturbation function for patch permutation experiments.
        download: Whether to download the dataset if it does not exist.
        '''
        urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json',
                'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json'}
        filenames = {'val':'flickr30k_val.json','test':'flickr30k_test.json'}
        
        if not os.path.exists(root_dir):
            print("Directory for Flickr30k could not be found!")
            flickr_url = "https://forms.illinois.edu/sec/229675"
            raise RuntimeError(f"You need to manually sign up and download the dataset from {flickr_url} and place it in the `root_dir`.")
        
        download_url(urls[split],root_dir)
        
        self.annotation = json.load(open(os.path.join(root_dir,filenames[split]),'r'))
        self.image_preprocess = image_preprocess
        self.image_perturb_fn = image_perturb_fn
        self.root_dir = root_dir
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        image_path = os.path.join(self.root_dir, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')   
        if self.image_preprocess is not None: 
            image = self.image_preprocess(image)  
        if self.image_perturb_fn is not None:
            image = self.image_perturb_fn(image) 
        
        return {"image": image, "idx": index}
    
    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
    
        else:
            scores_t2i = scores
            scores_i2t = scores

        print(f"Flickr30k Retrieval results across {scores_i2t.shape} samples. ")
        prec_at_1 = AverageMeter()
        prec_at_5 = AverageMeter()

        # Text retrieval
        tqdm_iterator = tqdm(range(len(self.img2txt)))
        for i in tqdm_iterator:
            top5_captions = np.argsort(scores_i2t[i])[-5:]
            true_captions = self.img2txt[i]

            prec_at_1.update(len(set(true_captions) & set(top5_captions[-1:]))>0)
            prec_at_5.update(len(set(true_captions) & set(top5_captions))>0)

            tqdm_iterator.set_description(f"Text Retrieval Prec@1: {prec_at_1.avg:.3f}, Prec@5: {prec_at_5.avg:.3f}")

        # Image Retrieval
        image_prec_at_1 = AverageMeter()
        image_prec_at_5 = AverageMeter()

        tqdm_iterator = tqdm(range(len(self.txt2img)))
        for i in tqdm_iterator:
            top5_images = np.argsort(scores_t2i[:, i])[-5:]
            true_image = self.txt2img[i]

            image_prec_at_1.update(true_image in top5_images[-1:])
            image_prec_at_5.update(true_image in top5_images)

            tqdm_iterator.set_description(f"Image Retrieval Prec@1: {image_prec_at_1.avg:.3f}, Prec@5: {image_prec_at_5.avg:.3f}")

        records = [{"ImagePrec@1": image_prec_at_1.avg, "ImagePrec@5": image_prec_at_5.avg, "TextPrec@1": prec_at_1.avg, "TextPrec@5": prec_at_5.avg}]
        return records
    
    def download(self):
        raise NotImplementedError("Flickr30k dataset is not available for download.")



class Cola_Multi(Dataset):
    def __init__(self, image_preprocess, root_dir):
        self.root_dir = f'../../../data/raw/GQA/images'
        self.dataset = json.load(open(f"../diffusion-itm/COLA/data/COLA_multiobjects_matching_benchmark.json", 'r'))
        self.image_preprocess = image_preprocess
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        ex = self.dataset[idx]
        cap0 = ex[1]
        cap1 = ex[3]
        img0 =  os.path.join(self.root_dir,ex[0].split('/')[-1])
        img1 = os.path.join(self.root_dir,ex[2].split('/')[-1])
        
        img0 = Image.open(img0).convert('RGB')
        img1 = Image.open(img1).convert('RGB')
        if self.image_preprocess is not None: 
            image0 = self.image_preprocess(img0)
            image1 = self.image_preprocess(img1)
            
        text = [cap0, cap1]

        item = edict({"image_options": [image0, image1], "caption_options": text})
        return item
        # if self.scoring_only:
        #     return text, idx
        # else:
        #     return (0, [img0_resize, img1_resize]), text, idx

    def evaluate_scores(self, scores):
        # print(scores.shape) # (210, 2, 2)
        # exit(0)
        if isinstance(scores, tuple):
            # print("true")
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
        else:
            # print("false") # here
            scores_t2i = scores
            scores_i2t = scores
        # exit(0)
        ground_truth_i2t = np.array([0,1] * (len(scores_i2t)))
        ground_truth_t2i = np.array([0,1] * (len(scores_t2i)))

        predicted_i2t = scores_i2t.argmax(axis=-1)

        predicted_t2i = scores_t2i.argmax(axis=1)
        correct_i2t = np.all(predicted_i2t == ground_truth_i2t.reshape(predicted_i2t.shape), axis=-1)
        correct_t2i = np.all(predicted_t2i == ground_truth_t2i.reshape(predicted_t2i.shape), axis=-1)
        # accuracy = correct_predictions.mean()
        result_records = [{"Accuracy i2t": np.mean(correct_i2t), "Accuracy t2i": np.mean(correct_t2i)}]
        return result_records

class SugarCrepe(Dataset):
    def __init__(self, image_preprocess, root_dir, resize=512, tokenizer=None, split = 'add_att'):
        self.root_dir = '../../../data/raw/COCO2017/val2017'
        self.resize = resize
        self.split = split
        # self.data = json.load(open(f"../diffusion-itm/sugar-crepe/data/{self.split}.json", 'r'))
        if self.split == 'att' or self.split == 'obj':
            data1 = json.load(open(f"/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/sugar-crepe/data/add_{self.split}.json", 'r'))
            data2 = json.load(open(f"/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/sugar-crepe/data/replace_{self.split}.json", 'r'))
            data3 = json.load(open(f"/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/sugar-crepe/data/swap_{self.split}.json", 'r'))
            self.data = {**data1, **data2, **data3}
        elif self.split == 'rel':
            self.data = json.load(open(f"/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/sugar-crepe/data/replace_{self.split}.json", 'r'))
        else:
            self.data = json.load(open(f"/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/diffusion-itm/sugar-crepe/data/{self.split}.json", 'r'))
        
        self.image_preprocess = image_preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not (self.split == 'swap_obj' and idx ==108):
            row = self.data[str(idx)]

            img_path = row['filename']
            # only get filename
            # img_path = img_path.split('/')[-1]
            img_path = f"{self.root_dir}/{img_path}"

            img = Image.open(img_path).convert("RGB")
            if self.image_preprocess is not None: 
                img = self.image_preprocess(img)
            item = edict({"image_options": [img], "caption_options": [row['caption'],row['negative_caption']]})
            return item
            # return [img_path, imgs], [row['caption'],row['negative_caption']], 0
        else:
            return self.__getitem__(idx+1)
    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T
        else:  
            scores_t2i = scores
            scores_i2t = scores
        
        preds = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_mask = (preds == 0)
        records = [{"Precision@1": np.mean(correct_mask)}]
        return records


class PETS(Dataset):
    def __init__(self, image_preprocess, root_dir, resize=512, scoring_only=False):
        root_dir =  "/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/raw/oxford-iiit-pet"
        self.root_dir = root_dir
        # read all imgs in root_dir with glob
        imgs = list(glob(root_dir + '/images/*.jpg'))
        self.resize = resize
        self.image_preprocess = image_preprocess
        self.classes = list(open(f'{self.root_dir}/classes.txt', 'r').read().splitlines())
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
        self.targets = [self.data[i][1] for i in range(len(self.data))]
        # print(self.targets)
        # exit(0)
    def __getitem__(self, idx):
        if not self.scoring_only:
            img, class_id = self.data[idx]
            img = Image.open(img)
            img = img.convert("RGB")
            if self.image_preprocess is not None:
                image = self.image_preprocess(img)
        else:
            class_id = idx // 50
        # print(class_id)
        # if self.scoring_only:
        #     return self.classes, class_id
        # else:
        item = edict({"image_options": [image], "caption_options": self.classes})
        return item
        # return [0, [img_resize]], self.classes, class_id

    def __len__(self):
        return len(self.data)
    

    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            # print("true")
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
        else:
            # print("false") # here
            scores_t2i = scores
            scores_i2t = scores
        
        predicted_i2t = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        # predicted_t2i = scores_t2i.argmax(axis=1)
        # print(predicted_i2t)
        # input()
        # print(self.targets)
        correct_i2t = (predicted_i2t == self.targets)

        result_records = [{"Accuracy t2i": np.mean(correct_i2t)}]
        return result_records

class VisMin(Dataset):
    def __init__(self, image_preprocess, root_dir, resize=512, scoring_only=False,split=None):
        self.root_dir = '../../../VisMin/images/train'
        self.image_preprocess = image_preprocess
        self.split = split
        self.dataset =  json.load(open(f"../diffusion-itm/vismin/vismin.json", 'r'))
        if self.split is not None:
            # Filter the dataset to include only entries where the 'category' matches the split
            self.dataset = [item for item in self.dataset if item['category'] == self.split]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        sample = self.dataset[idx]
        img0, img1 = sample["pos_image_path"], sample["neg_image_path"]
        if not (img0 == "../../../data/raw/VisMin/images/train/8f93d0cd.png" or img0 == "../../../data/raw/VisMin/images/train/380e1323.png" or img0=="../../../data/raw/VisMin/images/train/3b3072a0.png"):
            cap0, cap1 = sample["pos_caption"], sample["neg_caption"]

            img0 = Image.open(img0).convert('RGB')
            img1 = Image.open(img1).convert('RGB')

            if self.image_preprocess is not None: 
                img0 = self.image_preprocess(img0)
                img1 = self.image_preprocess(img1)

            text = [cap0, cap1]
            # return (0, [img0_resize, img1_resize]), text, idx
            item = edict({"image_options": [img0, img1], "caption_options": text})
            return item
        else:
            return self.__getitem__(idx+1)
    def evaluate_scores(self, scores):
        # print(scores.shape) # (210, 2, 2)
        # exit(0)
        if isinstance(scores, tuple):
            # print("true")
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
        else:
            # print("false") # here
            scores_t2i = scores
            scores_i2t = scores
        # exit(0)
        ground_truth_i2t = np.array([0,1] * (len(scores_i2t)))
        ground_truth_t2i = np.array([0,1] * (len(scores_t2i)))

        predicted_i2t = scores_i2t.argmax(axis=-1)

        predicted_t2i = scores_t2i.argmax(axis=1)
        correct_i2t = np.all(predicted_i2t == ground_truth_i2t.reshape(predicted_i2t.shape), axis=-1)
        correct_t2i = np.all(predicted_t2i == ground_truth_t2i.reshape(predicted_t2i.shape), axis=-1)
        # accuracy = correct_predictions.mean()
        result_records = [{"Accuracy i2t": np.mean(correct_i2t), "Accuracy t2i": np.mean(correct_t2i)}]
        return result_records

class VisMin_split(Dataset):
    def __init__(self, image_preprocess, root_dir, resize=512, scoring_only=False,split=None):
        # self.root_dir = '../../../VisMin/images/train'
        self.image_preprocess = image_preprocess
        self.split = split
        # self.dataset =  json.load(open(f"../diffusion-itm/vismin/vismin.json", 'r'))
        # if self.split is not None:
        #     # Filter the dataset to include only entries where the 'category' matches the split
        #     self.dataset = [item for item in self.dataset if item['category'] == self.split]

        with open("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/hf_token.txt", "r") as f:
                    hf_token = f.read().strip()
        try:
            dataset = load_dataset('mair-lab/vismin-bench', use_auth_token=hf_token)
        except:
            dataset = load_dataset('mair-lab/vismin-bench', token=hf_token)
        # dataset = load_dataset("mair-lab/vismin-bench")
        dataset = dataset['test']
        if self.split != None:
            self.dataset = dataset.filter(lambda x: x['category'] == self.split)
        else:
            self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        sample = self.dataset[idx]
        # img0, img1 = sample["pos_image_path"], sample["neg_image_path"]
        img0, img1 = sample["image_0"], sample["image_1"]
        if not (img0 == "../../../data/raw/VisMin/images/train/8f93d0cd.png" or img0 == "../../../data/raw/VisMin/images/train/380e1323.png" or img0=="../../../data/raw/VisMin/images/train/3b3072a0.png"):
            # cap0, cap1 = sample["pos_caption"], sample["neg_caption"]
            cap0, cap1 = sample["text_0"], sample["text_1"]

            img0 = img0.convert('RGB')
            img1 = img1.convert('RGB')
            # img0 = Image.open(img0).convert('RGB')
            # img1 = Image.open(img1).convert('RGB')

            if self.image_preprocess is not None: 
                img0 = self.image_preprocess(img0)
                img1 = self.image_preprocess(img1)

            text = [cap0, cap1]
            # return (0, [img0_resize, img1_resize]), text, idx
            item = edict({"image_options": [img0, img1], "caption_options": text})
            return item
        else:
            return self.__getitem__(idx+1)
    def evaluate_scores(self, scores):
        # print(scores.shape) # (210, 2, 2)
        # exit(0)
        if isinstance(scores, tuple):
            # print("true")
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
        else:
            # print("false") # here
            scores_t2i = scores
            scores_i2t = scores
        # exit(0)
        ground_truth_i2t = np.array([0,1] * (len(scores_i2t)))
        ground_truth_t2i = np.array([0,1] * (len(scores_t2i)))

        predicted_i2t = scores_i2t.argmax(axis=-1)

        predicted_t2i = scores_t2i.argmax(axis=1)
        correct_i2t = np.all(predicted_i2t == ground_truth_i2t.reshape(predicted_i2t.shape), axis=-1)
        correct_t2i = np.all(predicted_t2i == ground_truth_t2i.reshape(predicted_t2i.shape), axis=-1)
        # accuracy = correct_predictions.mean()
        result_records = [{"Accuracy i2t": np.mean(correct_i2t), "Accuracy t2i": np.mean(correct_t2i)}]
        return result_records

class WhatsUp(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=None, download=False, split='A', resize=512):
        root_dir = '../../../data/raw/whatsup'
        self.root_dir = root_dir
        if split == 'A':
            annotation_file = os.path.join(root_dir, "controlled_images_dataset.json")
            image_dir = os.path.join(root_dir, 'controlled_images')

            if not os.path.exists(image_dir):
                print("Image directory for Controlled Images A could not be found!")
                if download:
                    self.download()
                else:
                    raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")

            if not os.path.exists(annotation_file):
                subprocess.call(["gdown", "--id", "1ap8mmmpQjLIjPGuplkpBgc1hoEHCj4hm", "--output", annotation_file])

        else:
            annotation_file = os.path.join(root_dir, "controlled_clevr_dataset.json")
            image_dir = os.path.join(root_dir, 'controlled_clevr')
            if not os.path.exists(image_dir):
                print("Image directory for Controlled Images B could not be found!")
                if download:
                    self.download()
                else:
                    raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")

            if not os.path.exists(annotation_file):
                subprocess.call(["gdown", "--id", "1unNNosLbdy9NDjgj4l8fsQP3WiAAGA6z", "--output", annotation_file])


        self.dataset = json.load(open(annotation_file))
        self.split = split
        self.all_prepositions = []
        if self.split == 'A':
            for d in self.dataset:
                if 'left_of' in d['image_path']:
                    self.all_prepositions.append('left_of')
                elif 'right_of' in d['image_path']:
                    self.all_prepositions.append('right_of')
                elif '_on_' in d['image_path']:
                    self.all_prepositions.append('on')
                else:
                    self.all_prepositions.append('under')
            self.eval_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                {'left': 0, 'right': 0, \
                                'on': 0, 'under': 0} for d in self.dataset}
            self.pred_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                {'left': '', 'right': '', \
                                'on': '', 'under': ''} for d in self.dataset}


        else:
            for d in self.dataset:
                if 'left_of' in d['image_path']:
                    self.all_prepositions.append('left_of')
                elif 'right_of' in d['image_path']:
                    self.all_prepositions.append('right_of')
                elif '_in-front_of_' in d['image_path']:
                    self.all_prepositions.append('in-front_of')
                else:
                    self.all_prepositions.append('behind')
            self.eval_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                {'left': 0, 'right': 0, \
                                'in-front': 0, 'behind': 0} for d in self.dataset}
            self.pred_dict = {(d['image_path'].split('/')[-1].split('_')[0], \
                                d['image_path'].split('/')[-1].split('_')[-1][:-5]): \
                                {'left': '', 'right': '', \
                                'in-front': '', 'behind': ''} for d in self.dataset}

        self.image_preprocess = image_preprocess
        self.resize = resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image_path = test_case["image_path"].replace("data", self.root_dir)
        image = Image.open(image_path).convert('RGB')
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        # else:
        #     img0_resize = image.resize((self.resize, self.resize))
        #     img0_resize = diffusers_preprocess(img0_resize)
        
        item = edict({"image_options": [image], "caption_options": test_case['caption_options']})
        # return [image_path, [img0_resize]], test_case["caption_options"], 0
        return item

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
            if self.split == 'A':
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
                "Dataset": "Controlled Images - {}".format(self.split)
            })
        result_records.append({"Preposition": "All", "Accuracy": metrics['Accuracy'], "Count": len(all_prepositions), "Dataset": "Controlled Images - {}".format(self.split)})
        return result_records


class COCO_QA(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=None, download=False, split='one',resize=512):
        root_dir = '../../../data/raw/coco_qa'
        self.root_dir = root_dir
        if split == 'one':
            annotation_file = os.path.join(root_dir, "coco_qa_one_obj.json")
            image_dir = os.path.join(root_dir, 'val2017')
        else:
            annotation_file = os.path.join(root_dir, "coco_qa_two_obj.json")
            image_dir = os.path.join(root_dir, 'val2017')
        if not os.path.exists(image_dir):
            print("Image directory for COCO-QA could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")

        if not os.path.exists(annotation_file):
            if split == 'one':
                subprocess.call(["gdown", "--id", "1RsMdpE9mmwnK4zzMPpC1-wTU_hNis-dq", "--output", annotation_file])
            else:
                subprocess.call(["gdown", "--id", "1TCEoM0mgFmz8T4cF7PQ3XJmO6JjtiQ-s", "--output", annotation_file])


        self.dataset = json.load(open(annotation_file))
        self.split = split
        self.all_prepositions = []
        if self.split == 'one':
            self.all_prepositions = [d[1].split()[-1] for d in self.dataset]
        else:
            for d in self.dataset:
                if 'left of' in d[1]:
                    self.all_prepositions.append('left')
                elif 'right of' in d[1]:
                    self.all_prepositions.append('right')
                elif 'above' in d[1]:
                    self.all_prepositions.append('above')
                else:
                    self.all_prepositions.append('below')
        self.image_preprocess = image_preprocess
        self.resize = resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image_path = os.path.join(self.root_dir, 'val2017/{}.jpg'.format(str(test_case[0]).zfill(12)))
        image = Image.open(image_path).convert('RGB')
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        # else:
        #     img0_resize = image.resize((self.resize, self.resize))
        #     img0_resize = diffusers_preprocess(img0_resize)
        
        item = edict({"image_options": [image], "caption_options": [test_case[1], test_case[2]]})
        # return [image_path, [img0_resize]], [test_case[1], test_case[2]], 0
        return item
    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "val2017.zip")
        subprocess.call(["gdown", "--no-cookies",  "1zp5vBRRM4_nSik6o9PeVspDvOsHgPT4l", "--output", image_zip_file])
        subprocess.call(["unzip", "val2017.zip"], cwd=self.root_dir)

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
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
        print(metrics['Accuracy']*100)

        all_prepositions = np.array(self.all_prepositions)

        prepositions = list(set(self.all_prepositions))
        prep_counts = {p: {p1: 0 for p1 in prepositions} for p in prepositions}
        opposite = {'left': 'right', 'right': 'left', 'above': 'below', 'below': 'above', 'top': 'bottom', 'bottom': 'top'}

        for prep, pred in zip(self.all_prepositions, preds):
            if pred == 0:
                prep_counts[prep][prep] += 1
            else:
                prep_counts[prep][opposite[prep]] += 1
        #print(prep_counts)
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
                "Dataset": "COCO-QA {}-object".format(self.split)
            })
        result_records.append({"Preposition": "All", "Accuracy": metrics['Accuracy'], "Count": len(all_prepositions), "Dataset": "COCO-QA {}-object".format(self.split)})
        return result_records

class VG_QA(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=None, download=False, split='one', resize=512):
        root_dir = '../../../data/raw/VG_QA'
        self.root_dir = root_dir
        if split == 'one':
            annotation_file = os.path.join(root_dir, "vg_qa_one_obj.json")
            image_dir = os.path.join(root_dir, 'vg_images')
        else:
            annotation_file = os.path.join(root_dir, "vg_qa_two_obj.json")
            image_dir = os.path.join(root_dir, 'vg_images')
        if not os.path.exists(image_dir):
            print("Image directory for VG-QA could not be found!")
            if download:
                self.download()
            else:
                raise RuntimeError("Please either download the dataset by letting `--download` or specify the correct directory.")

        if not os.path.exists(annotation_file):
            if split == 'one':
                subprocess.call(["gdown", "--id", "1ARMRzRdohs9QTr1gpIfzyUzvW20wYp_p", "--output", annotation_file])
            else:
                subprocess.call(["gdown", "--id", "1sjVG5O3QMY8s118k7kQM8zzDZH12i_95", "--output", annotation_file])


        self.dataset = json.load(open(annotation_file))
        self.split = split
        self.all_prepositions = []
        if self.split == 'one':
            self.all_prepositions = [d[1].split()[-1] for d in self.dataset]
        else:
            for d in self.dataset:
                if 'left of' in d[1]:
                    self.all_prepositions.append('left')
                elif 'right of' in d[1]:
                    self.all_prepositions.append('right')
                elif 'front of' in d[1]:
                    self.all_prepositions.append('front')
                elif 'behind' in d[1]:
                    self.all_prepositions.append('behind')
                else:
                    self.all_prepositions.append('top')
        self.image_preprocess = image_preprocess
        self.resize = resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]
        image_path = os.path.join(self.root_dir, 'vg_images/{}.jpg'.format(test_case[0]))
        image = Image.open(image_path).convert('RGB')
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        # else:
        #     img0_resize = image.resize((self.resize, self.resize))
        #     img0_resize = diffusers_preprocess(img0_resize)
        
        item = edict({"image_options": [image], "caption_options": [test_case[1], test_case[2]]})
        # return [image_path, [img0_resize]], [test_case[1], test_case[2]], 0
        return item

    def download(self):
        os.makedirs(self.root_dir, exist_ok=True)
        image_zip_file = os.path.join(self.root_dir, "vg_images.tar.gz")
        subprocess.call(["gdown", "--no-cookies",  "1idW7Buoz7fQm4-670n-oERw9U-2JLJvE", "--output", image_zip_file])
        subprocess.call(["tar", "-xvf", "vg_images.tar.gz"], cwd=self.root_dir)


    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
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
        print(metrics['Accuracy']*100)

        all_prepositions = np.array(self.all_prepositions)
        
        prepositions = list(set(self.all_prepositions)) + ['below', 'bottom', 'front']
        prep_counts = {p: {p1: 0 for p1 in prepositions} for p in prepositions}
        opposite = {'left': 'right', 'right': 'left', 'behind': 'front', 'front': 'behind', 'above': 'below', 'below': 'above', 'bottom': 'top', 'top': 'bottom'}

        for prep, pred in zip(self.all_prepositions, preds):
            if pred == 0:
                prep_counts[prep][prep] += 1
            else:
                prep_counts[prep][opposite[prep]] += 1
        #print(prep_counts)
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
                "Dataset": "VG-QA {}-object".format(self.split)
            })
        result_records.append({"Preposition": "All", "Accuracy": metrics['Accuracy'], "Count": len(all_prepositions), "Dataset": "VG-QA {}-object".format(self.split)})
        return result_records

DATASET_ROOT = os.getenv('DATASET_ROOT', '/pfss/mlde/workspaces/mlde_wsp_Rohrbach/data/raw')

class MNIST(datasets.MNIST):
    """Simple subclass to override the property"""
    # class_to_idx = {str(i): i for i in range(10)}
    
    def __init__(self, image_preprocess=None, root_dir=None, *args, **kwargs):
        super(MNIST, self).__init__(root=DATASET_ROOT, train=False, transform=None, target_transform=None,
                        download=True, *args, **kwargs)
        self.root_dir = root_dir
        self.image_preprocess = image_preprocess
        
    def __getitem__(self, index):
        image, label = super(MNIST, self).__getitem__(index)

        # caption = [f"a photo of the number: '{i}'" for i in range(10)]
        # caption = [f"A rough white handwritten number '{i}' on a black canvas." for i in range(10)]
        # caption = [f"A rough black handwritten number '{i}' on a white canvas." for i in range(10)]
        
        # caption = [f"A freeform handwritten black digit '{i}' on a white surface." for i in range(10)]
        
        # caption = [f"A white background with a black handwritten number: '{i}'" for i in range(10)]

        caption = [f"A black background with a white handwritten number: '{i}'" for i in range(10)]
        # A photo of the number zero on a black background.
        number_to_text = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
        # caption = [f"A photo of the number {number_to_text[i]} on a black background." for i in range(10)]
        # Apply any necessary transformations to the image and label
        # For example, you could convert the label to a one-hot encoded vector
        # or apply additional preprocessing to the image
        
        # Return the processed image and label
        # return image, label
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        item = edict({"image_options": [image], "caption_options": caption})
        return item

    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            # print("true")
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
        else:
            # print("false") # here
            scores_t2i = scores
            scores_i2t = scores
        
        predicted_i2t = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        # predicted_t2i = scores_t2i.argmax(axis=1)
        # print(predicted_i2t)
        # input()
        # print(self.targets)
        correct_i2t = (predicted_i2t == self.targets.numpy())

        result_records = [{"Accuracy t2i": np.mean(correct_i2t)}]
        return result_records

class CIFAR100(datasets.CIFAR100):
    """Simple subclass to override the property"""
    # class_to_idx = {str(i): i for i in range(10)}
    
    def __init__(self, image_preprocess=None, root_dir=None, split = 'action-replacement', *args, **kwargs):
        super(CIFAR100, self).__init__(root=DATASET_ROOT, train=False, transform=None, target_transform=None,
                        download=False, *args, **kwargs)
        self.root_dir = root_dir
        self.image_preprocess = image_preprocess
        # print(self.classes)
        # self.class_to_idx = {j: i for i,j in enumerate(self.classes)}
        # print(list(self.class_to_idx.values()))
    def __getitem__(self, index):
        image, label = super(CIFAR100, self).__getitem__(index)

        caption = [f"a photo of {i}" for i in self.classes]
        # Apply any necessary transformations to the image and label
        # For example, you could convert the label to a one-hot encoded vector
        # or apply additional preprocessing to the image
        
        # Return the processed image and label
        # return image, label
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        item = edict({"image_options": [image], "caption_options": caption})
        return item
    def evaluate_scores(self, scores):
        if isinstance(scores, tuple):
            # print("true")
            scores_i2t = scores[0]
            scores_t2i = scores[1].T
        else:
            # print("false") # here
            scores_t2i = scores
            scores_i2t = scores
        
        predicted_i2t = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)

        correct_i2t = (predicted_i2t == self.targets)

        result_records = [{"Accuracy t2i": np.mean(correct_i2t)}]
        return result_records

class VALSE(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=None, download=False, split='A', resize=512):
        # root_dir = '../../../data/raw/SWiG/images_512'
        self.root_dir = root_dir
        self.split = split
        with open(f"../diffusion-itm/VALSE/data/{self.split}.json", "r") as f:
            self.dataset = json.load(f)

        self.all_prepositions = []

        self.image_preprocess = image_preprocess
        self.resize = resize
        self.keys = list(self.dataset.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        key = self.keys[index]
        test_case = self.dataset[key]
        image_path = os.path.join(self.root_dir, test_case["image_file"])
        image = Image.open(image_path).convert('RGB')
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        # else:
        #     img0_resize = image.resize((self.resize, self.resize))
        #     img0_resize = diffusers_preprocess(img0_resize)
        
        item = edict({"image_options": [image], "caption_options": [test_case['caption'], test_case['foil']]})
        # return [image_path, [img0_resize]], [test_case[1], test_case[2]], 0
        return item

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        predicted_i2t = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_i2t = (predicted_i2t == 0)

        result_records = [{"Accuracy t2i": np.mean(correct_i2t)}]
        return result_records

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VLCheck_Relation(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=None, download=False, split='A', resize=512):
        # root_dir = '../../../data/raw/SWiG/images_512'
        self.root_dir = root_dir
        self.split = split
        if self.split == 'hake':
            self.dataset = [
                item for item in json.load(open(f"../diffusion-itm/VL-CheckList/data/Relation/{self.split}_action.json", 'r'))
                if 'pic/' not in item[0]
            ] 
        elif self.split == 'swig':
            self.dataset = json.load(open(f"../diffusion-itm/VL-CheckList/data/Relation/{self.split}_action.json", 'r'))
            # I want self.dataset filtering by '/pic/' not in self.dataset[idx][0]

        elif 'vg' in self.split:
            split = self.split.replace('vg_', '')
            self.dataset = json.load(open(f"../diffusion-itm/VL-CheckList/data/Relation/vg/{split}.json", 'r'))
        
        self.all_prepositions = []

        self.image_preprocess = image_preprocess
        self.resize = resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:

            test_case = self.dataset[index]
            if self.split == 'hake':
                if 'coco' not in test_case[0]:
                    image_path =  os.path.join(self.root_dir,test_case[0])
                else:
                    image_path = os.path.join(self.root_dir,test_case[0])
                    image_path = image_path.replace('HAKE/vcoco','COCO2014')
            
            else:
                image_path =  os.path.join(self.root_dir,test_case[0].split('/')[-1])
            # image_path = os.path.join(self.root_dir, test_case["image_file"])
            image = Image.open(image_path).convert('RGB')
            if self.image_preprocess is not None:
                image = self.image_preprocess(image)
            # else:
            #     img0_resize = image.resize((self.resize, self.resize))
            #     img0_resize = diffusers_preprocess(img0_resize)
            
            item = edict({"image_options": [image], "caption_options": [test_case[1]["POS"][0], test_case[1]["NEG"][0]]})
            # return [image_path, [img0_resize]], [test_case[1], test_case[2]], 0
            return item
        except FileNotFoundError:
            if index+1 == len(self.dataset):
                return self.__getitem__(0)
            return self.__getitem__(index+1)

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        predicted_i2t = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_i2t = (predicted_i2t == 0)

        result_records = [{"Accuracy t2i": np.mean(correct_i2t)}]
        return result_records


class VLCheck_Object_Size(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=None, download=False, split='A', resize=512):
        # root_dir = '../../../data/raw/SWiG/images_512'
        self.root_dir = root_dir
        self.split = split
        if self.split == 'hake':
            self.dataset = [] 
            for i in os.listdir(f"../diffusion-itm/VL-CheckList/data/Object/Size/{self.split}_size"):
                # dataset = json.load(open(f"./VL-CheckList/data/Object/Size/{self.split}_size/{i}", 'r'))
                dataset = [
                item for item in json.load(open(f"../diffusion-itm/VL-CheckList/data/Object/Size/{self.split}_size/{i}", 'r'))
                if 'pic/' not in item[0]
            ] 
                self.dataset += dataset
        elif self.split == 'swig':
            self.dataset = []
            for i in os.listdir(f"../diffusion-itm/VL-CheckList/data/Object/Size/{self.split}_size"):
                for j in os.listdir(f"../diffusion-itm/VL-CheckList/data/Object/Size/{self.split}_size/{i}"):
                    dataset = json.load(open(f"../diffusion-itm/VL-CheckList/data/Object/Size/{self.split}_size/{i}/{j}", 'r'))
                    self.dataset += dataset
        elif self.split == 'vg':
            self.dataset = []
            for i in os.listdir(f"../diffusion-itm/VL-CheckList/data/Object/Size/{self.split}_size"):
                dataset = json.load(open(f"../diffusion-itm/VL-CheckList/data/Object/Size/{self.split}_size/{i}", 'r'))
                self.dataset += dataset

        self.all_prepositions = []

        self.image_preprocess = image_preprocess
        self.resize = resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:

            test_case = self.dataset[index]
            if self.split == 'hake':
                if 'coco' not in test_case[0]:
                    image_path =  os.path.join(self.root_dir,test_case[0])
                else:
                    image_path = os.path.join(self.root_dir,test_case[0])
                    image_path = image_path.replace('HAKE/vcoco','COCO2014')
            
            else:
                image_path =  os.path.join(self.root_dir,test_case[0].split('/')[-1])
            # image_path = os.path.join(self.root_dir, test_case["image_file"])
            image = Image.open(image_path).convert('RGB')
            if self.image_preprocess is not None:
                image = self.image_preprocess(image)
            # else:
            #     img0_resize = image.resize((self.resize, self.resize))
            #     img0_resize = diffusers_preprocess(img0_resize)
            
            item = edict({"image_options": [image], "caption_options": [test_case[1]["POS"][0], test_case[1]["NEG"][0]]})
            # return [image_path, [img0_resize]], [test_case[1], test_case[2]], 0
            return item
        except FileNotFoundError:
            if index+1 == len(self.dataset):
                return self.__getitem__(0)
            return self.__getitem__(index+1)

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        predicted_i2t = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_i2t = (predicted_i2t == 0)

        result_records = [{"Accuracy t2i": np.mean(correct_i2t)}]
        return result_records

class VLCheck_Object_Location(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=None, download=False, split='A', resize=512):
        # root_dir = '../../../data/raw/SWiG/images_512'
        self.root_dir = root_dir
        self.split = split
        if self.split == 'hake':
            self.dataset = [] 
            for i in os.listdir(f"../diffusion-itm/VL-CheckList/data/Object/Location/{self.split}_location"):
                dataset = [
                item for item in json.load(open(f"../diffusion-itm/VL-CheckList/data/Object/Location/{self.split}_location/{i}", 'r'))
                if 'pic/' not in item[0]
            ] 
                # dataset = json.load(open(f"./VL-CheckList/data/Object/Location/{self.split}_location/{i}", 'r'))
                self.dataset += dataset
        elif self.split == 'swig':
            self.dataset = []
            for i in os.listdir(f"../diffusion-itm/VL-CheckList/data/Object/Location/{self.split}_location"):
                for j in os.listdir(f"../diffusion-itm/VL-CheckList/data/Object/Location/{self.split}_location/{i}"):
                    dataset = json.load(open(f"../diffusion-itm/VL-CheckList/data/Object/Location/{self.split}_location/{i}/{j}", 'r'))
                    self.dataset += dataset
        elif self.split == 'vg':
            self.dataset = []
            for i in os.listdir(f"../diffusion-itm/VL-CheckList/data/Object/Location/{self.split}_location"):
                dataset = json.load(open(f"../diffusion-itm/VL-CheckList/data/Object/Location/{self.split}_location/{i}", 'r'))
                self.dataset += dataset

        self.all_prepositions = []

        self.image_preprocess = image_preprocess
        self.resize = resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:

            test_case = self.dataset[index]
            if self.split == 'hake':
                if 'coco' not in test_case[0]:
                    image_path =  os.path.join(self.root_dir,test_case[0])
                else:
                    image_path = os.path.join(self.root_dir,test_case[0])
                    image_path = image_path.replace('HAKE/vcoco','COCO2014')
            
            else:
                image_path =  os.path.join(self.root_dir,test_case[0].split('/')[-1])
            # image_path = os.path.join(self.root_dir, test_case["image_file"])
            image = Image.open(image_path).convert('RGB')
            if self.image_preprocess is not None:
                image = self.image_preprocess(image)
            # else:
            #     img0_resize = image.resize((self.resize, self.resize))
            #     img0_resize = diffusers_preprocess(img0_resize)
            
            item = edict({"image_options": [image], "caption_options": [test_case[1]["POS"][0], test_case[1]["NEG"][0]]})
            # return [image_path, [img0_resize]], [test_case[1], test_case[2]], 0
            return item
        except FileNotFoundError:
            if index+1 == len(self.dataset):
                return self.__getitem__(0)
            return self.__getitem__(index+1)

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        predicted_i2t = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_i2t = (predicted_i2t == 0)

        result_records = [{"Accuracy t2i": np.mean(correct_i2t)}]
        return result_records

class VLCheck_Attribute(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=None, download=False, split='A', resize=512):
        # root_dir = '../../../data/raw/SWiG/images_512'
        self.root_dir = root_dir
        self.split = split
        self.dataset = json.load(open(f"../diffusion-itm/VL-CheckList/data/Attribute/vaw/{self.split}.json", 'r'))
        self.dataset1 = json.load(open(f"../diffusion-itm/VL-CheckList/data/Attribute/vg/{self.split}.json", 'r'))
        self.dataset = self.dataset + self.dataset1
        self.all_prepositions = []

        self.image_preprocess = image_preprocess
        self.resize = resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        test_case = self.dataset[index]

        image_path =  os.path.join(self.root_dir,test_case[0].split('/')[-1])
        # image_path = os.path.join(self.root_dir, test_case["image_file"])
        image = Image.open(image_path).convert('RGB')
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        # else:
        #     img0_resize = image.resize((self.resize, self.resize))
        #     img0_resize = diffusers_preprocess(img0_resize)
        
        item = edict({"image_options": [image], "caption_options": [test_case[1]["POS"][0], test_case[1]["NEG"][0]]})
        # return [image_path, [img0_resize]], [test_case[1], test_case[2]], 0
        return item

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        predicted_i2t = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_i2t = (predicted_i2t == 0)

        result_records = [{"Accuracy t2i": np.mean(correct_i2t)}]
        return result_records


from collections import defaultdict

class EQBench(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=None, download=False, split='A', resize=512):
        # root_dir = '../../../data/raw/SWiG/images_512'
        self.root_dir = f"{root_dir}/image_jpg"
        self.split = split

        self.all_prepositions = []
        with open(f"{root_dir}/ann_json_finegrained_random.json", 'r') as f:
            data = [item for item in json.load(f) if (f'{self.split}/' in item['image']) and ('train' not in item['image']) and (item['private_info']['name']=='c0i0' or item['private_info']['name']=='c1i1')]

        grouped_data = defaultdict(list)
        for entry in data:
            # Split the path to get the common base (up to the second last element for grouping)
            base_path = "/".join(entry["image"].split("/")[:-1])
            grouped_data[base_path].append(entry)

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

        self.image_preprocess = image_preprocess
        self.resize = resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]

        image_path1 =  os.path.join(self.root_dir,test_case['image1'])
        image_path2 = os.path.join(self.root_dir,test_case['image2'])
        
        if '.npy' in image_path1:
                img0 = Image.fromarray(np.load(image_path1)[:, :, [2, 1, 0]], 'RGB')
                img1 = Image.fromarray(np.load(image_path2)[:, :, [2, 1, 0]], 'RGB')
        else:
            img0 = image_path1.replace('.png','.jpg')
            img1 = image_path2.replace('.png','.jpg')

            img0 = Image.open(img0).convert('RGB')
            img1 = Image.open(img1).convert('RGB')

        if self.image_preprocess is not None:
            img0 = self.image_preprocess(img0)
            img1 = self.image_preprocess(img1)

        item = edict({"image_options": [img0,img1], "caption_options": [test_case["caption1"], test_case["caption2"]]})
        # return [image_path, [img0_resize]], [test_case[1], test_case[2]], 0
        return item
    
    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
        else:
            # print("false") # here
            scores_t2i = scores
            scores_i2t = scores
        # exit(0)
        ground_truth_i2t = np.array([0,1] * (len(scores_i2t)))
        ground_truth_t2i = np.array([0,1] * (len(scores_t2i)))

        predicted_i2t = scores_i2t.argmax(axis=-1)

        predicted_t2i = scores_t2i.argmax(axis=1)
        correct_i2t = np.all(predicted_i2t == ground_truth_i2t.reshape(predicted_i2t.shape), axis=-1)
        correct_t2i = np.all(predicted_t2i == ground_truth_t2i.reshape(predicted_t2i.shape), axis=-1)
        # accuracy = correct_predictions.mean()
        result_records = [{"Accuracy i2t": np.mean(correct_i2t), "Accuracy t2i": np.mean(correct_t2i)}]
        return result_records


class EQBench_split(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=None, download=False, split='A', resize=512):
        # root_dir = '../../../../mlde_wsp_Shared_Datasets/SWiG/images_512'
        self.root_dir = f"{root_dir}/images"
        self.split = split

        dataset = json.load(open(f"{root_dir}/all_select.json", 'r'))
        self.dataset = [i for i in dataset if self.split in i["image0"]]
        self.image_preprocess = image_preprocess
        self.resize = resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        test_case = self.dataset[index]

        image_path1 =  os.path.join(self.root_dir,test_case['image0'])
        image_path2 = os.path.join(self.root_dir,test_case['image1'])
        
        if '.npy' in image_path1:
                img0 = Image.fromarray(np.load(image_path1)[:, :, [2, 1, 0]], 'RGB')
                img1 = Image.fromarray(np.load(image_path2)[:, :, [2, 1, 0]], 'RGB')
        else:
            img0 = Image.open(image_path1).convert('RGB')
            img1 = Image.open(image_path2).convert('RGB')

        if self.image_preprocess is not None:
            img0 = self.image_preprocess(img0)
            img1 = self.image_preprocess(img1)

        item = edict({"image_options": [img0,img1], "caption_options": [test_case["caption0"], test_case["caption1"]]})
        # return [image_path, [img0_resize]], [test_case[1], test_case[2]], 0
        return item
    
    def evaluate_scores(self, scores):
        """
        Scores: N x 2 x 2
        - scores[:, 0, 0] = similarity (Image A → Text A) # 1
        - scores[:, 0, 1] = similarity (Image A → Text B) # 0
        - scores[:, 1, 1] = similarity (Image B → Text B) # 1
        - scores[:, 1, 0] = similarity (Image B → Text A) # 0

        # Text retrieval: scores[:, 0, 0] > scores[:, 0, 1] and scores[:, 1, 1] > scores[:, 1, 0]
        # Image retrieval: scores[:, 0, 0] > scores[:, 1, 0] and scores[:, 1, 1] > scores[:, 0, 1]


        """
        # how can I see in the score there is a negative value?
        # check it
        
        if isinstance(scores, tuple):
            scores_i2t = scores[1]  # (N, 2, 2)
            scores_t2i = scores[0]  # (N, 2, 2)
        else:
            scores_t2i = scores
            scores_i2t = scores

        # Compute correctness for both queries
        correct_i2t = (scores_i2t[:, 0, 0] > scores_i2t[:, 0, 1]) & (scores_i2t[:, 1, 1] > scores_i2t[:, 1, 0])
        correct_t2i = (scores_t2i[:, 0, 0] > scores_t2i[:, 1, 0]) & (scores_t2i[:, 1, 1] > scores_t2i[:, 0, 1])
        # print(scores)
        # print(correct_i2t)
        # print(correct_t2i)
        result_records = [{"Accuracy i2t": np.mean(correct_i2t), "Accuracy t2i": np.mean(correct_t2i)}]
        # print(result_records)
        # Compute accuracy where both queries must be correct
        return result_records

    # def evaluate_scores(self, scores):
    #     """
    #     Scores: N x 1 x 2, i.e. first caption is right, next is wrong
    #     """
    #     if isinstance(scores, tuple):
    #         scores_i2t = scores[0]
    #         scores_t2i = scores[1].T # Make it N_ims x N_text
    #     else:
    #         # print("false") # here
    #         scores_t2i = scores
    #         scores_i2t = scores
    #     # exit(0)
    #     ground_truth_i2t = np.array([0,1] * (len(scores_i2t)))
    #     ground_truth_t2i = np.array([0,1] * (len(scores_t2i)))

    #     predicted_i2t = scores_i2t.argmax(axis=-1)

    #     predicted_t2i = scores_t2i.argmax(axis=1)
    #     correct_i2t = np.all(predicted_i2t == ground_truth_i2t.reshape(predicted_i2t.shape), axis=-1)
    #     correct_t2i = np.all(predicted_t2i == ground_truth_t2i.reshape(predicted_t2i.shape), axis=-1)
    #     # accuracy = correct_predictions.mean()
    #     result_records = [{"Accuracy i2t": np.mean(correct_i2t), "Accuracy t2i": np.mean(correct_t2i)}]
    #     return result_records

class SPEC(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=None, download=False, split='A', resize=512):
        # root_dir = '../../../data/raw/SWiG/images_512'
        self.root_dir = root_dir
        self.split = split

        self.dataset = json.load(open(f"{self.root_dir}/{self.split}/image2text.json", 'r'))
        self.keys = [i["label"] for i in self.dataset]
        self.image_preprocess = image_preprocess
        self.resize = resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        test_case = self.dataset[index]

        image_path =  os.path.join(self.root_dir,self.split,test_case['query'])
        # image_path = os.path.join(self.root_dir, test_case["image_file"])
        image = Image.open(image_path).convert('RGB')
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
        # else:
        #     img0_resize = image.resize((self.resize, self.resize))
        #     img0_resize = diffusers_preprocess(img0_resize)
        
        item = edict({"image_options": [image], "caption_options": test_case["keys"]})
        # return [image_path, [img0_resize]], [test_case[1], test_case[2]], 0
        return item

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        predicted_i2t = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)

        correct_i2t = (predicted_i2t == self.keys)

        result_records = [{"Accuracy i2t": np.mean(correct_i2t)}] # there was typo of t2i
        return result_records


class SPEC_IMG_RETRIEVAL(Dataset):
    def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=None, download=False, split='A', resize=512):
        # root_dir = '../../../data/raw/SWiG/images_512'
        self.root_dir = root_dir
        self.split = split

        self.dataset = json.load(open(f"{self.root_dir}/{self.split}/text2image.json", 'r'))
        self.keys = [i["label"] for i in self.dataset]
        self.image_preprocess = image_preprocess
        self.resize = resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        test_case = self.dataset[index]

        image_path0 =  os.path.join(self.root_dir,self.split,test_case['keys'][0])
        image_path1 = os.path.join(self.root_dir,self.split,test_case['keys'][1])
        if self.split != 'existence':
            image_path2 = os.path.join(self.root_dir,self.split,test_case['keys'][2])

        image0 = Image.open(image_path0).convert('RGB')
        image1 = Image.open(image_path1).convert('RGB')
        if self.split != 'existence':
            image2 = Image.open(image_path2).convert('RGB')
        if self.image_preprocess is not None:
            image0 = self.image_preprocess(image0)
            image1 = self.image_preprocess(image1)
            if self.split != 'existence':
                image2 = self.image_preprocess(image2)

        if self.split != 'existence':
            item = edict({"image_options": [image0,image1, image2], "caption_options": [test_case["query"]]})
        # return [image_path, [img0_resize]], [test_case[1], test_case[2]], 0
        else :
            item = edict({"image_options": [image0,image1], "caption_options": [test_case["query"]]})
        return item

    def evaluate_scores(self, scores):
        """
        Scores: N x2 x 1, i.e. first image is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        # print(scores_t2i.shape)
        # exit(0)
        predicted_t2i = np.argmax(np.squeeze(scores_t2i, axis=2), axis=-1)

        correct_t2i = (predicted_t2i == self.keys)

        result_records = [{"Accuracy t2i": np.mean(correct_t2i)}]
        return result_records

class Ours(Dataset):
    # def __init__(self, image_preprocess, text_perturb_fn=None, image_perturb_fn=None, root_dir=None, download=False, split='A', resize=512):
    def __init__(self, root_dir, image_preprocess, split, version, before=False, resize=512, scoring_only=False, domain = 'photo'):
        self.version = version
        root_dir = '/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/geneval/outputs'
        self.resize = resize
        self.image_preprocess = image_preprocess
        self.scoring_only = scoring_only
        self.split = split
        self.domain = domain
        self.before = before
        self.examples = []
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

        self.prompts = prompts[prompts['tag']==self.split]

        for i in self.prompts['filename']:
            self.examples.append(os.path.join(self.root_dir, i))
        if self.split == 'color_attr' or self.split == 'colors':
            choice = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"]
            self.choice = [self.with_article(i) for i in choice]
        elif self.split == 'position':
            self.choice = ["left of", "right of", "above", "below"]
        elif self.split == 'counting':
            self.choice = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        else:
            raise ValueError("Invalid split")
    
        color_pattern = re.compile(r'\b(' + '|'.join(self.choice) + r')\b')
        self.all_text = []
        self.keys = []
        if not self.before:
            for text in self.prompts['real_prompt']:
                self.split_text = []
                found_colors = color_pattern.findall(text)
                if len(found_colors) == 0:
                    raise ValueError(f"No colors found in text: '{text}'")
                elif self.split == 'color_attr' and len(found_colors) == 1:
                    raise ValueError(f"Only one color found in text: '{text}'")

                # Generate all combinations of replacements for found colors
                if self.split != "color_attr":
                    for replacements in itertools.product(self.choice, repeat=len(found_colors)):
                            for original_color, new_color in zip(found_colors, replacements):
                                modified_template = text.replace(original_color, new_color, 1)
                            self.split_text.add(modified_template)
                    assert len(self.split_text) == len(self.choice)
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

                        self.split_text.append(final_text)

                    # Convert to a set AFTER all replacements to preserve order
                    unique_prompts = list(dict.fromkeys(self.split_text))  # Removes accidental duplicates while keeping order
                    self.split_text = unique_prompts  # Store only unique prompts

                    assert len(self.split_text) == len(self.choice) * len(self.choice)
                
                self.all_text.append(self.split_text)
                self.keys.append(self.split_text.index(text))

        else:
            for text in self.prompts['prompt']:
                self.split_text = []
                found_colors = color_pattern.findall(text)
                if len(found_colors) == 0:
                    raise ValueError(f"No colors found in text: '{text}'")
                elif self.split == 'color_attr' and len(found_colors) == 1:
                    raise ValueError(f"Only one color found in text: '{text}'")

                # Generate all combinations of replacements for found colors
                if self.split != "color_attr":
                    for replacements in itertools.product(self.choice, repeat=len(found_colors)):
                            for original_color, new_color in zip(found_colors, replacements):
                                modified_template = text.replace(original_color, new_color, 1)
                            self.split_text.add(modified_template)
                    assert len(self.split_text) == len(self.choice)
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

                        self.split_text.append(final_text)

                    # Convert to a set AFTER all replacements to preserve order
                    unique_prompts = list(dict.fromkeys(self.split_text))  # Removes accidental duplicates while keeping order
                    self.split_text = unique_prompts  # Store only unique prompts

                    assert len(self.split_text) == len(self.choice) * len(self.choice)
                self.all_text.append(self.split_text)
                self.keys.append(self.split_text.index(text))

        

    def __len__(self):
        return len(self.examples)

    def with_article(self, name: str):
        if name[0] in "aeiou":
            return f"an {name}"
        return f"a {name}"

    def __getitem__(self, idx):

        image = Image.open(self.examples[idx]).convert('RGB')
        # if not self.scoring_only:
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)
            # if self.transform:
            #     img0_resize = self.transform(img0).unsqueeze(0)
            # else:
            #     img0_resize = img0.resize((self.resize, self.resize))
            #     img0_resize = diffusers_preprocess(img0_resize)
        if not self.before:
            text = self.prompts[self.prompts['filename']==self.examples[idx].split(self.root_dir)[1][1:]]['real_prompt'].iloc[0]
        else:
            text = self.prompts[self.prompts['filename']==self.examples[idx].split(self.root_dir)[1][1:]]['prompt'].iloc[0]
        if self.scoring_only:
            return text, idx
        else:
            if self.domain == 'photo':
                item = edict({"image_options": [image], "caption_options": self.all_text[idx]})
                # return [image_path, [img0_resize]], [test_case[1], test_case[2]], 0
                return item
                # return (self.examples[idx], [img0_resize]), self.all_text[idx], self.all_text[idx].index(text)
            else:
                raise ValueError("Invalid domain")
                # item = edict({"image_options": [img0_resize], "caption_options": self.all_text[idx]})
                # # return [image_path, [img0_resize]], [test_case[1], test_case[2]], 0
                # return item
                # return (self.examples[idx], [img0_resize]), self.all_text[idx], self.all_text[idx].index(text.replace('a photo',self.with_article(self.domain)))
    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        predicted_i2t = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)

        correct_i2t = (predicted_i2t == self.keys)

        result_records = [{"Accuracy t2i": np.mean(correct_i2t)}]
        return result_records



class CLEVRDataset(Dataset):
    def __init__(self, root_dir, image_preprocess, resize=512, scoring_only=False):
        # root_dir = '../clevr/validation'
        # root_dir = "data/clevr"
        root_dir = '../../../data/raw'
        self.root_dir = os.path.join(root_dir, 'clevr')
        subtasks = ['pair_binding_size', 'pair_binding_color', 'recognition_color', 'recognition_shape', 'spatial', 'binding_color_shape', 'binding_shape_color']
        data_ = []
        self.subtasks = []
        for subtask in subtasks:
            self.data = json.load(open(f'{self.root_dir}/captions/{subtask}.json', 'r')).items()
            for k, v in self.data:
                for i in range(len(v)):
                    if subtask == 'spatial':
                        texts = [v[i][1], v[i][0]]
                    else:
                        texts = [v[i][0], v[i][1]]
                    data_.append((k, texts, subtask))
                    self.subtask.append(subtask)
        self.data = data_
        self.resize = resize
        self.image_preprocess = image_preprocess
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
        image = Image.open(img_path0).convert('RGB')
        # if not self.scoring_only:
        if self.image_preprocess is not None:
            image = self.image_preprocess(image)

        text = [cap0, cap1]
        # if self.scoring_only:
        #     return text, 0
        # else:
        item = edict({"image_options": [image], "caption_options":text})
            # return [image_path, [img0_resize]], [test_case[1], test_case[2]], 0
        return item

    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        predicted_i2t = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_i2t = (predicted_i2t == 0)

        result_records = [{"Accuracy t2i": np.mean(correct_i2t)}]
        return result_records


from datasets import load_dataset
class WinogroundDataset(Dataset):
    def __init__(self, root_dir, image_preprocess, resize=512, scoring_only=False):
        # with open("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/hf_token.txt", 'r') as f:
        #     token = f.read().strip()
        # login(token) # datasets==2.14.6 works
        # # download_config = DownloadConfig(use_auth_token=token)
        # os.environ["HUGGINGFACE_TOKEN"] = token
        # self.examples = load_dataset("facebook/winoground",use_auth_token=True)["test"]
        with open("/pfss/mlde/workspaces/mlde_wsp_Rohrbach/users/yj29rogy/hf_token.txt", "r") as f:
            hf_token = f.read().strip()
        try:
            self.examples = load_dataset('facebook/winoground', use_auth_token=hf_token)
        except:
            self.examples = load_dataset('facebook/winoground', token = hf_token)
        self.resize = resize
        self.image_preprocess = image_preprocess
        self.scoring_only = scoring_only

    def __len__(self):
        return len(self.examples['test'])
    
    def __getitem__(self, idx):
        ex = self.examples['test'][idx]
        cap0 = ex['caption_0']
        cap1 = ex['caption_1']
        img_id = ex['id']
        if not self.scoring_only:
            img0 = ex['image_0'].convert('RGB')
            img1 = ex['image_1'].convert('RGB')
            if self.image_preprocess is not None:
                image0 = self.image_preprocess(img0)
                image1 = self.image_preprocess(img1)
        text = [cap0, cap1]
        item = edict({"image_options": [image0, image1], "caption_options":text})
            # return [image_path, [img0_resize]], [test_case[1], test_case[2]], 0
        return item

    def evaluate_scores(self, scores):
        """
        Scores: N x 2 x 2 # image x text
        """

        if isinstance(scores, tuple):
            # print("true")
            scores_i2t = scores[0]
            scores_t2i = scores[1].T # Make it N_ims x N_text
        else:
            # print("false") # here
            scores_t2i = scores
            scores_i2t = scores
        # exit(0)
        ground_truth_i2t = np.array([0,1] * (len(scores_i2t)))
        ground_truth_t2i = np.array([0,1] * (len(scores_t2i)))

        predicted_i2t = scores_i2t.argmax(axis=-1)
        predicted_t2i = scores_t2i.argmax(axis=1)
        

        correct_i2t = np.all(predicted_i2t == ground_truth_i2t.reshape(predicted_i2t.shape), axis=-1)
        correct_t2i = np.all(predicted_t2i == ground_truth_t2i.reshape(predicted_t2i.shape), axis=-1)
        # accuracy = correct_predictions.mean()
        result_records = [{"Accuracy i2t": np.mean(correct_i2t), "Accuracy t2i": np.mean(correct_t2i)}]
        return result_records

class MMVP_VLM(Dataset):
    def __init__(self, root_dir, image_preprocess, split, resize=512, scoring_only=False):
        self.resize = resize
        self.image_preprocess = image_preprocess
        self.scoring_only = scoring_only
        self.split = split

        tokenat = "../hf_token.txt"
        # check if exists
        if not os.path.exists(tokenat):
            # use env variable os.environ["HF_HOME"]HF_HOME
            tokenat = os.environ["HF_HOME"] + "/hf_token.txt"

        with open(tokenat, "r") as f:
                hf_token = f.read().strip()
        try:
            self.examples = []
            self.all_text = []
            examples = os.path.join(root_dir, 'hub/datasets--MMVP--MMVP_VLM/snapshots/211372d5357398f914d806d07dc305aea1f257d2')
            questions = pd.read_csv(os.path.join(examples,'Questions.csv'))
            self.questions = questions[questions['Type']==self.split]
            # for i in range(len(self.questions)):
            self.questions = self.questions.sort_values(by='Question ID')
            for i in range(0, len(self.questions), 2):
                example_set = []
                text_set = []
                example_set.append(os.path.join(examples, 'MLLM_VLM Images', self.split, str(self.questions['Question ID'].iloc[i])+'.jpg'))
                example_set.append(os.path.join(examples, 'MLLM_VLM Images', self.split, str(self.questions['Question ID'].iloc[i+1])+'.jpg'))
                text_set.append("a photo of "+ str(self.questions['Statement'].iloc[i]))
                text_set.append("a photo of "+ str(self.questions['Statement'].iloc[i+1]))
                self.examples.append(example_set)
                self.all_text.append(text_set)

        except:
            try:
                examples = load_dataset('MMVP/MMVP_VLM', use_auth_token=hf_token)
            except:
                examples = load_dataset('MMVP/MMVP_VLM', token = hf_token)
            self.examples = examples['train']
            raise ValueError("Please complete the code for the split")


    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # print(self.examples[idx][0], self.examples[idx][1])
        # print()
        img0 = Image.open(self.examples[idx][0]).convert('RGB')
        img1 = Image.open(self.examples[idx][1]).convert('RGB')
        
        if self.image_preprocess is not None:
            image0 = self.image_preprocess(img0)
            image1 = self.image_preprocess(img1)

        text = [self.all_text[idx][0], self.all_text[idx][1]]

        item = edict({"image_options": [image0, image1], "caption_options":text, "image_idx": [self.examples[idx][0], self.examples[idx][1]]})
        return item


    def evaluate_scores(self, scores):
        """
        Scores: N x 2 x 2
        - scores[:, 0, 0] = similarity (Image A → Text A) # 1
        - scores[:, 0, 1] = similarity (Image A → Text B) # 0
        - scores[:, 1, 1] = similarity (Image B → Text B) # 1
        - scores[:, 1, 0] = similarity (Image B → Text A) # 0

        # Text retrieval: scores[:, 0, 0] > scores[:, 0, 1] and scores[:, 1, 1] > scores[:, 1, 0]
        # Image retrieval: scores[:, 0, 0] > scores[:, 1, 0] and scores[:, 1, 1] > scores[:, 0, 1]


        """
        # how can I see in the score there is a negative value?
        # check it
        
        if isinstance(scores, tuple):
            scores_i2t = scores[1]  # (N, 2, 2)
            scores_t2i = scores[0]  # (N, 2, 2)
        else:
            scores_t2i = scores
            scores_i2t = scores

        # Compute correctness for both queries
        correct_i2t = (scores_i2t[:, 0, 0] > scores_i2t[:, 0, 1]) & (scores_i2t[:, 1, 1] > scores_i2t[:, 1, 0])
        correct_t2i = (scores_t2i[:, 0, 0] > scores_t2i[:, 1, 0]) & (scores_t2i[:, 1, 1] > scores_t2i[:, 0, 1])
        print(scores)
        # print(correct_i2t)
        print(correct_t2i)
        result_records = [{"Accuracy i2t": np.mean(correct_i2t), "Accuracy t2i": np.mean(correct_t2i)}]
        print(result_records)
        # Compute accuracy where both queries must be correct
        return result_records

    
    # def evaluate_scores(self, scores):

    #     if isinstance(scores, tuple):
    #         scores_i2t = scores[0]
    #         scores_t2i = scores[1].T
    #     else:
    #         scores_t2i = scores
    #         scores_i2t = scores
        
    #     ground_truth_i2t = np.array([0,1] * (len(scores_i2t)))
    #     ground_truth_t2i = np.array([0,1] * (len(scores_t2i)))
        
    #     predicted_i2t = scores_i2t.argmax(axis=-1)
    #     predicted_t2i = scores_t2i.argmax(axis=1)

    #     # print(predicted_i2t.shape) # 15,2
    #     # print(predicted_t2i.shape) # 15,2
    #     # exit(0)

    #     correct_i2t = np.all(predicted_i2t == ground_truth_i2t.reshape(predicted_i2t.shape), axis=-1)
    #     result_records = np.all(predicted_t2i == ground_truth_t2i.reshape(predicted_t2i.shape), axis=-1)


    #     result_records = [{"Accuracy i2t": np.mean(correct_i2t), "Accuracy t2i": np.mean(correct_t2i)}]
    #     return result_records
class Geneval(Dataset):
    def __init__(self, root_dir, image_preprocess, split, version, resize=512, scoring_only=False, domain = 'photo', cfg = 9.0):
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
        self.image_preprocess = image_preprocess
        self.scoring_only = scoring_only
        self.split = split
        self.domain = domain
        self.examples = []
        self.text = []
        self.keys = []
        
        if self.split == "two_object_subset":
            self.prompts = json.load(open(f'../diffusion-itm/two_object_subset.json', 'r'))
            split = "two_object"
        else:
            prompt = f'{root_dir}/../prompts/zero_shot_prompts.json'
            self.prompts = json.load(open(prompt, 'r'))[domain][self.split]

        for i in os.listdir(self.root_dir):
            metadata = os.path.join(self.root_dir, i, 'metadata.jsonl')
            metadata = json.load(open(metadata, 'r'))
            if metadata['tag'] == split:
                for j in range(4):
                    self.examples.append(os.path.join(self.root_dir, i, 'samples', f'0000{j}.png'))
                    if self.split == 'color_attr':
                        text_list = self.prompts[metadata["include"][0]["class"]][metadata["include"][1]["class"]]
                    elif self.split == 'position':
                        text_list = self.prompts[metadata["include"][1]["class"]][metadata["include"][0]["class"]]
                    elif self.split in ['single_object','two_object']:
                        text_list = self.prompts
                    elif self.split == "two_object_subset":
                        first = metadata["include"][0]["class"]
                        second = metadata["include"][1]["class"]
                        text_list = self.prompts[f'{first}_{second}']
                    else:
                        text_list = self.prompts[metadata["include"][0]["class"]]
                    self.text.append(text_list)
                    self.keys.append(text_list.index(metadata['prompt']))

    def __len__(self):
        return len(self.examples)

    def with_article(self, name: str):
        if name[0] in "aeiou":
            return f"an {name}"
        return f"a {name}"

    def __getitem__(self, idx):

        img0 = Image.open(self.examples[idx]).convert('RGB')


        if self.image_preprocess is not None:
            image0 = self.image_preprocess(img0)

        item = edict({"image_options": [image0], "caption_options":self.text[idx]})
        return item
    
    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        predicted_i2t = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_i2t = (predicted_i2t == self.keys)

        result_records = [{"Accuracy t2i": np.mean(correct_i2t)}]
        return result_records

class GenevalFiltered(Dataset):
    def __init__(self, root_dir, image_preprocess, split, version, resize=512, scoring_only=False, domain = 'photo', cfg = 9.0):
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
        self.image_preprocess = image_preprocess
        self.scoring_only = scoring_only
        self.split = split
        self.domain = domain
        self.examples = []
        self.text = []
        self.keys = []
        # self.include = []
        self.cfg = cfg
        prompt = f'{root_dir}/../filter/SD-{version}-CFG={str(int(self.cfg))}.json'
        prompt = json.load(open(prompt, 'r')) 
        if self.split =="two_object_subset":
            text = json.load(open(f'../diffusion-itm/two_object_subset.json', 'r'))
            split = "two_object"
        else:
            text = json.load(open(f'{root_dir}/../prompts/zero_shot_prompts.json', 'r'))[self.domain][self.split]
            
        for i in prompt:
            try:
                if version == "3-m" and self.cfg == 9.0:
                    if i['tag'] == split and any(j["label"] == "original_good" for j in i["labels"]) :
                        self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                        # self.prompts.append(i["original_prompt"])
                        # self.include.append(i["full_metadata"]["include"])
                        if self.split == 'color_attr':
                            self.text.append(text[i["full_metadata"]["include"][0]["class"]][i["full_metadata"]["include"][1]["class"]])
                        elif self.split == 'position':
                            self.text.append(text[i["full_metadata"]["include"][1]["class"]][i["full_metadata"]["include"][0]["class"]])
                        elif self.split in ['single_object','two_object']:
                            self.text.append(text)
                        elif self.split == "two_object_subset":
                            first = i["full_metadata"]["include"][0]["class"]
                            second = i["full_metadata"]["include"][1]["class"]
                            self.text.append(text[f"{first}_{second}"])
                        else:
                            self.text.append(text[i["full_metadata"]["include"][0]["class"]])

                        self.keys.append(self.text[-1].index(i["original_prompt"]))
                else:
                    if i['tag'] == split and all(j["label"] == "original_good" for j in i["labels"]) :
                        self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                        # self.prompts.append(i["original_prompt"])
                        # self.include.append(i["full_metadata"]["include"])
                        if self.split == 'color_attr':
                            self.text.append(text[i["full_metadata"]["include"][0]["class"]][i["full_metadata"]["include"][1]["class"]])
                        elif self.split == 'position':
                            self.text.append(text[i["full_metadata"]["include"][1]["class"]][i["full_metadata"]["include"][0]["class"]])
                        elif self.split in ['single_object','two_object']:
                            self.text.append(text)
                        elif self.split == "two_object_subset":
                            first = i["full_metadata"]["include"][0]["class"]
                            second = i["full_metadata"]["include"][1]["class"]
                            self.text.append(text[f"{first}_{second}"])
                        else:
                            self.text.append(text[i["full_metadata"]["include"][0]["class"]])

                        self.keys.append(self.text[-1].index(i["original_prompt"]))
            except:
                    if i['tag'] == split and i["human_label"]=="original_good":
                        # print(i["full_metadata"]["include"][1]["class"])
                        # print(i["full_metadata"]["include"][0]["class"])
                        # print(text.keys())
                        # exit(0)
                        self.examples.append(f"{root_dir}/{cfg}/" + "/".join(i["sample_path"].split("/")[-4:]))
                        # self.prompts.append(i["original_prompt"])
                        # self.include.append(i["full_metadata"]["include"])
                        if self.split == 'color_attr':
                            self.text.append(text[i["full_metadata"]["include"][0]["class"]][i["full_metadata"]["include"][1]["class"]])
                        elif self.split == 'position':
                            self.text.append(text[i["full_metadata"]["include"][1]["class"]][i["full_metadata"]["include"][0]["class"]])
                        elif self.split in ['single_object','two_object']:
                            self.text.append(text)
                        elif self.split == "two_object_subset":
                            first = i["full_metadata"]["include"][0]["class"]
                            second = i["full_metadata"]["include"][1]["class"]
                            self.text.append(text[f"{first}_{second}"])
                        else:
                            self.text.append(text[i["full_metadata"]["include"][0]["class"]])

                        self.keys.append(self.text[-1].index(i["original_prompt"]))

            
        # print(self.__len__())
        # exit(0)
        if self.__len__() == 0:
            
            raise ValueError('No examples found')

        
    def __len__(self):
        return len(self.examples)

    def with_article(self, name: str):
        if name[0] in "aeiou":
            return f"an {name}"
        return f"a {name}"

    def __getitem__(self, idx):

        img0 = Image.open(self.examples[idx]).convert('RGB')


        if self.image_preprocess is not None:
            image0 = self.image_preprocess(img0)

        item = edict({"image_options": [image0], "caption_options":self.text[idx]})
        return item
    
    def evaluate_scores(self, scores):
        """
        Scores: N x 1 x 2, i.e. first caption is right, next is wrong
        """
        if isinstance(scores, tuple):
            scores_i2t = scores[1]
            scores_t2i = scores[0]
        else:
            scores_t2i = scores
            scores_i2t = scores

        metrics = {"Accuracy": None}
        predicted_i2t = np.argmax(np.squeeze(scores_i2t, axis=1), axis=-1)
        correct_i2t = (predicted_i2t == self.keys)

        result_records = [{"Accuracy t2i": np.mean(correct_i2t)}]
        return result_records



def get_geneval(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="test", version='1.5', cfg = None):
    dataset = Geneval(root_dir=root_dir, image_preprocess=image_preprocess, split = split, version = version, cfg = 9.0)
    return dataset

def get_geneval_filter(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="test", version='1.5', cfg = None):
    dataset = GenevalFiltered(root_dir=root_dir, image_preprocess=image_preprocess, split = split, version = version, cfg = 9.0)
    return dataset


def get_mmvp(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="test"):
    dataset = MMVP_VLM(root_dir=os.environ["HF_HOME"], image_preprocess=image_preprocess, split = split)
    return dataset

def get_winoground(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="test"):
    dataset = WinogroundDataset(root_dir=root_dir, image_preprocess=image_preprocess)
    return dataset

def get_clevr(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="test", before=False, version='1.5'):
    dataset = CLEVRDataset(root_dir=root_dir, image_preprocess=image_preprocess)
    return dataset

def get_ours(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="test", before=False, version='1.5'):
    dataset = Ours(root_dir=root_dir, image_preprocess=image_preprocess, split = split, version = version, before=before)
    return dataset

def get_pets(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="test"):
    dataset = PETS(root_dir=root_dir, image_preprocess=image_preprocess)
    return dataset

def get_spec(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="test"):
    dataset = SPEC(root_dir=root_dir, image_preprocess=image_preprocess, split = split)
    return dataset

def get_spec_img_retrieval(image_preprocess, image_perturb_fn, text_perturb_fn, max_words=30, download=False, root_dir=None, split="test"):
    dataset = SPEC_IMG_RETRIEVAL(root_dir=root_dir, image_preprocess=image_preprocess, split = split)
    return dataset

def get_eqbench(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="test"):
    # dataset = EQBench(root_dir=root_dir, image_preprocess=image_preprocess, split = split)
    dataset = EQBench_split(root_dir=root_dir, image_preprocess=image_preprocess, split = split)
    return dataset

def get_vlcheck_attribute(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="hake"):
    dataset = VLCheck_Attribute(root_dir=root_dir, image_preprocess=image_preprocess, split = split)
    return dataset


def get_vlcheck_relation(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="hake"):
    dataset = VLCheck_Relation(root_dir=root_dir, image_preprocess=image_preprocess, split = split)
    return dataset

def get_vlcheck_object_size(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="hake"):
    dataset = VLCheck_Object_Size(root_dir=root_dir, image_preprocess=image_preprocess, split = split)
    return dataset

def get_vlcheck_object_location(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="hake"):
    dataset = VLCheck_Object_Location(root_dir=root_dir, image_preprocess=image_preprocess, split = split)
    return dataset

def get_valse(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="action-replacement"):
    dataset = VALSE(root_dir=root_dir, image_preprocess=image_preprocess,  split = split)
    return dataset

def get_cifar100(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="test"):
    dataset = CIFAR100(root_dir=root_dir, image_preprocess=image_preprocess)
    return dataset

def get_mnist(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="test"):
    dataset = MNIST(root_dir=root_dir, image_preprocess=image_preprocess)
    return dataset

def get_whatsup(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="test"):
    dataset = WhatsUp(root_dir=root_dir, image_preprocess=image_preprocess, split = split)
    return dataset

def get_vg_qa(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="test"):
    dataset = VG_QA(root_dir=root_dir, image_preprocess=image_preprocess, split = split)
    return dataset

def get_coco_qa(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="test"):
    dataset = COCO_QA(root_dir=root_dir, image_preprocess=image_preprocess, split = split)
    return dataset

def get_vismin(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="test"):
    # dataset = VisMin(root_dir=root_dir, image_preprocess=image_preprocess, split = split)
    dataset = VisMin_split(root_dir=root_dir, image_preprocess=image_preprocess, split = split)
    return dataset

def get_cola_multi(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="test"):
    dataset = Cola_Multi(root_dir=root_dir, image_preprocess=image_preprocess)
    return dataset

def get_sugar_crepe(image_preprocess, image_perturb_fn=None, text_perturb_fn=None, max_words=30, download=False, root_dir=None, split="test"):
    dataset = SugarCrepe(root_dir=root_dir, split=split, image_preprocess=image_preprocess)
    return dataset


def get_coco_retrieval(image_preprocess, image_perturb_fn, text_perturb_fn, max_words=30, download=False, root_dir=COCO_ROOT, split="test"):
    dataset = COCO_Retrieval(root_dir=root_dir, split=split, image_preprocess=image_preprocess, image_perturb_fn=image_perturb_fn, max_words=max_words, 
                            download=download)
    return dataset


def get_flickr30k_retrieval(image_preprocess, image_perturb_fn, text_perturb_fn, max_words=30, download=False, root_dir=FLICKR_ROOT, split="test"):
    dataset = Flickr30k_Retrieval(root_dir=root_dir, split=split, image_preprocess=image_preprocess, image_perturb_fn=image_perturb_fn, max_words=max_words, 
                            download=download)
    return dataset
