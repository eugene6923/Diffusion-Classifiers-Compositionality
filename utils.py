import numpy as np
import json
from math import floor
from typing import Union, Callable
import torch
import torch.nn.functional as F
import os
import wandb

RETRIEVAL_TASKS = ['imagecode', 'imagecode_video', 'flickr30k', 'imagenet', 'clevr', 'svo_verb', 'svo_subj', 'svo_obj', 'pets', 'flickr30k_text', 'vg_relation', 'vg_attribution', 
'coco_order', 'flickr30k_order', 'mscoco_val', 'whatsup_A', 'whatsup_B','COCO_QA_one','COCO_QA_two','VG_QA_one', 'VG_QA_two',
'sugar_add_obj','sugar_add_att','sugar_replace_att','sugar_replace_obj','sugar_replace_rel','sugar_swap_att','sugar_swap_obj','cola_multi',
'sugar_obj','sugar_rel','sugar_att',
'vismin','vismin_relation','vismin_attribution','vismin_counting','vismin_object','countbench','valse_action-replacement','valse_actant-swap',
'valse_existence','valse_counting-adversarial','valse_counting-hard','valse_counting-small-quant','valse_relations','valse_foil-it','valse_plurals',
'spec_absolute_size','spec_absolute_spatial','spec_count','spec_existence','spec_relative_size','spec_relative_spatial','spec_existence',
'vlcheck_action','vlcheck_color','vlcheck_material','vlcheck_size','vlcheck_state', 
'vlcheck_Object_Location_hake', 'vlcheck_Object_Location_swig', 'vlcheck_Object_Location_vg',
'vlcheck_Object_Size_hake', 'vlcheck_Object_Size_swig', 'vlcheck_Object_Size_vg',
'vlcheck_Relation_vg_action', 'vlcheck_Relation_vg_spatial', 'vlcheck_Relation_hake', 'vlcheck_Relation_swig',
'naturalbench', 'imagenet', 'stl10', 'cifar10',
'geneval_colors','geneval_color','geneval_position','geneval_single','geneval_counting','geneval_color_attr','geneval_two',
'geneval_1_5_color','geneval_1_5_position','geneval_1_5_single','geneval_1_5_counting','geneval_1_5_color_attr','geneval_1_5_two',
'geneval_2_0_color','geneval_2_0_position','geneval_2_0_single','geneval_2_0_counting','geneval_2_0_color_attr','geneval_2_0_two',
'geneval_3_m_color','geneval_3_m_position','geneval_3_m_single','geneval_3_m_counting','geneval_3_m_color_attr','geneval_3_m_two',
'geneval_filter_color','geneval_filter_position','geneval_filter_single','geneval_filter_counting','geneval_filter_color_attr','geneval_filter_two',
'geneval_1_5_filter_color','geneval_1_5_filter_position','geneval_1_5_filter_single','geneval_1_5_filter_counting','geneval_1_5_filter_color_attr','geneval_1_5_filter_two',
'geneval_2_0_filter_color','geneval_2_0_filter_position','geneval_2_0_filter_single','geneval_2_0_filter_counting','geneval_2_0_filter_color_attr','geneval_2_0_filter_two',
'geneval_3_m_filter_color','geneval_3_m_filter_position','geneval_3_m_filter_single','geneval_3_m_filter_counting','geneval_3_m_filter_color_attr','geneval_3_m_filter_two',
'geneval_two_subset', 'geneval_1_5_two_subset', 'geneval_2_0_two_subset', 'geneval_3_m_two_subset', 'geneval_filter_two_subset', 'geneval_1_5_filter_two_subset', 'geneval_2_0_filter_two_subset', 'geneval_3_m_filter_two_subset', 
'ours_colors','ours_color_attr','ours_position','ours_counting',
'ours_before_colors','ours_before_color_attr','ours_before_position','ours_before_counting',
'ours_1_5_colors','ours_1_5_color_attr','ours_1_5_position','ours_1_5_counting',
'ours_1_5_before_colors','ours_1_5_before_color_attr','ours_1_5_before_position','ours_1_5_before_counting',
'ours_2_0_colors','ours_2_0_color_attr','ours_2_0_position','ours_2_0_counting',
'ours_2_0_before_colors','ours_2_0_before_color_attr','ours_2_0_before_position','ours_2_0_before_counting',
'ours_3_m_colors','ours_3_m_color_attr','ours_3_m_position','ours_3_m_counting',
'ours_3_m_before_colors','ours_3_m_before_color_attr','ours_3_m_before_position','ours_3_m_before_counting',
'mmvp_camera' , 'mmvp_color', 'mmvp_orientation','mmvp_presence', 'mmvp_quantity', 'mmvp_spatial','mmvp_state','mmvp_structural','mmvp_text',
]

def evaluate_winoground(args, scores, batch, idx):
    text = batch[1]
    # img_idx = batch[-1]
    data_path = batch[0][0]
    # print(scores.shape) # | torch.Size([N, 4])

    text_score, img_score, group_score = [], [], []
    if args.save_results:
        text_list1 = [j for j in text[0]]
        text_list2 = [j for j in text[1]]
        image_list1 = [j for j in data_path[0]]
        image_list2 = [j for j in data_path[1]]
    for i, score_ in enumerate(scores):
        

        c0_i0, c0_i1, c1_i0, c1_i1 = score_
        
        text_score_ = 1 if c0_i0 > c1_i0 and c1_i1 > c0_i1 else 0
        img_score_ = 1 if c0_i0 > c0_i1 and c1_i1 > c1_i0 else 0
        group_score_ = 1 if text_score_ and img_score_ else 0 
        text_score.append(text_score_)
        img_score.append(img_score_)
        group_score.append(group_score_)

        if args.save_results:

            os.makedirs(f'{args.outdir}/confusion/{args.task}', exist_ok=True)
            file_path = f'{args.outdir}/confusion/{args.task}/{args.run_id}.txt'
            
            if idx==0:
                mode = 'w'
            else:
                mode = 'a'
            
            with open(file_path, mode) as f:
                f.write(f'text1: {text_list1[i]}, text2: {text_list2[i]}\n')
                f.write(f'img1: {image_list1[i]}, img2: {image_list2[i]}\n')
                f.write(f'c0_i0: {c0_i0}, c0_i1: {c0_i1}, c1_i0: {c1_i0}, c1_i1: {c1_i1}\n\n')

    return text_score, img_score, group_score

def evaluate_retrieval(args, scores, batch, idx, table = None):
    text = batch[1]

    img_idx = batch[-1]
    data_path = batch[0][0]

    if type(scores) != list:
        img_idx = img_idx.cpu().numpy()
        scores = scores.cpu().numpy()
    if args.task == 'naturalbench':
        print(scores)

    scores = np.stack(scores, axis=0)
    retrieval_accuracy = []
    max_more_than_once = 0
    # print(img_idx) # 4
    # print(idx)
    # print(len(text))

    for i in range(scores.shape[0]):
        number_of_argmax_appearances = np.sum(scores[i] == np.max(scores[i]))

        if number_of_argmax_appearances > 1:
            max_more_than_once += 1
        if img_idx[i] == np.argmax(scores[i]):
            retrieval_accuracy.append(1)
        else:
            retrieval_accuracy.append(0)

        if args.wandb and args.img_retrieval != True:
            
            text_list = [j[i] for j in text]

            guess_idx = np.argmax(scores[i])
            gt_id = text_list[img_idx[i]]
            
            for j, cont in enumerate(text_list):
                pass
                # wandb.log({f'text': cont, f'image': data_path[i], f'score': scores[i][j], 'gt': text == gt_id, 'guess': text == text_list[guess_id]})
                # table.add_data(cont, data_path[i], scores[i][j], cont == gt_id, cont == text_list[guess_id])
                # table.append({'text': cont, 'image': data_path[i], 'score': scores[i][j], 'gt': cont == gt_id, 'guess': cont == text_list[guess_idx]})
        
        if args.save_results and args.img_retrieval != True:
            # Extract top-5 indices (sorted by highest score)
            if not args.wandb:
                text_list = [j[i] for j in text]

                # Extract top-5 indices (sorted by highest score)
                guess_idx = np.argmax(scores[i])  # Get the guessed text index
            top5_idx = np.argsort(scores[i])[-5:][::-1]  # Get top 5 indices in descending order

            # Extract texts & scores for top 5 indices (including the guess)
            top5_texts = [text_list[j] for j in top5_idx]  # Extract corresponding text
            top5_scores = [scores[i][j] for j in top5_idx]  # 

            os.makedirs(f'{args.outdir}/confusion/{args.task}', exist_ok=True)
            file_path = f'{args.outdir}/confusion/{args.task}/{args.run_id}.txt'
            
            if idx==0 and i==0:
                mode = 'w'  # Create a new file
            else:
                mode = 'a'  # Append to the file

            with open(file_path, mode) as f:
                if mode == 'w':
                    f.write(f'run id: {args.run_id}\n')
                f.write(f'text list: {text_list}\n')  # Save full text list
                f.write(f'data path: {data_path[i]}\n')
                f.write(f'top5 text: {top5_texts}\n')  # ✅ Now including guess
                f.write(f'top5 scores: {top5_scores}\n')  # ✅ Scores aligned correctly
                f.write(f'top5 idx: {top5_idx.tolist()}\n')  # ✅ Save the actual indices
                f.write(f'truth idx: {img_idx[i]}, guess idx: {guess_idx}\n')  #  ✅ Correct truth index
                f.write(f'truth: {text_list[img_idx[i]]}\n')  #  ✅ Correct truth text
                f.write(f'guess: {text_list[guess_idx]}\n\n')  #  ✅ Correct guess text

    
    if args.task in ['flickr30k', 'imagecode', 'imagenet', 'flickr30k_text']:
        r5 = []
        for i in range(scores.shape[0]):
            if img_idx[i] in np.argsort(scores[i])[-5:]:
                r5.append(1)
            else:
                r5.append(0)
        return (retrieval_accuracy, r5, max_more_than_once), table
    else:
        return (retrieval_accuracy, max_more_than_once), table

def evaluate_bias(args, good_scores, bad_scores, img_idx):
    img_idx = img_idx.cpu().numpy()
    good_scores = good_scores.cpu().numpy()
    bad_scores = bad_scores.cpu().numpy()
    phis = {}
    for i in range(len(good_scores)): # rows of tensor are images, columns are the words
        # p val test just needs the phi(w,A,B) which i have!  just code it elionrrr
        class_idx = int(img_idx[i]) # get class, should be an integer {0,1,...,7}
        good_score = good_scores[i].mean() # mean_{a\in A} sigma(x,a)
        bad_score = bad_scores[i].mean() # mean_{b\in B} sigma(x,b)
        phi = good_score-bad_score # phi(w,A,B) = mean_{a\in A} sigma(x,a) - mean_{b\in B} sigma(x,b)
        if class_idx in phis:
            phis[class_idx].append(phi)
        else:
            phis[class_idx] = [phi]
    return phis#, raw_scores

def evaluate_gender_bias(args, m_attr_scores, f_attr_scores, class_ids):
    entity = class_ids[0].split('_')[-1] # either clothes, drinks, or bags
    male_filter = np.array(class_ids)==f'male_{entity}' # indices of scores of male images
    female_filter = np.array(class_ids)==f'female_{entity}' # indices of scores of female images
    m_attr_scores = m_attr_scores.cpu().numpy() # all the images scored with the male attr word
    f_attr_scores = f_attr_scores.cpu().numpy() # all the images scores w female attr word
    
    m_imgs_m_attr = m_attr_scores[male_filter]
    m_imgs_f_attr = f_attr_scores[male_filter]
    f_imgs_m_attr = m_attr_scores[female_filter]
    f_imgs_f_attr = f_attr_scores[female_filter]
    
    phi_male = m_imgs_m_attr - m_imgs_f_attr #phi(m,w_m,w_f) = sigma(m,w_m)-sigma(m,w_f)
    phi_female = f_imgs_m_attr - f_imgs_f_attr #phi(f,w_m,w_f) = sigma(f,w_m)-sigma(f,w_f)
    
    return {f'male_{entity}':phi_male,f'female_{entity}':phi_female}

def save_bias_scores(fname, bias_scores):
    with open(fname, 'w') as f:
            print(bias_scores)
            json.dump(bias_scores,f)
            f.close()
    return bias_scores

def save_bias_results(fname, bias_scores, task):
    if task == 'mmbias':
        with open(fname, 'w') as f:
                christian = bias_scores[0]
                muslim = bias_scores[1]
                jewish = bias_scores[2]
                hindu = bias_scores[3]
                american = bias_scores[4]
                arab = bias_scores[5]
                hetero = bias_scores[6]
                lgbt = bias_scores[7]
                buddhist = bias_scores[8]
                f.write(f'Christian-Muslim bias score {(np.mean(christian)-np.mean(muslim))/(np.concatenate((christian,muslim)).std())}\n')
                f.write(f'Christian-Jewish bias score {(np.mean(christian)-np.mean(jewish))/(np.concatenate((christian,jewish)).std())}\n')
                f.write(f'Jewish-Muslim bias score {(np.mean(jewish)-np.mean(muslim))/(np.concatenate((jewish,muslim)).std())}\n')
                f.write(f'Buddhist-Muslim bias score {(np.mean(buddhist)-np.mean(muslim))/(np.concatenate((buddhist,muslim)).std())}\n')
                f.write(f'Buddhist-Christian bias score {(np.mean(buddhist)-np.mean(christian))/(np.concatenate((christian,buddhist)).std())}\n')
                f.write(f'Buddhist-Hindu bias score {(np.mean(buddhist)-np.mean(hindu))/(np.concatenate((christian,hindu)).std())}\n')
                f.write(f'Buddhist-Jewish bias score {(np.mean(buddhist)-np.mean(jewish))/(np.concatenate((christian,jewish)).std())}\n')
                f.write(f'Hindu-Muslim bias score {(np.mean(hindu)-np.mean(muslim))/(np.concatenate((hindu,muslim)).std())}\n')
                f.write(f'American-Arab bias score {(np.mean(american)-np.mean(arab))/(np.concatenate((american,arab)).std())}\n')
                f.write(f'Hetero-LGBT bias score {(np.mean(hetero)-np.mean(lgbt))/(np.concatenate((hetero,lgbt)).std())}\n')
                f.write('Positive scores indicate bias towards the first group, closer to 0 is less bias')
                f.close()
    elif task == 'genderbias':
        with open(fname, 'w') as f:
                #bias_scores = {'male_clothes': [], 'female_clothes': [], 'male_bags': [], 'female_bags': [], 'male_drinks': [], 'female_drinks': []}
                f.write(f"Clothes bias score {(np.mean(bias_scores['male_clothes'])-np.mean(bias_scores['female_clothes']))/(np.concatenate((bias_scores['male_clothes'],bias_scores['female_clothes'])).std())}\n")
                f.write(f"Bags bias score {(np.mean(bias_scores['male_bags'])-np.mean(bias_scores['female_bags']))/(np.concatenate((bias_scores['male_bags'],bias_scores['female_bags'])).std())}\n")
                f.write(f"Drinks bias score {(np.mean(bias_scores['male_drinks'])-np.mean(bias_scores['female_drinks']))/(np.concatenate((bias_scores['male_drinks'],bias_scores['female_drinks'])).std())}\n")
                f.write('Positive scores indicate bias towards males, closer to 0 is less bias')
                f.close()
    return bias_scores # returns no changes

def evaluate_scores(args, scores, batch,i, table= None):
    if 'winoground' in args.task or args.task == 'cola_multi' or 'vismin' in args.task or 'eqbench' in args.task or 'mmvp' in args.task:
        score = evaluate_winoground(args, scores, batch, i)

    elif args.task == 'mmbias':
        # so we have a bunch of scores, which is a tensor Size([batchsize,len(texts)])
        # example for 4 texts and batchsize 2
        # scores = tensor([[ 0.0555,  0.0121,  0.0113,mmOKxRfPbYjE -0.0000],
        #         [ 0.0398, -0.0133, -0.0340, -0.0391]], device='cuda:7')
        text_len = floor(len(batch[1])/2) # number of good / bad texts
        good_scores = scores[:, :text_len]  # extract the first len(good_texts) cols for pleasant_texts
        bad_scores = scores[:, text_len:]   # extract the remaining cols for unpleasant_texts
        assert len(good_scores) == len(bad_scores)
        img_idx = batch[-1] # tensor of class_ids
        return evaluate_bias(args, good_scores, bad_scores, img_idx) # dictionary of lists of phis
    elif args.task == 'genderbias':
        # input is list of scores (tensors whatever), ill use batchsize 6 so its just one text and one fe/male_entity
        # evaluate_gender_bias should return a just the phi for the one class
        male_attr_scores = scores[:,0]
        female_attr_scores = scores[:,-1]
        class_ids = batch[-1]
        return evaluate_gender_bias(args, male_attr_scores, female_attr_scores, class_ids) 
    elif args.task == 'naturalbench':
        # img_idx = batch[-1]
        score, table = evaluate_retrieval(args, scores, batch, i)
    else:
        score, table = evaluate_retrieval(args, scores, batch, i, table)
    if table == None:
        return score
    return score, table
