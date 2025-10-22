from .aro_datasets import VG_Relation, VG_Attribution, COCO_Order, Flickr30k_Order
from .retrieval import COCO_Retrieval, Flickr30k_Retrieval


def get_dataset(dataset_name, image_preprocess=None, text_perturb_fn=None, image_perturb_fn=None, download=False, version = None, cfg= None, *args, **kwargs):
    """
    Helper function that returns a dataset object with an evaluation function. 
    dataset_name: Name of the dataset.
    image_preprocess: Preprocessing function for images.
    text_perturb_fn: A function that takes in a string and returns a string. This is for perturbation experiments.
    image_perturb_fn: A function that takes in a PIL image and returns a PIL image. This is for perturbation experiments.
    download: Whether to allow downloading images if they are not found.
    """
    if dataset_name == "VG_Relation": 
        from .aro_datasets import get_visual_genome_relation
        return get_visual_genome_relation(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "VG_Attribution":
        from .aro_datasets import get_visual_genome_attribution
        return get_visual_genome_attribution(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "COCO_Order":
        from .aro_datasets import get_coco_order
        # from .retrieval import get_coco_order
        return get_coco_order(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)

    elif dataset_name == "Flickr30k_Order":
        from .aro_datasets import get_flickr30k_order
        return get_flickr30k_order(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "COCO_Retrieval":
        from .retrieval import get_coco_retrieval
        return get_coco_retrieval(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "Flickr30k_Retrieval":
        from .retrieval import get_flickr30k_retrieval
        return get_flickr30k_retrieval(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "Cola_Multi":
        from .retrieval import get_cola_multi
        return get_cola_multi(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'sugar_add_att':
        from .retrieval import get_sugar_crepe
        return get_sugar_crepe(split='add_att', image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'sugar_add_obj':
        from .retrieval import get_sugar_crepe
        return get_sugar_crepe(split='add_obj', image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'sugar_replace_att':
        from .retrieval import get_sugar_crepe
        return get_sugar_crepe(split='replace_att', image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'sugar_replace_obj':
        from .retrieval import get_sugar_crepe
        return get_sugar_crepe(split='replace_obj', image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'sugar_replace_rel':
        from .retrieval import get_sugar_crepe
        return get_sugar_crepe(split='replace_rel', image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'sugar_swap_att':
        from .retrieval import get_sugar_crepe
        return get_sugar_crepe(split='swap_att', image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'sugar_swap_obj':
        from .retrieval import get_sugar_crepe
        return get_sugar_crepe(split='swap_obj', image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'sugar_att':
        from .retrieval import get_sugar_crepe
        return get_sugar_crepe(split='att', image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'sugar_rel':
        from .retrieval import get_sugar_crepe
        return get_sugar_crepe(split='rel', image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'sugar_obj':
        from .retrieval import get_sugar_crepe
        return get_sugar_crepe(split='obj', image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'vismin_relation':
        from .retrieval import get_vismin
        return get_vismin(split="relation",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'vismin_attribute':
        from .retrieval import get_vismin
        return get_vismin(split="attribute",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'vismin_object':
        from .retrieval import get_vismin
        return get_vismin(split="object",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'vismin_counting':
        from .retrieval import get_vismin
        return get_vismin(split="counting",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'whatsup_A':
        from .retrieval import get_whatsup
        return get_whatsup(split="A",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
        # return Controlled_Images(image_preprocess = transform, download = True, root_dir =f'{root_dir}/whatsup', subset='A')
    elif dataset_name == 'whatsup_B':
        from .retrieval import get_whatsup
        return get_whatsup(split="B",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
        # return Controlled_Images(image_preprocess = transform, download = True, root_dir =f'{root_dir}/whatsup', subset='B')
    elif dataset_name == 'COCO_QA_one':
        from .retrieval import get_coco_qa
        return get_coco_qa(split="one",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
        # return COCO_QA(image_preprocess = transform, download = True, root_dir =f'{root_dir}/coco_qa', subset='one')
    elif dataset_name == 'COCO_QA_two':
        from .retrieval import get_coco_qa
        return get_coco_qa(split="two",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
        # return COCO_QA(image_preprocess = transform, download = True, root_dir =f'{root_dir}/coco_qa', subset='two')
    elif dataset_name == 'VG_QA_one':
        from .retrieval import get_vg_qa
        return get_vg_qa(split="one",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
        # return VG_QA(image_preprocess = transform, download = True, root_dir =f'{root_dir}/VG_QA', subset='one')
    elif dataset_name == 'VG_QA_two':
        from .retrieval import get_vg_qa
        return get_vg_qa(split="two",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
        # return VG_QA(image_preprocess = transform, download = True, root_dir =f'{root_dir}/VG_QA', subset='two')
    elif dataset_name == 'mnist':
        from .retrieval import get_mnist
        return get_mnist(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'cifar100':
        from .retrieval import get_cifar100
        return get_cifar100(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'valse_action-replacement':
        from .retrieval import get_valse
        return get_valse(root_dir = '../../../data/raw/SWiG/images_512',split='action-replacement',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'valse_relations':
        from .retrieval import get_valse
        return get_valse(root_dir = '../../../data/raw/COCO2017/val2017',split='relations',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'valse_actant-swap':
        from .retrieval import get_valse
        return get_valse(root_dir = '../../../data/raw/SWiG/images_512',split='actant-swap',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'vlcheck_Relation_hake':
        from .retrieval import get_vlcheck_relation
        return get_vlcheck_relation(root_dir = '../../../data/raw/HAKE', split='hake',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'vlcheck_Relation_swig':
        from .retrieval import get_vlcheck_relation
        return get_vlcheck_relation(root_dir = '../../../data/raw/SWiG/images_512', split='swig',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'vlcheck_Relation_vg_action':
        from .retrieval import get_vlcheck_relation
        return get_vlcheck_relation(root_dir = '../../../data/raw/VG_100K/image', split='vg_action',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'vlcheck_Relation_vg_spatial':
        from .retrieval import get_vlcheck_relation
        return get_vlcheck_relation(root_dir = '../../../data/raw/VG_100K/image', split='vg_spatial',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'vlcheck_action':
        from .retrieval import get_vlcheck_attribute
        return get_vlcheck_attribute(root_dir = '../../../data/raw/VG_100K/image', split='action',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "vlcheck_Object_Size_hake":
        from .retrieval import get_vlcheck_object_size
        return get_vlcheck_object_size(root_dir = '../../../data/raw/HAKE', split='hake',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "vlcheck_Object_Location_hake":
        from .retrieval import get_vlcheck_object_location
        return get_vlcheck_object_location(root_dir = '../../../data/raw/HAKE', split='hake',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'eqbench_eqbenyoucook2':
        from .retrieval import get_eqbench
        return get_eqbench(root_dir = '../../../data/raw/eqbench/eqbench_subset', split='eqbenyoucook2',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'eqbench_eqbengebc':
        from .retrieval import get_eqbench
        return get_eqbench(root_dir = '../../../data/raw/eqbench/eqbench_subset', split='eqbengebc',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'eqbench_eqbenag':
        from .retrieval import get_eqbench
        return get_eqbench(root_dir = '../../../data/raw/eqbench/eqbench_subset', split='eqbenag',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'eqbench_eqbenkubric_attr':
        from .retrieval import get_eqbench
        return get_eqbench(root_dir = '../../../data/raw/eqbench/eqbench_subset', split='eqbenkubric_attr',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'eqbench_eqbenkubric_cnt':
        from .retrieval import get_eqbench
        return get_eqbench(root_dir = '../../../data/raw/eqbench/eqbench_subset', split='eqbenkubric_cnt',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'eqbench_eqbenkubric_loc':
        from .retrieval import get_eqbench
        return get_eqbench(root_dir = '../../../data/raw/eqbench/eqbench_subset', split='eqbenkubric_loc',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'eqbench_eqbensd':
        from .retrieval import get_eqbench
        return get_eqbench(root_dir = '../../../data/raw/eqbench/eqbench_subset', split='eqbensd',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'spec_absolute_spatial':
        from .retrieval import get_spec
        return get_spec(root_dir = '../../../data/raw/spec', split='absolute_spatial',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'spec_relative_spatial':
        from .retrieval import get_spec
        return get_spec(root_dir = '../../../data/raw/spec', split='relative_spatial',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'spec_relative_size':
        from .retrieval import get_spec
        return get_spec(root_dir = '../../../data/raw/spec', split='relative_size',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'spec_existence':
        from .retrieval import get_spec
        return get_spec(root_dir = '../../../data/raw/spec', split='existence',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'spec_count':
        from .retrieval import get_spec
        return get_spec(root_dir = '../../../data/raw/spec', split='count',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'spec_absolute_size':
        from .retrieval import get_spec
        return get_spec(root_dir = '../../../data/raw/spec', split='absolute_size',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'spec_absolute_spatial_img_retrieval':
        from .retrieval import get_spec_img_retrieval
        return get_spec_img_retrieval(root_dir = '../../../data/raw/spec', split='absolute_spatial',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'spec_relative_spatial_img_retrieval':
        from .retrieval import get_spec_img_retrieval
        return get_spec_img_retrieval(root_dir = '../../../data/raw/spec', split='relative_spatial',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'spec_relative_size_img_retrieval':
        from .retrieval import get_spec_img_retrieval
        return get_spec_img_retrieval(root_dir = '../../../data/raw/spec', split='relative_size',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'spec_existence_img_retrieval':
        from .retrieval import get_spec_img_retrieval
        return get_spec_img_retrieval(root_dir = '../../../data/raw/spec', split='existence',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'spec_count_img_retrieval':
        from .retrieval import get_spec_img_retrieval
        return get_spec_img_retrieval(root_dir = '../../../data/raw/spec', split='count',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'spec_absolute_size_img_retrieval':
        from .retrieval import get_spec_img_retrieval
        return get_spec_img_retrieval(root_dir = '../../../data/raw/spec', split='absolute_size',image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    
    elif dataset_name == 'pets':
        from .retrieval import get_pets
        return get_pets(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'ours_colors':
        from .retrieval import get_ours
        return get_ours(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, split="colors", before = False, version = version, *args, **kwargs)
    elif dataset_name == 'ours_color_attr':
        from .retrieval import get_ours
        return get_ours(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, split="color_attr", before = False, version = version, *args, **kwargs)
    elif dataset_name == 'ours_position':
        from .retrieval import get_ours
        return get_ours(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, split="position", before = False, version = version, *args, **kwargs)
    elif dataset_name == 'ours_counting':
        from .retrieval import get_ours
        return get_ours(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, split="counting", before = False, version = version, *args, **kwargs)
    elif dataset_name == 'ours_before_colors':
        from .retrieval import get_ours
        return get_ours(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, split="colors", before = True, version = version, *args, **kwargs)
    elif dataset_name == 'ours_before_color_attr':
        from .retrieval import get_ours
        return get_ours(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, split="color_attr", before = True, version = version, *args, **kwargs)
    elif dataset_name == 'ours_before_position':
        from .retrieval import get_ours
        return get_ours(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, split="position", before = True, version = version, *args, **kwargs)
    elif dataset_name == 'ours_before_counting':
        from .retrieval import get_ours
        return get_ours(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, split="counting", before = True, version = version, *args, **kwargs)
    elif dataset_name == 'clevr':
        from .retrieval import get_clevr
        return get_clevr(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'winoground':
        from .retrieval import get_winoground
        return get_winoground(image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'mmvp_camera':
        from .retrieval import get_mmvp
        return get_mmvp(split="Camera Perspective",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'mmvp_color':
        from .retrieval import get_mmvp
        return get_mmvp(split="Color",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'mmvp_orientation':
        from .retrieval import get_mmvp
        return get_mmvp(split="Orientation",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'mmvp_presence':
        from .retrieval import get_mmvp
        return get_mmvp(split="Presence",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'mmvp_quantity':
        from .retrieval import get_mmvp
        return get_mmvp(split="Quantity",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'mmvp_spatial':
        from .retrieval import get_mmvp
        return get_mmvp(split="Spatial",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'mmvp_state':
        from .retrieval import get_mmvp
        return get_mmvp(split="State",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'mmvp_structural':
        from .retrieval import get_mmvp
        return get_mmvp(split="Structural Character",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == 'mmvp_text':
        from .retrieval import get_mmvp
        return get_mmvp(split="Text",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "geneval_color":
        from .retrieval import get_geneval
        return get_geneval(root_dir = '../geneval/outputs', split="colors", version = version,cfg = cfg,image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "geneval_color_attr":
        from .retrieval import get_geneval
        return get_geneval(root_dir = '../geneval/outputs', split="color_attr", version = version,cfg = cfg,image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "geneval_position":
        from .retrieval import get_geneval
        return get_geneval(root_dir = '../geneval/outputs', split="position", version = version,cfg = cfg,image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "geneval_counting":
        from .retrieval import get_geneval
        return get_geneval(root_dir = '../geneval/outputs', split="counting", version = version,cfg = cfg,image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "geneval_single":
        from .retrieval import get_geneval
        return get_geneval(root_dir = '../geneval/outputs', split="single_object", version = version,cfg = cfg,image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "geneval_two":
        from .retrieval import get_geneval
        return get_geneval(root_dir = '../geneval/outputs', split="two_object", version = version,cfg = cfg,image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "geneval_two_subset":
        from .retrieval import get_geneval
        return get_geneval(root_dir = '../geneval/outputs', split="two_object_subset", version = version,cfg = cfg,image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "geneval_filter_color":
        from .retrieval import get_geneval_filter
        return get_geneval_filter(root_dir = '../geneval/outputs', split="colors", version = version,cfg = cfg,image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "geneval_filter_color_attr":
        from .retrieval import get_geneval_filter
        return get_geneval_filter(root_dir = '../geneval/outputs', split="color_attr", version = version,cfg = cfg,image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "geneval_filter_position":
        from .retrieval import get_geneval_filter
        return get_geneval_filter(root_dir = '../geneval/outputs', split="position", version = version,cfg = cfg,image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "geneval_filter_counting":
        from .retrieval import get_geneval_filter
        return get_geneval_filter(root_dir = '../geneval/outputs', split="counting", version = version,cfg = cfg,image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "geneval_filter_single":
        from .retrieval import get_geneval_filter
        return get_geneval_filter(root_dir = '../geneval/outputs', split="single_object", version = version, cfg = cfg,image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "geneval_filter_two":
        from .retrieval import get_geneval_filter
        return get_geneval_filter(root_dir = '../geneval/outputs', split="two_object", version = version,cfg = cfg,image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "geneval_filter_two_subset":
        from .retrieval import get_geneval_filter
        return get_geneval_filter(root_dir = '../geneval/outputs', split="two_object_subset", version = version,cfg = cfg,image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

def get_images_dataset(dataset_name, image_preprocess=None, text_perturb_fn=None, image_perturb_fn=None, download=False, version = None, cfg= None, *args, **kwargs):
    if dataset_name == "whatsup_A":
        from .image import get_whatsup
        return get_whatsup(split="A",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)
    elif dataset_name == "whatsup_B":
        from .image import get_whatsup
        return get_whatsup(split="B",image_preprocess=image_preprocess, text_perturb_fn=text_perturb_fn, image_perturb_fn=image_perturb_fn, download=download, *args, **kwargs)