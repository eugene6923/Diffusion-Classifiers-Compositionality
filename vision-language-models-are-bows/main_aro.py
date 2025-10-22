import argparse
import os
import pandas as pd

from torch.utils.data import DataLoader

from model_zoo import get_model
from dataset_zoo import get_dataset
from misc import seed_all, _default_collate, save_scores

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--model-name", default="openai-clip:ViT-B/32", type=str)
    parser.add_argument("--dataset", default="VG_Relation", type=str)
    parser.add_argument("--seed", default=1, type=int)
    
    parser.add_argument("--download", action="store_true", help="Whether to download the dataset if it doesn't exist. (Default: False)")
    parser.add_argument("--save-scores", action="store_true", help="Whether to save the scores for the retrieval to analyze later.")
    parser.add_argument("--output-dir", default="./outputs", type=str)
    parser.add_argument("--version", default=None, type=str)
    parser.add_argument("--cfg", default=None, type=float)
    return parser.parse_args()

    
def main(args):
    seed_all(args.seed)
    
    model, image_preprocess = get_model(args.model_name, args.device)
    if args.dataset == "ours_colors" and args.version is None:
        raise ValueError("Please specify the version of the dataset to use.")
    dataset = get_dataset(args.dataset, image_preprocess=image_preprocess, download=False, version = args.version, cfg = args.cfg)
    
    # For some models we just pass the PIL images, so we'll need to handle them in the collate_fn. 
    collate_fn = _default_collate if image_preprocess is None else None
    
    joint_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # else:
    scores = model.get_retrieval_scores_batched(joint_loader)
    result_records = dataset.evaluate_scores(scores)
    
    for record in result_records:
        record.update({"Model": args.model_name, "Dataset": args.dataset, "Seed": args.seed, "Sequence": dataset.__len__()})
    
    
    
    if '/' in args.model_name:
        model_name = args.model_name.replace('/', '_')
        output_dir = os.path.join(args.output_dir, model_name.split(":")[1])
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{model_name}_{args.dataset}{args.version if args.version !=None else ''}{f'_cfg{args.cfg}' if args.cfg != None else ''}.csv")
    else:
        output_dir = os.path.join(args.output_dir, args.model_name.split(":")[1])
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{args.model_name}_{args.dataset}{args.version if args.version !=None else ''}{f'_cfg{args.cfg}' if args.cfg != None else ''}.csv")
    df = pd.DataFrame(result_records)
    print(f"Saving results to {output_file}")
    if os.path.exists(output_file):
        # all_df = pd.read_csv(output_file, index_col=0)
        # all_df = pd.concat([all_df, df])
        # all_df.to_csv(output_file)
        df.to_csv(output_file)

    else:
        df.to_csv(output_file)
        
    if args.save_scores:
        save_scores(scores, args)

    
if __name__ == "__main__":
    args = config()
    if 'geneval' in args.dataset and args.cfg == None:
        raise ValueError("Please specify the cfg of the dataset to use.")
    if 'geneval' in args.dataset and args.version == None:
        raise ValueError("Please specify the version of the dataset to use.")
    main(args)