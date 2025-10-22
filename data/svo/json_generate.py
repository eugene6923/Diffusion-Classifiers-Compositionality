import pandas as pd
import requests
import json
import os
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor, as_completed

def is_downloadable(url):
    """
    Checks if a given url is downloadable by making a HEAD request
    """
    print(f"Checking URL: {url}")
    try:
        h = requests.head(url, allow_redirects=True)
        header = h.headers
        content_type = header.get('content-type')
        if 'text' in content_type.lower():
            print(f"URL is not an image file: {url}")
            return False
        if 'html' in content_type.lower():
            print(f"URL is an HTML file, not image: {url}")
            return False
        print(f"URL is valid: {url}")
        return True
    except:
        print(f"URL checking failed: {url}")
        return False

def download_image(url, file_name):
    """
    Downloads an image from a given url and saves it with a given file name
    """
    print(f"Downloading image from URL: {url}")
    response = requests.get(url, stream=True)
    file = open(file_name, 'wb')
    response.raw.decode_content = True
    file.write(response.raw.read())
    file.close()
    print(f"Image saved as: {file_name}")

import os
import json
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_row(index, row, image_dir, json_data):
    pos_url = row['pos_url']
    neg_url = row['neg_url']
    pos_image_id = str(row['pos_image_id']) + '.jpg'
    neg_image_id = str(row['neg_image_id']) + '.jpg'

    # Check if the images exist
    if os.path.exists(image_dir + pos_image_id) and os.path.exists(image_dir + neg_image_id):
        try:
            img1 = Image.open(image_dir + pos_image_id).convert("RGB")
        except UnidentifiedImageError:
            os.remove(image_dir + pos_image_id)
            return f"Skipping row {index + 1} due to invalid images."

        try:
            img1 = Image.open(image_dir + neg_image_id).convert("RGB")
        except UnidentifiedImageError:
            os.remove(image_dir + neg_image_id)
            return f"Skipping row {index + 1} due to invalid images."

        # Determine neg_type
        neg_type = "subj" if row['subj_neg'] else ("verb" if row['verb_neg'] else "obj")

        # Append a new dictionary to our list
        json_data.append({
            "pos_id": pos_image_id,
            "neg_id": neg_image_id,
            "sentence": row['sentence'],
            "pos_triplet": row['pos_triplet'],
            "neg_triplet": row['neg_triplet'],
            "neg_type": neg_type
        })

        return f"Row {index + 1} processed successfully."
    return f"Skipping row {index + 1} due to invalid images."

def main():
    # Load the csv
    print("Loading CSV file...")
    data = pd.read_csv('data/svo/svo_probes.csv')
    print("CSV file loaded successfully.")
    json_data = []

    image_dir = 'data/svo/images/'
    max_workers=16
    # Use ThreadPoolExecutor to process rows concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor for processing
        futures = [executor.submit(process_row, index, row, image_dir, json_data) 
                    for index, row in data.iterrows()]

        # Use tqdm to track progress
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result:
                print(result)

    # Write out the json file
    print("Writing data to JSON file...")
    with open('data/svo/svo.json', 'w') as outfile:
        json.dump(json_data, outfile, indent=4)
    print("JSON file created successfully.")
    # Limiting to first 10 rows for testing
    # data = data.head(10)

    # Initialize an empty list to hold valid entries
    # json_data = []

    # # Image directory
    # image_dir = 'data/svo/images/'
    # if not os.path.exists(image_dir):
    #     print(f"Creating directory: {image_dir}")
    #     os.makedirs(image_dir)

    # for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    #     pos_url = row['pos_url']
    #     neg_url = row['neg_url']
    #     pos_image_id = str(row['pos_image_id']) + '.jpg'
    #     neg_image_id = str(row['neg_image_id']) + '.jpg'
        # print(image_dir + neg_image_id)
        # if os.path.exists(image_dir + pos_image_id) and os.path.exists(image_dir + neg_image_id):
        # if os.path.exists(image_dir + pos_image_id) and neg_image_id == "1901.jpg":
            # try:
            #     Image.open(image_dir + pos_image_id)
            # except:
            #     os.remove(image_dir + pos_image_id)
            #     try:
            #         Image.open(image_dir + neg_image_id)
            #     except:
            #         os.remove(image_dir + neg_image_id)
            #         continue
            #     continue
            # try: 
            #     Image.open(image_dir + neg_image_id).convert("RGB")
            # except UnidentifiedImageError:
            #     os.remove(image_dir + pos_image_id)
            #     print(image_dir + pos_image_id)
            #     input()
            #     try:
            #         Image.open(image_dir + neg_image_id).convert("RGB") 
            #     except:
            #         os.remove(image_dir + neg_image_id)
            #         print(image_dir + neg_image_id)
            #         input()
            #     continue
            # input()
        # if is_downloadable(pos_url) and is_downloadable(neg_url):
            # Download and save images
            
            # download_image(pos_url, image_dir + pos_image_id)
            # download_image(neg_url, image_dir + neg_image_id)
            
            # Determine neg_type
        #     neg_type = "subj" if row['subj_neg'] else ("verb" if row['verb_neg'] else "obj")
            
        #     # Append a new dictionary to our list
        #     json_data.append({
        #         "pos_id": pos_image_id,
        #         "neg_id": neg_image_id,
        #         "sentence": row['sentence'],
        #         "pos_triplet": row['pos_triplet'],
        #         "neg_triplet": row['neg_triplet'],
        #         "neg_type": neg_type
        #     })
        #     print(f"Row {index + 1} processed successfully.")
        # else:
        #     print(f"Skipping row {index + 1} due to invalid URLs.")

        # # Write out the json file
        # print("Writing data to JSON file...")
        # with open('data/svo/svo.json', 'w') as outfile:
        #     json.dump(json_data, outfile, indent=4)
        # print("JSON file created successfully.")

if __name__ == '__main__':
    main()
