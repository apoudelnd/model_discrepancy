import transformers
from datasets import load_dataset
import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torchvision.transforms as transforms
import requests
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from dataloader import COCODataset, Fakeddit, create_data_loader, create_fakedata_loader
import datetime
import pandas
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import PIL
from PIL import UnidentifiedImageError


def get_data_loaders(dataset, batch_size = 20, label = 0):
    data_loader = create_fakedata_loader(dataset, batch_size = 20, label = label)
    return data_loader

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define the image and text preprocessing functions
def preprocess_image(img_fpath):
    # url = os.path.join(img_fpath)
    image = Image.open(requests.get(img_fpath, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")
    return inputs

def preprocess_text(text):
    return processor(text, padding=True, max_length=77, truncation=True, return_tensors="pt")

def keep_first_occurrence(lst):
    seen = {}
    result = []
    for item in lst:
        if item not in seen:
            result.append(item)
            seen[item] = True
    return result

def main(data_loader, sets, label):

    """
    data_loader takes the type of data_loader [test, train, validation]
    sets - takes train, test, valid for naming purposes
    """

    ids = list()
    image_embeddings = list()
    text_embeddings = list()
    counterid = list()
    caption_ids = list()
    captions = list()
    path = list()
    all_way_labels = list()


    counter = 0

    print(f"Working on {sets} data_loader")
    count = 0
    # Iterate over each element in the batch

    for batch in tqdm(data_loader):
        counter += 1
        if counter == 26: break

        image_path = batch['image_url']
        fakeid = batch['fakeid']
        clean_title = batch['clean_title']
        way_labels = batch['way_labels']

        set_ids = set()
        title_id = list()
        img_id = list()
        image_fpath = list()
        clean_titlef = list()
        labels = list()

        image_inputs = list()

        for i, idx in enumerate(fakeid):

            count+=1
            if idx not in set_ids:
                set_ids.add(idx)

                title_id.append(torch.tensor(count))
                img_id.append(fakeid[i])
                image_fpath.append(image_path[i])
                clean_titlef.append(clean_title[i])
                labels.append(way_labels[i])

        for img in image_fpath:
            try:
                image_inputs.append(preprocess_image(img))

            except:
                print(img)
        # image_inputs = [preprocess_image(img) for img in image_fpath]

        text_inputs = preprocess_text(clean_titlef)

        image_inputs = torch.stack([x['pixel_values'] for x in image_inputs])
        image_inputs = image_inputs.squeeze(1)

        with torch.no_grad():
            image_features = model.get_image_features(image_inputs)
            text_features = model.get_text_features(**text_inputs)
        
        path.extend(image_fpath)
        captions.extend(clean_titlef)
        caption_ids.extend(title_id)
        ids.extend(img_id)
        all_way_labels.extend(labels)
        image_embeddings.extend(image_features)
        text_embeddings.extend(text_features)

    # Stack the caption tensors

    path_tensor = path
    caption_tensor = captions
    id_tensor = torch.stack(ids)
    cid_tensor = torch.stack(caption_ids)
    labels_tensor = torch.stack(all_way_labels)
    image_tensor = torch.stack(image_embeddings)
    text_tensor = torch.stack(text_embeddings)


    #sentid here -- title_id self generated using count
    #ids -- real ids from the dataset itself
    data_dict = {
        'ids': id_tensor,
        'sentid': cid_tensor,
        'image_embeddings': image_tensor,
        'text_embeddings': text_tensor,
        'captions': caption_tensor,
        'img_path': path_tensor
    }

    out_dir = os.path.join(os.getcwd(), 'results_final', 'fakeddit', f'{label}')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    out_file = os.path.join(
        out_dir, f'{sets}_data.pt'
        )

    # Save the dictionary using torch.save
    torch.save(data_dict, out_file)


    print(f'{sets} done')


if __name__ == "__main__":

    start_time = datetime.datetime.now()
    fakeddit_path = '../multimodal_only_samples'
    all_files = os.listdir(fakeddit_path)
    print(all_files)

    columns_to_read = ['created_utc', 'clean_title', 'image_url', '6_way_label']

    for sets in ['train', 'test', 'validate']:
        
        label = 2

        data_path = os.path.join(fakeddit_path, f'multimodal_{sets}.tsv')
        dataset = pd.read_csv(data_path, delimiter = '\t')
        
        start_time = datetime.datetime.now()

        data_loader = get_data_loaders(dataset, batch_size = 20, label = label)
        main(data_loader, sets, label = label)
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print(f"Total time taken by {sets} dataset: {total_time}")

    # end_time = datetime.datetime.now()
    # total_time = end_time - start_time
    # print(f"Total time taken {total_time}")