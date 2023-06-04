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
from dataloader import COCODataset, create_data_loader
import datetime
import pandas
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
import PIL
from PIL import UnidentifiedImageError


from datasets import load_dataset

# dataset = load_dataset("Apter/tiny_coco")
# print(dataset)
# print(dataset['test'][0])
# exit()
#load dataset
# dataset = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval")

dataset = load_dataset("HuggingFaceM4/COCO")
# print(dataset)
# print(dataset['test'][:10])
# exit()
# Drop duplicates based on one colum

def get_data_loaders(dataset, batch_size = 20):
    data_loader = create_data_loader(dataset, batch_size = 20)
    return data_loader

# data_loader = create_data_loader(dataset['test'], batch_size = 16)

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define the image and text preprocessing functions
def preprocess_image(img_fpath):
    url = os.path.join(img_fpath)
    image = Image.open(requests.get(url, stream=True).raw)
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

def main(data_loader, sets):

    """
    data_loader takes the type of data_loader [test, train, validation]
    sets - takes train, test, valid for naming purposes
    """

    ids = list()
    image_embeddings = list()
    text_embeddings = list()
    caption_ids = list()
    captions = list()
    path = list()


    counter = 0

    print(f"Working on {sets} data_loader")

    for batch in tqdm(data_loader):
        counter += 1
        if counter == 1000: break

        image_path = batch['img_fpath']
        cocoid = batch['cocoid']
        caption = batch['caption']
        sentid = batch['sentid']

        set_ids = set()
        cocoidf = list()
        image_fpath = list()
        captionf = list()
        sentidf = list()
        image_inputs = list()

        for i, idx in enumerate(cocoid.tolist()):
            if idx not in set_ids:
                set_ids.add(idx)

                cocoidf.append(cocoid[i])
                image_fpath.append(image_path[i])
                captionf.append(caption[i])
                sentidf.append(sentid[i])

        for img in image_fpath:
            try:
                image_inputs.append(preprocess_image(img))

            except PIL.UnidentifiedImageError:
                print(img)
        
        # image_inputs = [preprocess_image(img) for img in image_fpath]

        text_inputs = preprocess_text(captionf)

        image_inputs = torch.stack([x['pixel_values'] for x in image_inputs])
        image_inputs = image_inputs.squeeze(1)

        with torch.no_grad():
            image_features = model.get_image_features(image_inputs)
            text_features = model.get_text_features(**text_inputs)
        
        path.extend(image_fpath)
        captions.extend(captionf)
        caption_ids.extend(sentidf)
        ids.extend(cocoidf)
        image_embeddings.extend(image_features)
        text_embeddings.extend(text_features)

    # Convert strings to tensors

    # Stack the caption tensors
    path_tensor = path
    caption_tensor = captions
    cid_tensor = torch.stack(caption_ids)
    id_tensor = torch.stack(ids)
    image_tensor = torch.stack(image_embeddings)
    text_tensor = torch.stack(text_embeddings)

    # exit()
    data_dict = {
        'ids': id_tensor,
        'sentid': cid_tensor,
        'image_embeddings': image_tensor,
        'text_embeddings': text_tensor,
        'captions': caption_tensor,
        'img_path': path_tensor
    }

    out_dir = os.path.join(os.getcwd(), 'results_final')
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

    for sets in ['train', 'test', 'validation']:
        start_time = datetime.datetime.now()

        data_loader = get_data_loaders(dataset[sets], batch_size = 20)
        main(data_loader, sets)
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        print(f"Total time taken by {sets} dataset: {total_time}")

    # end_time = datetime.datetime.now()
    # total_time = end_time - start_time
    # print(f"Total time taken {total_time}")