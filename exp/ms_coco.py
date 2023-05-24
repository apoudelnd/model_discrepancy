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


from datasets import load_dataset

# dataset = load_dataset("Apter/tiny_coco")
# print(dataset)
# print(dataset['test'][0])
# exit()
#load dataset
# dataset = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval")

dataset = load_dataset("HuggingFaceM4/COCO")
# Drop duplicates based on one colum

def get_data_loaders(dataset, batch_size = 16):
    data_loader = create_data_loader(dataset, batch_size = 16)
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


def main(data_loader, sets):

    """
    data_loader takes the type of data_loader [test, train, validation]
    sets - takes train, test, valid for naming purposes
    """

    ids = list()
    image_embeddings = list()
    text_embeddings = list()
    caption_ids = list()

    counter = 0

    print(f"Working on {sets} data_loader")

    for batch in tqdm(data_loader):
        counter += 1
        if counter == 1000: break

        image_path = batch['img_fpath']
        cocoid = batch['cocoid']
        caption = batch['caption']
        sentid = batch['sentid']

        #remove duplicates
        cocoid, unique_index = torch.unique(cocoid, sorted=False, return_inverse=True)

        # print(cocoid)
        #since there are 5 different text for an image - we only take the first one! -- 
        u_i = set(unique_index.tolist())
        # exit()
        image_path = [image_path[i] for i in u_i]
        caption = [caption[i] for i in u_i]
        sentid = [sentid[i] for i in u_i]
        
        # image_inputs = preprocess_image(image_path)
        # print(image_path)
        # exit()
        image_inputs = [preprocess_image(img) for img in image_path]

        text_inputs = preprocess_text(caption)

        image_inputs = torch.stack([x['pixel_values'] for x in image_inputs])
        image_inputs = image_inputs.squeeze(1)

        with torch.no_grad():
            image_features = model.get_image_features(image_inputs)
            text_features = model.get_text_features(**text_inputs)
        
        caption_ids.extend(sentid)
        ids.extend(cocoid)
        image_embeddings.extend(image_features)
        text_embeddings.extend(text_features)

    # Convert lists to tensors
    cid_tensor = torch.stack(caption_ids)
    id_tensor = torch.stack(ids)
    image_tensor = torch.stack(image_embeddings)
    text_tensor = torch.stack(text_embeddings)

    # exit()
    data_dict = {
        'ids': id_tensor,
        'sentid': cid_tensor,
        'image_embeddings': image_tensor,
        'text_embeddings': text_tensor
    }

    out_dir = os.path.join(os.getcwd(), 'results')
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