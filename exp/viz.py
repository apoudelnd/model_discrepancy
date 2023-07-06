import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.metrics.pairwise import cosine_similarity
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
import seaborn as sns
import os
# import sys
# sys.path.append('./')

print("Loading data ...")
# Load the saved dictionary using torch.load

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_list(lst):
    return [sigmoid(x) for x in lst]

def euclidean_dist(image_simmilarity, text_similarity):

    img_dist = np.sqrt(np.abs(2 * (1-image_simmilarity)))+ 0.0000001
    text_dist = np.sqrt(np.abs(2 * (1-text_similarity)))+ 0.0000001

    distance_ratio  = img_dist / text_dist
    
    normalized_ratio = distance_ratio / np.max(distance_ratio)
    return distance_ratio

#compute cosine_similarity between all pair of rows
def img_text_sim(image_similarity, text_similarity):
    return cosine_similarity(image_similarity, text_similarity)


import torch

def find_topk_neighbors(embedding1, embedding2, k):
    # Convert input tensors to float dtype
    embedding1 = embedding1.float()
    embedding2 = embedding2.float()

    # Compute cosine similarity between embedding1 and embedding2
    similarity_matrix = torch.cosine_similarity(embedding1.unsqueeze(1), embedding2.unsqueeze(0), dim=2)

    topk_similarities, topk_indices = torch.topk(similarity_matrix, k, dim=1)

    return topk_indices


def combine_neighbors(image_sim0, img_sim1, text_sim0, text_sim1, sent_id0, sent_id1):

    k = 2
    topk_neighbors = find_topk_neighbors(image_sim0, img_sim1, k)

    ratios = []

    for i, j, neighbors in zip(image_sim0, text_sim0, topk_neighbors):
        topk_text_embeddings = text_sim1[neighbors]
        topk_image_embeddings = img_sim1[neighbors]

        sim_query = torch.cosine_similarity(i.unsqueeze(0), j.unsqueeze(0), dim=1)
        # print(sim_query)

        for t_embed, i_embed in zip(topk_text_embeddings, topk_image_embeddings):
            ratio = sim_query / torch.cosine_similarity(t_embed.unsqueeze(0), i_embed.unsqueeze(0), dim=1)
            ratios.append(ratio.item())

    # print(ratios)

    plt.figure()

    out_path = os.path.join(os.getcwd(), 'neigh_plots', 'fakeddit', 'test05')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # Create distribution plot
    sns.set_style('darkgrid')

    sns.distplot(ratios, kde=True, rug=True)
    plt.xlabel('Normalized Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Normalized Similarity')
    final_path = os.path.join(out_path, 'test05.png')
    plt.savefig(final_path)

    exit()
        



def neighbor_sim_exact_neigh(image_similarity, text_similarity, sets, id_tensor, sent_id, caption_tensor, path_tensor, project, label):

    top_k = 2
    normalized_sims = []
    ratio_list = []
    new_ratio_list = list()
    top_ratios = []
    bottom_ratios = []
    sentence_ids = sent_id

    # for i in range(len(id_tensor)):
    for i in range(id_tensor.shape[0]): 

        image_nearest_neighbors = torch.topk(torch.tensor(image_similarity[i]), k=top_k+1, largest=True)
        image_neighbors_indices = image_nearest_neighbors.indices
        # print(image_neighbors_indices)
        image_top_k_values = image_similarity[i][image_neighbors_indices]

        coco_ids = id_tensor[image_neighbors_indices]
        sent_ids = sentence_ids[image_neighbors_indices]
        caption = [caption_tensor[i] for i in image_neighbors_indices.tolist()]
        path = [path_tensor[i] for i in image_neighbors_indices.tolist()]

        # print(image_top_k_values)

        image_embeddings = image_tensor[image_neighbors_indices]
        text_embeddings = text_tensor[image_neighbors_indices]
        similarities = cosine_similarity(image_embeddings, text_embeddings)

        similarities_tensor = torch.from_numpy(similarities)
        # print(similarities_tensor)
        sim_query = torch.diag(similarities_tensor)[0]

        # ratio_list = [sim_query/sim for sim in torch.diag(similarities_tensor)[1:]]

        for sim, coco_id, sent_id, cap, pat in zip(torch.diag(similarities_tensor)[1:], coco_ids[1:], sent_ids[1:], caption[1:], path[1:]):
            ratio = sim_query / sim
            ratio_list.append(ratio)
            #caption[0] and path[0] are for the query! 
            new_ratio_list.append((ratio, coco_id, sent_id, cap, pat, caption[0], path[0]))
        
        
    ratio_sorted = sorted(new_ratio_list)

    top_ratios.extend(ratio_sorted[-10:])  # Append top 10 ratios
    bottom_ratios.extend(ratio_sorted[:10])  # Append bottom 10 ratios

    # Get the middle 10 ratios
    middle_10_ratios = ratio_sorted[len(ratio_sorted)//2 - 5: len(ratio_sorted)//2 + 5]

    # Print the middle 10 ratios
    print("Middle ------------------")
    print(middle_10_ratios)

    #extract the items from the dictionary work on this later
    print("Top ratio ----------------")
    print(top_ratios)

    print("Botoom ratio -------------")
    print(bottom_ratios)

    # exit()

    flattened_sims = np.array(ratio_list).flatten()
    normalized_flattened_sims = (flattened_sims - np.min(flattened_sims)) / (np.max(flattened_sims) - np.min(flattened_sims))

    if project == 'fakeddit':
        out_path = os.path.join(os.getcwd(), 'neigh_plots', 'fakeddit', f'{label}')
    else:
        out_path = os.path.join(os.getcwd(), 'neigh_plots')
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # print(sim)
    print(len(ratio_list))
    plt.figure()

    # Create distribution plot
    sns.set_style('darkgrid')

    sns.distplot(flattened_sims, kde=True, rug=True)
    plt.xlabel('Normalized Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Normalized Similarity')
    final_path = os.path.join(out_path, f'kn_{sets}_{top_k}.png')
    plt.savefig(final_path)
    # plt.show()

def random_neighbors(image_similarity, text_similarity, sets, id_tensor, sent_id, caption_tensor, path_tensor, normalize):

    top_k = 5
    sentence_ids = sent_id
    all_ratios = list()

    #10k samples -- run this
    for i in range(id_tensor.shape[0]):

        image_nearest_neighbors = torch.topk(torch.tensor(image_similarity[i]), k=top_k+1, largest=True)
        image_neighbors_indices = image_nearest_neighbors.indices
        # print(image_neighbors_indices)
        image_top_k_values = image_similarity[i][image_neighbors_indices]

        coco_ids = id_tensor[image_neighbors_indices]
        sent_ids = sentence_ids[image_neighbors_indices]
        caption = [caption_tensor[i] for i in image_neighbors_indices.tolist()]
        path = [path_tensor[i] for i in image_neighbors_indices.tolist()]


        image_embeddings = image_tensor[image_neighbors_indices]
        text_embeddings = text_tensor[image_neighbors_indices]

        similarities = cosine_similarity(image_embeddings, text_embeddings)
        # all_similarities = cosine_similarity(image_embeddings, text_tensor)

        similarities_tensor = torch.from_numpy(similarities)
    #     # print(similarities_tensor)
        sim_query = torch.diag(similarities_tensor)[0]

        import random
        random.seed(42)

        num_random_texts = 5
        random_indices = random.sample(range(len(text_tensor)), num_random_texts)

        random_text_tensors = text_tensor[random_indices]
        random_sent_ids = [sentence_ids[index] for index in random_indices]

        similarities_ran = cosine_similarity(image_embeddings, random_text_tensors)

        ratio_values = sim_query / similarities_ran.flatten()
        # print(ratio_values)

        flattened_sims = np.array(ratio_values).flatten()
       
        all_ratios.extend(flattened_sims)

    normalized_flattened_sims = (all_ratios - np.min(all_ratios)) / (np.max(all_ratios) - np.min(all_ratios))
    out_path = os.path.join(os.getcwd(), 'ran_plots')
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # print(sim)
    print(len(all_ratios))
    plt.figure()

    # Create distribution plot
    sns.set_style('darkgrid')

    if normalize:
        sns.distplot(normalized_flattened_sims, kde=True, rug=True)
        final_path = os.path.join(out_path, f'kn_{sets}_{top_k}nom.png')
    else:
        sns.distplot(all_ratios, kde=True, rug=True)
        final_path = os.path.join(out_path, f'kn_{sets}_{top_k}unnom.png')

    plt.xlabel('Normalized Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Normalized Similarity')
    # if normalize:
    #     final_path = os.path.join(out_path, f'kn_{sets}_{top_k}nom.png')
    # else:
    #     final_path = os.path.join(out_path, f'kn_{sets}_{top_k}nom.png')

    plt.savefig(final_path)


def norm_exp(image_similarity, text_similarity):

    #fix me later for euclidean -- for now just the cosine

    # Calculate the ratio of image similarity to text similarity
    similarity_ratio = image_similarity / text_similarity

    return similarity_ratio


def plot_dist(ratio, exp_type, sets, cal_type):

    #later add os.path for exp_type and others here

    if cal_type == "sigmoid":
        ratio = sigmoid_list(ratio)

    plt.figure()

    # Create distribution plot
    sns.set_style('darkgrid')

    sns.distplot(ratio, kde=True, rug=True)
    plt.xlabel('Similarity Ratio')
    plt.ylabel('Density')
    plt.title('Neighbors distribuition')
    #maybe write a function later for making dir - done manually for now
    out_path = f'./plot_f/{exp_type}/n_{sets}'
    # path = os.path.join(out_path, f'{cal_type}.png')
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    f_path = os.path.join(out_path, f'{cal_type}.png')
    # plt.savefig(f'./plot_f/{exp_type}/n_{sets}_{cal_type}.png')
    plt.savefig(f_path)


project = "fakeddit"
label = 5


if __name__ == "__main__":

    for project in ['fakeddit', 'ms_coco']:
        print(f"Working on {project}")
        for sets in ['train', 'test', 'validate']:
            if project == "ms_coco":
                loaded_data = torch.load(f'./results_final/{sets}_data.pt')
            else:
                loaded_data = torch.load(f'./results_final/fakeddit/{label}/{sets}_data.pt')
        
            id_tensor = loaded_data['ids']
            image_tensor = loaded_data['image_embeddings']
            text_tensor = loaded_data['text_embeddings']
            sent_id = loaded_data['sentid']


            caption_tensor = loaded_data['captions']
            path_tensor = loaded_data['img_path']

            #loading for label == 0

            loaded_data0 = torch.load(f'./results_final/fakeddit/0/{sets}_data.pt')
            image_tensor0 = loaded_data0['image_embeddings']
            text_tensor0 = loaded_data0['text_embeddings']
            sent_id0 = loaded_data0['sentid']

            combine_neighbors(image_tensor0, image_tensor, text_tensor0, text_tensor, sent_id0, sent_id)

            exit()

            # Calculate cosine similarity between image embeddings
            image_similarity = cosine_similarity(image_tensor)
    
            text_similarity = cosine_similarity(text_tensor)

            # random_neighbors(image_similarity, text_similarity, sets, id_tensor, sent_id, caption_tensor, path_tensor, False)
            neighbor_sim_exact_neigh(image_similarity, text_similarity, sets, id_tensor, sent_id, caption_tensor, path_tensor, project, label = label)
            exit()
    



