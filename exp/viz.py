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

def neighbor_sim_exact_neigh(image_similarity, text_similarity, sets):

    top_k = 60
    normalized_sims = []

    for i in range(id_tensor.shape[0]):
        # print(id_tensor[i])
        # print(image_similarity[i])
        # exit()
        image_nearest_neighbors = torch.topk(torch.tensor(image_similarity[i]), k=top_k, largest=True)
        image_neighbors_indices = image_nearest_neighbors.indices
        image_top_k_values = image_similarity[i][image_neighbors_indices]

        image_embeddings = image_tensor[image_neighbors_indices]
        text_embeddings = text_tensor[image_neighbors_indices]
        similarities = cosine_similarity(image_embeddings, text_embeddings)
        # print(similarities)

        similarities_tensor = torch.from_numpy(similarities)

        diagonal_sum = torch.diag(similarities_tensor).sum()

        for sim in torch.diag(similarities_tensor):
            # Calculate the normalized similarity value
            normalized_sim = sim / diagonal_sum
            normalized_sims.append(normalized_sim)  # Add t
        
    # Flatten the normalized_sims list
    flattened_sims = np.array(normalized_sims).flatten()
    normalized_flattened_sims = (flattened_sims - np.min(flattened_sims)) / (np.max(flattened_sims) - np.min(flattened_sims))

    # print(sim)
    plt.figure()

    # Create distribution plot
    sns.set_style('darkgrid')

    # sns.distplot(flattened_sims, kde=True, rug=True)

    sns.distplot(normalized_flattened_sims, kde=True, rug=True)

    # Plotting the distribution of normalized similarity values
    # plt.hist(flattened_sims, bins=10)  # Adjust the number of bins as needed
    plt.xlabel('Normalized Similarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Normalized Similarity')
    plt.savefig(f'histogram_{sets}_{top_k}.png')
    # plt.show()
    


def neighbor_sim(image_similarity, text_similarity):

    top_k = 5

    # zip(range(image_tensor.shape[0]), range(text_tensor.shape[0])):
    #both of these are lists of tensors
    img_ls_of_indices = list()
    img_ls_of_values = list()

    txt_ls_of_indices = list()
    txt_ls_of_values = list()

    print("Top 5 neigbors, and the ratio - processing....")

    for i, j in zip(range(image_tensor.shape[0]), range(text_tensor.shape[0])):

        image_nearest_neighbors = torch.topk(torch.tensor(image_similarity[i]), k=top_k, largest=True)
        image_neighbors_indices = image_nearest_neighbors.indices
        image_top_k_values = image_similarity[i][image_neighbors_indices]
        # print(image_neighbors_indices)
        img_ls_of_indices.append(image_neighbors_indices)
        img_ls_of_values.append(image_top_k_values)

        # print(image_top_k_values)

        # Find the top k closest neighbors for the text
        text_nearest_neighbors = torch.topk(torch.tensor(text_similarity[j]), k=top_k, largest=True)
        text_neighbors_indices = text_nearest_neighbors.indices
        text_top_k_values = text_similarity[j][text_neighbors_indices]

        txt_ls_of_indices.append(text_neighbors_indices)
        txt_ls_of_values.append(text_top_k_values)


        # print(text_neighbors_indices)
        # print(text_top_k_values)
    
    img_plain_list_val = [elem for tensor in img_ls_of_values for elem in tensor.tolist()]
    txt_plain_list_val = [elem for tensor in txt_ls_of_values for elem in tensor.tolist()]
    # Take the element-wise ratio of the two lists
    ratio_list = [a / b for a, b in zip(img_plain_list_val, txt_plain_list_val)]

    return ratio_list

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


#train's already done! work on test and validation for now!!
for sets in ['train', 'test', 'validation']:

    print(f"Working on {sets} set")

    loaded_data = torch.load(f'../results_f/{sets}_data.pt')
    id_tensor = loaded_data['ids']
    image_tensor = loaded_data['image_embeddings']
    text_tensor = loaded_data['text_embeddings']
    sent_id = loaded_data['sentid']


    # Calculate cosine similarity between image embeddings
    image_similarity = cosine_similarity(image_tensor)
    
    # print(image_similarity)
    # print(image_similarity[0])
    # print(image_similarity.shape[0])  
    # exit()
    # Calculate cosine similarity between text embeddings
    text_similarity = cosine_similarity(text_tensor)

    neighbor_sim_exact_neigh(image_similarity, text_similarity, sets)

    exit()

    #for top 5 neighbors experiment
    ratio = neighbor_sim(image_similarity, text_similarity)
    plot_dist(ratio, "topk_exp", sets, "nosig")
    plot_dist(ratio, "topk_exp", sets, "sigmoid")


    #for normal experiment -- base case

    ratio = norm_exp(image_similarity, text_similarity)

    plot_dist(ratio.flatten(), "norm_exp", sets, "nosig")
    plot_dist(ratio.flatten(), "norm_exp", sets,  "sigmoid")

    print(f"Completed Working on {sets} set ************************")

    # exit()






# # Get the indices that sort the sigmoid_similarity_ratio array in ascending order
# sorted_indices = np.argsort(sigmoid_similarity_ratio)

# # Get the top 10 and bottom 10 combinations
# top_10_combinations = [(i, j) for i, j in zip(*np.unravel_index(sorted_indices[-10:], similarity_ratio.shape))]
# bottom_10_combinations = [(i, j) for i, j in zip(*np.unravel_index(sorted_indices[:10], similarity_ratio.shape))]

# print(top_10_combinations)
# print(bottom_10_combinations)