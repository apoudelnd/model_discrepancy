import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.metrics.pairwise import cosine_similarity
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
import seaborn as sns
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

# def neighbor_sim_exact_neigh(image_similarity, text_similarity):

#     top_k = 5


#     for i in range(id_tensor.shape[0]):
#         image_nearest_neighbors = torch.topk(torch.tensor(image_similarity[i]), k=top_k, largest=True)
#         image_neighbors_indices = image_nearest_neighbors.indices



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

        # print("looop")
        # print(i,j)

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

    #using sigmoid to constraint the range from 0-1
    # sigmoid_similarity_ratio = sigmoid_list(similarity_ratio.flatten())

    # distance_ratio = euclidean_dist(image_similarity, text_similarity)

    # sigmoid_dist_ratio = sigmoid_list(distance_ratio.flatten())

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
    plt.savefig(f'./plots/{exp_type}/n_{sets}_{cal_type}.png')


#train's already done! work on test and validation for now!!
for sets in ['test', 'validation']:

    print(f"Working on {sets} set")

    loaded_data = torch.load(f'../results/{sets}_data.pt')
    id_tensor = loaded_data['ids']
    image_tensor = loaded_data['image_embeddings']
    text_tensor = loaded_data['text_embeddings']


    # Calculate cosine similarity between image embeddings
    image_similarity = cosine_similarity(image_tensor)
    print(image_similarity.shape)
    print(image_similarity.shape[0])  

    # Calculate cosine similarity between text embeddings
    text_similarity = cosine_similarity(text_tensor)


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