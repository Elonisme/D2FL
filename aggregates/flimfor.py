import copy
import math
from collections import Counter, OrderedDict

import hdbscan
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from decorators.timing import record_time


def replace_fc_layers(normal_model, fc_model):
    for fc_key, fc_value in fc_model.items():
        if fc_key in normal_model:
            normal_model[fc_key] = fc_value
    return normal_model


def extract_fc_layers(global_model):
    fc_layers = OrderedDict()
    for key, value in global_model.items():
        if "linear" in key or "fc" in key or "classifier" in key:
            fc_layers.update({key: value})
    return fc_layers


def extract_feature_layers(global_model):
    feature_layers = OrderedDict()
    for key, value in global_model.items():
        if "linear" not in key and "fc" not in key and "classifier" not in key:
            feature_layers.update({key: value})
    return feature_layers


def create_fc_layers_models(model_list, type_of_model):
    if type_of_model == 'global_model':
        fc_layers_model = extract_fc_layers(model_list)
        return fc_layers_model
    else:
        fc_model_list = []
        for model in model_list:
            fc_layers_model = extract_fc_layers(model)
            fc_model_list.append(fc_layers_model)
        return fc_model_list



def get_update_norm(user_model, global_model):
    squared_sum = 0
    for key in user_model.keys():
        squared_sum += torch.sum(torch.pow(user_model[key] - global_model[key], 2)).item()
    update_norm = math.sqrt(squared_sum)
    return update_norm

def vectorize_net(model_weight):
    vectorized_weight = []
    for key, value in model_weight.items():
        flattened_tensor = torch.flatten(value)
        vectorized_weight.append(flattened_tensor)
    vectorized_tensor = torch.cat(vectorized_weight, dim=0)
    return vectorized_tensor


@record_time
def shannon_module(user_model_weights, fc_model_list, global_fc_model, clients_entropy):
    # label entropy
    clients_entropy = np.array(clients_entropy).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0, init='k-means++', n_init='auto')
    kmeans.fit(clients_entropy)

    entropy_class = kmeans.labels_
    centroids = kmeans.cluster_centers_
    smaller_centroid = np.min(centroids, axis=0)
    larger_centroid = np.max(centroids, axis=0)
    larger_centroid_label = np.argmax(centroids)
    print(f"smaller_centroid: {smaller_centroid}")
    print(f"large_centroid: {larger_centroid}")

    # if larger_centroid - smaller_centroid > 0.5:
    schema = 'entropy'

    if schema == 'entropy':
        entropy_score = []
        for item in entropy_class:
            if item == larger_centroid_label:
                entropy_score.append(False)
            else:
                entropy_score.append(True)
        final_score = entropy_score
    elif schema == 'cosine':
        # cosine similarity
        user_num = len(user_model_weights)
        cos_tensor = torch.zeros(user_num, user_num)
        user_model_fc_vec_weights = [vectorize_net(user_model_weight) for user_model_weight in fc_model_list]
        global_model_fc_vec_weights = vectorize_net(global_fc_model)
        for i, user_x_model_fc_vec_weight in enumerate(user_model_fc_vec_weights):
            update_x = user_x_model_fc_vec_weight - global_model_fc_vec_weights
            for j, user_y_model_fc_vec_weight in enumerate(user_model_fc_vec_weights):
                fc_update_y = user_y_model_fc_vec_weight - global_model_fc_vec_weights
                adjust_cosine_similarity = torch.cosine_similarity(update_x-torch.mean(update_x), fc_update_y-torch.mean(update_x), dim=0).detach().cpu()
                cos_tensor[i][j] = adjust_cosine_similarity

        cosine_cluster = hdbscan.HDBSCAN(min_cluster_size=2)
        cosine_cluster_labels = cosine_cluster.fit_predict(cos_tensor)
        cosine_majority = Counter(cosine_cluster_labels)
        cosine_most_common_clusters = cosine_majority.most_common(user_num)
        cosine_candidate = cosine_most_common_clusters[0][0]
        cosine_score = np.array([True if i == cosine_candidate else False for i in cosine_cluster_labels])
        print(f"cosine_score: {cosine_score}")
        final_score = cosine_score
    else:
        raise ValueError(f"Unrecognized schema: {schema}")

    # # 创建数据框
    # data = {'Entropy': clients_entropy.flatten(), 'Entropy Cluster': entropy_class}
    # df = pd.DataFrame(data)
    #
    # # 绘制散点图
    # sns.set_theme(style='white')
    # golden_ratio = 1.618033988749895
    # width = 10
    # plt.figure(figsize=(width, width/golden_ratio))
    # if larger_centroid_label == 0:
    #     palette = ['#377eb8', '#e41a1c']
    # else:
    #     palette = ['#e41a1c', '#377eb8']
    #
    # scatter = sns.scatterplot(data=df, x='Entropy', y='Entropy Cluster', hue='Entropy Cluster', palette=palette, legend="full", s=100)
    #
    # plt.xlabel('Entropy')
    # plt.ylabel('Cluster')
    # plt.yticks([0, 1])
    # plt.title('Entropy Clustering')
    #
    # handles, label = scatter.get_legend_handles_labels()
    # if int(label[0]) == larger_centroid_label:
    #     label[0] = 'Normal Cluster'
    #     label[1] = 'Malicious Cluster'
    # else:
    #     label[0] = 'Malicious Cluster'
    #     label[1] = 'Normal Cluster'
    #
    # scatter.legend(handles=handles, labels=label, loc='best')
    # path = "save/csv/efficient/entropy_clustering.png"
    # plt.savefig(path, dpi=900, bbox_inches='tight')
    # plt.show()

    print(f"guess malicious: {final_score}")
    safe_model_weights = []
    for i, malicious_slog in enumerate(final_score):
        if malicious_slog == False:
            safe_model_weights.append(user_model_weights[i])

    next_global_weight = copy.deepcopy(safe_model_weights[0])
    for k in next_global_weight.keys():
        for i in range(1, len(safe_model_weights)):
            next_global_weight[k] += safe_model_weights[i][k]
        next_global_weight[k] = torch.div(next_global_weight[k], len(safe_model_weights))

    return next_global_weight, final_score


def flclaude(user_model_weights, global_model_weight, entropy):
    global_fc_model = create_fc_layers_models(global_model_weight, "global_model")
    fc_model_list = create_fc_layers_models(copy.deepcopy(user_model_weights), "fc_model_list")
    new_global_model_weight, malicious_score = shannon_module(user_model_weights, fc_model_list, global_fc_model, entropy)
    runtime = shannon_module.runtime
    return new_global_model_weight, runtime, malicious_score
