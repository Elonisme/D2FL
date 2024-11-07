import copy
from collections import OrderedDict

import numpy
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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



def vectorize_net(model_weight):
    vectorized_weight = []
    for key, value in model_weight.items():
        flattened_tensor = torch.flatten(value)
        vectorized_weight.append(flattened_tensor)
    vectorized_tensor = torch.cat(vectorized_weight, dim=0)
    return vectorized_tensor

def cosine_filter(user_model_weight_vecs):
    # cosine similarity
    user_num = len(user_model_weight_vecs)
    half_user_num = int(user_num // 2)
    cosine_similarity_matrix = np.zeros((user_num, user_num))
    for i in range(0, user_num):
        for j in range(i + 1, user_num):
            adjust_cosine_similarity = torch.cosine_similarity(user_model_weight_vecs[i] - torch.mean(user_model_weight_vecs[i]),
                                                               user_model_weight_vecs[j] - torch.mean(user_model_weight_vecs[j]), dim=0).detach().cpu()
            cosine_similarity_matrix[i, j] = adjust_cosine_similarity
            cosine_similarity_matrix[j, i] = adjust_cosine_similarity

    sum_dists = []
    for i in range(user_num):
        sorted_indices = np.argsort(cosine_similarity_matrix[i])
        sum_dists = np.append(sum_dists, np.sum(cosine_similarity_matrix[i, sorted_indices[0:half_user_num]]))

    dist_indexes = np.argsort(sum_dists)
    maliciousness_list = []
    for i in range(user_num):
        if i not in dist_indexes[0:half_user_num]:
            maliciousness_list.append(False)
        else:
            maliciousness_list.append(True)

    return maliciousness_list


def loss_filter(labels, centers_1d):
    maliciousness_list = []
    min_cluster = np.argmin(centers_1d)
    for cluster in labels:
        if cluster == min_cluster:
            maliciousness_list.append(False)
        else:
            maliciousness_list.append(True)

    return maliciousness_list

def simple_filter(model_loss_list):
    maliciousness_list = []
    for loss_group in model_loss_list:
        bad_loss = 0
        for loss in loss_group:
            if loss < 0.01:
                bad_loss += 1
        threshold = 0
        if bad_loss > threshold:
            maliciousness_list.append(True)
        else:
            maliciousness_list.append(False)
    return maliciousness_list

def judgment_module(model_weights_list, model_loss_list):
    features = []
    for group in model_loss_list:
        variance = np.var(group, ddof=1)
        min = np.min(group)
        max = np.max(group)
        median = np.median(group)
        features.append(np.array([variance, min, max, median]))

    kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')
    kmeans.fit(features)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # pca = PCA(n_components=2)
    # features_2d = pca.fit_transform(features)
    # centers_2d = pca.transform(centers)
    #
    # plt.figure(figsize=(10, 6))
    # scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis')
    # plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=200, label='Centers')
    # plt.colorbar(scatter)
    # plt.title("K-means Clustering of Data")
    # plt.xlabel("Principal Component 1")
    # plt.ylabel("Principal Component 2")
    # plt.legend()
    # plt.show()

    pca = PCA(n_components=1)
    _ = pca.fit_transform(features)
    centers_1d = pca.transform(centers)
    smaller_center = np.min(centers_1d)
    larger_centroid = np.max(centers_1d)
    print(f"smaller_center: {smaller_center}")
    print(f"larger_centroid: {larger_centroid}")

    if larger_centroid - smaller_center > 0.5:
        maliciousness_list = loss_filter(labels, centers_1d)
        print(f"loss_filter: {maliciousness_list}")
    else:
        user_model_fc_weights = create_fc_layers_models(model_weights_list, "user_model")
        user_model_fc_weight_vecs = [vectorize_net(fc_vec) for fc_vec in user_model_fc_weights]
        maliciousness_list = cosine_filter(user_model_fc_weight_vecs)
        print(f"cosine_filter: {maliciousness_list}")
    return maliciousness_list


@record_time
def floss_module(model_weights_list, model_loss_list):
    maliciousness_list = judgment_module(model_weights_list, model_loss_list)
    safe_model_weights_list = []
    for i, bool_value in enumerate(maliciousness_list):
        if bool_value is False:
            safe_model_weights_list.append(model_weights_list[i])

    next_global_weights = copy.deepcopy(safe_model_weights_list[0])
    for k in next_global_weights.keys():
        for i in range(1, len(safe_model_weights_list)):
            next_global_weights[k] += safe_model_weights_list[i][k]
        next_global_weights[k] = torch.div(next_global_weights[k], len(safe_model_weights_list))
    return next_global_weights



def floss(model_weights_list, model_loss_list):
    new_global_weights = floss_module(model_weights_list, model_loss_list)
    runtime = floss_module.runtime
    return copy.deepcopy(new_global_weights), runtime
