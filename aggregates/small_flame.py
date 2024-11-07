import copy
from collections import OrderedDict, Counter

import hdbscan
import torch

from decorators.timing import record_time


def adjust_cosine_similarity(vec1, vec2):
    mean_vec1 = vec1.mean(dim=0)
    mean_vec2 = vec2.mean(dim=0)
    adjust_vec1 = vec1 - mean_vec1
    adjust_vec2 = vec2 - mean_vec2
    return torch.cosine_similarity(adjust_vec1, adjust_vec2, dim=0)

def vectorize_net(model_weight):
    vectorized_weight = []
    for key, value in model_weight.items():
        flattened_tensor = torch.flatten(value)
        vectorized_weight.append(flattened_tensor)
    vectorized_tensor = torch.cat(vectorized_weight, dim=0)
    return vectorized_tensor

@record_time
def small_flame_module(model_list, global_model, model_fc_list, global_fc_model, device=None):
    pre_global_fc_model = copy.deepcopy(global_fc_model)
    # 将全局模型向量化
    pre_global_fc_model_vectorized = vectorize_net(pre_global_fc_model)

    # 创建全零张量
    user_num = len(model_list)
    cos_tensor = torch.zeros(user_num, user_num)

    # 计算余弦相似度
    for i, user_fc_model in enumerate(model_fc_list):
        x1 = vectorize_net(user_fc_model) - pre_global_fc_model_vectorized
        for j, other_user_fc_model in enumerate(model_fc_list):
            x2 = vectorize_net(other_user_fc_model) - pre_global_fc_model_vectorized
            # cosine_similarity = adjust_cosine_similarity(x1, x2)
            cosine_similarity =torch.cosine_similarity(x1, x2, dim=0).detach().cpu()
            cos_tensor[i][j] = cosine_similarity

    cluster = hdbscan.HDBSCAN(min_cluster_size=2)
    cluster_labels = cluster.fit_predict(cos_tensor)
    majority = Counter(cluster_labels)
    most_common_clusters = majority.most_common(user_num)

    most_common_cluster_label = most_common_clusters[0][0]
    good_model_indices = [i for i, label in enumerate(cluster_labels) if label == most_common_cluster_label]

    # 欧氏距离张量
    euclidean_distances = torch.zeros(user_num)

    # 遍历每个用户模型
    for index in range(user_num):
        # 计算当前用户模型l与上次全局模型间的参数向量差
        parameter_vector_difference = pre_global_fc_model_vectorized - vectorize_net(model_fc_list[index])

        # 计算欧氏距离，即向量差的L2范数
        euclidean_distance = torch.sqrt(torch.sum(parameter_vector_difference ** 2))

        # 将欧氏距离添加到列表中
        euclidean_distances[index] = euclidean_distance

    # 计算欧氏距离的中值
    median_euclidean_distance = torch.median(euclidean_distances)

    # 最常见簇的数量
    num_most_common_clusters = most_common_clusters[0][1]

    # 初始化距离比率
    distance_ratios = torch.zeros(user_num)

    # 计算每个好模型与中值距离的比率
    for index, good_indices in enumerate(good_model_indices):
        distance_ratios[index] = min(1, median_euclidean_distance / euclidean_distances[good_indices])

    # 设置噪声的标准差
    mu = 0
    sigma = 0.01
    # 进行聚合
    for index, (key, layer_parameter) in enumerate(global_model.items()):
        # 层参数尺寸
        layer_size = layer_parameter.size()
        # 初始化参数聚合器
        aggregator = torch.zeros(layer_size).to(device)
        # 对每个好模型进行参数聚合
        for good_indices in good_model_indices:
            good_model_weight = model_list[good_indices]
            aggregator += (good_model_weight[key] - layer_parameter) * distance_ratios[good_indices]

        # 聚合
        gaussian_noise = mu + (sigma ** 2) * torch.randn(layer_size).to(device)
        global_model[key] = layer_parameter + aggregator / num_most_common_clusters + gaussian_noise

    return global_model


def replace_fc_layers(normal_model, fc_model):
    for fc_key, fc_value in fc_model.items():
        if fc_key in normal_model:
            normal_model[fc_key] = fc_value
    return normal_model


def extract_fc_layers(global_model):
    fc_layers = OrderedDict()
    for key, value in global_model.items():
        if "linear" in key or "fc" in key:
            fc_layers.update({key: value})
    return fc_layers


def extract_feature_layers(global_model):
    feature_layers = OrderedDict()
    for key, value in global_model.items():
        if "linear" not in key and "fc" not in key:
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


def small_flame(model_weights_list, global_model_weights, device):
    global_fc_model = create_fc_layers_models(global_model_weights, "global_model")
    fc_model_list = create_fc_layers_models(model_weights_list, "fc_model_list")
    new_global_model = small_flame_module(model_list=model_weights_list,
                                          global_model=global_model_weights,
                                          model_fc_list=fc_model_list,
                                          global_fc_model=global_fc_model,
                                          device=device)
    runtime = small_flame_module.runtime
    return copy.deepcopy(new_global_model), runtime
