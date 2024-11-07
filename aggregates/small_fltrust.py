import copy
from collections import OrderedDict

import torch
from torch import nn, optim

from decorators.timing import record_time
from models.model import Model
from torch.cuda.amp import autocast, GradScaler

def replace_layers(normal_model, other_layers_model):
    for layer_key, layer_value in other_layers_model.items():
        if layer_key in normal_model:
            normal_model[layer_key] = layer_value
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


def create_feature_layers_models(model_list, type_of_model):
    if type_of_model == 'global_model':
        fc_layers_model = extract_feature_layers(model_list)
        return fc_layers_model
    else:
        fc_model_list = []
        for model in model_list:
            fc_layers_model = extract_feature_layers(model)
            fc_model_list.append(fc_layers_model)
        return fc_model_list


def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.values()])


def root_train(root_net, root_train_loader, root_optimizer, criterion, device):
    root_net.train()
    scaler = GradScaler()

    num_epochs = 3
    root_avg_loss = 0
    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, data in enumerate(root_train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            root_optimizer.zero_grad()

            with autocast():
                outputs = root_net(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(root_optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(root_train_loader)
        root_avg_loss += avg_loss

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    root_avg_loss /= num_epochs

    return root_net.state_dict(), root_avg_loss


def adjust_cosine_similarity(vec1, vec2):
    mean_vec1 = vec1.mean(dim=0)
    mean_vec2 = vec2.mean(dim=0)
    adjust_vec1 = vec1 - mean_vec1
    adjust_vec2 = vec2 - mean_vec2
    return torch.cosine_similarity(adjust_vec1, adjust_vec2, dim=0)

@record_time
def small_fltrust_module(model_weights_list, model_fc_weights_list,
                   global_model_weights, global_fc_model_weights,
                   root_train_loader, model_name, device):

    root_model = Model(model_name)
    root_net = root_model.get_model()
    root_net.to(device)
    root_net.load_state_dict(copy.deepcopy(global_model_weights))

    root_optimizer = optim.Adam(root_net.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    root_model_trained_weight, root_avg_loss = root_train(root_net, root_train_loader, root_optimizer, criterion,
                                                          device)

    print("root model is being trained")
    print("root model average loss is: ", root_avg_loss)

    # 深拷贝全局模型权重
    pre_global_fc_model_weights = copy.deepcopy(global_fc_model_weights)

    # 计算根模型的更新
    root_update = copy.deepcopy(root_model_trained_weight)
    root_fc_update = create_fc_layers_models(model_list=root_update, type_of_model="global_model")
    for key in root_fc_update.keys():
        root_fc_update[key] = root_fc_update[key] - pre_global_fc_model_weights[key]

    # 计算用户模型的fc权重更新
    user_model_fc_update_list = copy.deepcopy(model_fc_weights_list)
    for user_model_fc_update in user_model_fc_update_list:
        for key in pre_global_fc_model_weights.keys():
            user_model_fc_update[key] = user_model_fc_update[key] - pre_global_fc_model_weights[key]

    # 计算 Trust Score for all users
    user_num = len(user_model_fc_update_list)
    root_fc_update_vec = vectorize_net(root_fc_update)
    trust_scores = torch.zeros(user_num)
    user_model_fc_update_vecs = []
    for index, user_model_fc_update in enumerate(user_model_fc_update_list):
        user_model_fc_update_vec = vectorize_net(user_model_fc_update)
        user_model_fc_update_vecs.append(user_model_fc_update_vec)
        cos_sim = torch.cosine_similarity(user_model_fc_update_vec, root_fc_update_vec, dim=0)
        ts = torch.relu(cos_sim)
        trust_scores[index] = ts

    # 如果分数全为零，则返回上一次的全局模型
    if all(x == 0 for x in trust_scores):
        final_global_model_weights = copy.deepcopy(global_model_weights)
        return final_global_model_weights

    trust_scores_normalized = torch.div(trust_scores, trust_scores.sum())
    print(f"trust score: {trust_scores_normalized}")

    # 规范化用户更新，通过与根更新对齐
    norm_list = []
    fc_root_normal_number = torch.norm(root_fc_update_vec)
    for user_model_fc_update_vec in user_model_fc_update_vecs:
        user_norm_number = torch.norm(user_model_fc_update_vec)
        scale_factor = torch.div(fc_root_normal_number, user_norm_number)
        norm_list.append(scale_factor)

    for i, user_fc_model_update in enumerate(user_model_fc_update_list):
        for key, fc_update_value in user_fc_model_update.items():
            model_weights_list[i][key] = torch.mul(norm_list[i], fc_update_value) + global_model_weights[key]


    # 聚合：获取全局更新
    final_global_model_weights = copy.deepcopy(global_model_weights)
    for key in final_global_model_weights.keys():
        update = torch.zeros(final_global_model_weights[key].size()).to(device)
        for net_index, user_model in enumerate(model_weights_list):
            update += torch.mul(trust_scores_normalized[net_index], user_model[key])
        final_global_model_weights[key] = update

    return final_global_model_weights


def small_fltrust(model_weights_list, global_model_weights, root_train_loader, model_name, device):
    model_fc_weights_list = create_fc_layers_models(
        model_list=copy.deepcopy(model_weights_list), type_of_model="user_model")
    global_fc_model_weights = create_fc_layers_models(
        model_list=copy.deepcopy(global_model_weights), type_of_model="global_model")

    final_global_model = small_fltrust_module(model_weights_list, model_fc_weights_list,
                                              global_model_weights, global_fc_model_weights,
                                              root_train_loader, model_name, device)
    runtime = small_fltrust_module.runtime

    return final_global_model, runtime
