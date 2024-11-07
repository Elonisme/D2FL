import copy
import logging
from collections import OrderedDict

import torch

from decorators.timing import record_time

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


@record_time
def small_krum_module(model_weights_list, fc_model_list):
    """Aggregate weight updates from the clients using multi-krum."""
    flatten_models_weights = flatten_weights(fc_model_list)

    num_attackers_selected = 5

    distances = []
    for weight in flatten_models_weights:
        distance = torch.norm((flatten_models_weights - weight), dim=1) ** 2
        distances = (
            distance[None, :]
            if not len(distances)
            else torch.cat((distances, distance[None, :]), 0)
        )

    distances = torch.sort(distances, dim=1)[0]

    scores = torch.sum(
        distances[:, : len(flatten_models_weights) - 2 - num_attackers_selected],
        dim=1,
    )
    indices = torch.argsort(scores)

    krum_update = copy.deepcopy(model_weights_list[indices[0]])
    logging.info(f"Finished multi-krum server aggregation.")
    return krum_update


def flatten_weights(weights):
    flattened_weights = []

    for weight in weights:
        flattened_weight = []
        for name in weight.keys():
            flattened_weight = (
                weight[name].view(-1)
                if not len(flattened_weight)
                else torch.cat((flattened_weight, weight[name].view(-1)))
            )

        flattened_weights = (
            flattened_weight[None, :]
            if not len(flattened_weights)
            else torch.cat((flattened_weights, flattened_weight[None, :]), 0)
        )
    return flattened_weights


def small_krum(model_weights_list):
    fc_model_list = create_fc_layers_models(copy.deepcopy(model_weights_list), "fc_model_list")
    return small_krum_module(model_weights_list, fc_model_list), small_krum_module.runtime
