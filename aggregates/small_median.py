import copy
import logging

import torch

from decorators.timing import record_time


def flatten_fc_weights(weights):
    flattened_weights = []

    for weight in weights:
        flattened_weight = []
        for name in weight.keys():
            if "fc" in name or "linear" in name:
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


@record_time
def small_median_module(model_weights_list):
    update_weight = copy.deepcopy(model_weights_list[0])
    for key in update_weight.keys():
        if "fc" not in key and "linear" not in key:
            for weight in model_weights_list:
                update_weight[key].add_(weight[key])
        update_weight[key].div_(len(model_weights_list))

    """Aggregate weight updates from the clients using median."""

    flattened_fc_weights = flatten_fc_weights(model_weights_list)

    median_fc_weight = torch.median(flattened_fc_weights, dim=0)[0]

    # Update global model
    start_index = 0
    for name, weight_value in model_weights_list[0].items():
        if "fc" in name or "linear" in name:
            update_weight[name] = median_fc_weight[
                                  start_index: start_index + len(weight_value.view(-1))
                                  ].reshape(weight_value.shape)
            start_index = start_index + len(weight_value.view(-1))

    return update_weight


def small_median(model_weights_list):
    weight = small_median_module(model_weights_list)
    run_time = small_median_module.runtime
    return weight, run_time
