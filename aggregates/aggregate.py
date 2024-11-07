import torch
from torch.utils.data import Subset

from aggregates.deepsight import deepsight
from aggregates.fedavg import federated_averaging
from aggregates.flame import flame
from aggregates.floss import floss
from aggregates.fltrust import fltrust
from aggregates.flclaude import flclaude
from aggregates.krum import krum
from aggregates.median import median
from aggregates.rflbat import rflbat
from aggregates.small_flame import small_flame
from aggregates.small_fltrust import small_fltrust
from aggregates.small_krum import small_krum
from aggregates.small_median import small_median
from subset.fl_subset import get_iid_subset


class Aggregate:
    def __init__(self, aggregate_name, model_name=None, train_set=None):
        self.aggregate_name = aggregate_name
        self.model_name = model_name
        if 'fltrust' in self.aggregate_name:
            root_dataset = Subset(train_set, get_iid_subset(len(train_set), num_samples=128))
            self.root_dataset_loader = torch.utils.data.DataLoader(root_dataset, batch_size=64, shuffle=True,
                                                                   num_workers=2)

    def aggregate_function(self, model_weights, weights, device, group_loss, clients_labels_entropy):
        if self.aggregate_name == "fedavg":
            return federated_averaging(model_weights_list=model_weights)
        elif self.aggregate_name == "median":
            return median(model_weights_list=model_weights)
        elif self.aggregate_name == "small_median":
            return small_median(model_weights_list=model_weights)
        elif self.aggregate_name == "krum":
            return krum(model_weights_list=model_weights)
        elif self.aggregate_name == "small_krum":
            return small_krum(model_weights_list=model_weights)
        elif self.aggregate_name == "small_flame":
            return small_flame(model_weights_list=model_weights, global_model_weights=weights, device=device)
        elif self.aggregate_name == "flame":
            return flame(model_weights_list=model_weights, global_model_weights=weights, device=device)
        elif self.aggregate_name == "fltrust":
            return fltrust(model_weights_list=model_weights,
                           global_model_weights=weights,
                           root_train_loader=self.root_dataset_loader,
                           model_name=self.model_name,
                           device=device)
        elif self.aggregate_name == "small_fltrust":
            return small_fltrust(model_weights_list=model_weights,
                                 global_model_weights=weights,
                                 root_train_loader=self.root_dataset_loader,
                                 model_name=self.model_name,
                                 device=device)
        elif self.aggregate_name == "deepsight":
            return deepsight(user_model_weights=model_weights,
                             global_model_weight=weights,
                             model_name=self.model_name,
                             device=device)
        elif self.aggregate_name == "rflbat":
            return rflbat(model_weights_list=model_weights,
                          global_model_weights=weights)
        elif self.aggregate_name == "floss":
            return floss(model_weights_list=model_weights, model_loss_list=group_loss)
        elif self.aggregate_name == "flclaude":
            return flclaude(user_model_weights=model_weights, entropy=clients_labels_entropy)
        else:
            raise ValueError(f"Aggregate {self.aggregate_name} not implemented")
