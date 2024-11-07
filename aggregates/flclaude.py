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



@record_time
def shannon_module(user_model_weights, clients_entropy):
    # label entropy
    clients_entropy = np.array(clients_entropy).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0, init='k-means++', n_init='auto')
    kmeans.fit(clients_entropy)

    entropy_class = kmeans.labels_
    centroids = kmeans.cluster_centers_
    larger_centroid_label = np.argmax(centroids)

    entropy_score = []
    for item in entropy_class:
        if item == larger_centroid_label:
            entropy_score.append(False)
        else:
            entropy_score.append(True)
    final_score = entropy_score

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


def flclaude(user_model_weights, entropy):
    new_global_model_weight, malicious_score = shannon_module(user_model_weights, entropy)
    runtime = shannon_module.runtime
    return new_global_model_weight, runtime, malicious_score
