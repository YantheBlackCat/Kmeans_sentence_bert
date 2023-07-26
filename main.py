"""
==========================
# -*- coding: utf8 -*-
# @Author   : Miya
# @Time     : 2023/7/25 14:08
# @FileName : cluster.py
# @Email    : Miya.n@foxmail.com
==========================
"""

import json
import torch
import os
from collections import defaultdict
from random import uniform
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICE']="1"

device = torch.device(1)
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
model.to(device)


def point_avg(points):
    """
    :param points: List(List()) Embeddings of a cluster;
    :return: new_center List() A new cluster center
    This function calculates a new center point of a cluster.
    """
    dimensions = len(points[0])
    new_center = [0 for _ in range(len(dimensions))]
    point_num = float(len(points))
    for point in points:
        for dimension in dimensions:
            new_center[dimension] += point[dimension]
    return [num/point_num for num in new_center]


def update_centers(data_set, assignments):
    """
    :param data_set:  The embeddings list of all data.
    :param assignments:  The assigned cluster of each embedding.
    :return:  The new centers.
    This function updates the centers based on the cluster obtained in last updating.
    """

    new_means = defaultdict(list)
    centers = list()
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)
    for points in new_means.values():
        centers.append(point_avg(points))
    return centers


def lm_score(emb1, emb2): 
    """
    :param emb1:  Embedding of corresponding sentence.
    :param emb2:  Embedding of corresponding sentence.
    :return:  Semantic similarity score.
    """
    cos_sim = util.dot_score(emb1, emb2)
    return cos_sim[0][0].item()

def assign_points(data_points, centers): 
    """
    :param data_points:  All embedding datas.
    :param centers:  Center points (k).
    :return:  The assignments of all data.
    This function assign each data a index of k values as a sign of cluster the data belongs.
    """
    labels = dict()
    labels.setdefault(0, 0)
    labels.setdefault(1, 0)
    labels.setdefault(2, 0)
    labels.setdefault(3, 0)

    assignments = []
    for point in data_points:
        shortest = float('inf')
        index = 0
        for i in tqdm(range(len(centers))):
            val = lm_score(point, centers[i])
            if val < shortest:
                shortest = val
                index = i
        labels[index]+=1
        assignments.append(ndex)
    print(labels)
    return assignments


def generate_k(data_set, k): 
    """
    :param data_set:  Embedding of all data.
    :param k:  The number of cluster.
    :return:  K random points as centers within the range.
    This function randomly generates k centers of the dataset.
    """

    centers = list()
    dimensions = len(data_set[0])
    min_max = defaultdict(int)

    for data in data_set:
        for i in tqdm(range(dimensions)):
            val = data[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    for _k in range(k):
        rand_point = list()
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]
            rand_point.append(uniform(min_val, max_val))
        centers.append(rand_point)

    return centers


def k_means(sentence_list, k):
    """
    :param sentence_list:  The dataset of sentences.
    :param k:  The num of cluster.
    :return:  Finall assignments.
    This function is the main body of clustering.
    """
    dataset = list()
    for sen in sentence_list:
        emb = model.encode(sen, device=device, normalize_embeddings=True, convert_to_tensor=True).tolist()
        dataset.append(emb)

    k_points = generate_k(dataset, k) 
    assignments = assign_points(dataset, k_points) 
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    return zip(assignments, dataset)

if __name__ == "__main__":
    with open('../data/cluster_data.json', 'r') as f:
        dataset = json.load(f)
        print(k_means(dataset, 4))
