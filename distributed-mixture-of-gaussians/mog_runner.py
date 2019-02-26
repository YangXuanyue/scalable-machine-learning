import random
import time
import numpy as np
import matplotlib.pyplot as plt
import json
from copy import deepcopy
import celery
from mog_worker import receive_partial_points, calc_new_cluster_probs_and_mean_vecs, calc_new_cov_mats
from utils import plot_results

start_time = time.time()

points = np.array(
    [
        list(map(float, line.split(',')))
        for line in open('mog_dataset.csv')
    ]
)

cluster_num = 3
worker_num = 4
iteration_num = 10
point_num, point_dim = points.shape
shard_size = point_num // worker_num

celery.group(
    receive_partial_points.s(points[i:min(i + shard_size, point_num)])
    for i in range(0, point_num, shard_size)
)().get()

cluster_probs = np.ones(cluster_num, dtype=np.float32) / cluster_num
mean_vecs = np.array(
    [[1, 0], [-5, 2], [-2, -2]],
    dtype=np.float32
)
cov_mats = np.zeros((cluster_num, point_dim, point_dim), dtype=np.float32)

for i in range(cluster_num):
    np.fill_diagonal(cov_mats[i], 2.)

# cmt = 0.
# cct = 0.

for i in range(iteration_num):
    # st = time.time()

    results = celery.group(
        calc_new_cluster_probs_and_mean_vecs.s(cluster_probs, mean_vecs, cov_mats)
        for _ in range(worker_num)
    )().get()

    # cmt += time.time() - st

    # print(cmt)

    mean_vecs.fill(0.)
    cluster_probs.fill(0.)
    cond_prob_sums = np.zeros_like(cluster_probs)

    for partial_cond_prob_sums, partial_new_mean_vecs in results:
        # [cluster_num, point_dim]
        mean_vecs += partial_new_mean_vecs
        # [cluster_num]
        cond_prob_sums += partial_cond_prob_sums

    # [cluster_num]
    cluster_probs = cond_prob_sums / point_num
    # [cluster_num, point_dim]
    mean_vecs /= cond_prob_sums.reshape(cluster_num, 1)

    st = time.time()

    results = celery.group(
        calc_new_cov_mats.s(cond_prob_sums, mean_vecs)
        for _ in range(worker_num)
    )().get()

    # cct += time.time() - st

    cov_mats.fill(0.)

    for partial_cov_mats in results:
        # [cluster_num, point_dim, point_dim]
        cov_mats += partial_cov_mats

# print(cmt, cct)

print(f'finished in {time.time() - start_time} secs with {worker_num} worker(s)')

plot_results(points, mu=mean_vecs, cov=cov_mats)