import celery
import numpy as np
from copy import deepcopy
import json
from mog import MixtureOfGaussians, GaussianDistribution
import time

app = celery.Celery('mog_worker')
app.config_from_object('celery_configs')

# [partial_point_num, point_dim]
_partial_points = np.array([])
_cond_prob_mat = np.array([])

@app.task()
def receive_partial_points(partial_points):
    global _partial_points
    _partial_points = partial_points
    print(f'received partial_points of shape {_partial_points.shape}')

# _1 = 0
# _2 = 0

@app.task
def calc_new_cluster_probs_and_mean_vecs(
        # [cluster_num]
        cluster_probs,
        # [cluster_num, point_dim]
        mean_vecs,
        # [cluster_num, point_dim, point_dim]
        cov_mats
):
    # st = time.time()

    global _cond_prob_mat

    cluster_num, point_dim = mean_vecs.shape
    mog = MixtureOfGaussians(
        [
            GaussianDistribution(mean_vecs[z], cov_mats[z])
            for z in range(cluster_num)
        ], cluster_probs
    )
    # [cluster_num, partial_point_num]
    # [P(z | x_i)]_{z, i}
    _cond_prob_mat = np.stack(
        [
            # [cluster_num]
            mog.calc_cond_probs(point)
            for point in _partial_points
        ], axis=1
    )
    # [cluster_num]
    # [sum_i P(z | x_i)]_z
    partial_cond_prob_sums = _cond_prob_mat.sum(axis=1)
    # [cluster_num, point_dim] = [cluster_num, partial_point_num] @ [partial_point_num, point_dim]
    # [sum_i P(z | x_i) * x_i]_z
    partial_new_mean_vecs = _cond_prob_mat @ _partial_points

    # print(time.time() - st)

    # global _1
    # _1 += time.time() - st
    # print(_1)

    return partial_cond_prob_sums, partial_new_mean_vecs

@app.task
def calc_new_cov_mats(
        # [cluster_num]
        cond_prob_sums,
        # [cluster_num, point_dim]
        mean_vecs
):
    # st = time.time()

    global _cond_prob_mat

    cluster_num = len(cond_prob_sums)
    partial_point_num, point_dim = _partial_points.shape
    partial_new_cov_mats = []
    # [cluster_num, partial_point_num, point_dim]
    centered_points_clusters = _partial_points.reshape(1, partial_point_num, point_dim) \
                               - mean_vecs.reshape(cluster_num, 1, point_dim)
    # [cluster_num, partial_point_num]
    _cond_prob_mat /= cond_prob_sums.reshape(cluster_num, 1)
    # [cluster_num, point_dim, partial_point_num] = [cluster_num, 1, partial_point_num]
    #                                               * [cluster_num, point_dim, partial_point_num]
    partial_new_cov_mats = _cond_prob_mat.reshape(cluster_num, 1, partial_point_num) \
                           * centered_points_clusters.transpose(0, 2, 1)
    # [cluster_num, point_dim, point_dim] = [cluster_num, point_dim, partial_point_num]
    #                                       @ [cluster_num, partial_point_num, point_dim]
    partial_new_cov_mats = partial_new_cov_mats @ centered_points_clusters

    # print(time.time() - st)

    # global _2
    # _2 += time.time() - st
    # print(_2)

    return partial_new_cov_mats

# app.start()