"""
multiscale operator processing pipeline.
features are generated for points in the query cloud, using geometry from the search cloud.
some features won't work every time-- for instance, if a neighborhood has only one point in it, then
the PCA feature is undefined. all undefined features are represented by zeros.
"""

import time

import numpy as np
from scipy.spatial import cKDTree

from nimrud.minimal import features
from nimrud.utils import geometry
from nimrud.utils import generic

# size of a leaf on the kdtree. not a terribly sensitive parameter on my machine.
LEAFSIZE = 300

# number of query points to process at once. fairly sensitive parameter.
QUERY_CHUNK_SIZE = 1000

# how often do you want to be notified?
VERBOSITY_INTERVAL = 100


def process_single_core(query_cloud, search_cloud, edge_lengths, radii, verbose=False):
    """
    compute features at multiple scales. returns an array of feature vectors aligned with the query
    cloud.
    """
    assert(len(edge_lengths) == len(radii)), \
        "edge_lengths and radii should be equal-length sequences."

    accumulator = []
    outer_start = time.perf_counter()
    for this_edge, this_radius in zip(edge_lengths, radii):
        inner_start = time.perf_counter()
        this_feature = one_scale_single_core(
            query_cloud,
            search_cloud,
            this_edge,
            this_radius,
            verbose=verbose)
        accumulator.append(this_feature)

        if verbose:
            inner_stop = time.perf_counter()
            print("============")
            inner = inner_stop - inner_start
            inner_rate = query_cloud.shape[0] / inner
            print("this scale took {}s".format(np.around(inner, 3)))
            print("one scale rate of {} points per second".format(np.around(inner_rate, 3)))
            print("===================================")

    all_features = np.concatenate(accumulator, axis=1)

    if verbose:
        outer_stop = time.perf_counter()
        print("===============================")
        print("===============================")
        outer = outer_stop - outer_start
        outer_rate = query_cloud.shape[0] / outer
        print("calculating all scales took {}s".format(np.around(outer, 3)))
        print("final rate of {} points per second".format(np.around(outer_rate, 3)))
    
    return all_features


def one_scale_single_core(query_cloud, search_cloud, edge_length, radius, verbose=False):
    """
    generate a 4d feature vector representing one analysis scale.
    """

    # resample the search cloud
    voxel_filter = geometry.VoxelFilter(search_cloud, edge_length)
    search_voxels = voxel_filter.unique_voxels(search_cloud)

    if verbose:
        print("querying {} points against a search space of {} voxels".format(
            query_cloud.shape[0],
            search_voxels.shape[0]))
        print("using a voxel edge length of {} and radius of {}".format(edge_length, radius))
        print("-------------")

    # build a kdtree for it
    search_tree = cKDTree(search_voxels, leafsize=LEAFSIZE)

    accumulator = []

    # segment the query cloud
    # TODO: this loop is where i would use multiprocessing.pool to parallelize. 4-6 processes would
    # maybe yield a 2-3x speedup. system dependent, of course.
    for num, chunk in enumerate(generic.batcher(query_cloud, QUERY_CHUNK_SIZE)):

        if verbose and num % VERBOSITY_INTERVAL == 0:
            print("processing chunk {} of {}".format(num+1, len(query_cloud) // QUERY_CHUNK_SIZE+1))

        # build a tree
        chunk_tree = cKDTree(chunk, leafsize=LEAFSIZE)

        # query against the search space
        neighbor_idx = chunk_tree.query_ball_tree(search_tree, radius)

        # find all the neighbors of all the query points
        neighborhoods = [features.take(idx, search_voxels) for idx in neighbor_idx]

        # compute the feature vectors
        population = np.array([features.population(this_neighborhood)\
                    for this_neighborhood in neighborhoods]).reshape(-1, 1)

        centroid = np.array([features.centroid(query_point, this_neighborhood)\
                    for query_point, this_neighborhood in zip(chunk, neighborhoods)]).reshape(-1, 1)

        pca = np.array([features.pca(this_neighborhood)\
                    for this_neighborhood in neighborhoods]).reshape(-1, 2)


        new_features = np.concatenate((population, centroid, pca), axis=1)
        accumulator.append(new_features)

    accumulator = np.concatenate(accumulator, axis=0)
    return accumulator














