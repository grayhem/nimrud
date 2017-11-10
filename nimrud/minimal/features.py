"""
functions to be mapped over point neighborhoods
"""


import numpy as np


# for handling empty neighborhoods in centroid
np.seterr(invalid="raise")



def take(neighborhood_idx, search_space_cloud):
    """
    return an array of points from the search space (needed for the following)
    """
    return search_space_cloud.take(neighborhood_idx, axis=0)


def centroid(query_point, neighborhood_points):
    """
    compute the distance between the query point and the geometric mean of its neighborhood.
    """
    try:
        norm = np.linalg.norm(query_point - neighborhood_points.mean(0))
    except FloatingPointError:
        norm = 0
    return norm


def population(neighborhood_points):
    """
    count the points in the neighborhood
    """
    return np.atleast_2d(neighborhood_points).shape[0]


def pca(neighborhood_points):
    """
    return the normalized variance of the first two principal components of the neighborhood
    """
    covariance = np.cov(neighborhood_points, rowvar=False)
    # note we use eigvalsh here because it guarantees ascending order of the eigenvalues
    try:
        eigvals = np.linalg.eigvalsh(covariance)
    except np.linalg.LinAlgError:
        # one point in neighborhood
        eigvals = np.zeros(3)
    except FloatingPointError:
        # no points in neighborhood
        eigvals = np.zeros(3)
    else:
        # normalize to sum to 1
        eigvals /= eigvals.sum()
    # return the two largest. this is a weird slice.
    return eigvals[:0:-1]

