# pylint: disable=

"""
implements a class which facilitates point cloud classification by tracking associations between
points, features and labels.
"""

import numpy as np


class FlexCloud(object):
    """
    given a 3d point cloud as a 2d numpy array, shift its points close to the origin and track its
    features. any supplemental information must be added separately, after the object has been 
    instantiated with its geometry. supplemental information is stored as "assets" which can be 
    1d or 2d arrays of floats or integers.

    the asset index dictionary looks like the following:
    {
        intensity: {
            index: index_array,
            asset: 1d array of floats,
            meta: "this intensity data is not calibrated"
        },
        geometry_mso_1: {
            index: index_array,
            asset: 2d array of floats,
            meta: {
                voxel: 0.05,
                scales: [0.15, 0.2, 0.25]
            }
        },
        known_label: {
            index: index_array,
            asset: 1d array of ints
        },
        multilabel_predicted: {
            index: index_array,
            asset: 2d array of ints,
            meta: "this is why i think this is a good idea..."
        }
    }
    """

    #==================================

    def __init__(self, input_cloud):
        input_dim = input_cloud.shape[1]
        if input_dim != 3:
            raise ValueError("must be initialized with a 3D point cloud")
        # now bring the point cloud in to the origin
        self.corner = input_cloud[0, :3]
        self.xyz = input_cloud[:, :3] - self.corner
        # count how many points we have in the original point cloud
        self.num_points = input_cloud.shape[0]
        # initialize the asset dictionary
        self.assets = {}

    #==================================

    def add_asset(self, asset_array, index_array, asset_name, meta=None):
        """
        add a new asset array to the cloud's asset index. its index array should be unique, but
        it doesn't necessarily need to be sorted.
        """

        # first make sure this is a good idea
        if asset_name in self.assets:
            raise ValueError("asset {} already exists in asset dictionary".format(asset_name))
        self._validate_asset(asset_array, index_array)

        # assemble the asset
        asset = {
            "asset": asset_array,
            "index": index_array
        }
        if meta is not None:
            asset["meta"] = meta

        self.assets[asset_name] = asset

    #==================================

    def _validate_asset(self, asset_array, index_array):
        """
        raise an exception if the asset to be added to the asset dictionary won't fit in the cloud
        """

        # make sure the asset has no more than 2 dimensions-- we don't support 2d assets per
        # point (yet?)
        if asset_array.ndim > 2:
            raise ValueError("asset array has too many dimensions")
        # make sure the asset and its index array are aligned
        if asset_array.shape[0] != index_array.size:
            raise ValueError("asset and index arrays misaligned")
        # make sure all indices are unique
        unique_indices = np.unique(index_array)
        if unique_indices.size != index_array.size:
            raise ValueError("index array is not unique")
        # make sure the index array will index into the cloud
        if index_array.min() < 0 or index_array.max() >= self.num_points:
            raise ValueError("index array addresses outside the extant cloud")

    #==================================

    #==================================

    #==================================

    #==================================
