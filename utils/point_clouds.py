# pylint: disable=E0401

"""
implements a class which facilitates point cloud classification by tracking associations between
points, features and labels.
"""

import numpy as np


class FlexCloud(object):
    """
    given a 3d point cloud as a 2d numpy array, shift its points close to the origin and track its
    features. any supplemental information must be added separately, after the object has been 
    instantiated with its points. supplemental information is stored as "assets" which can be 
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
            asset: 1d array of ints,
            meta: None
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

        if input_cloud.ndim != 2: 
            raise ValueError("input point cloud must be a 2D array")
        if input_cloud.shape[1] != 3:
            raise ValueError("must be initialized with a 3D point cloud")
        # now bring the point cloud in to the origin
        self.corner = input_cloud[0]
        self.points = input_cloud - self.corner
        # count how many points we have in the original point cloud
        self.num_points = input_cloud.shape[0]
        self.id_index = np.arange(self.num_points)
        # initialize the asset dictionary
        self.assets = {}

    #==================================

    def add_asset(self, asset_array, index_array, asset_name, meta=None):
        """
        add a new asset array to the cloud's asset index. input index array does not need to be
        sorted or unique, but it will be stored sorted and unique to simplify the set ops later.
        """

        # first make sure this is a good idea
        if asset_name in self.assets:
            raise ValueError("asset {} already exists in asset dictionary".format(asset_name))
        asset_array, index_array = self._validate_asset(asset_array, index_array)

        # assemble the asset
        asset = {
            "asset": asset_array,
            "index": index_array,
            "meta": meta
        }

        self.assets[asset_name] = asset

    #==================================

    def _validate_asset(self, asset_array, index_array):
        """
        unique and sort the index array, and align the asset array to match it.
        raise an exception if the asset to be added to the asset dictionary won't fit in the cloud
        """

        # make sure the asset has no more than 2 dimensions-- we don't support 2d assets per
        # point (yet?)
        if asset_array.ndim > 2:
            raise ValueError("asset array has too many dimensions")
        # make sure the asset and its index array are aligned
        if asset_array.shape[0] != index_array.size:
            raise ValueError("asset and index arrays misaligned")
        # make sure all indices are unique and sorted
        unique_indices, index_to_unique = np.unique(index_array, return_index=True)
        # make sure the index array will index into the cloud
        if index_array.min() < 0 or index_array.max() >= self.num_points:
            raise ValueError("index array addresses outside the extant cloud")

        # now return (assets, indices)
        return asset_array.take(index_to_unique, axis=0), unique_indices

    #==================================

    def intersection(self, asset_names):
        """
        given a list of names of assets, compute the intersection of their index sets and return
        that, along with the horizontal concatenation of all the corresponding assets.
        """

        # fold down the sequence of asset names, accumulating to the identity index set using the
        # intersection operator.
        index_accumulator = self.id_index
        for name in asset_names:
            this_index = self.assets[name]["index"]
            index_accumulator = np.intersect1d(index_accumulator, this_index, assume_unique=True)

        # how many points are there?
        num_points = index_accumulator.size
        
        # we can now use in1d to find each asset that is present in the intersection
        asset_accumulator = []
        for name in asset_names:
            this_index = self.assets[name]["index"]
            this_asset = self.assets[name]["asset"]
            # find which assets are present in the output index set
            mask = np.in1d(this_index, index_accumulator, assume_unique=True)
            # put them on the accumulator
            asset_accumulator.append(np.compress(mask, this_asset, axis=0).reshape(num_points, -1))

        return_assets = np.concatenate(asset_accumulator, axis=1)

        return index_accumulator, return_assets

    #==================================

    def take(self, index_array=None, original_coordinates=True):
        """
        equivalent to ndarray.take(). return a subset of the FlexCloud's points addressed by an
        index array, in the original coordinates if desired. if no index given, return all.
        """
        if original_coordinates:
            return_points = self.points + self.corner
        else:
            return_points = self.points
        if index_array is not None:
            return return_points.take(index_array, axis=0)
        else:
            return return_points

    #==================================

    #==================================
