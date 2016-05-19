# pylint: disable=E0401, E1101

"""
tests for the FlexCloud class
"""

import numpy as np

from nimrud.utils import point_clouds

SEED = 10
np.random.seed(SEED)

#---------------------------------------------------------------------------------------------------

def test_instantiation():
    """
    test instantiation of the FlexCloud
    """
    good_geometry = np.random.rand(1000, 3)
    small_geometry = np.random.rand(1000, 2)
    large_geometry = np.random.rand(1000, 4)
    flat_geometry = np.random.rand(3)

    # this should work
    cloud = point_clouds.FlexCloud(good_geometry)
    assert np.array_equal(cloud.corner, good_geometry[0]), "didn't store the corner"
    assert np.array_equal(cloud.points + cloud.corner, good_geometry),\
        "didn't subtract the corner"
    assert hasattr(cloud, "assets"), "cloud didn't make an asset directory"
    assert cloud.num_points == good_geometry.shape[0], "cloud didn't count right number of points"
    assert np.array_equal(cloud.id_index, np.arange(good_geometry.shape[0])),\
        "cloud didn't store a full index array"

    # these should not
    try:
        cloud = point_clouds.FlexCloud(small_geometry)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a cloud with too few dimensions")

    try:
        cloud = point_clouds.FlexCloud(large_geometry)
    except ValueError:
        pass    
    else:
        raise AssertionError("accepted a cloud with too many dimensions")

    try:
        cloud = point_clouds.FlexCloud(flat_geometry)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted points with incorrect array shape")

#---------------------------------------------------------------------------------------------------

def test_add_asset():
    """
    test adding assets to the FlexCloud. it should sort/ make unique.
    """

    points = np.random.rand(1000, 3)
    cloud = point_clouds.FlexCloud(points)

    asset_1 = np.random.rand(100, 2)
    asset_1_idx = np.random.permutation(1000)[:100]
    cloud.add_asset(asset_1, asset_1_idx, "asset_1")

    # should be sorted in the cloud
    asset_1_sorting_idx = np.argsort(asset_1_idx)
    asset_1_idx_sorted = asset_1_idx.take(asset_1_sorting_idx)
    asset_1_sorted = asset_1.take(asset_1_sorting_idx, axis=0)
    assert np.array_equal(asset_1_sorted, cloud.assets["asset_1"]["asset"]),\
        "didn't store asset_1 correctly"
    assert np.array_equal(asset_1_idx_sorted, cloud.assets["asset_1"]["index"]),\
        "didn't store asset_1 index correctly"

    # this should come out identical to asset_1 (not unique version of asset_1)
    asset_2 = np.vstack((asset_1, asset_1))
    asset_2_idx = np.hstack((asset_1_idx, asset_1_idx))
    cloud.add_asset(asset_2, asset_2_idx, "asset_2")
    assert np.array_equal(asset_1_sorted, cloud.assets["asset_2"]["asset"]),\
        "didn't store asset_2 correctly"
    assert np.array_equal(asset_1_idx_sorted, cloud.assets["asset_2"]["index"]),\
        "didn't store asset_2 index correctly" 

    # same (not unique and not sorted version of asset_1)
    shuffle_index = np.random.permutation(200)
    asset_3 = asset_2.take(shuffle_index, axis=0)
    asset_3_idx = asset_2_idx.take(shuffle_index)
    cloud.add_asset(asset_3, asset_3_idx, "asset_3")
    assert np.array_equal(asset_1_sorted, cloud.assets["asset_3"]["asset"]),\
        "didn't store asset_3 correctly"
    assert np.array_equal(asset_1_idx_sorted, cloud.assets["asset_3"]["index"]),\
        "didn't store asset_3 index correctly"

    # now let's do it with scalar assets
    cloud.add_asset(asset_3_idx, asset_3_idx, "asset_4")
    assert np.array_equal(asset_1_idx_sorted, cloud.assets["asset_4"]["asset"]),\
        "didn't store asset_4 correctly"
    assert np.array_equal(asset_1_idx_sorted, cloud.assets["asset_4"]["index"]),\
        "didn't store asset_4 index correctly"

#---------------------------------------------------------------------------------------------------

def test_intersection():
    """
    test intersecting the assets
    """

    points = np.random.rand(1000, 3)
    cloud = point_clouds.FlexCloud(points)

    asset_1 = np.random.rand(100, 2)
    asset_1_idx = np.arange(100)
    cloud.add_asset(asset_1, asset_1_idx, "asset_1")

    asset_2 = np.random.rand(100)
    asset_2_idx = np.arange(100)+50
    cloud.add_asset(asset_2, asset_2_idx, "asset_2")

    known_asset = np.hstack((asset_1[50:], asset_2[:50].reshape(-1, 1)))
    known_idx = asset_1_idx[50:]

    test_idx, test_asset = cloud.intersection(["asset_1", "asset_2"])
    assert np.array_equal(known_idx, test_idx), "intersection produced wrong index set"
    assert np.array_equal(known_asset, test_asset), "intersection produced wrong asset block"

#---------------------------------------------------------------------------------------------------

def test_take():
    """
    .take should function like ndarray.take, but add the corner back to the points (if told to)
    """

    points = np.random.rand(1000, 3)
    cloud = point_clouds.FlexCloud(points)

    idx = np.random.permutation(1000)[:100]
    assert np.array_equal(cloud.take(idx), points.take(idx, axis=0)), "take failed with given idx"
    assert np.array_equal(cloud.take(), points), "take failed with no idx given"

    off_center_points = points - points[0]
    assert np.array_equal(
        cloud.take(idx, original_coordinates=False),
        off_center_points.take(idx, axis=0)), "take failed with given idx and centering"
    assert np.array_equal(
        cloud.take(original_coordinates=False),
        off_center_points), "take failed with no idx and centering"

#---------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    print("testing instantiation")
    test_instantiation()
    print("FlexCloud instantiated properly")
    print("testing adding of assets")
    test_add_asset()    
    print("assets added correctly")
    print("testing asset intersection")
    test_intersection()
    print("intersection operation tests out")
    print("testing take")
    test_take()
    print("take took")


