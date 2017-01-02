# pylint: disable=E0401, E1101, C0103

"""
tests for the geometry module, encompassing voxel filtering and nested partitioning
"""

from itertools import product
import numpy as np

from nimrud.utils import geometry

SEED = 10
np.random.seed(SEED)

#---------------------------------------------------------------------------------------------------

def test_voxel_init():
    """
    initialize the voxel filter
    """
    num = 1000
    scale = 100
    dims = [2, 3]
    edge_length = 0.5

    for dim in dims:
        # first let's do it with only one point, which makes no sense
        points = np.random.rand(1, dim) * scale
        try:
            vf = geometry.VoxelFilter(points, edge_length)
        except ValueError:
            pass
        else:
            raise AssertionError("built a voxel filter with a single point at dim {}".format(dim))

        # now let's do it for real
        points = np.random.rand(num, dim) * scale

        # the first voxel (all 0s address) should be centered on the *actual* minimum corner of 
        # the cloud-- though there likely won't be a point there.
        minimum_corner = points.min(0) - edge_length / 2
        # the maximum corner here doesn't actually correspond to a voxel location. we just don't
        # want to be responsible for addressing points outside the rectangular prism bounding the
        # original point cloud on one side and not the other. that feels weird and arbitrary.
        maximum_corner = points.max(0) + edge_length / 2

        vf = geometry.VoxelFilter(points, edge_length)

        assert np.array_equal(vf.minimum_corner, minimum_corner),\
            "set wrong minimum corner with dim {}".format(dim)
        assert np.array_equal(vf.maximum_corner, maximum_corner),\
            "set wrong maximum corner with dim {}".format(dim)
        assert edge_length == vf.edge_length, "set wrong edge length with dim {}".format(dim)


    # now let's try doing it with too many dimensions
    dims = [1, 4]
    for dim in dims:
        points = np.random.rand(num, dim) * scale
        try:
            vf = geometry.VoxelFilter(points, edge_length)
        except ValueError:
            pass
        else:
            raise AssertionError("built a VoxelFilter for a {}D space. that's bad.".format(dim))

    # and some arrays that cannot be  point clouds
    try:
        vf = geometry.VoxelFilter(np.random.rand(10), edge_length)
    except ValueError:
        pass
    else:
        raise AssertionError("built a VoxelFilter on a 1D array. that's bad.")

    try:
        vf = geometry.VoxelFilter(np.random.rand(10, 10, 10), edge_length)
    except ValueError:
        pass
    else:
        raise AssertionError("built a VoxelFilter on a 3D array. that's bad.")

#---------------------------------------------------------------------------------------------------

def test_voxel_shift():
    """
    calculate the number of bits each address has to be shifted over to fit in the address integer
    """
    dims = [2, 3]

    def get_widths(point_cloud, edge_length):
        """
        compute the widths
        """
        minimum_corner = point_cloud.min(0) - edge_length / 2
        maximum_corner = point_cloud.max(0) + edge_length / 2
        span = maximum_corner - minimum_corner
        return np.ceil(np.log2(span / edge_length))

    for dim in dims:
        good_edge_length = 0.001
        points = np.asarray([
            [0, 0, 0],
            [100, 100, 100]])[:, :dim]

        # first let's compute what it should be (17 bits each)
        shifts = [17, 34][:dim-1]
        vf = geometry.VoxelFilter(points, good_edge_length)
        assert np.array_equal(shifts, vf.shifts),\
            "computed incorrect shifts for {}d cloud".format(dim)
        widths = [17, 17, 17][:dim]
        assert np.array_equal(
            widths,
            vf.widths), "computed incorrect widths for {}d cloud".format(dim)

        # now, what about a voxel edge length that's incompatible with the point cloud shape?
        # we should detect that and raise a ValueError.
        # well, first we'll need to figure out how small we have to go to make it impossible.
        if dim == 3:
            # this should be too small in 3d. in a test in the repl it yielded 72 address bits.
            bad_edge_length = 0.00001
        else:
            # and 68 here for a 2d cloud
            bad_edge_length = 0.00000001

        # but let's check anyway.
        address_length = sum(get_widths(points, bad_edge_length))
        assert address_length > 64,\
            "our test edge length is not short enough to overflow at dim {}. try div by 10.".\
                format(dim)

        try:
            vf = geometry.VoxelFilter(points, bad_edge_length)
        except ValueError:
            pass
        else:
            raise AssertionError\
            ("built a VoxelFilter for a space we should not be able to voxelize at dim {}".\
                format(dim))

#---------------------------------------------------------------------------------------------------

def test_masks():
    """
    test that masks used to decalate addresses into grid coordinates are computed correctly
    """

    for dim in [2, 3]:
        boundary_points = np.asarray([
            [0, 0, 0],
            [100, 100, 100]])[:, :dim]
        edge_length = 1
        vf = geometry.VoxelFilter(boundary_points, edge_length)
        # each axis should have 7 bits
        masks = [
            0b1111111,
            0b11111110000000,
            0b111111100000000000000][:dim]
        assert np.array_equal(masks, vf.masks), "computed masks incorrectly"

#---------------------------------------------------------------------------------------------------

def test_in_bounds():
    """
    assert that boundary testing works
    """
    for dim in [2, 3]:
        boundary_points = np.asarray([
            [0, 0, 0],
            [100, 100, 100]])[:, :dim]
        edge_length = 1
        vf = geometry.VoxelFilter(boundary_points, edge_length)

        def check_in_bounds(point):
            """
            try/ catch around the in bounds method. returns True if it works.
            """
            try:
                vf._check_in_bounds(point)
            except ValueError as err:
                # print(err)
                return False
            else:
                return True

        assert check_in_bounds(np.zeros((1, dim)) - 0.5), "should be in bounds"
        assert not check_in_bounds(np.zeros((1, dim)) - 1.5), "should not be in bounds"
        assert check_in_bounds(np.zeros((1, dim)) + 0.5), "should be in bounds"
        assert check_in_bounds(np.zeros((1, dim)) + 100.5), "should be in bounds"
        assert not check_in_bounds(np.zeros((1, dim)) + 101.5), "should not be in bounds"
        assert not check_in_bounds(np.zeros((1, dim+1))), "should not be in bounds"
        assert check_in_bounds(np.zeros(dim)), "should be in bounds"
        assert not check_in_bounds(np.zeros(dim+1)), "should be in bounds"

#---------------------------------------------------------------------------------------------------

def test_voxel_address():
    """
    convert point coordinates to integer addresses
    """
    edge_length = 1
    boundary_points = np.asarray([
        [0, 0, 0],
        [100, 100, 100]])
    vf = geometry.VoxelFilter(boundary_points, edge_length)
    
    test_point = np.arange(3) + 10
    # first get the minimum corner. 
    minimum_corner = vf.minimum_corner
    # now convert the test point to grid coordinates
    grid_point = np.floor((test_point - minimum_corner) / edge_length).astype(np.int64)
    assert np.array_equal(grid_point, [10, 11, 12]), "got wrong grid coordinate"
    # now the bit shifts
    shifts = vf.shifts
    # this operation is going to look like x ^ (y << y_shift) ^ (z << z_shift)
    address = grid_point[0] ^ (grid_point[1] << shifts[0]) ^ (grid_point[2] << shifts[1])
    # if each axes' domain in the bit string is properly separated, we can actually just add the
    # grid coordinates together.
    x = 10
    y = 11 << 7
    z = 12 << 14
    known_address = 198026
    assert known_address == x + y + z
    assert known_address == x ^ y ^ z
    assert address == known_address, "test logic computed wrong address"
    assert address == vf.coordinate_to_address(test_point), "VoxelFilter computed wrong address"


#---------------------------------------------------------------------------------------------------

def test_voxel_transform():
    """
    convert integer addresses to point coordinates
    """
    
    edge_length = 1
    boundary_points = np.asarray([
        [0, 0, 0],
        [100, 100, 100]])
    vf = geometry.VoxelFilter(boundary_points, edge_length)
    
    # we'll use the same address we used before
    known_address = 198026
    known_coordinates = np.arange(3)+10

    assert np.allclose(
        known_coordinates,
        vf.address_to_coordinate(known_address).flatten()),\
        "failed to recover correct coordinates in 3D"

    # and a quick 2d test
    vf = geometry.VoxelFilter(boundary_points[:, :2], edge_length)
    assert np.allclose(
        known_coordinates[:2],
        vf.address_to_coordinate(
            vf.coordinate_to_address(known_coordinates[:2]).flatten())),\
        "failed to recover correct coordinates in 2D"


#---------------------------------------------------------------------------------------------------

def test_voxel_unique():
    """
    convert point coordinates to integer addresses, unique and convert back to point coordinates
    """
    
    for dim in [2, 3]:
        boundary_points = np.asarray([
            [0, 0, 0],
            [100, 100, 100]])[:, :dim]
        edge_length = 1
        vf = geometry.VoxelFilter(boundary_points, edge_length)
        # we should have 10 points here
        test_points =\
            np.concatenate([np.zeros((1, dim)) + offset for offset in np.arange(0, 20, 2)])
        # 20, with 10 unique
        duplicated_test_points = np.vstack((test_points, test_points))
        unique_voxels = vf.unique_voxels(duplicated_test_points)
        assert np.array_equal(test_points, unique_voxels),\
            "failed to get correct set of unique voxels at dimension {}".format(dim)


#---------------------------------------------------------------------------------------------------

def test_octree_init():
    """
    initialize the NestedOctree and check the attributes it sets
    """
    num_points = 1000
    scale = 10
    buffer_radius = 0.5
    search_space = np.random.rand(num_points, 3) * scale
    query_set = np.random.rand(num_points, 3) * scale

    tree = geometry.NestedOctree(query_set, search_space, buffer_radius)

    assert tree.buffer_radius == buffer_radius, "buffer radius set wrong"
    assert np.array_equal(tree.search_space, search_space), "search space set wrong"
    assert np.array_equal(tree.query_set, query_set), "query set set wrong"
    assert tree.cubes == [], "didn't initialize cube list"

    # bounds refer to the query set
    known_max = query_set.max(0)
    known_min = query_set.min(0)
    assert np.array_equal(tree.maximum_corner, known_max), "max corner set wrong"
    assert np.array_equal(tree.minimum_corner, known_min), "min corner set wrong"

    bad_query_sets = [
        query_set.flatten(),
        query_set.reshape(-1, 2),
        query_set.reshape(-1, 6),
        query_set[0:1]]

    for this_bad_query_set in bad_query_sets:
        try:
            tree = geometry.NestedOctree(this_bad_query_set, search_space, buffer_radius)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a query set with shape {}"\
                    .format(this_bad_query_set.shape))

    bad_search_spaces = [
        search_space.flatten(),
        search_space.reshape(-1, 2),
        search_space.reshape(-1, 6),
        search_space[0:1]]

    for this_bad_search_space in bad_search_spaces:
        try:
            tree = geometry.NestedOctree(query_set, this_bad_search_space, buffer_radius)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a search space with shape {}"\
                    .format(this_bad_search_space.shape))

    # should not accept negative buffer radii
    try:
        tree = geometry.NestedOctree(query_set, search_space, -buffer_radius)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a negative buffer radius")

    # and we need our cube generator algorithms
    algorithms = ["naive", "take_one", "take_three"]
    for this_algorithm in algorithms:
        assert this_algorithm in tree.cube_generators,\
            "missing cube generator algorithm {}".format(this_algorithm)

#---------------------------------------------------------------------------------------------------

def test_nested_regions():
    """
    assert that nested_regions returns indices to the correct points
    """

    query_set = np.random.rand(5000, 3) # 0,0,0 to 1,1,1
    search_space = np.random.rand(20000, 3) * 3 - 1  # -1,-1,-1 to 2,2,2
    minimum_corner = np.array([0.25, 0.25, 0.25])
    maximum_corner = np.array([0.75, 0.75, 0.75])
    buffer_radius = 0.5

    query_set_index, search_space_index = geometry.nested_regions(
        query_set,
        search_space,
        buffer_radius,
        minimum_corner,
        maximum_corner)

    cull_query_set = query_set.take(query_set_index, axis=0)
    cull_search_space = search_space.take(search_space_index, axis=0)
    assert all(cull_query_set.min(0) >= minimum_corner), "query set min error" 
    assert all(cull_query_set.max(0) <= maximum_corner), "query set max error" 
    assert all(cull_search_space.min(0) >= minimum_corner - buffer_radius), "search space min error" 
    assert all(cull_search_space.max(0) <= maximum_corner + buffer_radius), "search space max error" 

    # now but what happens if we try to extract a region with no points?
    minimum_corner = np.ones(3) * 100
    maximum_corner = minimum_corner + 10

    query_set_index, search_space_index = geometry.nested_regions(
        query_set,
        search_space,
        buffer_radius,
        minimum_corner,
        maximum_corner)
    assert query_set_index.size == 0, "returned indices for empty region in query set"
    assert search_space_index.size == 0, "returned indices for empty region in search space"    

#---------------------------------------------------------------------------------------------------

def test_octree_partition_accept():
    """
    if the population of a NestedOctree search space _within the buffered bounds of the region of 
    interest_ is acceptable, then it should create only one partition.
    if it has only one partition, then why call it an octree?
    """

    # build a query set and two search spaces
    query_set = np.random.rand(1000, 3)

    # first search space has fewer points than the max
    search_space_low = np.random.rand(999, 3)

    max_population = 1000
    buffer_radius = 0.01    # just as long as this is smaller than about 100 it doesn't matter here

    # second search space has more points than the max, but within the query set ROI it has fewer.
    search_space_high = np.vstack((
        search_space_low,
        np.random.rand(1000, 3) + 100))

    # for each search space:
    for num, this_search_space in enumerate([search_space_low, search_space_high]):
        # build a NestedOctree
        tree = geometry.NestedOctree(query_set, this_search_space, buffer_radius)
        # partition it with a point count that should yield one partition
        tree.partition(max_population)

        # check how many partitions we have
        assert len(tree.cubes) == 1,\
            "got {} partitions on search space {}. expected 1.".format(len(tree.cubes), num)

#---------------------------------------------------------------------------------------------------

def test_octree_cube_generator():
    """
    cube_generator yields 8 cubic nested regions covering the query set bounds of the NestedOctree
    """

    # testing the bounds-finding logic (using itertools.product).
    # it should work on any arbitrary coordinates.
    offsets = [
        np.zeros(3),
        np.random.rand(3)]

    cube_edge = 0.5
    buffer_radius = 0.1

    def is_in_bounds(points, min_bounds, max_bounds):
        """
        return bool indicating whether given points are included in given bounds
        """
        return all(points.min(0) >= min_bounds), all(points.max(0) <= max_bounds)

    algorithms = [
        "naive",
        "take_one",
        "take_three"]

    for algorithm in algorithms:
        for this_offset in offsets:
            query_set = np.random.rand(1000, 3) * 2 * cube_edge
            search_space = np.random.rand(4000, 3) * 4 * cube_edge- 0.5
            # just so we know what the bounds are a priori
            query_set[0] *= 0
            query_set[1] *= 0
            query_set[1] += 2 * cube_edge
            # and apply the offset
            query_set += this_offset
            search_space += this_offset
            minimum_corner = query_set.min(0)

            tree = geometry.NestedOctree(query_set, search_space, buffer_radius)

            # this this the part that could get goofed up w/r/t the arbitrary coordinate offset
            known_centers = np.asarray(list(product([0, 1], repeat=3)))
            known_min_corners = known_centers * cube_edge + minimum_corner
            known_max_corners = known_min_corners + cube_edge
            # print("min corner {}".format(minimum_corner))
            # print("max corner {}".format(maximum_corner))
            # for mi, ma in zip(known_min_corners, known_max_corners):
            #     print("min {}".format(mi))
            #     print("max {}".format(ma))

            # get the right number of cubes (8)
            assert len(list(tree.cube_generator(cube_edge, algorithm=algorithm))) == 8,\
                "failed to generate 8 cubes at offset {} using algorithm {}".format(
                    this_offset,
                    algorithm)

            # get the right cubes in the right order. we enforce the right order because it's
            # easier to test than getting any arbitrary order.
            for num, (query_cube, search_cube) in\
                enumerate(tree.cube_generator(cube_edge, algorithm=algorithm)):
                low = known_min_corners[num]
                high = known_max_corners[num]
                assert is_in_bounds(query_cube, low, high),\
                    "query cube {} failed at offset {}".format(num, this_offset)

                assert is_in_bounds(search_cube, low - buffer_radius, high + buffer_radius),\
                    "search cube {} failed at offset {}".format(num, this_offset)

    # now try with a bogus algorithm
    try:
        gen = tree.cube_generator(cube_edge, algorithm="bogus")
    except NameError:
        raise AssertionError("failed to raise a NameError on a nonexistent algorithm choice")

#---------------------------------------------------------------------------------------------------

def performance_octree_cube_generator():
    """
    test the performance of the various cube generator algorithms
    """
#---------------------------------------------------------------------------------------------------

def test_octree_partition_octree():
    """
    
    """
    assert False

#---------------------------------------------------------------------------------------------------

def test_octree_partition_grid():
    """
    """

    assert False
#---------------------------------------------------------------------------------------------------




if __name__ == '__main__':
    print("testing voxel filter")
    test_voxel_init()
    print("voxel filter initialized")
    test_voxel_shift()
    print("voxel filter shifts computed correctly")
    test_masks()
    print("masks computed correctly")
    test_in_bounds()
    print("boundary checking works")
    test_voxel_address()
    print("voxel address functions as intended")
    test_voxel_transform()
    print("voxels transform back to correct coordinates")
    test_voxel_unique()
    print("unique voxel transform functions")
    print("that does it for the voxel filter")
    print("testing nested partitions")
    test_nested_regions()
    print("nested regions found")
    test_octree_init()
    print("octree initialized")
    test_octree_cube_generator()
    print("cubes generated")
    test_octree_partition_accept()
    test_octree_partition_octree()
    test_octree_partition_grid()
    print("octree partitioned correctly")

