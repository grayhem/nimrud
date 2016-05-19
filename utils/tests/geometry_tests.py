# pylint: disable=E0401, E1101

"""
tests for the geometry module, encompassing voxel filtering and nested partitioning
"""

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
    assert 1==2

#---------------------------------------------------------------------------------------------------

def test_voxel_unique():
    """
    convert point coordinates to integer addresses, unique and convert back to point coordinates
    """
    assert 1==2

#---------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------




if __name__ == '__main__':
    print("testing voxel filter")
    test_voxel_init()
    print("voxel filter initialized")
    test_voxel_shift()
    print("voxel filter shifts computed correctly")
    test_in_bounds()
    print("boundary checking works")
    test_voxel_address()
    print("voxel address functions as intended")
    test_voxel_transform()
    print("voxels transform back to correct coordinates")
    test_voxel_unique()
    print("unique voxel transform functions")
    print("that does it for the voxel filter")
