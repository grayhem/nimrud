# pylint: disable=E0401, E1101

"""
implements a voxel filter and a spatial partitioning algorithm
"""

import numpy as np


MAX_ADDRESS_LENGTH = 64

#---------------------------------------------------------------------------------------------------

class VoxelFilter(object):
    """
    given a 2d or 3d point cloud, define a cubic grid of specified edge length enclosing it. exposes
    methods for converting point coordinates (within the enclosed region) into 64bit integer
    addresses (encoding grid coordinates) and back into 2d/ 3d floating point coordinates.
    """

    def __init__(self, points, edge_length):
        """
        points = sequence of 2d or 3d points. should be at least 2 of them.
        edge_length = edge length of voxel grid-- i.e. the spacing between two voxel centers along
            one of the coordinate axes
        """

        if points.ndim != 2:
            raise ValueError("wrong point cloud array shape")
        elif points.shape[1] not in [2, 3]:
            raise ValueError("only 2D and 3D spaces supported")
        elif points.shape[0] < 2:
            raise ValueError("need at least 2 points to define a voxel grid")

        self.minimum_corner = points.min(0) - edge_length / 2
        self.maximum_corner = points.max(0) + edge_length / 2
        self.edge_length = edge_length

        # now set the bit shifts and widths for each dimension
        self.shifts, self.widths = self._calculate_shift()

        # and get the masks
        self.masks = self._calculate_masks()

    #==================================

    def _calculate_shift(self):
        """
        calculate the number of address bits each dimension gets, and return the address widths
        while we're at it
        """

        span = self.maximum_corner - self.minimum_corner
        address_widths = np.ceil(np.log2(span / self.edge_length))

        # first check whether we can address this region in space at this edge length        
        if sum(address_widths) > MAX_ADDRESS_LENGTH:
            raise ValueError("edge length is too small to address this space")
        else:
            shifts = np.cumsum(address_widths)[:-1]

        return shifts.astype(np.int64), address_widths.astype(np.int64)

    #==================================

    def _calculate_masks(self):
        """
        create a mask for extracting each coordinate axis' grid coordinate from the integer address
        """

        # stack '1' bits to the proper widths
        masks = [int("0b" + "1" * this_width, base=2) for this_width in self.widths]
        # now shift them as necessary
        for num, this_shift in enumerate(self.shifts):
            masks[num+1] = masks[num+1] << this_shift

        return masks

    #==================================

    def _check_in_bounds(self, points):
        """
        confirm that any new collection of points to be used with this voxel filter is within its
        bounds with the right number of spatial dimensions
        """

        check_points = np.atleast_2d(points)

        if check_points.ndim != 2:
            raise ValueError("wrong array shape")
        if check_points.shape[1] != self.shifts.size+1:
            raise ValueError("wrong number of spatial dimensions")
        if np.any(check_points.min(0) < self.minimum_corner)\
            or np.any(check_points.max(0) > self.maximum_corner):
            raise ValueError("some points fall outside filter bounding region")

        return check_points

    #==================================

    def coordinate_to_address(self, points):
        """
        transform real-world coordinates into voxel coordinates and convert to integer addresses
        """
        points = self._check_in_bounds(points)
        voxel_coordinates = np.floor((points-self.minimum_corner)/self.edge_length).astype(np.int64)

        # now do the bit shifts
        for col, this_shift in enumerate(self.shifts):
            voxel_coordinates[:, col+1] = voxel_coordinates[:, col+1] << this_shift

        # in this special case, bitwise or is the same as addition
        voxel_addresses = voxel_coordinates.sum(1)
        return voxel_addresses

    #==================================

    def address_to_coordinate(self, addresses):
        """
        transform integer addresses into real-world coordinates
        """

        # we might want to give it just one address.
        addresses = np.atleast_1d(addresses)

        # extract voxel coordinates
        voxel_coordinate_list = [(addresses & this_mask).reshape(-1, 1) for this_mask in self.masks]
        # shift back to the right
        for num, this_shift in enumerate(self.shifts):
            voxel_coordinate_list[num+1] = voxel_coordinate_list[num+1] >> this_shift
        # get the right shape
        voxel_coordinates = np.concatenate(voxel_coordinate_list, axis=1)
        # bring them into real world coordinates
        # (add a half edge length to get the center of the voxel, instead of the minimum corner)
        points = voxel_coordinates * self.edge_length + self.minimum_corner + self.edge_length*0.5
        return points

    #==================================

    def unique_voxels(self, points):
        """
        return unique center coordinates of all grid cells that contain a point in "points"
        """

        # first convert to voxel addresses
        addresses = self.coordinate_to_address(points)
        # now unique
        unique_addresses = np.unique(addresses)
        # now back to real world coordinates
        coordinates = self.address_to_coordinate(unique_addresses)

        return coordinates

    #==================================

    def find_neighbors(self, address):
        """
        given an integer address, find the address of each directly adjacent voxel. 
        up to 8 voxels are adjacent in 2D, and up to 26 in 3D.
        """
        # TODO: not relevant right now
        raise NameError("find_neighbors not implemented yet")

    def find_facing_neighbors(self, address):
        """
        given an integer address, find the address of each voxel sharing an edge (in 2D) or a face
        (in 3D). up to 4 voxels will be adjacent in 2D, and 6 in 3D. 
        """
        # TODO: not relevant right now
        raise NameError("find_facing_neighbors not implemented yet")


#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

# nested partitioning:

# given two colocated point clouds (representing query set and search space) partition both 
# simultaneously such that the partitions of the query set enclose all of its points without
# overlap, and partitions of the search space totally enclose all of the query set partitions,
# extending past their boundaries in each coordinate direction by a fixed length. partitions of
# the search space are constrained in their maximum number of member points, while partitions of
# the query set are unbounded in population.

# i have two ideas how to do this: 
#   build an octree to get close and then evenly subdivide each leaf of the octree (if necessary)
#       NestedOctree/ NestedGrid
#   iteratively partition by glomming small cubic cells together
#       ProceduralNestedPartitioner

# both classes expose a public method partition_generator that iterates over the partitions they
# create. the partition_generator yields a tuple (query_set_indices, search_space_indices)


#---------------------------------------------------------------------------------------------------

class NestedOctree(object):
    """
    recursive object for octree-like nested partitioning.
    the structure resulting from a hierarchy of embedded NestedOctree instances is not strictly an
    octree because each one computes its own bounds given a collection of points. therefore the
    volume enclosed by the union of the bounding boxes of a parent tree's subtrees is 
    nearly always smaller than the volume enclosed by the parent tree's bounding box.
    """

    def __init__(self, query_set, search_space, buffer_radius):
        """
        when the object is initialized, it sets the boundaries of the region to be partitioned.
        query_set and search_space should be nx3 arrays with at least two elements. buffer_radius
        must be >= 0.
        """

        def validate_input(points):
            """
            check the shape of the point clouds
            """
            if points.ndim != 2:
                raise ValueError("wrong point cloud array shape")
            elif points.shape[1] != 3:
                raise ValueError("only 3D spaces are supported")
            elif points.shape[0] < 2:
                raise ValueError("need at least 2 points to partition")

        validate_input(query_set)
        validate_input(search_space)
        if buffer_radius <= 0:
            raise ValueError("buffer radius cannot be negative")

        self.query_set = query_set
        self.search_space = search_space
        self.buffer_radius = buffer_radius

        # the bounds we are interested in belong to the query set
        self.maximum_corner = query_set.max(0)
        self.minimum_corner = query_set.min(0)

    #==================================

    def partition(self, max_population, minimum_factor=3):
        """
        if necessary, subdivide the region into 8 equal cubes.
        for each of those cubes there are two options: OCTREE or GRID.
            OCTREE is chosen if the cube edge length is greater than 
                minimum_factor * buffer_radius. a NestedOctree is initialized for the cube.
            GRID is chosen otherwise. a NestedGrid is initialized for the cube.
        """


    #==================================

    def partition_generator(self):
        """
        iterate over the 8 cubes and yield tuples of (query_set_indices, search_space_indices)
        """

        for this_cube in self.cubes:
            try:
                for query_set_idx, search_space_idx in this_cube.partition_generator():
                    yield query_set_idx, search_space_idx
            except AttributeError:
                yield this_cube

    #==================================

#-------------------------------------------------

class NestedGrid(object):
    """
    single-level tree for nested partitioning. partitions are a set of identical cubes covering the
    region of interest. cube radius is reduced until the search space population constraint is met.
    """

    def __init__(
            self,
            query_set,
            search_space,
            buffer_radius,
            max_population):
        self.query_set = query_set
        self.search_space = search_space

    #==================================

    def partition_generator(self):
        """
        iterate over all cubes in the grid and yield tuples of 
        (query_set_indices, search_space_indices)
        """

        for this_cube in self.cubes:
            yield this_cube

    #==================================

#---------------------------------------------------------------------------------------------------
    
# this idea isn't totally fleshed out. as it stands now there is no guarantee against making
# concave partitions, and we can't know if the search space is sparse enough to successfully
# partition with the requested buffer radius without evaluating all possible partitions.

# when this is done, partition some point clouds with it and visualize the results. 

class ProceduralNestedPartitioner(object):
    """
    build a voxel space encompassing query set and search space
    convert query set and search space into dictionaries of {int_address : index_array}
    make a generator that does the following while cells remain in the query set dict:
        pick a query set cell at random
        use VoxelFilter.find_neighbors to find the addresses of every adjacent cell
        look up those cells in the search space dictionary
        count the total number of points in all of the search space cells
        if total number of search space points is over the max:
            pop the query cell from the query set dictionary
            yield query set and search space indices
            (we'll let the user decide what to do in this case)
        else:
            put the search space cell addresses in a set
            put the query set cell address in a set
            start a new set for rejected query set cells
            start a new set for potential query set cells
            while True:
                find the face neighbors of the last added query set cell with 
                    VoxelFilter.find_facing_neighbors and add those not in rejected query set or 
                    accepted query set to potential query set.
                    ??? what happens if we do not add any new cells to potential in this step ???
                for _ in range(max_tries):
                    find all adjacent cells of one cell in potential query set
                    compute the union of these adjacent cells and the existing search set
                    if the proposed search set is larger than the max population:
                        move this query set cell to rejected query set cells
                        remove this query set cell from the potential query set
                    else:
                        move this query set cell to accepted query set
                        remove this query set cell from potential query set
                        break
                else:
                    break
            pop the accepted query set cells from the query set dict
            cat and yield index arrays
    """

    def __init__(
            self,
            query_set,
            search_space,
            buffer_radius,
            max_population,
            num_tries=5):
        raise NameError("ProceduralNestedPartitioner has not been implemented yet.")

    #==================================

    def partition_generator(self):
        """
        iteratively creates partitions of the given query set and search space. yields a tuple of
        (query_set_indices, search_space_indices)
        """


    #==================================



#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
