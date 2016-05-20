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

#---------------------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------------------
