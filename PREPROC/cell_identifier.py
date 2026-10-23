import xarray as xr

class cell_identifier:
    #
    def __init__(self, meshmask: xr.Dataset, lon_cut: float | None = None):
        """
        Initialize the cell identifier using a NEMO meshmask dataset.
        Parameters
        ----------
        meshmask : xr.Dataset
            NEMO meshmask dataset containing:
            - gphit : latitude at T points
            - glamt : longitude at T points
            - tmask : mask at T points
            - umask : mask at U points
            - vmask : mask at V points
        lon_cut : float, optional
              Longitude threshold. Points west of this longitude are
              considered outside the domain. If None, no longitude
              restriction is applied.
        """
        self.meshmask = meshmask
        self.lon_cut = lon_cut
        # Latitude and longitude (discard time dimension if present)
        self.gphit = meshmask["gphit"].isel(time=0, z_a=0) 
        self.glamt = meshmask["glamt"].isel(time=0, z_a=0) 
        # T, U, V longitudes
        self.glam = {"T": self.glamt,
                     "U": meshmask["glamu"].isel(time=0, z_a=0),
                     "V": meshmask["glamv"].isel(time=0, z_a=0)}
        # Surface-level masks
        self.masks = {"T": meshmask["tmask"].isel(time=0, z=0),
                      "U": meshmask["umask"].isel(time=0, z=0),
                      "V": meshmask["vmask"].isel(time=0, z=0)}
        # Surface-level masks (discard time and depth dimensions)
        self.tmask = meshmask["tmask"].isel(time=0, z=0)
        self.umask = meshmask["umask"].isel(time=0, z=0)
        self.vmask = meshmask["vmask"].isel(time=0, z=0)
    #
    def is_in_domain(self, j: int, i: int) -> dict:
        """
        Check whether a grid point belongs to the domain at T, U, V points.
        A point is outside the domain if its indices are out of bounds
        or its longitude is west of lon_cut.
        ----------
        Parameters:
        j : int, Row index.
        i : int, Column index.
        -------
        Returns:
        dict
            Dictionary with keys 'T', 'U', and 'V'.
            Each value is True if the corresponding point is in the
            domain, and False otherwise.
        """
        result = {}
        for grid, lon in self.glam.items():
            # Check indices before accessing the array.
            ny, nx = lon.shape[-2:]
            if not (0 <= j < ny and 0 <= i < nx):
                result[grid] = False
                continue
            # Apply the optional longitude restriction.
            if self.lon_cut is None:
                result[grid] = True
            else:
                point_lon = float(lon.isel(y=j, x=i).values)
                result[grid] = point_lon >= self.lon_cut
        return result
    #
    def is_water_idx(self, j: int, i: int) -> dict:
        """
        Check whether a grid point is water or land at T, U, and V points.
        ------
        Raises:
        ValueError,If the requested point is outside the domain on any of the T, U, or V grids.
        ------
        Parameters:
        j : int, Latitude (row) index.
        i : int, Longitude (column) index.
        -------
        Returns:
        dict
            Dictionary with keys 'T', 'U', and 'V'.
            Each value is True if the corresponding grid point is water,
            and False if it is land.
        """
        in_domain = self.is_in_domain(j, i)
        outside = [grid for grid, inside in in_domain.items() if not inside]
        if outside:
            raise ValueError(f"Grid point (j={j}, i={i}) is outside the domain "
                             f"for grid(s): {', '.join(outside)}. "
                             f"lon_cut={self.lon_cut}.")
        return {grid: bool(mask.isel(y=j, x=i).values) for grid, mask in self.masks.items()}
        #return {"T": bool(self.tmask.isel(y=j, x=i).values),
        #        "U": bool(self.umask.isel(y=j, x=i).values),
        #        "V": bool(self.vmask.isel(y=j, x=i).values)}
    #
    def print_coords(self, j: int, i: int) -> dict:
        """
        Return the latitude and longitude of a T-grid point.
        ---------- 
        Parameters:
        j : int, Latitude (row) index.
        i : int, Longitude (column) index.
        ------- 
        Returns:
        dict
            Dictionary containing 'lat' and 'lon'.
        """
        if not self.is_in_domain(j, i)["T"]:
            raise ValueError(f"T-grid point (j={j}, i={i}) is outside the domain.")
        #
        return {"lat": float(self.gphit.isel(y=j, x=i).values),
                "lon": float(self.glamt.isel(y=j, x=i).values)}

