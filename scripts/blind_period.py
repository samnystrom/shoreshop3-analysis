import sys

import numpy as np
import xarray as xr

from shoreshop3_analysis import data


def main() -> int:
    ds = xr.open_dataset(data.nc_1980_2023_cube_path)
    start = np.datetime64('2020-01-01')
    indices = (ds.time.values >= start).nonzero()[0]
    blind = ds.isel(time=xr.DataArray(indices, dims=['time']))
    blind.to_netcdf(data.blind_path)

    ds = xr.open_zarr(data.coastsat_data_path, consolidated=False)
    start = np.datetime64('2020-01-01')
    indices = (ds.time.values >= start).nonzero()[0]
    blind = ds.isel(time=xr.DataArray(indices, dims=['time']))
    blind.to_zarr(data.coastsat_blind_path, mode='w', consolidated=False, zarr_format=3)

    return 0


if __name__ == '__main__':
    sys.exit(main())