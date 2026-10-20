import sys

import numpy as np
import xarray as xr

from shoreshop3_analysis import data


def main() -> int:
    fromdir = 'nc-1980-2023'
    todir = 'nc-2020-2023'

    ds = xr.open_dataset(data.basedir / fromdir / 'models.nc')
    start = np.datetime64('2020-01-01')
    indices = (ds.time.values >= start).nonzero()[0]
    blind = ds.isel(time=xr.DataArray(indices, dims=['time']))
    blind.to_netcdf(data.basedir / todir / 'models.nc')

    ds = xr.open_zarr(data.basedir / fromdir / 'observations.zarr', consolidated=False)
    start = np.datetime64('2020-01-01')
    indices = (ds.time.values >= start).nonzero()[0]
    blind = ds.isel(time=xr.DataArray(indices, dims=['time']))
    blind.to_zarr(data.basedir / todir / 'observations.zarr', mode='w', consolidated=False, zarr_format=3)

    return 0


if __name__ == '__main__':
    sys.exit(main())