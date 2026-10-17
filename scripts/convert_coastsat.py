import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from shoreshop3_analysis import data, wavelet, util


def main() -> int:
    dt = 16

    print('loading CSVs...', file=sys.stderr)

    timer = util.Timer()
    with timer:
        dfs = []
        with zipfile.ZipFile(data.input_coastsat_path, 'r') as zf:
            i = 0
            for info in zf.infolist():
                if not info.filename.startswith('NC_CoastSat_smoothed/usa_NC_'):
                    continue
                i += 1
                print(f'\r{i}', end='', file=sys.stderr)
    
                transect = info.filename.removeprefix('NC_CoastSat_smoothed/usa_NC_').removesuffix('.csv')
    
                with zf.open(info.filename, 'r') as file:
                    df = pd.read_csv(file, names=('time', 'shoreline_position'))
                    df['time'] = pd.to_datetime(df['time'])
                    df['transect'] = transect
                    dfs.append(df)
    
        cs_data = pd.concat(dfs)
    
    start = cs_data['time'].min()

    print(f'\nloaded {len(cs_data)} rows of coastsat data in {timer}', file=sys.stderr)

    with timer:
        dfs = []
        transects = cs_data['transect'].unique()
        # using an index makes this much faster than selecting by equality
        cs_data = cs_data.set_index(['transect', 'time'])
        for i, transect in enumerate(transects):
            print(f'\rinterpolating transect {i+1}/{len(transects)}...', end='', file=sys.stderr)
            selected = cs_data.loc[transect].reset_index(names='time')
            df = wavelet.time_regularize(selected, dt=dt, start=start)
            df['transect'] = transect
            dfs.append(df)
    print(f'\nfinished interpolating transects in {timer}', file=sys.stderr)

    regularized = pd.concat(dfs).set_index(['transect', 'time'])

    print('writing Zarr store...', file=sys.stderr)

    with timer:
        ds = regularized.to_xarray()
        ds.attrs = {
            'title': 'North Carolina CoastSat Transects 1980-2023',
            'source': 'ShoreShop3 input CSVs',
            'comment': '',
            'dt': dt,
        }
        ds.to_zarr(data.coastsat_data_path, mode='w', consolidated=False, zarr_format=3)
    print(f'written Zarr store in {timer}')

    return 0


if __name__ == '__main__':
    sys.exit(main())