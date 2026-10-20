import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from shoreshop3_analysis import data, wavelet, util


def main() -> int:
    dt = 8

    print('loading CSVs...', file=sys.stderr)

    timer = util.Timer()
    with timer:
        dfs = []
        with zipfile.ZipFile(data.input_frf_path, 'r') as zf:
            for transect_id, transect in [('1.0', 1), ('1006.0', 1006)]:
                with zf.open(f'FRF_Profiles/{transect:05}/shorelinePosAtyFRF{transect}.csv') as file:
                    df = pd.read_csv(file).rename(columns={'xFRF': 'shoreline_position'})
                    df = df.drop(columns=['yFRF', 'latitude', 'longitude'])
                    df['time'] = pd.to_datetime(df['time'])
                    df['transect'] = transect_id
                    dfs.append(df)

        frf_data = pd.concat(dfs)

    start = frf_data['time'].min()

    print(f'\nloaded {len(frf_data)} rows of coastsat data in {timer}', file=sys.stderr)

    with timer:
        dfs = []
        transects = frf_data['transect'].unique()
        # using an index makes this much faster than selecting by equality
        frf_data = frf_data.set_index(['transect', 'time'])
        for i, transect in enumerate(transects):
            print(f'\rinterpolating transect {i+1}/{len(transects)}...', end='', file=sys.stderr)
            selected = frf_data.loc[transect].reset_index(names='time')
            df = wavelet.time_regularize(selected, dt=dt, start=start)
            df['transect'] = transect
            dfs.append(df)
    print(f'\nfinished interpolating transects in {timer}', file=sys.stderr)

    regularized = pd.concat(dfs).set_index(['transect', 'time'])

    print('writing Zarr store...', file=sys.stderr)

    with timer:
        ds = regularized.to_xarray()
        ds.attrs = {
            'title': 'North Carolina FRF Survey Transects 1980-2023',
            'source': 'ShoreShop3 input CSVs',
            'comment': '',
            'dt': dt,
        }
        ds.to_zarr(data.basedir / 'duck-2020-2023' / 'observations.zarr', mode='w', consolidated=False, zarr_format=3)
    print(f'written Zarr store in {timer}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
