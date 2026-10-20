import os
import sqlite3
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gp
import pycwt

from shoreshop3_analysis import wavelet, util


workdir = Path('/work/sln33')
scratchdir = Path('/scratch')

shoreshop_path = Path.home() / 'abmurraylab/shoreshop3'

basedir = shoreshop_path / 'analysis/wavelets'
datadir = basedir / 'nc-1980-2023'

observations_path = datadir / 'observations.zarr'
model_data_path = datadir / 'models.nc'
model_coherence_path = datadir / 'coherence.zarr'
global_coherence_path = datadir / 'global-coherence.zarr'

shoreline_path = Path(__file__).parent.parent / 'nc_80k.gpkg'

input_coastsat_path = shoreshop_path / 'inputdata/NC_CoastSat_smoothed.zip'
input_profiles_path = shoreshop_path / 'inputdata/hindcast_1980_2023/shorelines_and_profiles/NC_transects.geojson'
input_frf_path = shoreshop_path / 'inputdata/hindcast_1980_2023/shorelines_and_profiles/FRF_Profiles.zip'
submissions_path = shoreshop_path / 'submissions/UserSubmissionsCleaned'

datacubes_path = shoreshop_path / 'submissions/MIP_datacubes'
duck_1871_2012_cube_path = datacubes_path / 'MIP_Duck_1871-2012.nc'
duck_1980_2023_cube_path = datacubes_path / 'MIP_Duck_1980-2023.nc'
duck_2024_2100_cube_path = datacubes_path / 'MIP_Duck_2024-2100.nc'
nc_1871_2012_cube_path = datacubes_path / 'MIP_NC_1871-2012.nc'
nc_1980_2023_cube_path = datacubes_path / 'MIP_NC_1980-2023.nc'
nc_2024_2100_cube_path = datacubes_path / 'MIP_NC_2024-2100.nc'


def load_profiles() -> gp.GeoDataFrame:
    # UTM zone 18N
    return gp.read_file(input_profiles_path).to_crs(epsg=26918)


def load_shoreline() -> gp.GeoDataFrame:
    # UTM zone 18N
    return gp.read_file(shoreline_path).to_crs(epsg=26918)


class Observations:
    def __init__(self) -> None:
        self._ds = xr.open_zarr(observations_path, consolidated=False)

    def dt(self) -> float:
        return self._ds.attrs['dt']

    def name(self) -> str:
        return self._ds.attrs['name']

    def get_range(self) -> tuple[np.datetime64, np.datetime64]:
        return self._ds.time.min().values, self._ds.time.max().values

    def get_transects(self) -> list[str]:
        return self._ds.transect.values.tolist()

    def get_single(self, transect: str) -> pd.DataFrame:
        subset = self._ds.shoreline_position.sel(transect=transect).dropna('time')
        return subset.to_pandas().reset_index().rename(columns={'shoreline_position': 'x'})


class Models:
    def __init__(self) -> None:
        self._ds = xr.open_dataset(model_data_path)

    def get_range(self) -> tuple[np.datetime64, np.datetime64]:
        return self._ds.time.min().values, self._ds.time.max().values

    def get_models(self) -> list[str]:
        return self._ds.model.values.tolist()

    def get_transects(self) -> list[str]:
        return self._ds.transect.values.tolist()

    def get_model_transect_data(self, model: str, transect: str) -> pd.DataFrame:
        '''
        Returns: dataframe with columns time (datetime64), x (float)
        '''
        subset = self._ds.shoreline_position.sel(model=model, transect=transect)
        subset = subset.dropna('time')
        return subset.to_pandas().reset_index().rename(columns={'shoreline_position': 'x'})
