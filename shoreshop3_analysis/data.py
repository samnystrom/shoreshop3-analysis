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

datadir = shoreshop_path / 'analysis/wavelets'
coastsat_data_path = datadir / 'coastsat.zarr'
model_coherence_path = datadir / 'coherence.zarr'
blind_path = datadir / 'MIP_NC_2020-2023.nc'

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


duck_1980_2023_paths = {
    'CSIRO_nearshore': submissions_path / 'CSIRO_nearshore/mipDuck_1980-2023_CWaCS.csv',
    'CSIRO_nearshore-smoothed': submissions_path / 'CSIRO_nearshore/mipDuck_1980-2023_CWaCS_smoothed.csv',
    'GEOOCEAN': submissions_path / 'GEOOCEAN/mipDuck_1980-2023_GeoOcean_bulk_SLP_Qs_partitions_N.csv',
    'GEOOCEAN-cross_eps': submissions_path / 'GEOOCEAN/mipDuck_1980-2023_GeoOcean_cross_eps_bulk_SLP_Qs_partitions_N.csv',
    'ICM-TCN': submissions_path / 'ICM-TCN/mipDuck_1980-2023_ICM-TCN_transect.csv',
    'IdSOR': submissions_path / 'IdSOR/mipDuck_1980-2023_WaveBulkParameters_IdSOR.csv',
    'IHCantabriaCMEG-Crossformer': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/Crossformer/mipDuck_1980-2023-Crossformer.csv',
    'IHCantabriaCMEG-DLinear': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/DLinear/mipDuck_1980-2023-DLinear.csv',
    'IHCantabriaCMEG-ETSformer': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/ETSformer/mipDuck_1980-2023-ETSformer.csv',
    'IHCantabriaCMEG-FEDformer': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/FEDformer/mipDuck_1980-2023-FEDformer.csv',
    'IHCantabriaCMEG-iTransformer': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/iTransformer/mipDuck_1980-2023-iTransformer.csv',
    'IHCantabriaCMEG-MillerDean': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/MillerDean/mipDuck_1980-2023-MillerDean.csv',
    'IHCantabriaCMEG-moirai': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/moirai/mipDuck_1980-2023-moirai.csv',
    'IHCantabriaCMEG-MultiPatchFormer': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/MultiPatchFormer/mipDuck_1980-2023-MultiPatchFormer.csv',
    'IHCantabriaCMEG-Nonstationary_Transformer': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/Nonstationary_Transformer/mipDuck_1980-2023-Nonstationary_Transformer.csv',
    'IHCantabriaCMEG-PatchTST': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/PatchTST/mipDuck_1980-2023-PatchTST.csv',
    'IHCantabriaCMEG-SPADS': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/SPADS/mipDuck_1980-2023-SPADS.csv',
    'IHCantabriaCMEG-SPADS-Huber': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/SPADS-Huber/mipDuck_1980-2023-SPADS-Huber.csv',
    'IHCantabriaCMEG-SPADS-Scale': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/SPADS-Scale/mipDuck_1980-2023-SPADS-Scale.csv',
    'IHCantabriaCMEG-TiDE': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/TiDE/mipDuck_1980-2023-TiDE.csv',
    'IHCantabriaCMEG-TimeFilter': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/TimeFilter/mipDuck_1980-2023-TimeFilter.csv',
    'IHCantabriaCMEG-TimeMixer': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/TimeMixer/mipDuck_1980-2023-TimeMixer.csv',
    'IHCantabriaCMEG-timer_xl': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/timer_xl/mipDuck_1980-2023-timer_xl.csv',
    'IHCantabriaCMEG-TimesNet': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/TimesNet/mipDuck_1980-2023-TimesNet.csv',
    'IHCantabriaCMEG-TimeXer': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/TimeXer/mipDuck_1980-2023-TimeXer.csv',
    'IHCantabriaCMEG-ttm': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/ttm/mipDuck_1980-2023-ttm.csv',
    'IHCantabriaCMEG-Yates09': submissions_path / 'IHCantabriaCMEG/submission_Duck_objective_1/Yates09/mipDuck_1980-2023-Yates09.csv',
    'LEGI-ShoreFor': submissions_path / 'LEGI/mipDuck_1980-2023_waves_bulk-LEGIShoreFor.csv',
    'LEGI-ShoreForeKFcp45': submissions_path / 'LEGI/mipDuck_1980-2023_waves_bulk-LEGIShoreForeKFcp45.csv',
    'LEGI-ShoreForeKFcp55': submissions_path / 'LEGI/mipDuck_1980-2023_waves_bulk-LEGIShoreForeKFcp55.csv',
    'mtraboulsi689': submissions_path / 'mtraboulsi689/mipDuck_1980-2023_waves_simplified-MohammadTraboulsi.csv',
    # Missing: ANTOLINEZ (directory is empty)
    # Missing: ShEPreMo (24 files, don't know which to use)
    # use the p50 for shepremo
    'UN-IHE-Elghandour-DA_N05': submissions_path / 'UN-IHE-Elghandour/mipDuck_1980-2023_waves_simplified-ShorelineS-DA_N05.csv',
    'UN-IHE-Elghandour-DA_N125': submissions_path / 'UN-IHE-Elghandour/mipDuck_1980-2023_waves_simplified-ShorelineS-DA_N125.csv',
    'UN-IHE-Elghandour-DA_C02': submissions_path / 'UN-IHE-Elghandour/mipDuck_1980-2023_waves_simplified-ShorelineS-DA_C02.csv',
    'UN-IHE-Elghandour-DA_STORM_B05': submissions_path / 'UN-IHE-Elghandour/mipDuck_1980-2023_waves_simplified-ShorelineS-DA_STORM_B05.csv',
    'UN-IHE-Elghandour-DA_STORM_B125': submissions_path / 'UN-IHE-Elghandour/mipDuck_1980-2023_waves_simplified-ShorelineS-DA_STORM_B125.csv',
    'UNSW-WRL-Calcraft': submissions_path / 'UNSW-WRL-Calcraft/mipDuck_1980-2023_MTFB.csv',
    'UNSW-WRL-Mao': submissions_path / 'UNSW-WRL-Mao/shoreshop3Duck/mipDuck_1980-2023_GraphShore.csv',
}


def load_profiles() -> gp.GeoDataFrame:
    # UTM zone 18N
    return gp.read_file(input_profiles_path).to_crs(epsg=26918)


def load_shoreline() -> gp.GeoDataFrame:
    # UTM zone 18N
    return gp.read_file(shoreline_path).to_crs(epsg=26918)


class CoastsatData:
    def __init__(self) -> None:
        self._ds = xr.open_zarr(coastsat_data_path, consolidated=False)

    def dt(self) -> float:
        return self._ds.attrs['dt']

    def get_range(self) -> tuple[np.datetime64, np.datetime64]:
        return self._ds.time.min().values, self._ds.time.max().values

    def get_transects(self) -> list[str]:
        return [name.removeprefix('usa_NC_') for name in self._ds.transect.values]

    def get_single(self, transect_id: str) -> pd.DataFrame:
        subset = self._ds.shoreline_position.sel(transect=transect_id).dropna('time')
        return subset.to_pandas().reset_index().rename(columns={'shoreline_position': 'x'})


def load_frf_single(transect_id: str) -> pd.DataFrame:
    '''
    returns df with columns time, xFRF, yFRF, latitude, longitude
    '''
    with zipfile.ZipFile(input_frf_path, 'r') as zf:
        with zf.open(f'FRF_Profiles/{transect_id}/shorelinePosAtyFRF1.csv') as file:
            df = pd.read_csv(file)

    df['time'] = pd.to_datetime(df['time'])

    return df


def load_single_profile_models() -> pd.DataFrame:
    '''
    returns df with columns date, frf1, frf1006, model
    '''

    dfs = []
    for model, path in duck_1980_2023_paths.items():
        df = pd.read_csv(path, header=1, names=('time', 'frf1', 'frf1006')).dropna()
        df['model'] = model
        dfs.append(df)

    df = pd.concat(dfs)

    def fix_date(date: str) -> str:
        if '/' not in date:
            return date
        m, d, y = list(map(int, date.split('/')))
        return f'{y:04}-{m:02}-{d:02}'

    df['time'] = pd.to_datetime(df['time'].apply(fix_date))

    return df


class MultiProfileModels:
    def __init__(self) -> None:
        self._ds = xr.open_dataset(nc_1980_2023_cube_path)

    def get_models(self) -> list[str]:
        return self._ds.model.values.tolist()

    def get_all_transects(self) -> list[str]:
        return [name.removeprefix('usa_NC_') for name in self._ds.transect.values]

    def get_transects(self, model: str) -> list[str]:
        return self.get_all_transects()

    def get_model_transect_data(self, model: str, transect: str) -> pd.DataFrame:
        '''
        Returns: dataframe with columns time (datetime64), x (float)
        '''
        subset = self._ds.shoreline_position.sel(model=model, transect='usa_NC_' + transect)
        subset = subset.dropna('time')
        return subset.to_pandas().reset_index().rename(columns={'shoreline_position': 'x'})
