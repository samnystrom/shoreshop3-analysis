import json
import time
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from pyleoclim import Series

from shoreshop3_analysis import data, wavelet


_coastsat_data: data.CoastsatData | None = None
_models: data.MultiProfileModels | None = None
_out: xr.Dataset | None = None


@dataclass(frozen=True)
class WorkerResult:
    model_idx: int
    transect_idx: int
    dt: float
    start: np.datetime64
    n: int
    j: int
    period: np.ndarray
    sig: np.ndarray
    coi: np.ndarray


def run(model: str, transect: str, ids: tuple[int, int]) -> WorkerResult | None:
    try:
        return _run_internal(model, transect, ids)
    except Exception as e:
        print(e)
        raise e


def _run_internal(model: str, transect: str, idx: tuple[int, int]) -> WorkerResult | None:
    global _coastsat_data
    global _models
    global _out
    _coastsat_data = _coastsat_data or data.CoastsatData()
    _models = _models or data.MultiProfileModels()
    #_out = _out or xr.open_zarr(data.model_coherence_path, consolidated=False)
    _out = _out or zarr.open_group(data.model_coherence_path, mode='r+', zarr_format=3)['coherence']

    dt = 8

    obs = _coastsat_data.get_single(transect)
    model_data = _models.get_model_transect_data(model, transect)
    if len(model_data) == 0: # this transect isn't present for the model
        return None
    obs['date'] = obs['time'].dt.date
    model_data['date'] = model_data['time'].dt.date
    joined = obs.merge(model_data, on='date', how='inner', suffixes=('_obs', '_model'))

    wave = wavelet.WCT.fromdata(joined['time_obs'], joined['x_obs'].values, joined['x_model'].values, dt)

    if np.isnan(wave._wct).any():
        print(f'Warning: NaN WCT result for model={model} transect={transect}', file=sys.stderr)

    wct = wave.power()
    j, n = wct.shape
    _out[*idx, :j, :n] = wct.astype(np.float32)
    #_out.coherence[*idx, :j, :n] = wct.astype(np.float32)
    #_out.to_zarr(
    #    data.model_coherence_path,
    #    region={'model': slice(idx[0], idx[0]+1), 'transect': slice(idx[1], idx[1]+1)},
    #    consolidated=False,
    #    zarr_format=3,
    #)

    period = wave.period()
    start = joined['time_obs'].values.min()

    return WorkerResult(*idx, dt, start, n, j, period, wave.sig(), wave.coi())
