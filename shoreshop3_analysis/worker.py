import time
import sys

import numpy as np
import pandas as pd
import zarr

from shoreshop3_analysis import data, wavelet


_obs: data.Observations | None = None
_models: data.Models | None = None
_arr: zarr.Array | None = None


def run(model: str, transect: str, ids: tuple[int, int]) -> None:
    try:
        return _run_internal(model, transect, ids)
    except Exception as e:
        print(e)
        raise e


def _run_internal(model: str, transect: str, idx: tuple[int, int]) -> None:
    global _obs
    global _models
    global _arr
    _obs = _obs or data.Observations()
    _models = _models or data.Models()
    #_out = _out or xr.open_zarr(data.model_coherence_path, consolidated=False)
    _arr = _arr or zarr.open_group(data.model_coherence_path, mode='r+', zarr_format=3)

    dt = _obs.dt()

    obs = _obs.get_single(transect)
    model_data = _models.get_model_transect_data(model, transect)
    if len(model_data) == 0: # this transect isn't present for the model
        print(f'skipping (empty): {model} {transect} ({idx[0]}, {idx[1]})')
        return None
    obs['date'] = obs['time'].dt.date
    model_data['date'] = model_data['time'].dt.date
    joined = obs.merge(model_data, on='date', how='inner', suffixes=('_obs', '_model'))
    joined = joined.dropna(subset=['x_obs', 'x_model'])
    if len(joined) < 4:
        print(f'skipping (empty join): {model} {transect} ({idx[0]}, {idx[1]})')
        return None # too short for WCT

    wave = wavelet.WCT.fromdata(joined['time_obs'], joined['x_obs'].values, joined['x_model'].values, dt)

    if np.isnan(wave._wct).any():
        print(f'Warning: NaN WCT result for model={model} transect={transect}', file=sys.stderr)

    wct = wave.power()
    j, n = wct.shape
    _arr['coherence'][*idx, :j, :n] = wct.astype(np.float32)

    start = joined['time_obs'].values.min()
    _arr['start'][*idx] = int((start - np.datetime64(0, 'D')) / np.timedelta64(1, 'D'))
    _arr['period'][*idx, :j] = wave.period().astype(np.float64)
    _arr['coi'][*idx, :n] = wave.coi().astype(np.float64)
    _arr['sig'][*idx, :j] = wave.sig().astype(np.float64)

    print(f'finished: {model} {transect} ({idx[0]}, {idx[1]})')

    return None
