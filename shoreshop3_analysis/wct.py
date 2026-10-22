import json
import os
import sys
import sqlite3
from concurrent import futures

import numpy as np
import pandas as pd
import xarray as xr
import dask.array
import zarr

from shoreshop3_analysis import util, data, worker, wavelet


class ModelsWCT:
    def __init__(self) -> None:
        self._obs = data.Observations() # may raise FileNotFoundError
        self._models = data.Models()

        obs_min_time, obs_max_time = self._obs.get_range()
        models_min_time, models_max_time = self._models.get_range()
        min_time = max(obs_min_time, models_min_time)
        max_time = min(obs_max_time, models_max_time)
        self._range = min_time, max_time

        if data.model_coherence_path.exists():
            self._ds = xr.open_zarr(data.model_coherence_path, consolidated=False)
            return

        all_models = self._models.get_models()
        all_transects = self._models.get_transects()
        M, T = len(all_models), len(all_transects)

        dt = self._obs.dt()

        diff = (max_time - min_time) / np.timedelta64(1, 'D')
        n = int(np.ceil(diff / dt))

        flambda = 4*np.pi / (6 + np.sqrt(2 + 6**2))
        j = int(np.round(12 * np.log2(n * flambda / 2))) + 1

        self._ds = xr.Dataset(
            data_vars={
                'start':     (('model', 'transect'),          np.full((M, T), 0, dtype=int)),
                'period':    (('model', 'transect', 'scale'), np.full((M, T, j), np.nan, dtype=np.float64)),
                'sig':       (('model', 'transect', 'scale'), np.full((M, T, j), np.nan, dtype=np.float64)),
                'coi':       (('model', 'transect', 'time'),  np.full((M, T, n), np.nan, dtype=np.float64)),
                'coherence': (('model', 'transect', 'scale', 'time'), dask.array.full((M, T, j, n), np.nan, dtype=np.float32, chunks=(1, 1, j, n))),
            },
            coords={
                'model': all_models,
                'transect': all_transects,
                'time': np.arange(n),
                'scale': np.arange(j),
            },
            attrs={
                'title': '',
                'source': '',
                'dt': dt,
            },
        )
        encoding = {
            'start': {'chunks': (1, 1)},
            'period': {'chunks': (1, 1, j)},
            'sig': {'chunks': (1, 1, j)},
            'coi': {'chunks': (1, 1, n)},
        }
        self._ds.to_zarr(data.model_coherence_path, compute=False, consolidated=False, zarr_format=3, encoding=encoding)

    def dt(self) -> float:
        return self._ds.attrs['dt']

    def compute(self, task: int = 0, ntasks: int = 1) -> None:
        timer = util.Timer()
        print('Initializing...')

        models = data.Models()

        mem_gb = util.get_peak_mem() / 1024**3

        with futures.ProcessPoolExecutor(max_workers=os.process_cpu_count()) as executor:
            fs = []
            all_models = models.get_models()
            all_transects = models.get_transects()
            models_slice = slice(None)
            transects_slice = slice(task, None, ntasks)
            if len(all_models) > len(all_transects):
                models_slice, transects_slice = transects_slice, models_slice
            for i, j in np.array(np.isnan(self._ds.period[models_slice, transects_slice, 0].values).nonzero()).T:
                if transects_slice == slice(None):
                    i = i * ntasks + task
                else:
                    j = j * ntasks + task
                print(f'queueing ({i}, {j}): {all_models[i]} {all_transects[j]}', flush=True)
                fs.append(executor.submit(worker.run, all_models[i], all_transects[j], (i, j)))

            for i, future in enumerate(futures.as_completed(fs)):
                try:
                    timer.stop()
                    elapsed = timer.ns
                    predicted_time = elapsed / (i+1) * len(fs)
                    predicted = util.Timer()
                    predicted._start = timer._start
                    predicted._end = predicted._start + predicted_time
                    if i % 10 == 0:
                        mem_gb = util.get_peak_mem() / 1024**3
                    print(f'Processing result {i+1}/{len(fs)}: peak mem={mem_gb:.1f} GiB time={timer}/{predicted}', flush=True)
    
                    r = future.result()
                except Exception as e:
                    # catch everything so we don't leave the workers to run without progress reports
                    print(e)

        print(f'\nFinished in {timer}')

    def get_range(self) -> tuple[np.datetime64, np.datetime64]:
        return self._range

    def get_models(self) -> list[str]:
        return self._ds.model.values.tolist()

    def get_transects(self) -> list[str]:
        return self._ds.transect.values.tolist()

    def get_wct(self, model: str, transect: str) -> wavelet.WCT | None:
        model_idx = self.get_models().index(model)
        transect_idx = self.get_transects().index(transect)

        coh = self._ds.coherence[model_idx, transect_idx, :, :].values
        n = len(coh[0][~np.isnan(coh[0])])
        if n == 0:
            return None
        coh = coh[~np.isnan(coh)].reshape((-1, n))
        j, n = coh.shape

        start = int(self._ds.start[model_idx, transect_idx].values)
        start = np.datetime64(0, 'D') + np.timedelta64(start, 'D')
        period = self._ds.period[model_idx, transect_idx, :j].values.astype(np.float64)
        sig = self._ds.sig[model_idx, transect_idx, :j].values
        coi = self._ds.coi[model_idx, transect_idx, :n].values.astype(np.float64)

        dt = self.dt()

        times = start + (np.arange(n) * np.timedelta64(dt, 'D'))
        times = pd.Series(times)

        return wavelet.WCT(dt, times, coh, coi, 1 / period, sig)
